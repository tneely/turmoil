//! Simulated filesystem types mirroring `std::fs`.
//!
//! Provides drop-in replacements for:
//! - [`File`] - Simulated file handle with `std::io::{Read, Write, Seek}` impls
//! - [`OpenOptions`] - Options for opening files
//! - [`Metadata`] - File metadata
//! - [`ReadDir`] - Iterator over directory entries
//! - [`DirEntry`] - An entry inside a directory
//! - [`FileType`] - Representation of a file type
//! - Free functions: [`canonicalize`], [`copy`], [`create_dir`], [`create_dir_all`],
//!   [`metadata`], [`read`], [`read_dir`], [`read_to_string`], [`remove_dir`],
//!   [`remove_dir_all`], [`remove_file`], [`rename`], [`sync_dir`], [`write`]
//!
//! # Durability Model
//!
//! This shim implements POSIX durability semantics. Understanding these semantics
//! is critical for writing crash-safe code.
//!
//! ## File Data vs Directory Entries
//!
//! In POSIX, a file has two distinct pieces that must be persisted:
//!
//! 1. **File data and inode** - The file's contents and metadata (size, timestamps)
//! 2. **Directory entry** - The name→inode mapping in the parent directory
//!
//! These are stored separately and synced separately:
//!
//! - [`File::sync_all`] syncs the file's data and inode
//! - [`sync_dir`] syncs the directory's entries
//!
//! ## Why Both Are Needed
//!
//! After creating a file and calling only `sync_all()`:
//! - The file's data exists on disk
//! - But the directory entry pointing to it may not
//! - On crash, the file is "orphaned" - data exists but is unreachable
//!
//! To make a new file fully durable:
//! ```ignore
//! // Create and write
//! let file = File::create("/data/important.txt")?;
//! file.write_all(b"critical data")?;
//!
//! // Sync file data
//! file.sync_all()?;
//!
//! // Sync directory entry - THIS IS THE PART MOST PEOPLE FORGET
//! sync_dir("/data")?;
//! ```
//!
//! ## Why `sync_dir` Exists
//!
//! In POSIX C, you sync a directory by opening it and calling `fsync()`:
//! ```c
//! int dir_fd = open("/data", O_RDONLY);
//! fsync(dir_fd);
//! ```
//!
//! However, Rust's `std::fs::File::open()` does not support opening directories -
//! it returns an error on most platforms. This means **there is no portable way
//! to sync a directory using `std::fs`**.
//!
//! The [`sync_dir`] function fills this gap for testing. In production code,
//! you would need platform-specific code (e.g., `libc::open` + `libc::fsync`
//! or the `nix` crate) to achieve the same effect.
//!
//! ## Testing Patterns
//!
//! For code that needs to be crash-safe, use conditional compilation:
//!
//! ```ignore
//! #[cfg(test)]
//! use turmoil::fs::shim::std::fs::{File, sync_dir};
//! #[cfg(not(test))]
//! use std::fs::File;
//!
//! fn write_durable(path: &str, data: &[u8]) -> std::io::Result<()> {
//!     let file = File::create(path)?;
//!     file.write_all(data)?;
//!     file.sync_all()?;
//!
//!     // In tests, this calls turmoil's sync_dir
//!     // In production, implement platform-specific directory sync
//!     #[cfg(test)]
//!     if let Some(parent) = std::path::Path::new(path).parent() {
//!         sync_dir(parent)?;
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! # Limitations
//!
//! This simulation has some differences from real filesystems:
//!
//! - **Permissions are informational only**: File modes (e.g., 0o644) are stored
//!   and returned by [`Metadata::permissions`], but they are **not enforced**.
//!   Opening a file with mode 0o000 will still succeed. This simplifies testing
//!   while still allowing code to set and read permission bits.
//!
//! - **All paths must be absolute**: Relative paths are treated as absolute from
//!   root. There is no concept of a current working directory.
//!
//! - **No file locking**: `flock()`/`fcntl()` locking is not implemented.
//!
//! - **Symlink loops**: Resolving a path follows at most 40 symlinks and then
//!   returns an error, matching POSIX ELOOP behavior.

use crate::tree::{Create, FileKind, Stat};
use crate::{stale_handle, FileOwner, Fs, FsContext, OpenFile, MAX_FILE_LEN};
use std::io::{Error, ErrorKind, Result, SeekFrom};
use std::os::fd::RawFd;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// O_DIRECT flag value for bypassing page cache in the simulation.
///
/// We use Linux's value (0x4000) as the canonical constant. macOS doesn't have
/// O_DIRECT (it uses fcntl F_NOCACHE), but since this is a simulation, we just
/// need a consistent value users can pass to indicate "bypass cache" behavior.
const O_DIRECT: i32 = 0x4000;

/// Creates a new directory at the provided path.
///
/// Like file creation, directory creation is not durable until the parent
/// directory is synced.
pub fn create_dir<P: AsRef<Path>>(path: P) -> Result<()> {
    DirBuilder::new().create(path)
}

/// Creates a directory and all of its parent components if they are missing.
pub fn create_dir_all<P: AsRef<Path>>(path: P) -> Result<()> {
    DirBuilder::new().recursive(true).create(path)
}

/// Removes an empty directory.
pub fn remove_dir<P: AsRef<Path>>(path: P) -> Result<()> {
    FsContext::current(|ctx| ctx.fs.tree.rmdir(path.as_ref(), ctx.now))
}

/// Syncs a directory, making its entries durable.
///
/// # Why This Function Exists
///
/// In POSIX, you sync a directory by opening it and calling `fsync()`:
/// ```c
/// int dir_fd = open("/data", O_RDONLY);
/// fsync(dir_fd);
/// ```
///
/// However, **Rust's `std::fs::File::open()` does not support opening directories** -
/// it returns an error on most platforms. This means there is no portable way to
/// sync a directory using `std::fs`.
///
/// This function provides the missing capability for turmoil's crash-consistency
/// testing. In production code, you would need platform-specific code (e.g.,
/// `libc::open` + `libc::fsync` or the `nix` crate) to achieve the same effect.
///
/// # What It Does
///
/// Equivalent to calling `fsync()` on a directory file descriptor.
/// It makes the following operations durable:
/// - File and directory creations within this directory
/// - File and directory deletions from this directory. A subdirectory goes
///   together with the renames that emptied it
/// - Renames into or out of this directory. A rename is durable as a whole:
///   the name leaves one directory as it arrives in the other. The
///   directories above its new place become durable as far as it takes for
///   a crash that finds the old directory to find the new one, so that the
///   name is never lost
/// - The directory's own creation, if it was recently created and its name
///   does not durably belong to something else
///
/// # POSIX Durability Model
///
/// A file has two distinct pieces that must be persisted separately:
/// 1. **File data and inode** - synced via [`File::sync_all`]
/// 2. **Directory entry** - synced via `sync_dir`
///
/// After `sync_all()` alone, the file's data exists on disk but the directory
/// entry may not. On crash, the file would be "orphaned" - unreachable because
/// no directory entry points to it.
///
/// # Example
///
/// ```ignore
/// use turmoil::fs::shim::std::fs::{File, sync_dir, create_dir};
///
/// // Create directory and make it durable
/// create_dir("/data")?;
/// sync_dir("/")?;  // /data entry is now durable
///
/// // Create file and make it fully durable
/// let file = File::create("/data/important.txt")?;
/// file.write_all_at(b"data", 0)?;
/// file.sync_all()?;      // File data is now durable
/// sync_dir("/data")?;    // File's directory entry is now durable
/// ```
///
/// # Testing Pattern
///
/// Use conditional compilation to call `sync_dir` in tests:
/// ```ignore
/// #[cfg(test)]
/// turmoil::fs::shim::std::fs::sync_dir(parent)?;
///
/// #[cfg(not(test))]
/// platform_specific_dir_sync(parent)?;
/// ```
pub fn sync_dir<P: AsRef<Path>>(path: P) -> Result<()> {
    FsContext::current(|ctx| ctx.fs.tree.sync_dir(path.as_ref()))
}

/// Removes a file from the filesystem.
///
/// Note: The deletion is not durable until the parent directory is synced.
pub fn remove_file<P: AsRef<Path>>(path: P) -> Result<()> {
    FsContext::current(|ctx| ctx.fs.tree.unlink(path.as_ref(), ctx.now))
}

/// Renames a file or directory.
///
/// This is atomic - if the destination exists, it will be replaced.
/// The rename is not durable until one of the parent directories is synced,
/// and a crash never leaves it half done.
pub fn rename<P: AsRef<Path>, Q: AsRef<Path>>(from: P, to: Q) -> Result<()> {
    let (from, to) = (from.as_ref(), to.as_ref());
    FsContext::current(|ctx| ctx.fs.tree.rename(from, to, ctx.now))
}

/// Returns `true` if the path points at an existing entity.
///
/// This function follows symbolic links, so if `path` is a symbolic link,
/// it returns whether the target exists.
///
/// # Example
///
/// ```ignore
/// use turmoil::fs::shim::std::fs::{exists, write, create_dir};
///
/// assert!(!exists("/data"));
/// create_dir("/data")?;
/// assert!(exists("/data"));
///
/// write("/data/file.txt", b"hello")?;
/// assert!(exists("/data/file.txt"));
/// ```
pub fn exists<P: AsRef<Path>>(path: P) -> bool {
    stat(path.as_ref(), true).is_ok()
}

fn stat(path: &Path, follow: bool) -> Result<Metadata> {
    FsContext::current(|ctx| {
        let ino = ctx.fs.tree.resolve(path, follow)?;
        Ok(Metadata(ctx.fs.tree.stat(ino)))
    })
}

/// Returns metadata for a file or directory.
///
/// This function follows symbolic links. Use [`symlink_metadata`] to get
/// metadata for the symlink itself.
pub fn metadata<P: AsRef<Path>>(path: P) -> Result<Metadata> {
    stat(path.as_ref(), true)
}

/// Returns an iterator over the entries within a directory.
///
/// The iterator yields instances of `Result<DirEntry>`. Each entry can be
/// inspected for its path, file name, and metadata.
pub fn read_dir<P: AsRef<Path>>(path: P) -> Result<ReadDir> {
    let entries = FsContext::current(|ctx| ctx.fs.tree.read_dir(path.as_ref()))?;
    let entries = entries.into_iter();
    Ok(ReadDir { entries })
}

/// Read the entire contents of a file into a bytes vector.
///
/// This is a convenience function for using [`File::open`] and reading the
/// entire file contents.
pub fn read<P: AsRef<Path>>(path: P) -> Result<Vec<u8>> {
    let file = File::open(path)?;
    let len = usize::try_from(file.len()?).map_err(|_| ErrorKind::FileTooLarge)?;
    let mut buf = Vec::new();
    buf.try_reserve_exact(len)
        .map_err(|_| ErrorKind::OutOfMemory)?;
    buf.resize(len, 0);
    if len > 0 {
        file.read_at_internal(&mut buf, 0)?;
    }
    Ok(buf)
}

/// Read the entire contents of a file into a string.
///
/// This is a convenience function for using [`File::open`] and reading the
/// entire file contents as a UTF-8 string.
pub fn read_to_string<P: AsRef<Path>>(path: P) -> Result<String> {
    let bytes = read(path)?;
    String::from_utf8(bytes).map_err(|e| Error::new(ErrorKind::InvalidData, e))
}

/// Write a slice as the entire contents of a file.
///
/// This function will create a file if it does not exist, and will entirely
/// replace its contents if it does.
pub fn write<P: AsRef<Path>, C: AsRef<[u8]>>(path: P, contents: C) -> Result<()> {
    let file = File::create(path)?;
    let contents = contents.as_ref();
    if !contents.is_empty() {
        file.write_at_internal(contents, 0)?;
    }
    Ok(())
}

/// Copies the contents of one file to another.
///
/// This function will overwrite the contents of `to`. Returns the number of
/// bytes copied.
///
/// Note: This does not copy file permissions or other metadata.
pub fn copy<P: AsRef<Path>, Q: AsRef<Path>>(from: P, to: Q) -> Result<u64> {
    let contents = read(from)?;
    let len = contents.len() as u64;
    write(to, contents)?;
    Ok(len)
}

/// Iterator over directory entries.
///
/// Returned by [`read_dir`].
pub struct ReadDir {
    entries: std::vec::IntoIter<std::path::PathBuf>,
}

impl Iterator for ReadDir {
    type Item = Result<DirEntry>;

    fn next(&mut self) -> Option<Self::Item> {
        self.entries.next().map(|path| Ok(DirEntry { path }))
    }
}

/// An entry inside a directory.
///
/// Returned by iterating over [`ReadDir`].
pub struct DirEntry {
    path: std::path::PathBuf,
}

impl DirEntry {
    /// Returns the full path to this entry.
    pub fn path(&self) -> std::path::PathBuf {
        self.path.clone()
    }

    /// Returns the file name of this entry.
    pub fn file_name(&self) -> std::ffi::OsString {
        self.path
            .file_name()
            .map(|s| s.to_owned())
            .unwrap_or_default()
    }

    /// Returns the metadata for this entry.
    ///
    /// This does not follow symlinks - if the entry is a symlink, it returns
    /// metadata for the symlink itself, not the target.
    pub fn metadata(&self) -> Result<Metadata> {
        symlink_metadata(&self.path)
    }

    /// Returns the file type for this entry.
    pub fn file_type(&self) -> Result<FileType> {
        // Use symlink_metadata to not follow symlinks
        Ok(symlink_metadata(&self.path)?.file_type())
    }
}

/// Representation of a file type.
///
/// Returned by [`DirEntry::file_type`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileType(FileKind);

impl FileType {
    /// Returns true if this file type is a directory.
    pub fn is_dir(&self) -> bool {
        self.0 == FileKind::Dir
    }

    /// Returns true if this file type is a regular file.
    pub fn is_file(&self) -> bool {
        self.0 == FileKind::File
    }

    /// Returns true if this file type is a symbolic link.
    pub fn is_symlink(&self) -> bool {
        self.0 == FileKind::Symlink
    }
}

/// A simulated file handle.
///
/// Drop-in replacement for `std::fs::File` that stores data in turmoil's
/// per-host simulated filesystem.
#[derive(Debug)]
pub struct File {
    /// The filesystem the file is on. Dropping the file closes its
    /// descriptor there, whichever filesystem the thread has entered.
    owner: FileOwner,
    /// File descriptor used to identify this file in the host's Fs.
    ///
    /// Allocated from a high range (see [`crate::SIM_FD_BASE`]) so it
    /// is safe to expose via [`AsRawFd`](std::os::fd::AsRawFd) — the
    /// io_uring shim resolves these back to files through the host's
    /// open-file table.
    fd: RawFd,
    /// Whether the file is in append mode
    append_mode: bool,
    /// Whether this file bypasses the page cache (O_DIRECT)
    direct_io: bool,
    /// Current cursor position for read()/write()
    cursor: u64,
}

impl File {
    /// Attempts to open a file in read-only mode.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<File> {
        OpenOptions::new().read(true).open(path)
    }

    /// Opens a file in write-only mode, creating it if it doesn't exist,
    /// and truncating it if it does.
    pub fn create<P: AsRef<Path>>(path: P) -> Result<File> {
        OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
    }

    /// Runs `f` on the filesystem the file is on. That has to be the one
    /// entered, since the descriptor means nothing on any other.
    pub(crate) fn context<R>(&self, f: impl FnOnce(FsContext<'_>) -> Result<R>) -> Result<R> {
        self.owner.entered()?.context(f)
    }

    /// Current length of the file.
    fn len(&self) -> Result<u64> {
        Ok(self.metadata()?.len())
    }

    /// Queries metadata about the file.
    pub fn metadata(&self) -> Result<Metadata> {
        self.context(|ctx| {
            let file = ctx.fs.open_file(self.fd).ok_or_else(stale_handle)?;
            Ok(Metadata(ctx.fs.tree.stat(file.ino)))
        })
    }

    /// Truncates or extends the file to the specified size.
    pub fn set_len(&self, size: u64) -> Result<()> {
        self.context(|ctx| ctx.fs.set_len(self.fd, size))
    }

    /// Attempts to sync all OS-internal metadata and data to disk.
    ///
    /// This is equivalent to calling `fsync()` on the file descriptor. It makes
    /// the file's **data and inode** durable, including:
    /// - File contents (all writes)
    /// - File size (truncate/extend operations)
    /// - File timestamps (mtime, ctime)
    ///
    /// # Important: Directory Entry Not Synced
    ///
    /// **This does NOT sync the directory entry.** In POSIX, a file's data/inode
    /// and its directory entry are separate. After `sync_all()`:
    /// - The file's data is durable on disk
    /// - But the directory entry pointing to it may not be
    /// - On crash, the file could be "orphaned" (data exists but unreachable)
    ///
    /// To make a newly created file fully durable, you must also call
    /// [`sync_dir`] on the parent directory.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use turmoil::fs::shim::std::fs::{File, sync_dir};
    ///
    /// let file = File::create("/data/file.txt")?;
    /// file.write_all(b"important")?;
    ///
    /// // Sync file data - necessary but not sufficient for new files
    /// file.sync_all()?;
    ///
    /// // Sync directory entry - required for new files to survive crash
    /// sync_dir("/data")?;
    /// ```
    ///
    /// # When Directory Sync Is Not Needed
    ///
    /// If overwriting an **existing** file that was previously synced, you only
    /// need `sync_all()` - the directory entry already exists and is durable.
    pub fn sync_all(&self) -> Result<()> {
        self.context(|ctx| ctx.fs.sync(self.fd))
    }

    /// Attempts to sync file data to disk, but not all metadata.
    ///
    /// This is equivalent to calling `fdatasync()` on the file descriptor.
    /// It syncs file contents and size, but may not sync all metadata.
    ///
    /// This leaves changes to permissions and timestamps pending.
    /// The directory entry is synced separately with [`sync_dir`].
    pub fn sync_data(&self) -> Result<()> {
        self.context(|ctx| ctx.fs.sync_data(self.fd))
    }

    /// Creates a new independently owned handle to the underlying file.
    ///
    /// The returned `File` is a reference to the same state that this object
    /// references. Both handles will read and write at independent cursors.
    pub fn try_clone(&self) -> Result<File> {
        let fd = self.context(|ctx| ctx.fs.dup(self.fd).ok_or_else(stale_handle))?;
        let file = File::from_parts(self.owner.clone(), fd, self.append_mode, self.direct_io);
        Ok(file)
    }

    /// Returns whether this file was opened with O_DIRECT (bypasses page cache).
    pub fn is_direct_io(&self) -> bool {
        self.direct_io
    }

    /// Returns the path the file was opened by. Renaming or removing the
    /// file does not change it.
    ///
    /// Returns `None` if the file handle is no longer valid.
    pub fn path(&self) -> Option<PathBuf> {
        let path = |ctx: FsContext<'_>| {
            let file = ctx.fs.open_file(self.fd).ok_or_else(stale_handle)?;
            Ok(file.path.clone())
        };
        self.context(path).ok()
    }

    /// Internal read implementation.
    pub(crate) fn read_at_internal(&self, buf: &mut [u8], offset: u64) -> Result<usize> {
        self.context(|ctx| ctx.fs.read_at(self.fd, buf, offset))
    }

    /// Internal write implementation.
    pub(crate) fn write_at_internal(&self, buf: &[u8], offset: u64) -> Result<usize> {
        self.context(|ctx| ctx.fs.write_at(self.fd, buf, offset))
    }

    /// Create a File from internal state (used by OpenOptions).
    fn from_parts(owner: FileOwner, fd: RawFd, append_mode: bool, direct_io: bool) -> Self {
        Self {
            owner,
            fd,
            append_mode,
            direct_io,
            cursor: 0,
        }
    }
}

impl std::io::Read for File {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
        let n = self.read_at_internal(buf, self.cursor)?;
        self.cursor += n as u64;
        Ok(n)
    }
}

impl std::io::Write for File {
    fn write(&mut self, buf: &[u8]) -> Result<usize> {
        let offset = if self.append_mode {
            self.len()?
        } else {
            self.cursor
        };
        let n = self.write_at_internal(buf, offset)?;
        self.cursor = offset + n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}

impl std::io::Seek for File {
    fn seek(&mut self, pos: SeekFrom) -> Result<u64> {
        let len = self.len()?;
        let new_pos = match pos {
            SeekFrom::Start(offset) => Some(offset),
            SeekFrom::End(offset) => len.checked_add_signed(offset),
            SeekFrom::Current(offset) => self.cursor.checked_add_signed(offset),
        };
        self.cursor = new_pos
            .filter(|&p| p <= MAX_FILE_LEN)
            .ok_or(ErrorKind::InvalidInput)?;
        Ok(self.cursor)
    }
}

impl Drop for File {
    fn drop(&mut self) {
        self.owner.close(self.fd);
    }
}

/// Exposes the simulated file descriptor.
///
/// The returned [`RawFd`] is allocated from a high range
/// ([`crate::SIM_FD_BASE`]) so it cannot collide with any real OS
/// fd. It is meaningful only inside the turmoil simulation that minted
/// it; passing it to a real syscall will reliably fail with `EBADF`.
///
/// This impl exists so consumers can hand the fd to the
/// [`crate::io_uring`] shim.
impl std::os::fd::AsRawFd for File {
    fn as_raw_fd(&self) -> RawFd {
        self.fd
    }
}

/// Metadata information about a file.
#[derive(Debug, Clone)]
pub struct Metadata(Stat);

impl Metadata {
    /// Returns the size of the file in bytes.
    #[allow(clippy::len_without_is_empty)] // is_empty() is not part of std::fs::Metadata
    pub fn len(&self) -> u64 {
        self.0.len
    }

    /// Returns true if this metadata is for a regular file.
    pub fn is_file(&self) -> bool {
        self.file_type().is_file()
    }

    /// Returns true if this metadata is for a directory.
    pub fn is_dir(&self) -> bool {
        self.file_type().is_dir()
    }

    /// Returns true if this metadata is for a symbolic link.
    pub fn is_symlink(&self) -> bool {
        self.file_type().is_symlink()
    }

    /// Returns the file type for this metadata.
    pub fn file_type(&self) -> FileType {
        FileType(self.0.kind)
    }

    /// Returns the last modification time.
    ///
    /// This is updated when file contents are modified (writes, truncate).
    pub fn modified(&self) -> Result<SystemTime> {
        Ok(UNIX_EPOCH + self.0.meta.mtime)
    }

    /// Returns the last access time.
    ///
    /// Note: turmoil simulates noatime behavior, so this always returns
    /// the creation time (atime is not updated on reads).
    pub fn accessed(&self) -> Result<SystemTime> {
        // noatime: return creation time instead of tracking access time
        Ok(UNIX_EPOCH + self.0.meta.crtime)
    }

    /// Returns the creation time (birthtime).
    ///
    /// Note: Not all platforms support creation time. On platforms that don't
    /// support it, `std::fs::Metadata::created()` returns an error. Our
    /// simulation always tracks it.
    pub fn created(&self) -> Result<SystemTime> {
        Ok(UNIX_EPOCH + self.0.meta.crtime)
    }

    /// Returns the permissions of the file.
    pub fn permissions(&self) -> Permissions {
        Permissions::from_mode(self.0.meta.mode)
    }
}

impl std::os::unix::fs::MetadataExt for Metadata {
    fn dev(&self) -> u64 {
        0 // Simulated device
    }

    fn ino(&self) -> u64 {
        self.0.ino
    }

    fn mode(&self) -> u32 {
        // Include file type bits: regular file (0o100000), directory (0o40000), symlink (0o120000)
        let type_bits = match self.0.kind {
            FileKind::Symlink => 0o120000,
            FileKind::Dir => 0o040000,
            FileKind::File => 0o100000,
        };
        type_bits | self.0.meta.mode
    }

    fn nlink(&self) -> u64 {
        self.0.nlink
    }

    fn uid(&self) -> u32 {
        0 // Simulated uid
    }

    fn gid(&self) -> u32 {
        0 // Simulated gid
    }

    fn rdev(&self) -> u64 {
        0 // Not a device file
    }

    fn size(&self) -> u64 {
        self.0.len
    }

    fn atime(&self) -> i64 {
        // noatime: return creation time
        self.0.meta.crtime.as_secs() as i64
    }

    fn atime_nsec(&self) -> i64 {
        self.0.meta.crtime.subsec_nanos() as i64
    }

    fn mtime(&self) -> i64 {
        self.0.meta.mtime.as_secs() as i64
    }

    fn mtime_nsec(&self) -> i64 {
        self.0.meta.mtime.subsec_nanos() as i64
    }

    fn ctime(&self) -> i64 {
        self.0.meta.ctime.as_secs() as i64
    }

    fn ctime_nsec(&self) -> i64 {
        self.0.meta.ctime.subsec_nanos() as i64
    }

    fn blksize(&self) -> u64 {
        4096 // Simulated block size
    }

    fn blocks(&self) -> u64 {
        self.0.blocks
    }
}

/// Options and flags for opening files.
///
/// Drop-in replacement for `std::fs::OpenOptions`.
#[derive(Debug, Clone)]
pub struct OpenOptions {
    read: bool,
    write: bool,
    append: bool,
    truncate: bool,
    create: bool,
    create_new: bool,
    /// Unix mode bits for file creation (default: 0o644)
    mode: u32,
    /// Custom flags (for OpenOptionsExt compatibility)
    custom_flags: i32,
    /// Direct I/O mode - bypasses page cache (turmoil-specific)
    direct_io: bool,
}

impl Default for OpenOptions {
    fn default() -> Self {
        Self {
            read: false,
            write: false,
            append: false,
            truncate: false,
            create: false,
            create_new: false,
            mode: 0o644,
            custom_flags: 0,
            direct_io: false,
        }
    }
}

impl OpenOptions {
    /// Creates a blank new set of options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the option for read access.
    pub fn read(&mut self, read: bool) -> &mut Self {
        self.read = read;
        self
    }

    /// Sets the option for write access.
    pub fn write(&mut self, write: bool) -> &mut Self {
        self.write = write;
        self
    }

    /// Sets the option for append mode.
    pub fn append(&mut self, append: bool) -> &mut Self {
        self.append = append;
        self
    }

    /// Sets the option for truncating a file.
    pub fn truncate(&mut self, truncate: bool) -> &mut Self {
        self.truncate = truncate;
        self
    }

    /// Sets the option for creating a new file.
    pub fn create(&mut self, create: bool) -> &mut Self {
        self.create = create;
        self
    }

    /// Sets the option for creating a new file, failing if it already exists.
    pub fn create_new(&mut self, create_new: bool) -> &mut Self {
        self.create_new = create_new;
        self
    }

    /// Sets the option for direct I/O (bypasses page cache).
    ///
    /// This is a turmoil-specific extension that works on all platforms.
    /// On real systems, you would use `custom_flags(libc::O_DIRECT)` on Linux
    /// or `fcntl(F_NOCACHE)` on macOS.
    ///
    /// When enabled, reads and writes bypass the simulated page cache,
    /// always incurring full I/O latency. Additionally, all reads and writes
    /// enforce alignment requirements: the buffer pointer, file offset, and
    /// buffer length must each be a multiple of `direct_io_alignment` (default
    /// 512 bytes). Misaligned operations fail with `InvalidInput`, matching
    /// the `EINVAL` behavior of the Linux kernel.
    ///
    /// # Durability Note
    ///
    /// In this simulation, O_DIRECT only affects **latency**, not **durability**.
    /// Writes still require `sync_all()` to become crash-safe, regardless of
    /// whether O_DIRECT is used. This is actually realistic: on real systems,
    /// O_DIRECT bypasses the OS page cache but data can still sit in the drive's
    /// volatile write cache. Only `fsync()` guarantees data reaches persistent
    /// media.
    ///
    /// In practice, O_DIRECT writes are "closer" to disk (smaller device buffer
    /// vs large OS page cache), but this simulation takes the conservative
    /// approach of treating all non-synced writes equally for crash testing.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use turmoil::fs::shim::std::fs::OpenOptions;
    ///
    /// let file = OpenOptions::new()
    ///     .read(true)
    ///     .direct_io(true)
    ///     .open("/data/file.txt")?;
    /// ```
    pub fn direct_io(&mut self, direct_io: bool) -> &mut Self {
        self.direct_io = direct_io;
        self
    }

    /// Sets the mode bits for file creation.
    ///
    /// This is used internally by the OpenOptionsExt trait implementation.
    pub(crate) fn set_mode(&mut self, mode: u32) {
        self.mode = mode;
    }

    /// Sets custom flags for file opening.
    ///
    /// This is used internally by the OpenOptionsExt trait implementation.
    /// Note: custom_flags are currently stored but not used in the simulation.
    pub(crate) fn set_custom_flags(&mut self, flags: i32) {
        self.custom_flags = flags;
    }

    /// Opens a file at `path` with the options specified by `self`.
    pub fn open<P: AsRef<Path>>(&self, path: P) -> Result<File> {
        let path = path.as_ref();
        let create = match (self.create_new, self.create) {
            (true, _) => Create::New(self.mode),
            (false, true) => Create::IfAbsent(self.mode),
            (false, false) => Create::No,
        };
        // Check for direct I/O (bypasses page cache)
        // Can be set via direct_io() method or custom_flags(O_DIRECT)
        let direct_io = self.direct_io || (self.custom_flags & O_DIRECT) != 0;
        let writable = self.write || self.append;

        let fs = Fs::current();
        let fd = fs.context(|ctx| {
            let ino = ctx.fs.tree.open(path, create, ctx.now)?;

            // Handle truncate (pending until sync, like other operations)
            if self.truncate && self.write {
                ctx.fs.resize(ino, 0);
            }

            let file = OpenFile {
                ino,
                direct_io,
                readable: self.read,
                writable,
                path: path.to_owned(),
            };
            Ok::<_, Error>(ctx.fs.open(file))
        })?;
        let owner = FileOwner::new(&fs);
        Ok(File::from_parts(owner, fd, self.append, direct_io))
    }
}

/// Representation of file permissions.
///
/// Drop-in replacement for `std::fs::Permissions`.
///
/// # Note
///
/// Permissions are **informational only** in turmoil's simulation. The mode bits
/// are stored and can be queried, but they are not enforced during file operations.
/// For example, a file with mode 0o000 can still be opened for reading and writing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Permissions {
    mode: u32,
}

impl Permissions {
    /// Returns true if these permissions describe a readonly (unwritable) file.
    pub fn readonly(&self) -> bool {
        // Check if write bits are not set for owner
        (self.mode & 0o200) == 0
    }

    /// Modifies the readonly flag for this set of permissions.
    pub fn set_readonly(&mut self, readonly: bool) {
        if readonly {
            // Clear write bits for owner, group, others
            self.mode &= !0o222;
        } else {
            // Set write bit for owner
            self.mode |= 0o200;
        }
    }

    /// Create permissions from mode bits.
    pub fn from_mode(mode: u32) -> Self {
        Self { mode }
    }

    /// Returns the underlying raw mode bits.
    pub fn mode(&self) -> u32 {
        self.mode
    }
}

impl std::os::unix::fs::PermissionsExt for Permissions {
    fn mode(&self) -> u32 {
        self.mode
    }

    fn set_mode(&mut self, mode: u32) {
        self.mode = mode;
    }

    fn from_mode(mode: u32) -> Self {
        Permissions { mode }
    }
}

/// Set the permissions of a file or directory.
pub fn set_permissions<P: AsRef<Path>>(path: P, perm: Permissions) -> Result<()> {
    FsContext::current(|ctx| ctx.fs.tree.set_mode(path.as_ref(), perm.mode, ctx.now))
}

/// Creates a new symbolic link on the filesystem.
///
/// The `link` path will be a symbolic link pointing to the `original` path.
pub fn symlink<P: AsRef<Path>, Q: AsRef<Path>>(original: P, link: Q) -> Result<()> {
    let (original, link) = (original.as_ref(), link.as_ref());
    FsContext::current(|ctx| ctx.fs.tree.symlink(original, link, ctx.now))
}

/// Reads the target of a symbolic link.
pub fn read_link<P: AsRef<Path>>(path: P) -> Result<std::path::PathBuf> {
    FsContext::current(|ctx| ctx.fs.tree.read_link(path.as_ref()))
}

/// Returns metadata for a path without following symlinks.
///
/// If `path` is a symlink, this returns metadata for the symlink itself,
/// not the target.
pub fn symlink_metadata<P: AsRef<Path>>(path: P) -> Result<Metadata> {
    stat(path.as_ref(), false)
}

/// Creates a hard link on the filesystem.
///
/// The `link` path will be a hard link to the `original` file.
/// Both paths will refer to the same file content.
pub fn hard_link<P: AsRef<Path>, Q: AsRef<Path>>(original: P, link: Q) -> Result<()> {
    let (original, link) = (original.as_ref(), link.as_ref());
    FsContext::current(|ctx| ctx.fs.tree.hard_link(original, link, ctx.now))
}

/// A builder for creating directories with options.
///
/// Drop-in replacement for `std::fs::DirBuilder`.
#[derive(Debug, Default)]
pub struct DirBuilder {
    recursive: bool,
    mode: u32,
}

impl DirBuilder {
    /// Creates a new DirBuilder with default options.
    pub fn new() -> Self {
        Self {
            recursive: false,
            mode: 0o755,
        }
    }

    /// Sets the option for recursive directory creation.
    pub fn recursive(&mut self, recursive: bool) -> &mut Self {
        self.recursive = recursive;
        self
    }

    /// Creates the directory at the given path.
    pub fn create<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let (path, mode) = (path.as_ref(), self.mode);
        FsContext::current(|ctx| match self.recursive {
            true => ctx.fs.tree.mkdir_all(path, mode, ctx.now),
            false => ctx.fs.tree.mkdir(path, mode, ctx.now),
        })
    }
}

impl std::os::unix::fs::DirBuilderExt for DirBuilder {
    fn mode(&mut self, mode: u32) -> &mut Self {
        self.mode = mode;
        self
    }
}

/// Returns the canonical, absolute form of a path with all intermediate
/// components normalized and symbolic links resolved.
///
/// # Errors
///
/// Returns an error if the path does not exist or if any component of the
/// path is not a directory (except the final component).
///
/// # Example
///
/// ```ignore
/// use turmoil::fs::shim::std::fs::{canonicalize, create_dir, symlink, write};
///
/// create_dir("/data")?;
/// write("/data/file.txt", b"hello")?;
/// symlink("/data/file.txt", "/link")?;
///
/// // Resolves symlink to actual path
/// assert_eq!(canonicalize("/link")?, PathBuf::from("/data/file.txt"));
///
/// // Normalizes . and ..
/// assert_eq!(canonicalize("/data/../data/./file.txt")?, PathBuf::from("/data/file.txt"));
/// ```
pub fn canonicalize<P: AsRef<Path>>(path: P) -> Result<std::path::PathBuf> {
    FsContext::current(|ctx| ctx.fs.tree.canonicalize(path.as_ref()))
}

/// Removes a directory and all of its contents.
///
/// This function recursively removes all files and subdirectories within
/// the specified directory, then removes the directory itself.
///
/// # Errors
///
/// Returns an error if:
/// - The path does not exist
/// - The path is not a directory
/// - Any file or subdirectory cannot be removed
///
/// # Example
///
/// ```ignore
/// use turmoil::fs::shim::std::fs::{create_dir_all, write, remove_dir_all, metadata};
///
/// // Create nested structure
/// create_dir_all("/data/subdir")?;
/// write("/data/file1.txt", b"hello")?;
/// write("/data/subdir/file2.txt", b"world")?;
///
/// // Remove everything
/// remove_dir_all("/data")?;
///
/// // Directory no longer exists
/// assert!(metadata("/data").is_err());
/// ```
pub fn remove_dir_all<P: AsRef<Path>>(path: P) -> Result<()> {
    FsContext::current(|ctx| ctx.fs.tree.remove_dir_all(path.as_ref(), ctx.now))
}
