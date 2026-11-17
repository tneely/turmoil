//! The inode table and namespace behind [`FsState`](crate::FsState).
//!
//! Every file, directory and symlink is an inode, and a directory entry is a
//! [`Link`] to one. A link is only a reference: a rename moves it, a hard
//! link adds another, and an open file holds the inode itself.
//!
//! Whatever syncs on its own is stored as a [`Versioned`] pair. `live` is what
//! the application sees and `durable` is what a crash leaves behind: syncing
//! copies live to durable, and crashing copies durable back to live.

use indexmap::{IndexMap, IndexSet};
use rand::{Rng, RngCore};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{OsStr, OsString};
use std::io::{Error, ErrorKind, Result};
use std::iter;
use std::mem;
use std::os::unix::ffi::OsStrExt;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::Arc;
use std::time::Duration;

/// Identity of a file, directory or symlink. A filesystem never reuses one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Ino(u64);

/// Identity of a directory entry, which a rename moves rather than replaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct LinkId(u64);

const ROOT: Ino = Ino(0);

/// Symlinks one path resolution may follow, as on Linux.
const MAX_SYMLINK_HOPS: usize = 40;

/// Bytes of a file that one chunk holds.
const CHUNK: u64 = 4096;

type Entries = IndexMap<OsString, LinkId>;

/// The names a sync is about to make durable.
type Plan = IndexSet<LinkId>;

#[derive(Clone, Copy)]
enum View {
    Live,
    Durable,
}

#[derive(Default)]
struct Versioned<T> {
    live: T,
    durable: T,
}

impl<T: Clone> Versioned<T> {
    fn new(value: T) -> Self {
        let (live, durable) = (value.clone(), value);
        Self { live, durable }
    }

    fn sync(&mut self) {
        self.durable.clone_from(&self.live);
    }

    fn crash(&mut self) {
        self.live.clone_from(&self.durable);
    }
}

impl<T> Versioned<T> {
    fn get(&self, view: View) -> &T {
        match view {
            View::Live => &self.live,
            View::Durable => &self.durable,
        }
    }

    fn get_mut(&mut self, view: View) -> &mut T {
        match view {
            View::Live => &mut self.live,
            View::Durable => &mut self.durable,
        }
    }
}

/// Inode attributes. Times are simulated durations since the epoch.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Meta {
    pub(crate) mode: u32,
    pub(crate) crtime: Duration,
    pub(crate) mtime: Duration,
    pub(crate) ctime: Duration,
}

impl Meta {
    fn new(mode: u32, now: Duration) -> Self {
        Self {
            mode,
            crtime: now,
            mtime: now,
            ctime: now,
        }
    }

    /// Stamps a change to the inode's content.
    fn modified(&mut self, now: Duration) {
        self.mtime = now;
        self.ctime = now;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FileKind {
    File,
    Dir,
    Symlink,
}

/// A snapshot of an inode's live attributes.
#[derive(Debug, Clone)]
pub(crate) struct Stat {
    pub(crate) ino: u64,
    pub(crate) kind: FileKind,
    pub(crate) len: u64,
    pub(crate) blocks: u64,
    pub(crate) nlink: u64,
    pub(crate) meta: Meta,
}

/// Whether [`Tree::open`] creates the file, and with which mode.
#[derive(Clone, Copy)]
pub(crate) enum Create {
    /// The file must exist.
    No,
    /// The file is created unless it exists.
    IfAbsent(u32),
    /// The file is created and must not exist.
    New(u32),
}

/// A change to a file's bytes.
pub(crate) enum Change {
    /// Bytes written at an offset.
    Write(u64, Vec<u8>),
    /// Truncation or zero-extension to a length.
    SetLen(u64),
}

/// Splits `len` bytes at `offset` into the part that falls in each chunk:
/// the chunk's index, where the part starts in the chunk, and its length.
fn spans(offset: u64, len: usize) -> impl Iterator<Item = (u64, usize, usize)> {
    let (mut at, end) = (offset, offset + len as u64);
    iter::from_fn(move || {
        let n = (CHUNK - at % CHUNK).min(end - at);
        let span = (at / CHUNK, (at % CHUNK) as usize, n as usize);
        at += n;
        (n > 0).then_some(span)
    })
}

fn growth(offset: u64, len: usize, held: impl Fn(u64) -> usize) -> u64 {
    spans(offset, len)
        .map(|(index, start, n)| (start + n).saturating_sub(held(index)) as u64)
        .sum()
}

/// A file's bytes, in chunks. A chunk ends with the last byte written to
/// it, and whatever is missing reads as zeros, so a hole that covers whole
/// chunks takes neither memory nor room on the simulated disk.
#[derive(Clone, Default)]
struct Bytes {
    chunks: BTreeMap<u64, Vec<u8>>,
    len: u64,
    /// Bytes the chunks hold between them.
    stored: u64,
    /// Allocated 512-byte sectors. Separate chunks cannot share a sector.
    blocks: u64,
}

fn sectors(bytes: usize) -> u64 {
    bytes.div_ceil(512) as u64
}

impl Bytes {
    fn apply(&mut self, change: &Change) {
        match change {
            Change::Write(offset, data) => self.write(*offset, data),
            Change::SetLen(len) => self.set_len(*len),
        }
    }

    fn write(&mut self, offset: u64, data: &[u8]) {
        let mut written = 0;
        for (index, start, n) in spans(offset, data.len()) {
            let chunk = self.chunks.entry(index).or_default();
            let old = chunk.len();
            let end = old.max(start + n);
            chunk.resize(end, 0);
            chunk[start..start + n].copy_from_slice(&data[written..written + n]);
            self.stored += (end - old) as u64;
            self.blocks += sectors(end) - sectors(old);
            written += n;
        }
        self.len = self.len.max(offset + data.len() as u64);
    }

    /// Bytes a write of `len` at `offset` would add to those stored.
    fn growth(&self, offset: u64, len: usize) -> u64 {
        growth(offset, len, |i| self.chunks.get(&i).map_or(0, Vec::len))
    }

    fn set_len(&mut self, len: u64) {
        let (index, keep) = (len / CHUNK, (len % CHUNK) as usize);
        let mut cut = self.chunks.split_off(&index);
        for chunk in cut.values() {
            self.stored -= chunk.len() as u64;
            self.blocks -= sectors(chunk.len());
        }
        // The chunk the new end falls in keeps what lies before the end.
        if let Some(mut chunk) = cut.remove(&index).filter(|_| keep > 0) {
            chunk.truncate(keep);
            self.stored += chunk.len() as u64;
            self.blocks += sectors(chunk.len());
            self.chunks.insert(index, chunk);
        }
        self.len = len;
    }

    fn read(&self, buf: &mut [u8], offset: u64) -> usize {
        let len = self.len.saturating_sub(offset).min(buf.len() as u64) as usize;
        let mut read = 0;
        for (index, start, n) in spans(offset, len) {
            let chunk = self.chunks.get(&index).map_or(&[][..], Vec::as_slice);
            let held = chunk.get(start..).unwrap_or_default();
            let copied = n.min(held.len());
            buf[read..read + copied].copy_from_slice(&held[..copied]);
            buf[read + copied..read + n].fill(0);
            read += n;
        }
        len
    }
}

/// Once a truncate can roll back independently of torn writes, each chunk
/// can recover its largest allocation since the last sync.
struct Reservation {
    ends: BTreeMap<u64, usize>,
    stored: u64,
}

impl Reservation {
    fn from_bytes(bytes: &Bytes) -> Self {
        let ends = bytes.chunks.iter().map(|(&i, chunk)| (i, chunk.len()));
        Self {
            ends: ends.collect(),
            stored: bytes.stored,
        }
    }

    fn grow(&mut self, offset: u64, len: usize) {
        for (i, start, n) in spans(offset, len) {
            let end = self.ends.entry(i).or_default();
            self.stored += (start + n).saturating_sub(*end) as u64;
            *end = (*end).max(start + n);
        }
    }
}

#[derive(Default)]
struct FileContent {
    bytes: Versioned<Bytes>,
    /// Changes to the live bytes since the last sync, oldest first.
    unsynced: Vec<Change>,
    reservation: Option<Reservation>,
}

impl FileContent {
    fn change(&mut self, change: Change, reserve_torn_writes: bool) {
        let live = &self.bytes.live;
        match (&mut self.reservation, &change) {
            (None, Change::SetLen(len)) if reserve_torn_writes && *len < live.len => {
                self.reservation = Some(Reservation::from_bytes(live));
            }
            (Some(held), Change::Write(offset, data)) => held.grow(*offset, data.len()),
            _ => {}
        }
        self.bytes.live.apply(&change);
        self.unsynced.push(change);
    }

    fn sync(&mut self) {
        for change in self.unsynced.drain(..) {
            self.bytes.durable.apply(&change);
        }
        self.reservation = None;
    }

    /// Bytes the file takes up on the disk. A file shrunk since its last
    /// sync still holds what a crash would bring back.
    fn stored(&self) -> u64 {
        match &self.reservation {
            Some(held) => held.stored,
            None => self.bytes.live.stored.max(self.bytes.durable.stored),
        }
    }

    fn growth(&self, offset: u64, len: usize) -> u64 {
        if let Some(held) = &self.reservation {
            return growth(offset, len, |i| held.ends.get(&i).copied().unwrap_or(0));
        }
        let live = &self.bytes.live;
        (live.stored + live.growth(offset, len)).saturating_sub(self.stored())
    }

    /// Drops unsynced changes. With a `block_size`, a leading run of each
    /// unsynced write's blocks may have reached the disk first.
    fn crash(&mut self, block_size: Option<u64>, rng: &mut dyn RngCore) {
        // With nothing unsynced the live bytes are the durable ones already.
        if self.unsynced.is_empty() {
            return;
        }
        for change in self.unsynced.drain(..) {
            let (Some(block_size), Change::Write(offset, data)) = (block_size, change) else {
                continue;
            };
            let blocks = (data.len() as u64).div_ceil(block_size) as usize;
            let survivors = rng.random_range(0..=blocks) as u64;
            let kept = (survivors * block_size).min(data.len() as u64) as usize;
            if kept > 0 {
                self.bytes.durable.write(offset, &data[..kept]);
            }
        }
        self.bytes.crash();
        self.reservation = None;
    }
}

#[derive(Default)]
struct Dir {
    entries: Versioned<Entries>,
    /// Names the live and durable entries disagree on, which are all that a
    /// sync of the directory has to look at.
    unsynced: IndexSet<OsString>,
    /// The entry that names this directory. Only the root has none.
    link: Option<LinkId>,
    /// Departures can outlive this parent. Only syncing it commits them.
    next_sync: Option<Arc<AtomicBool>>,
    /// Earlier names that a sync of this directory could still make durable.
    departures: Vec<Departure>,
}

struct Departure {
    slot: Slot,
    parent_synced: Arc<AtomicBool>,
}

enum Kind {
    File(FileContent),
    Dir(Dir),
    Symlink(PathBuf),
}

struct Inode {
    kind: Kind,
    meta: Versioned<Meta>,
    /// Directory entries naming this inode.
    links: Versioned<u32>,
    /// File descriptors open on this inode.
    opens: u32,
}

impl Inode {
    fn new(kind: Kind, mode: u32, now: Duration) -> Self {
        Self {
            kind,
            meta: Versioned::new(Meta::new(mode, now)),
            links: Versioned::default(),
            opens: 0,
        }
    }
}

/// A name in a directory.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Slot {
    dir: Ino,
    name: OsString,
}

/// A directory entry. A rename moves where it lives and leaves where it is
/// durable for a sync to catch up, so a crash finds the name in one place or
/// the other, never in both and never in neither.
struct Link {
    ino: Ino,
    slot: Versioned<Option<Slot>>,
    overwrites: Overwrites,
}

/// Persisting either half of an overwrite requires the other half.
#[derive(Default)]
struct Overwrites {
    victims: BTreeSet<LinkId>,
    replaced_by: Option<LinkId>,
}

impl Link {
    /// An entry for `ino` that has yet to be placed in a directory.
    fn new(ino: Ino) -> Self {
        let slot = Versioned::default();
        Self {
            ino,
            slot,
            overwrites: Overwrites::default(),
        }
    }

    fn is_synced(&self) -> bool {
        self.slot.live == self.slot.durable
    }
}

/// Where a path leads.
struct Walk<'a> {
    /// The inodes passed through, from the root to the last one that exists.
    inos: Vec<Ino>,
    /// The name each inode after the root was reached by, then the final
    /// name if nothing exists there.
    names: Vec<&'a OsStr>,
    /// Whether the path ends in a name, rather than at the root, `.` or `..`.
    named: bool,
    /// A trailing slash permits a vacant final name only for a directory.
    requires_dir: bool,
}

impl Walk<'_> {
    fn from_root(requires_dir: bool) -> Self {
        Self {
            inos: vec![ROOT],
            names: Vec::new(),
            named: false,
            requires_dir,
        }
    }

    fn top(&self) -> Ino {
        self.inos[self.inos.len() - 1]
    }

    fn is_vacant(&self) -> bool {
        self.names.len() == self.inos.len()
    }

    /// The inode the path names.
    fn ino(&self) -> Result<Ino> {
        if self.is_vacant() {
            return Err(ErrorKind::NotFound.into());
        }
        Ok(self.top())
    }

    /// The directories from the root down to the one holding the final name.
    fn dirs(&self) -> &[Ino] {
        &self.inos[..self.names.len()]
    }

    /// The directory entry the path names. The root has none, and a path
    /// that ends in `..` names a directory but not the entry for it.
    fn slot(&self) -> Option<Slot> {
        let (&dir, name) = (self.dirs().last()?, self.names.last()?);
        let name = name.to_os_string();
        self.named.then_some(Slot { dir, name })
    }

    /// The path with every `.`, `..` and symlink resolved.
    fn path(&self) -> PathBuf {
        let mut path = PathBuf::from("/");
        path.extend(&self.names);
        path
    }
}

fn trailing_slash(path: &Path) -> bool {
    path.as_os_str().as_bytes().ends_with(b"/")
}

/// `Path::components` normalizes away a terminal `.` that must still be
/// looked up in a directory, including at the end of a symlink target.
fn components(path: &Path) -> impl DoubleEndedIterator<Item = Component<'_>> {
    let bytes = path.as_os_str().as_bytes();
    let last = bytes.rsplit(|&b| b == b'/').find(|part| !part.is_empty());
    let parts = path.components();
    let dot = last == Some(b".") && parts.clone().next_back() != Some(Component::CurDir);
    parts.chain(dot.then_some(Component::CurDir))
}

pub(crate) struct Tree {
    inodes: IndexMap<Ino, Inode>,
    links: IndexMap<LinkId, Link>,
    next_ino: u64,
    next_link: u64,
    reserve_torn_writes: bool,
    /// Files that have lost their last name and descriptor since
    /// [`Tree::take_unreadable`], so that nothing can read them any more.
    unreadable: IndexSet<Ino>,
}

impl Tree {
    pub(crate) fn new(reserve_torn_writes: bool) -> Self {
        let mut root = Inode::new(Kind::Dir(Dir::default()), 0o755, Duration::ZERO);
        // No entry names the root, so it holds a link to itself to stay alive.
        root.links = Versioned::new(1);
        Self {
            inodes: IndexMap::from([(ROOT, root)]),
            links: IndexMap::new(),
            next_ino: ROOT.0 + 1,
            next_link: 0,
            reserve_torn_writes,
            unreadable: IndexSet::new(),
        }
    }

    fn is_dir(&self, ino: Ino) -> bool {
        matches!(self.inodes[&ino].kind, Kind::Dir(_))
    }

    fn dir(&self, ino: Ino) -> Result<&Dir> {
        match &self.inodes[&ino].kind {
            Kind::Dir(dir) => Ok(dir),
            _ => Err(ErrorKind::NotADirectory.into()),
        }
    }

    fn dir_mut(&mut self, ino: Ino) -> &mut Dir {
        match &mut self.inodes[&ino].kind {
            Kind::Dir(dir) => dir,
            _ => unreachable!("{ino:?} is not a directory"),
        }
    }

    /// The entry `view` has in `slot`.
    fn entry(&self, slot: &Slot, view: View) -> Option<LinkId> {
        let entries = self.dir(slot.dir).ok()?.entries.get(view);
        entries.get(&slot.name).copied()
    }

    /// Follows `path` from the root through the live namespace. Relative
    /// paths start at the root too. A symlink in the final position is
    /// followed only if `follow`.
    fn walk<'a>(&'a self, path: &'a Path, follow: bool) -> Result<Walk<'a>> {
        if path.as_os_str().is_empty() {
            return Err(ErrorKind::NotFound.into());
        }
        let mut walk = Walk::from_root(trailing_slash(path));
        let mut rest: Vec<_> = components(path).rev().collect();
        let mut hops = 0;
        while let Some(component) = rest.pop() {
            let entries = &self.dir(walk.top())?.entries.live;
            walk.named = matches!(component, Component::Normal(_));
            let name = match component {
                Component::Prefix(_) | Component::RootDir => {
                    walk.inos.truncate(1);
                    walk.names.clear();
                    continue;
                }
                Component::CurDir => continue,
                Component::ParentDir => {
                    if walk.inos.len() > 1 {
                        walk.inos.pop();
                        walk.names.pop();
                    }
                    continue;
                }
                Component::Normal(name) => name,
            };
            let Some(id) = entries.get(name) else {
                if rest.is_empty() {
                    walk.names.push(name);
                    break;
                }
                return Err(ErrorKind::NotFound.into());
            };
            let ino = self.links[id].ino;
            match &self.inodes[&ino].kind {
                Kind::Symlink(target) if follow || !rest.is_empty() => {
                    hops += 1;
                    if hops > MAX_SYMLINK_HOPS {
                        return Err(Error::other("too many levels of symbolic links"));
                    }
                    if rest.is_empty() {
                        walk.requires_dir |= trailing_slash(target);
                    }
                    rest.extend(components(target).rev());
                }
                _ => {
                    walk.inos.push(ino);
                    walk.names.push(name);
                }
            }
        }
        if walk.requires_dir && !walk.is_vacant() {
            self.dir(walk.top())?;
        }
        Ok(walk)
    }

    /// The inode `path` names.
    pub(crate) fn resolve(&self, path: &Path, follow: bool) -> Result<Ino> {
        self.walk(path, follow || trailing_slash(path))?.ino()
    }

    /// The unused directory entry `path` names.
    fn vacancy(&self, path: &Path, kind: FileKind) -> Result<Slot> {
        let walk = self.walk(path, false)?;
        if walk.requires_dir && kind != FileKind::Dir && walk.is_vacant() {
            return Err(ErrorKind::NotFound.into());
        }
        match walk.slot() {
            Some(slot) if walk.is_vacant() => Ok(slot),
            _ => Err(ErrorKind::AlreadyExists.into()),
        }
    }

    /// The directory entry `path` names and the inode it refers to.
    fn occupant(&self, path: &Path) -> Result<(Slot, Ino)> {
        let walk = self.walk(path, false)?;
        let ino = walk.ino()?;
        // Without an entry there is nothing to remove or move.
        let slot = walk.slot().ok_or(ErrorKind::ResourceBusy)?;
        Ok((slot, ino))
    }

    pub(crate) fn stat(&self, ino: Ino) -> Stat {
        let inode = &self.inodes[&ino];
        let (kind, len, blocks, nlink) = match &inode.kind {
            Kind::File(content) => {
                let bytes = &content.bytes.live;
                (FileKind::File, bytes.len, bytes.blocks, inode.links.live)
            }
            // `.` and the parent's entry.
            Kind::Dir(_) => (FileKind::Dir, 0, 0, 2),
            Kind::Symlink(_) => (FileKind::Symlink, 0, 0, inode.links.live),
        };
        Stat {
            ino: ino.0,
            kind,
            len,
            blocks,
            nlink: nlink.into(),
            meta: inode.meta.live,
        }
    }

    pub(crate) fn canonicalize(&self, path: &Path) -> Result<PathBuf> {
        let walk = self.walk(path, true)?;
        walk.ino()?;
        Ok(walk.path())
    }

    pub(crate) fn read_link(&self, path: &Path) -> Result<PathBuf> {
        match &self.inodes[&self.resolve(path, false)?].kind {
            Kind::Symlink(target) => Ok(target.clone()),
            _ => Err(ErrorKind::InvalidInput.into()),
        }
    }

    /// Paths of the entries in the directory at `path`.
    pub(crate) fn read_dir(&self, path: &Path) -> Result<Vec<PathBuf>> {
        let entries = &self.dir(self.resolve(path, true)?)?.entries.live;
        Ok(entries.keys().map(|name| path.join(name)).collect())
    }

    /// Notes whether the two views agree on what `slot` names.
    fn mark(&mut self, slot: &Slot) {
        let dir = self.dir_mut(slot.dir);
        if dir.entries.live.get(&slot.name) == dir.entries.durable.get(&slot.name) {
            dir.unsynced.swap_remove(&slot.name);
        } else {
            dir.unsynced.insert(slot.name.clone());
        }
    }

    /// Moves the name `id` has in `view` to the vacant `slot`, or drops it.
    fn place(&mut self, view: View, id: LinkId, slot: Option<Slot>) {
        let link = &mut self.links[&id];
        let ino = link.ino;
        let old = mem::replace(link.slot.get_mut(view), slot.clone());
        if let Some(old) = &old {
            let entries = self.dir_mut(old.dir).entries.get_mut(view);
            entries.swap_remove(&old.name);
            self.mark(old);
        }
        if let Some(slot) = &slot {
            let entries = self.dir_mut(slot.dir).entries.get_mut(view);
            entries.insert(slot.name.clone(), id);
            self.mark(slot);
        }
        let inode = &mut self.inodes[&ino];
        if let (View::Durable, Some(_), Kind::Dir(dir)) = (view, &slot, &mut inode.kind) {
            dir.departures = Vec::new();
        }
        let names = inode.links.get_mut(view);
        *names = *names + slot.is_some() as u32 - old.is_some() as u32;
    }

    /// Forgets `id` if it no longer names anything, and its inode if
    /// nothing else keeps that.
    fn retire(&mut self, id: LinkId) {
        let Some(link) = self.links.get(&id) else {
            return;
        };
        let ino = link.ino;
        if link.slot.live.is_none() && link.slot.durable.is_none() {
            self.forget_overwrites(id);
            self.links.swap_remove(&id);
        }
        self.reap(ino);
    }

    /// Frees `ino` once nothing names it and no descriptor holds it open.
    fn reap(&mut self, ino: Ino) {
        let Some(inode) = self.inodes.get(&ino) else {
            return;
        };
        if inode.links.live + inode.opens > 0 {
            return;
        }
        if matches!(inode.kind, Kind::File(_)) {
            self.unreadable.insert(ino);
        }
        if inode.links.durable > 0 {
            return;
        }
        let dead = self.inodes.swap_remove(&ino).map(|inode| inode.kind);
        let Some(Kind::Dir(dir)) = dead else {
            return;
        };
        // A directory is empty by the time its name goes, but the durable
        // view may still have entries in it. Their slots went with it.
        for id in dir.entries.durable.into_values() {
            let link = &mut self.links[&id];
            link.slot.durable = None;
            self.inodes[&link.ino].links.durable -= 1;
            self.retire(id);
        }
    }

    /// Names `ino` by the vacant `slot`.
    fn link(&mut self, slot: &Slot, ino: Ino) -> LinkId {
        let id = LinkId(self.next_link);
        self.next_link += 1;
        self.links.insert(id, Link::new(ino));
        self.place(View::Live, id, Some(slot.clone()));
        id
    }

    /// Creates an inode named by the vacant `slot`.
    fn create(&mut self, slot: &Slot, kind: Kind, mode: u32, now: Duration) -> Ino {
        let ino = Ino(self.next_ino);
        self.next_ino += 1;
        self.inodes.insert(ino, Inode::new(kind, mode, now));
        let id = self.link(slot, ino);
        if let Kind::Dir(dir) = &mut self.inodes[&ino].kind {
            dir.link = Some(id);
        }
        self.inodes[&slot.dir].meta.live.modified(now);
        ino
    }

    /// Removes the live name in `slot`.
    fn remove(&mut self, slot: &Slot, now: Duration) {
        let id = self.entry(slot, View::Live).expect("the slot is occupied");
        let ino = self.links[&id].ino;
        self.inodes[&slot.dir].meta.live.modified(now);
        self.inodes[&ino].meta.live.ctime = now;
        self.place(View::Live, id, None);
        self.retire(id);
    }

    pub(crate) fn mkdir(&mut self, path: &Path, mode: u32, now: Duration) -> Result<()> {
        let slot = self.vacancy(path, FileKind::Dir)?;
        self.create(&slot, Kind::Dir(Dir::default()), mode, now);
        Ok(())
    }

    pub(crate) fn mkdir_all(&mut self, path: &Path, mode: u32, now: Duration) -> Result<()> {
        // As with `std::fs`, there is nothing to create for the empty path.
        if path.as_os_str().is_empty() {
            return Ok(());
        }
        let made = match (self.mkdir(path, mode, now), path.parent()) {
            (Err(missing), Some(parent)) if missing.kind() == ErrorKind::NotFound => {
                self.mkdir_all(parent, mode, now)?;
                self.mkdir(path, mode, now)
            }
            (made, _) => made,
        };
        // Whatever went wrong, a directory that is there is what was asked for.
        match made {
            Err(_) if self.resolve(path, true).is_ok_and(|ino| self.is_dir(ino)) => Ok(()),
            made => made,
        }
    }

    pub(crate) fn rmdir(&mut self, path: &Path, now: Duration) -> Result<()> {
        let (slot, ino) = self.occupant(path)?;
        if !self.dir(ino)?.entries.live.is_empty() {
            return Err(ErrorKind::DirectoryNotEmpty.into());
        }
        self.remove(&slot, now);
        Ok(())
    }

    /// Removes a directory and everything in it. A symlink is removed
    /// rather than followed.
    pub(crate) fn remove_dir_all(&mut self, path: &Path, now: Duration) -> Result<()> {
        let (slot, ino) = self.occupant(path)?;
        match &self.inodes[&ino].kind {
            Kind::File(_) => return Err(ErrorKind::NotADirectory.into()),
            Kind::Dir(_) => self.clear(ino, now),
            Kind::Symlink(_) => {}
        }
        self.remove(&slot, now);
        Ok(())
    }

    /// Removes everything inside the directory `dir`.
    fn clear(&mut self, dir: Ino, now: Duration) {
        for (name, id) in self.dir_mut(dir).entries.live.clone() {
            let child = self.links[&id].ino;
            if self.is_dir(child) {
                self.clear(child, now);
            }
            self.remove(&Slot { dir, name }, now);
        }
    }

    pub(crate) fn unlink(&mut self, path: &Path, now: Duration) -> Result<()> {
        let (slot, ino) = self.occupant(path)?;
        if self.is_dir(ino) {
            return Err(ErrorKind::IsADirectory.into());
        }
        self.remove(&slot, now);
        Ok(())
    }

    pub(crate) fn rename(&mut self, from: &Path, to: &Path, now: Duration) -> Result<()> {
        // Both paths are followed before either final name is examined.
        let source = self.walk(from, false)?;
        let target = self.walk(to, false)?;
        let ino = source.ino()?;
        let from = source.slot().ok_or(ErrorKind::ResourceBusy)?;
        let to = target.slot().ok_or(ErrorKind::ResourceBusy)?;
        let replaced = target.ino().ok();
        if target.requires_dir && !self.is_dir(ino) {
            return Err(ErrorKind::NotADirectory.into());
        }
        // A directory cannot move into its own subtree, and nothing can
        // replace a directory that contains it.
        if target.dirs().contains(&ino) {
            return Err(ErrorKind::InvalidInput.into());
        }
        if replaced.is_some_and(|dir| source.dirs().contains(&dir)) {
            return Err(ErrorKind::DirectoryNotEmpty.into());
        }
        if replaced == Some(ino) {
            return Ok(());
        }
        let id = self.entry(&from, View::Live).expect("the source exists");
        if let Some(replaced) = replaced {
            match (&self.inodes[&ino].kind, &self.inodes[&replaced].kind) {
                (Kind::Dir(_), Kind::Dir(dir)) if dir.entries.live.is_empty() => {}
                (Kind::Dir(_), Kind::Dir(_)) => return Err(ErrorKind::DirectoryNotEmpty.into()),
                (Kind::Dir(_), _) => return Err(ErrorKind::NotADirectory.into()),
                (_, Kind::Dir(_)) => return Err(ErrorKind::IsADirectory.into()),
                _ => {}
            }
            let victim = self.entry(&to, View::Live).expect("the target exists");
            self.displace(id, victim);
            self.remove(&to, now);
        }

        for dir in [from.dir, to.dir] {
            self.inodes[&dir].meta.live.modified(now);
        }
        if self.is_dir(ino) && self.links[&id].slot.durable.is_none() {
            self.note_departure(ino, from);
        }
        self.place(View::Live, id, Some(to));
        self.inodes[&ino].meta.live.ctime = now;
        Ok(())
    }

    fn note_departure(&mut self, ino: Ino, slot: Slot) {
        let parent = self.dir_mut(slot.dir);
        let next_sync = parent.next_sync.get_or_insert_with(Arc::default);
        let parent_synced = Arc::clone(next_sync);
        let departures = &mut self.dir_mut(ino).departures;
        // Consecutive renames within a parent share one commit point.
        let same_sync = |d: &Departure| Arc::ptr_eq(&d.parent_synced, &parent_synced);
        if !departures.last().is_some_and(same_sync) {
            departures.push(Departure {
                slot,
                parent_synced,
            });
        }
    }

    fn displace(&mut self, id: LinkId, victim: LinkId) {
        let replaced = &mut self.links[&victim];
        if replaced.slot.durable.is_some() {
            replaced.overwrites.replaced_by = Some(id);
            self.links[&id].overwrites.victims.insert(victim);
            return;
        }
        // A transient victim retires immediately. Its earlier overwrites
        // still connect their durable entries to the surviving name.
        let victims = mem::take(&mut replaced.overwrites.victims);
        for old in &victims {
            self.links[old].overwrites.replaced_by = Some(id);
        }
        self.links[&id].overwrites.victims.extend(victims);
    }

    fn forget_overwrites(&mut self, id: LinkId) {
        let overwrites = mem::take(&mut self.links[&id].overwrites);
        if let Some(owner) = overwrites.replaced_by {
            self.links[&owner].overwrites.victims.remove(&id);
        }
        for victim in overwrites.victims {
            self.links[&victim].overwrites.replaced_by = None;
        }
    }

    pub(crate) fn symlink(&mut self, target: &Path, link: &Path, now: Duration) -> Result<()> {
        if target.as_os_str().is_empty() {
            return Err(ErrorKind::NotFound.into());
        }
        let slot = self.vacancy(link, FileKind::Symlink)?;
        self.create(&slot, Kind::Symlink(target.to_owned()), 0o777, now);
        Ok(())
    }

    pub(crate) fn hard_link(&mut self, original: &Path, link: &Path, now: Duration) -> Result<()> {
        let ino = self.resolve(original, false)?;
        let slot = self.vacancy(link, FileKind::File)?;
        if self.is_dir(ino) {
            return Err(ErrorKind::PermissionDenied.into());
        }
        self.link(&slot, ino);
        self.inodes[&slot.dir].meta.live.modified(now);
        self.inodes[&ino].meta.live.ctime = now;
        Ok(())
    }

    pub(crate) fn set_mode(&mut self, path: &Path, mode: u32, now: Duration) -> Result<()> {
        let ino = self.resolve(path, true)?;
        let meta = &mut self.inodes[&ino].meta.live;
        meta.mode = mode;
        meta.ctime = now;
        Ok(())
    }

    /// The regular file `path` names, created first if `create` allows.
    pub(crate) fn open(&mut self, path: &Path, create: Create, now: Duration) -> Result<Ino> {
        // An exclusive create refuses any existing name, even a dangling symlink.
        let walk = self.walk(path, !matches!(create, Create::New(_)))?;
        match (walk.ino(), walk.slot(), create) {
            (Ok(_), _, Create::New(_)) => Err(ErrorKind::AlreadyExists.into()),
            (Ok(ino), ..) if self.is_dir(ino) => Err(ErrorKind::IsADirectory.into()),
            (Ok(ino), ..) => Ok(ino),
            (Err(_), Some(slot), Create::IfAbsent(mode) | Create::New(mode))
                if !walk.requires_dir =>
            {
                Ok(self.create(&slot, Kind::File(FileContent::default()), mode, now))
            }
            (Err(missing), ..) => Err(missing),
        }
    }

    /// Keeps `ino` alive for an open descriptor.
    pub(crate) fn pin(&mut self, ino: Ino) {
        self.inodes[&ino].opens += 1;
    }

    pub(crate) fn unpin(&mut self, ino: Ino) {
        self.inodes[&ino].opens -= 1;
        self.reap(ino);
    }

    /// The files that became unreadable since the last call.
    pub(crate) fn take_unreadable(&mut self) -> IndexSet<Ino> {
        mem::take(&mut self.unreadable)
    }

    fn file(&self, ino: Ino) -> &FileContent {
        match &self.inodes[&ino].kind {
            Kind::File(content) => content,
            _ => unreachable!("{ino:?} is not a regular file"),
        }
    }

    fn file_mut(&mut self, ino: Ino) -> &mut FileContent {
        match &mut self.inodes[&ino].kind {
            Kind::File(content) => content,
            _ => unreachable!("{ino:?} is not a regular file"),
        }
    }

    pub(crate) fn read(&self, ino: Ino, buf: &mut [u8], offset: u64) -> usize {
        self.file(ino).bytes.live.read(buf, offset)
    }

    pub(crate) fn change(&mut self, ino: Ino, change: Change, now: Duration) {
        let reserve = self.reserve_torn_writes;
        self.file_mut(ino).change(change, reserve);
        self.inodes[&ino].meta.live.modified(now);
    }

    pub(crate) fn sync_data(&mut self, ino: Ino) {
        self.file_mut(ino).sync();
    }

    /// Makes a file's content and attributes durable. Its names are not.
    pub(crate) fn sync_file(&mut self, ino: Ino) {
        self.sync_data(ino);
        self.inodes[&ino].meta.sync();
    }

    /// Makes the entries of the directory at `path` durable.
    pub(crate) fn sync_dir(&mut self, path: &Path) -> Result<()> {
        let ino = self.resolve(path, true)?;
        let mut plan = Plan::new();
        if let Some(own) = self.dir(ino)?.link {
            self.sync_creation(own, &mut plan);
        }
        let dir = self.dir(ino)?;
        for name in &dir.unsynced {
            let entries = [&dir.entries.live, &dir.entries.durable];
            for &id in entries.into_iter().filter_map(|entries| entries.get(name)) {
                self.plan(id, &mut plan);
            }
        }
        // What one name brings along changes where another will be found,
        // so every name is anchored again until that adds none.
        let mut anchored = 0;
        while anchored < plan.len() {
            anchored = plan.len();
            for index in 0..anchored {
                let id = plan[index];
                while let Some(above) = self.needed_ancestor(id, &plan) {
                    self.plan(above, &mut plan);
                }
            }
        }
        self.apply(plan);
        if let Some(synced) = self.dir_mut(ino).next_sync.take() {
            synced.store(true, Relaxed);
        }
        self.inodes[&ino].meta.sync();
        Ok(())
    }

    /// A directory sync flushes its own creation. Each later rename belongs
    /// to its parents, so keep the earliest name their syncs still allow.
    fn sync_creation(&mut self, id: LinkId, plan: &mut Plan) {
        let link = &self.links[&id];
        let (Some(live), None) = (&link.slot.live, &link.slot.durable) else {
            return;
        };
        let dir = self.dir(link.ino).expect("directory owns its link");
        let committed = |d: &Departure| d.parent_synced.load(Relaxed);
        let last_sync = dir.departures.iter().rposition(committed);
        let pending = &dir.departures[last_sync.map_or(0, |i| i + 1)..];
        let has_parent = |d: &&Departure| self.inodes.contains_key(&d.slot.dir);
        if let Some(Departure { slot, .. }) = pending.iter().find(has_parent) {
            let vacant = self.entry(slot, View::Durable).is_none();
            let cycle = self.reach(Some(slot.dir), plan).any(|p| p == link.ino);
            if vacant && !cycle {
                self.place(View::Durable, id, Some(slot.clone()));
                return;
            }
        }
        if self.entry(live, View::Durable).is_none() {
            self.plan(id, plan);
        }
    }

    /// Adds `id` to the names a sync makes durable, along with the names
    /// whose durable place it takes or does away with.
    fn plan(&self, id: LinkId, plan: &mut Plan) {
        let pending = |id: &LinkId| !self.links[id].is_synced();
        let mut next = plan.len();
        if !pending(&id) || !plan.insert(id) {
            return;
        }
        while let Some(&id) = plan.get_index(next) {
            next += 1;
            let link = &self.links[&id];
            let victims = link.overwrites.victims.iter().copied();
            let related = victims.chain(link.overwrites.replaced_by);
            plan.extend(related.filter(pending));
            if let Some(slot) = &link.slot.live {
                plan.extend(self.entry(slot, View::Durable).filter(pending));
            } else if let Kind::Dir(dir) = &self.inodes[&link.ino].kind {
                // A removed directory takes its durable entries with it.
                // Anything renamed out must land somewhere first.
                plan.extend(dir.entries.durable.values().copied().filter(pending));
            }
        }
    }

    /// Keep a moved name reachable wherever its old directory survives.
    /// A newly named directory needs only an acyclic ancestry.
    fn needed_ancestor(&self, id: LinkId, plan: &Plan) -> Option<LinkId> {
        let link = &self.links[&id];
        let from = link.slot.durable.as_ref().map(|durable| durable.dir);
        // Only a name that arrives in a directory depends on what is above.
        let arrival = |slot: &&Slot| from != Some(slot.dir);
        let slot = link.slot.live.as_ref().filter(arrival)?;
        if from.is_none() && !self.is_dir(link.ino) {
            return None;
        }
        let mut destination = IndexSet::new();
        let repeated = |dir| !destination.insert(dir);
        let cycle = self.reach(Some(slot.dir), plan).any(repeated);
        if cycle && !destination.contains(&link.ino) {
            // A planned move on the cycle will reconnect this ancestry.
            return None;
        }
        if !cycle && (from.is_none() || destination.contains(&ROOT)) {
            return None;
        }
        let origin: IndexSet<_> = self.reach(from, plan).collect();
        if !cycle && !destination.is_disjoint(&origin) {
            return None;
        }

        // The highest shared directory is where the live path leaves the
        // disconnected (or cyclic) durable ancestry. Only its parent edge
        // needs to change before reachability is checked again.
        let (mut dir, mut bridge) = (slot.dir, None);
        while let Some((above, parent)) = self.above(dir) {
            if !cycle && origin.contains(&dir) {
                break;
            }
            if destination.contains(&dir) {
                bridge = Some(above);
            }
            dir = parent;
        }
        bridge
    }

    /// The entry that names `dir`, and the directory it lives in.
    fn above(&self, dir: Ino) -> Option<(LinkId, Ino)> {
        let id = self.dir(dir).ok()?.link?;
        let slot = self.links[&id].slot.live.as_ref()?;
        Some((id, slot.dir))
    }

    /// `dir` and each directory it will be durably inside once `plan` is
    /// applied.
    fn reach<'a>(&'a self, dir: Option<Ino>, plan: &'a Plan) -> impl Iterator<Item = Ino> + 'a {
        let above = move |dir: &Ino| {
            let id = self.dir(*dir).ok()?.link?;
            let slot = &self.links[&id].slot;
            // A planned name will be durable where it lives, if it lives.
            let live = slot.live.as_ref().filter(|_| plan.contains(&id));
            Some(live.or(slot.durable.as_ref())?.dir)
        };
        // A plan that is not complete yet may describe a loop.
        iter::successors(dir, above).take(self.inodes.len())
    }

    /// Makes each planned name durable where it lives now.
    fn apply(&mut self, plan: Plan) {
        // Names may trade places, so every one leaves before any arrives.
        for &id in &plan {
            self.place(View::Durable, id, None);
        }
        for &id in &plan {
            let live = self.links[&id].slot.live.clone();
            self.place(View::Durable, id, live);
            self.forget_overwrites(id);
        }
        for id in plan {
            self.retire(id);
        }
    }

    /// Bytes the files take up on the disk.
    pub(crate) fn used_bytes(&self) -> u64 {
        let stored = |inode: &Inode| match &inode.kind {
            Kind::File(content) => content.stored(),
            _ => 0,
        };
        self.inodes.values().map(stored).sum()
    }

    /// Bytes a write of `len` at `offset` would add to those the file `ino`
    /// takes up on the disk.
    pub(crate) fn growth(&self, ino: Ino, offset: u64, len: usize) -> u64 {
        self.file(ino).growth(offset, len)
    }

    /// Rolls everything back to its durable version and drops the inodes no
    /// durable name reaches.
    pub(crate) fn crash(&mut self, block_size: Option<u64>, rng: &mut dyn RngCore) {
        // Durable names per inode, found breadth-first from the root.
        let mut names = IndexMap::from([(ROOT, 1)]);
        let mut next = 0;
        while let Some((&ino, _)) = names.get_index(next) {
            if let Kind::Dir(dir) = &self.inodes[&ino].kind {
                for id in dir.entries.durable.values() {
                    *names.entry(self.links[id].ino).or_insert(0) += 1;
                }
            }
            next += 1;
        }

        self.unreadable.clear();
        self.inodes.retain(|ino, _| names.contains_key(ino));
        self.links.retain(|_, link| {
            link.slot.crash();
            link.overwrites = Overwrites::default();
            let reached = |slot: &Slot| names.contains_key(&slot.dir);
            link.slot.live.as_ref().is_some_and(reached)
        });
        for (ino, inode) in &mut self.inodes {
            inode.links = Versioned::new(names[ino]);
            inode.opens = 0;
            inode.meta.crash();
            match &mut inode.kind {
                Kind::File(content) => content.crash(block_size, rng),
                Kind::Dir(dir) => {
                    dir.entries.crash();
                    dir.unsynced.clear();
                    dir.next_sync = None;
                }
                Kind::Symlink(_) => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    use std::collections::BTreeMap;

    const NOW: Duration = Duration::from_secs(1);
    const SEEDS: u64 = 100;

    /// A path for the next operation. It mostly follows entries that exist,
    /// since an operation on a missing path does nothing.
    fn path(tree: &Tree, rng: &mut SmallRng) -> PathBuf {
        let mut path = PathBuf::from("/");
        for _ in 0..rng.random_range(0..=3) {
            let entries = tree.read_dir(&path).unwrap_or_default();
            let name = ["a", "b", "c"][rng.random_range(0..3)];
            match entries.get(rng.random_range(0..=entries.len())) {
                Some(entry) => path.clone_from(entry),
                None => path.push(name),
            }
        }
        path
    }

    /// Makes one random change to the live tree.
    fn mutate(tree: &mut Tree, open: &mut Vec<Ino>, rng: &mut SmallRng) {
        let (a, b) = (path(tree, rng), path(tree, rng));
        let held = (!open.is_empty()).then(|| rng.random_range(0..open.len()));
        match (rng.random_range(0..16), held) {
            (0, _) => drop(tree.mkdir(&a, 0o755, NOW)),
            (1, _) => drop(tree.mkdir_all(&a, 0o755, NOW)),
            (2, _) => drop(tree.rmdir(&a, NOW)),
            (3, _) => drop(tree.remove_dir_all(&a, NOW)),
            (4, _) => drop(tree.unlink(&a, NOW)),
            (5..=7, _) => drop(tree.rename(&a, &b, NOW)),
            (8, _) => drop(tree.symlink(&a, &b, NOW)),
            (9, _) => drop(tree.hard_link(&a, &b, NOW)),
            (10, _) => drop(tree.set_mode(&a, 0o700, NOW)),
            (11, _) => {
                if let Ok(ino) = tree.open(&a, Create::IfAbsent(0o644), NOW) {
                    tree.pin(ino);
                    open.push(ino);
                }
            }
            (12, Some(held)) => tree.unpin(open.swap_remove(held)),
            (13, Some(held)) => {
                let len = rng.random_range(0..8);
                tree.change(open[held], Change::SetLen(len), NOW);
            }
            (_, Some(held)) => {
                let data = vec![rng.random(); rng.random_range(1..4)];
                let write = Change::Write(rng.random_range(0..8), data);
                tree.change(open[held], write, NOW);
            }
            _ => {}
        }
    }

    /// The bytes a read of the whole file returns.
    fn read_all(bytes: &Bytes) -> Vec<u8> {
        let mut all = vec![1; bytes.len as usize];
        assert_eq!(bytes.read(&mut all, 0), all.len());
        all
    }

    /// Everything the live namespace reaches, by path.
    fn snapshot(tree: &Tree) -> BTreeMap<PathBuf, (Ino, String)> {
        let mut found = BTreeMap::new();
        let mut dirs = vec![PathBuf::from("/")];
        while let Some(dir) = dirs.pop() {
            for path in tree.read_dir(&dir).unwrap() {
                let ino = tree.resolve(&path, false).unwrap();
                let inode = &tree.inodes[&ino];
                let what = match &inode.kind {
                    Kind::File(content) => format!("file {:?}", read_all(&content.bytes.live)),
                    Kind::Dir(_) => "dir".to_owned(),
                    Kind::Symlink(target) => format!("link {target:?}"),
                };
                if tree.is_dir(ino) {
                    dirs.push(path.clone());
                }
                let mode = inode.meta.live.mode;
                found.insert(path, (ino, format!("{what} mode {mode:o}")));
            }
        }
        found
    }

    /// `dir` and each directory it is durably inside.
    fn reach(tree: &Tree, dir: Ino) -> Vec<Ino> {
        tree.reach(Some(dir), &Plan::new()).collect()
    }

    /// Syncs the directory at `path`. A name the sync moves to another
    /// directory has to stay within reach of the one it was durable in, so
    /// that a crash which finds the old directory finds the new one.
    fn sync_dir(tree: &mut Tree, path: &Path) -> Result<()> {
        let durable_in = |link: &Link| Some(link.slot.durable.as_ref()?.dir);
        let place = |(id, link): (&LinkId, &Link)| {
            let dir = durable_in(link)?;
            Some((*id, (dir, reach(tree, dir))))
        };
        let before: IndexMap<_, _> = tree.links.iter().filter_map(place).collect();
        tree.sync_dir(path)?;

        for (id, link) in &tree.links {
            let (Some(now), Some((was, above))) = (durable_in(link), before.get(id)) else {
                continue;
            };
            // A directory that is gone was reached through what was above it.
            let old = match tree.inodes.contains_key(was) {
                true => reach(tree, *was),
                false => above.clone(),
            };
            let new = reach(tree, now);
            let within = new.contains(&ROOT) || new.iter().any(|dir| old.contains(dir));
            assert!(within, "{id:?} is out of reach of {was:?} in {now:?}");
        }
        Ok(())
    }

    /// Syncs a directory picked at random.
    fn sync_any(tree: &mut Tree, rng: &mut SmallRng) {
        let dir = path(tree, rng);
        drop(sync_dir(tree, &dir));
    }

    /// Syncs every directory and file the live namespace reaches, in no
    /// particular order.
    fn sync_all(tree: &mut Tree, open: &[Ino], rng: &mut SmallRng) {
        let mut reached: Vec<_> = snapshot(tree).into_iter().collect();
        reached.push((PathBuf::from("/"), (ROOT, String::new())));
        reached.shuffle(rng);
        for (path, (ino, _)) in reached {
            match &tree.inodes[&ino].kind {
                Kind::File(_) => tree.sync_file(ino),
                Kind::Dir(_) => sync_dir(tree, &path).unwrap(),
                Kind::Symlink(_) => {}
            }
            check(tree, open);
        }
    }

    /// Asserts that the counts and indexes the tree keeps agree with the
    /// links it holds.
    fn check(tree: &Tree, open: &[Ino]) {
        let mut names = IndexMap::from([(ROOT, (1, 1))]);
        for (id, link) in &tree.links {
            let (live, durable) = (&link.slot.live, &link.slot.durable);
            assert!(live.is_some() || durable.is_some(), "{id:?} names nothing");
            for victim in &link.overwrites.victims {
                assert_eq!(tree.links[victim].overwrites.replaced_by, Some(*id));
            }
            if let Some(owner) = link.overwrites.replaced_by {
                assert!(live.is_none() && tree.links[&owner].overwrites.victims.contains(id));
            }
            let named = names.entry(link.ino).or_insert((0, 0));
            named.0 += live.is_some() as u32;
            named.1 += durable.is_some() as u32;
            for (view, slot) in [(View::Live, live), (View::Durable, durable)] {
                let entry = slot.as_ref().map(|slot| tree.entry(slot, view));
                assert!(entry.is_none_or(|entry| entry == Some(*id)), "{id:?}");
            }
            // A directory has one entry, so it has one parent in either view.
            if let Kind::Dir(dir) = &tree.inodes[&link.ino].kind {
                assert_eq!(dir.link, Some(*id), "{:?} has two names", link.ino);
            }
        }
        for (&ino, inode) in &tree.inodes {
            let (live, durable) = names.swap_remove(&ino).unwrap_or((0, 0));
            let opens = open.iter().filter(|&&held| held == ino).count() as u32;
            let counted = (inode.links.live, inode.links.durable, inode.opens);
            assert_eq!(counted, (live, durable, opens), "{ino:?}");
            assert!(live + durable + opens > 0, "{ino:?} is garbage");
            match &inode.kind {
                Kind::Dir(dir) => check_dir(tree, ino, dir),
                Kind::File(content) => check_file(ino, content),
                Kind::Symlink(_) => {}
            }
        }
        assert!(names.is_empty(), "links name missing inodes: {names:?}");
    }

    fn check_dir(tree: &Tree, ino: Ino, dir: &Dir) {
        let (live, durable) = (&dir.entries.live, &dir.entries.durable);
        for (view, entries) in [(View::Live, live), (View::Durable, durable)] {
            for (name, id) in entries {
                let (dir, name) = (ino, name.clone());
                let slot = Some(Slot { dir, name });
                assert_eq!(tree.links[id].slot.get(view), &slot, "{id:?}");
            }
        }
        let differs = |name: &&OsString| live.get(*name) != durable.get(*name);
        let unsynced: IndexSet<_> = live.keys().chain(durable.keys()).filter(differs).collect();
        let noted: IndexSet<_> = dir.unsynced.iter().collect();
        assert_eq!(noted, unsynced, "{ino:?}");

        let above = reach(tree, ino);
        let distinct: IndexSet<_> = above.iter().collect();
        assert_eq!(distinct.len(), above.len(), "{ino:?} is inside itself");
        let home = dir.link.map(|own| &tree.links[&own].slot.live);
        let removed = home.is_some_and(Option::is_none);
        assert!(!removed || live.is_empty(), "{ino:?} was removed full");
    }

    fn check_file(ino: Ino, content: &FileContent) {
        let mut replayed = content.bytes.durable.clone();
        for change in &content.unsynced {
            replayed.apply(change);
        }
        let live = &content.bytes.live;
        assert_eq!(read_all(&replayed), read_all(live), "{ino:?}");
        for bytes in [live, &content.bytes.durable] {
            let held: usize = bytes.chunks.values().map(Vec::len).sum();
            assert_eq!(bytes.stored, held as u64, "{ino:?}");
            let allocated: u64 = bytes.chunks.values().map(|c| sectors(c.len())).sum();
            assert_eq!(bytes.blocks, allocated, "{ino:?}");
        }
    }

    #[test]
    fn counts_stay_consistent() {
        for seed in 0..SEEDS {
            let rng = &mut SmallRng::seed_from_u64(seed);
            let (mut tree, mut open) = (Tree::new(false), Vec::new());
            // Bytes each file held when it was last synced.
            let mut synced = IndexMap::new();
            for _ in 0..300 {
                match rng.random_range(0..12) {
                    0 => sync_any(&mut tree, rng),
                    1 => {
                        if let Some(&ino) = open.first() {
                            synced.insert(ino, read_all(&tree.file(ino).bytes.live));
                            tree.sync_file(ino);
                        }
                    }
                    2 => {
                        tree.crash(None, rng);
                        open.clear();
                        let reached: IndexMap<_, _> = snapshot(&tree).into_values().collect();
                        assert_eq!(reached.len() + 1, tree.inodes.len(), "seed {seed}");
                        for (ino, inode) in &tree.inodes {
                            let Kind::File(content) = &inode.kind else {
                                continue;
                            };
                            let expected = synced.get(ino).cloned().unwrap_or_default();
                            let live = read_all(&content.bytes.live);
                            assert_eq!(live, expected, "seed {seed} {ino:?}");
                        }
                    }
                    _ => mutate(&mut tree, &mut open, rng),
                }
                check(&tree, &open);
            }
        }
    }

    #[test]
    fn crash_restores_the_last_full_sync() {
        for seed in 0..SEEDS {
            let rng = &mut SmallRng::seed_from_u64(seed);
            let (mut tree, mut open) = (Tree::new(false), Vec::new());
            for _ in 0..5 {
                for _ in 0..rng.random_range(0..60) {
                    mutate(&mut tree, &mut open, rng);
                }
                sync_all(&mut tree, &open, rng);
                let synced = snapshot(&tree);
                for _ in 0..rng.random_range(0..60) {
                    mutate(&mut tree, &mut open, rng);
                }
                tree.crash(None, rng);
                open.clear();
                assert_eq!(snapshot(&tree), synced, "seed {seed}");
            }
        }
    }

    /// The inode each name in the directory `ino` refers to.
    fn listing(tree: &Tree, ino: Ino) -> BTreeMap<OsString, Ino> {
        let entries = &tree.dir(ino).unwrap().entries.live;
        let named = |(name, id): (&OsString, &LinkId)| (name.clone(), tree.links[id].ino);
        entries.iter().map(named).collect()
    }

    /// A crash right after a directory is synced finds the directory's
    /// entries as they were, wherever it finds the directory.
    #[test]
    fn crash_keeps_a_synced_dir_as_it_was() {
        for seed in 0..SEEDS {
            let rng = &mut SmallRng::seed_from_u64(seed);
            let (mut tree, mut open) = (Tree::new(false), Vec::new());
            for _ in 0..20 {
                for _ in 0..rng.random_range(0..30) {
                    match rng.random_range(0..5) {
                        0 => sync_any(&mut tree, rng),
                        _ => mutate(&mut tree, &mut open, rng),
                    }
                    check(&tree, &open);
                }
                let dir = path(&tree, rng);
                if sync_dir(&mut tree, &dir).is_err() {
                    continue;
                }
                check(&tree, &open);
                let ino = tree.resolve(&dir, true).unwrap();
                let synced = listing(&tree, ino);
                tree.crash(None, rng);
                open.clear();
                if tree.inodes.contains_key(&ino) {
                    assert_eq!(listing(&tree, ino), synced, "seed {seed}");
                }
            }
        }
    }

    /// Every slot each name has lived in, and the names that were removed.
    #[derive(Default)]
    struct Trails {
        lived: IndexMap<LinkId, Vec<Slot>>,
        removed: IndexSet<LinkId>,
    }

    impl Trails {
        fn follow(&mut self, tree: &Tree) {
            for (&id, link) in &tree.links {
                let trail = self.lived.entry(id).or_default();
                trail.extend(link.slot.live.clone());
            }
            let live = |id: &LinkId| tree.links.get(id)?.slot.live.as_ref();
            let removed = self.lived.keys().filter(|id| live(id).is_none());
            self.removed.extend(removed);
        }
    }

    /// However directories are synced, a crash finds each name in a place
    /// it has lived since it was last durable everywhere, and no name is
    /// lost that was durable and never removed.
    #[test]
    fn crash_keeps_names_where_they_have_lived() {
        for seed in 0..SEEDS {
            let rng = &mut SmallRng::seed_from_u64(seed);
            let (mut tree, mut open) = (Tree::new(false), Vec::new());
            for _ in 0..5 {
                for _ in 0..rng.random_range(0..60) {
                    mutate(&mut tree, &mut open, rng);
                }
                sync_all(&mut tree, &open, rng);
                let durable: Vec<_> = tree.links.keys().copied().collect();
                let mut trails = Trails::default();
                trails.follow(&tree);
                // Mostly what a sync has to get right: names that move, and
                // directories that come and go under them.
                for _ in 0..rng.random_range(0..60) {
                    match rng.random_range(0..10) {
                        0..3 => sync_any(&mut tree, rng),
                        3..6 => drop(tree.rename(&path(&tree, rng), &path(&tree, rng), NOW)),
                        6 => drop(tree.rmdir(&path(&tree, rng), NOW)),
                        7 => drop(tree.mkdir(&path(&tree, rng), 0o755, NOW)),
                        _ => mutate(&mut tree, &mut open, rng),
                    }
                    check(&tree, &open);
                    trails.follow(&tree);
                }
                tree.crash(None, rng);
                open.clear();
                check(&tree, &open);

                for (id, link) in &tree.links {
                    let (slot, lived) = (link.slot.live.as_ref().unwrap(), &trails.lived[id]);
                    assert!(lived.contains(slot), "seed {seed} {id:?} {slot:?}");
                }
                let kept = durable.iter().filter(|id| !trails.removed.contains(*id));
                for id in kept {
                    assert!(tree.links.contains_key(id), "seed {seed} lost {id:?}");
                }
            }
        }
    }

    /// Chunked bytes read back as the flat buffer they stand for.
    #[test]
    fn bytes_match_a_flat_buffer() {
        for seed in 0..SEEDS {
            let rng = &mut SmallRng::seed_from_u64(seed);
            let (mut bytes, mut flat) = (Bytes::default(), Vec::new());
            for _ in 0..100 {
                let at = rng.random_range(0..3 * CHUNK as usize);
                if rng.random_bool(0.2) {
                    bytes.set_len(at as u64);
                    flat.resize(at, 0);
                } else {
                    let data = vec![rng.random(); rng.random_range(1..2 * CHUNK as usize)];
                    let grown = bytes.stored + bytes.growth(at as u64, data.len());
                    bytes.write(at as u64, &data);
                    assert_eq!(bytes.stored, grown, "seed {seed}");
                    flat.resize(flat.len().max(at + data.len()), 0);
                    flat[at..at + data.len()].copy_from_slice(&data);
                }
                assert_eq!(read_all(&bytes), flat, "seed {seed}");
                let held: usize = bytes.chunks.values().map(Vec::len).sum();
                assert_eq!(bytes.stored, held as u64, "seed {seed}");
            }
        }
    }
}
