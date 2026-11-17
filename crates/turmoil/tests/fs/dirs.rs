//! Directory operation tests.

use std::ffi::OsString;
use std::io::ErrorKind;
use std::os::unix::fs::FileExt;
use std::sync::{Arc, Mutex};
use turmoil::fs::shim::std::fs::{
    create_dir, create_dir_all, exists, metadata, read, read_dir, remove_dir, remove_dir_all,
    remove_file, rename, symlink, sync_dir, write, OpenOptions,
};
use turmoil::{Builder, Result};

#[test]
fn mkdir_rmdir() -> Result {
    let mut sim = Builder::new().build();
    sim.client("test", async {
        create_dir("/mydir")?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open("/mydir/file.txt")?;
        file.write_all_at(b"hello", 0)?;
        assert!(remove_dir("/mydir").is_err()); // not empty
        drop(file);
        remove_file("/mydir/file.txt")?;
        remove_dir("/mydir")?;
        Ok(())
    });
    sim.run()
}

#[test]
fn mkdir_requires_parent() -> Result {
    let mut sim = Builder::new().build();
    sim.client("test", async {
        assert!(create_dir("/a/b/c").is_err());
        create_dir("/a")?;
        create_dir("/a/b")?;
        create_dir("/a/b/c")?;
        Ok(())
    });
    sim.run()
}

#[test]
fn create_dir_all_works() -> Result {
    let mut sim = Builder::new().build();
    sim.client("test", async {
        create_dir_all("/a/b/c/d")?;
        let _file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open("/a/b/c/d/file.txt")?;
        Ok(())
    });
    sim.run()
}

#[test]
fn remove_file_basic() -> Result {
    let mut sim = Builder::new().build();
    sim.client("test", async {
        create_dir("/data")?;
        write("/data/file.txt", b"hello")?;
        remove_file("/data/file.txt")?;
        assert!(OpenOptions::new()
            .read(true)
            .open("/data/file.txt")
            .is_err());
        assert!(remove_file("/data/file.txt").is_err());
        Ok(())
    });
    sim.run()
}

#[test]
fn read_dir_basic() -> Result {
    use std::collections::HashSet;
    let mut sim = Builder::new().build();
    sim.client("test", async {
        create_dir_all("/data/subdir")?;
        write("/data/file1.txt", b"content1")?;
        write("/data/file2.txt", b"content2")?;
        let entries: Vec<_> = read_dir("/data")?.collect::<std::io::Result<Vec<_>>>()?;
        assert_eq!(entries.len(), 3);
        let names: HashSet<_> = entries
            .iter()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        assert!(
            names.contains("subdir") && names.contains("file1.txt") && names.contains("file2.txt")
        );
        Ok(())
    });
    sim.run()
}

#[test]
fn remove_dir_all_basic() -> Result {
    let mut sim = Builder::new().build();
    sim.client("test", async {
        create_dir_all("/data/level1/level2")?;
        write("/data/file1.txt", b"hello")?;
        write("/data/level1/file2.txt", b"world")?;
        remove_dir_all("/data")?;
        assert!(metadata("/data").is_err());
        Ok(())
    });
    sim.run()
}

#[test]
fn remove_dir_all_empty() -> Result {
    let mut sim = Builder::new().build();
    sim.client("test", async {
        create_dir("/empty")?;
        remove_dir_all("/empty")?;
        assert!(metadata("/empty").is_err());
        Ok(())
    });
    sim.run()
}

#[test]
fn rename_file() -> Result {
    let mut sim = Builder::new().build();
    sim.client("test", async {
        create_dir("/data")?;
        write("/data/old.txt", b"hello")?;
        rename("/data/old.txt", "/data/new.txt")?;
        assert!(OpenOptions::new().read(true).open("/data/old.txt").is_err());
        assert!(metadata("/data/new.txt").is_ok());
        Ok(())
    });
    sim.run()
}

#[test]
fn rename_atomic_replace() -> Result {
    let mut sim = Builder::new().build();
    sim.client("test", async {
        create_dir("/data")?;
        write("/data/file1.txt", b"original")?;
        write("/data/file2.txt", b"replacement")?;
        rename("/data/file2.txt", "/data/file1.txt")?;
        use turmoil::fs::shim::std::fs::read;
        assert_eq!(read("/data/file1.txt")?, b"replacement");
        assert!(OpenOptions::new()
            .read(true)
            .open("/data/file2.txt")
            .is_err());
        Ok(())
    });
    sim.run()
}

#[test]
fn rename_dir() -> Result {
    let mut sim = Builder::new().build();
    sim.client("test", async {
        create_dir("/old")?;
        sync_dir("/")?;
        rename("/old", "/new")?;
        assert!(metadata("/old").is_err());
        let meta = metadata("/new")?;
        assert!(meta.is_dir());
        assert!(!meta.is_file());
        Ok(())
    });
    sim.run()
}

#[test]
fn rename_unsynced_dir() -> Result {
    let mut sim = Builder::new().build();
    sim.client("test", async {
        create_dir("/old")?;
        rename("/old", "/new")?;
        assert!(metadata("/old").is_err());
        let meta = metadata("/new")?;
        assert!(meta.is_dir());
        assert!(!meta.is_file());
        Ok(())
    });
    sim.run()
}

#[test]
fn repeated_renames_preserve_each_entry_kind() -> Result {
    use turmoil::fs::shim::std::fs::{read_link, symlink, symlink_metadata};

    for synced in [false, true] {
        for kind in 0..3 {
            let mut sim = Builder::new().build();
            sim.client("test", async move {
                match kind {
                    0 => write("/a", b"data")?,
                    1 => create_dir("/a")?,
                    _ => symlink("/target", "/a")?,
                }
                if synced {
                    sync_dir("/")?;
                }
                for n in 0..128 {
                    let (from, to) = if n % 2 == 0 {
                        ("/a", "/b")
                    } else {
                        ("/b", "/a")
                    };
                    rename(from, to)?;
                    assert!(symlink_metadata(from).is_err());
                    let meta = symlink_metadata(to)?;
                    assert_eq!(meta.is_file(), kind == 0);
                    assert_eq!(meta.is_dir(), kind == 1);
                    assert_eq!(meta.is_symlink(), kind == 2);
                    if kind == 2 {
                        assert_eq!(read_link(to)?, std::path::Path::new("/target"));
                    }
                }
                Ok(())
            });
            sim.run()?;
        }
    }
    Ok(())
}

#[test]
fn reused_source_does_not_change_renamed_entry_kind() -> Result {
    use turmoil::fs::shim::std::fs::{symlink, symlink_metadata};

    let mut sim = Builder::new().build();
    sim.client("test", async {
        create_dir("/a")?;
        rename("/a", "/b")?;
        write("/a", b"new file")?;
        assert!(metadata("/a")?.is_file());
        assert!(metadata("/b")?.is_dir());
        rename("/b", "/c")?;
        symlink("/target", "/b")?;
        assert!(metadata("/c")?.is_dir());
        assert!(symlink_metadata("/b")?.is_symlink());
        Ok(())
    });
    sim.run()
}

#[test]
fn remove_dir_all_nonexistent() -> Result {
    let mut sim = Builder::new().build();
    sim.client("test", async {
        assert!(remove_dir_all("/nonexistent").is_err());
        Ok(())
    });
    sim.run()
}

#[test]
fn rename_dir_moves_its_contents() -> Result {
    let mut sim = Builder::new().build();
    sim.client("test", async {
        create_dir_all("/a/sub")?;
        write("/a/file", b"one")?;
        write("/a/sub/file", b"two")?;
        rename("/a", "/b")?;
        assert_eq!(read("/b/file")?, b"one");
        assert_eq!(read("/b/sub/file")?, b"two");
        assert!(metadata("/a").is_err());
        assert!(metadata("/a/file").is_err());
        let mut names: Vec<_> = read_dir("/b")?.map(|e| e.unwrap().file_name()).collect();
        names.sort();
        assert_eq!(names, ["file", "sub"]);
        Ok(())
    });
    sim.run()
}

#[test]
fn rename_dir_into_itself_fails() -> Result {
    let mut sim = Builder::new().build();
    sim.client("test", async {
        create_dir_all("/a/b")?;
        let err = rename("/a", "/a/b/c").unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert!(metadata("/a/b")?.is_dir());
        Ok(())
    });
    sim.run()
}

#[test]
fn paths_resolve_through_symlinked_dirs_and_dot_dot() -> Result {
    let mut sim = Builder::new().build();
    sim.client("test", async {
        create_dir("/data")?;
        write("/data/file", b"x")?;
        symlink("/data", "/link")?;
        assert_eq!(read("/link/file")?, b"x");
        assert_eq!(read("/data/../link/./file")?, b"x");
        create_dir("/link/sub")?;
        assert!(metadata("/data/sub")?.is_dir());
        Ok(())
    });
    sim.run()
}

/// The same simulation must list a directory in the same order every time.
#[test]
fn read_dir_order_is_reproducible() -> Result {
    fn listing() -> Result<Vec<OsString>> {
        let names = Arc::new(Mutex::new(Vec::new()));
        let listed = names.clone();
        let mut sim = Builder::new().build();
        sim.client("test", async move {
            for n in 0..16 {
                write(format!("/file{n}"), b"")?;
            }
            let entries = read_dir("/")?.map(|e| e.unwrap().file_name());
            listed.lock().unwrap().extend(entries);
            Ok(())
        });
        sim.run()?;
        let names = names.lock().unwrap().clone();
        Ok(names)
    }

    let first = listing()?;
    assert_eq!(first.len(), 16);
    assert_eq!(first, listing()?);
    Ok(())
}

/// A path that ends in `..` names a directory but no entry, so nothing can
/// be renamed or removed through it.
#[test]
fn dot_dot_names_no_entry() -> Result {
    let mut sim = Builder::new().build();
    sim.client("test", async {
        create_dir_all("/a/b")?;
        write("/a/keep", b"x")?;
        for busy in [rename("/a/b/..", "/c"), rename("/a/keep", "/a/b/..")] {
            assert_eq!(busy.unwrap_err().kind(), ErrorKind::ResourceBusy);
        }
        assert!(remove_dir_all("/a/b/..").is_err());
        assert!(remove_dir("/a/b/..").is_err());
        assert!(remove_file("/a/b/..").is_err());
        assert_eq!(read("/a/b/../keep")?, b"x");
        assert!(metadata("/a/b/..")?.is_dir());
        assert!(!exists("/c"));
        Ok(())
    });
    sim.run()
}

#[test]
fn trailing_slash_and_dot_require_a_directory() -> Result {
    use turmoil::fs::shim::std::fs::symlink_metadata;

    let mut sim = Builder::new().build();
    sim.client("test", async {
        write("/file", b"kept")?;
        symlink("/file", "/link")?;
        for path in ["/file/", "/file/.", "/link/", "/link/.//"] {
            assert_eq!(metadata(path).unwrap_err().kind(), ErrorKind::NotADirectory);
            let removal = remove_dir_all(path).unwrap_err();
            assert_eq!(removal.kind(), ErrorKind::NotADirectory);
        }
        assert_eq!(read("/link")?, b"kept");
        symlink("/file/", "/indirect")?;
        assert_eq!(
            metadata("/indirect").unwrap_err().kind(),
            ErrorKind::NotADirectory
        );
        assert!(write("/missing/", []).is_err());
        assert!(!exists("/missing"));

        create_dir("/dir/")?;
        symlink("/dir", "/dirlink")?;
        assert!(symlink_metadata("/dirlink/")?.is_dir());
        assert!(remove_dir_all("/dir/.").is_err());
        rename("/dir/", "/renamed/")?;
        remove_dir_all("/renamed/")?;
        Ok(())
    });
    sim.run()
}

/// The empty path names nothing, as with `std::fs`. It is what
/// `Path::parent` returns for a bare file name.
#[test]
fn empty_path_is_not_found() -> Result {
    let mut sim = Builder::new().build();
    sim.client("test", async {
        let results = [
            metadata("").map(drop),
            read_dir("").map(drop),
            sync_dir(""),
            remove_dir_all(""),
        ];
        for result in results {
            assert_eq!(result.unwrap_err().kind(), ErrorKind::NotFound);
        }
        assert!(!exists(""));
        create_dir_all("")?;
        Ok(())
    });
    sim.run()
}
