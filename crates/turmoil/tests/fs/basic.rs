//! Basic file operation tests.

use std::io::{ErrorKind, Seek, SeekFrom};
use std::os::unix::fs::{FileExt, MetadataExt};
use turmoil::fs::shim::std::fs::{
    copy, create_dir, create_dir_all, read, read_to_string, remove_file, rename, write, File,
    OpenOptions,
};
use turmoil::fs::Fs;
use turmoil::{Builder, Result};

const TEST_PATH: &str = "/test/data.db";

fn open_rw(path: &str) -> std::io::Result<File> {
    let mut options = OpenOptions::new();
    options.read(true).write(true).create(true).open(path)
}

#[test]
fn basic_read_write() -> Result {
    let mut sim = Builder::new().build();
    sim.client("test", async {
        create_dir_all("/test")?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(TEST_PATH)?;
        file.write_all_at(b"hello world", 0)?;
        let mut buf = [0u8; 11];
        let n = file.read_at(&mut buf, 0)?;
        assert_eq!(n, 11);
        assert_eq!(&buf, b"hello world");
        Ok(())
    });
    sim.run()
}

#[test]
fn write_at_offset() -> Result {
    let mut sim = Builder::new().build();
    sim.client("test", async {
        create_dir_all("/test")?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(TEST_PATH)?;
        file.write_all_at(b"at offset", 100)?;
        let mut buf = [0u8; 9];
        file.read_at(&mut buf, 100)?;
        assert_eq!(&buf, b"at offset");
        Ok(())
    });
    sim.run()
}

#[test]
fn set_len_truncate() -> Result {
    let mut sim = Builder::new().build();
    sim.client("test", async {
        create_dir_all("/test")?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(TEST_PATH)?;
        file.write_all_at(b"0123456789", 0)?;
        file.sync_all()?;
        file.set_len(5)?;
        assert_eq!(file.metadata()?.len(), 5);
        Ok(())
    });
    sim.run()
}

#[test]
fn multiple_files() -> Result {
    let mut sim = Builder::new().build();
    sim.client("test", async {
        let f1 = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open("/file1")?;
        let f2 = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open("/file2")?;
        f1.write_all_at(b"file one", 0)?;
        f2.write_all_at(b"file two", 0)?;
        let mut b1 = [0u8; 8];
        let mut b2 = [0u8; 8];
        f1.read_at(&mut b1, 0)?;
        f2.read_at(&mut b2, 0)?;
        assert_eq!(&b1, b"file one");
        assert_eq!(&b2, b"file two");
        Ok(())
    });
    sim.run()
}

#[test]
fn read_write_convenience() -> Result {
    let mut sim = Builder::new().build();
    sim.client("test", async {
        create_dir("/data")?;
        write("/data/file.txt", b"hello world")?;
        assert_eq!(read("/data/file.txt")?, b"hello world");
        assert_eq!(read_to_string("/data/file.txt")?, "hello world");
        Ok(())
    });
    sim.run()
}

#[test]
fn copy_file() -> Result {
    let mut sim = Builder::new().build();
    sim.client("test", async {
        create_dir("/data")?;
        write("/data/source.txt", b"copy me")?;
        let bytes = copy("/data/source.txt", "/data/dest.txt")?;
        assert_eq!(bytes, 7);
        assert_eq!(read("/data/dest.txt")?, b"copy me");
        Ok(())
    });
    sim.run()
}

#[test]
fn set_len_extend() -> Result {
    let mut sim = Builder::new().build();
    sim.client("test", async {
        create_dir_all("/test")?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open("/test/data.db")?;
        file.set_len(100)?;
        assert_eq!(file.metadata()?.len(), 100);
        let mut buf = [0xffu8; 100];
        file.read_at(&mut buf, 0)?;
        assert!(buf.iter().all(|&b| b == 0));
        Ok(())
    });
    sim.run()
}

#[test]
fn open_options_create_new() -> Result {
    let mut sim = Builder::new().build();
    sim.client("test", async {
        create_dir_all("/test")?;
        let _file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(TEST_PATH)?;
        let result = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(TEST_PATH);
        assert!(result.is_err());
        Ok(())
    });
    sim.run()
}

#[test]
fn open_options_truncate() -> Result {
    let mut sim = Builder::new().build();
    sim.client("test", async {
        create_dir_all("/test")?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(TEST_PATH)?;
        file.write_all_at(b"existing data", 0)?;
        file.sync_all()?;
        drop(file);
        let file = OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(TEST_PATH)?;
        assert_eq!(file.metadata()?.len(), 0);
        Ok(())
    });
    sim.run()
}

#[test]
fn read_exact_at() -> Result {
    let mut sim = Builder::new().build();
    sim.client("test", async {
        create_dir_all("/test")?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(TEST_PATH)?;
        file.write_all_at(b"exact read test", 0)?;
        let mut buf = [0u8; 15];
        file.read_exact_at(&mut buf, 0)?;
        assert_eq!(&buf, b"exact read test");
        let mut buf2 = [0u8; 100];
        assert!(file.read_exact_at(&mut buf2, 0).is_err());
        Ok(())
    });
    sim.run()
}

#[test]
fn overlapping_writes() -> Result {
    let mut sim = Builder::new().build();
    sim.client("test", async {
        create_dir_all("/test")?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(TEST_PATH)?;
        file.write_all_at(&[b'A'; 10], 0)?;
        file.write_all_at(&[b'B'; 3], 3)?;
        let mut buf = [0u8; 10];
        file.read_at(&mut buf, 0)?;
        assert_eq!(&buf, b"AAABBBAAAA");
        Ok(())
    });
    sim.run()
}

#[test]
fn cursor_based_io() -> Result {
    use std::io::{Read, Seek, SeekFrom, Write};
    let mut sim = Builder::new().build();
    sim.client("test", async {
        create_dir("/data")?;
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open("/data/file.txt")?;
        file.write_all(b"hello world")?;
        file.seek(SeekFrom::Start(0))?;
        let mut buf = [0u8; 11];
        file.read_exact(&mut buf)?;
        assert_eq!(&buf, b"hello world");
        file.seek(SeekFrom::End(-5))?;
        let mut buf2 = [0u8; 5];
        file.read_exact(&mut buf2)?;
        assert_eq!(&buf2, b"world");
        Ok(())
    });
    sim.run()
}

#[test]
fn append_mode() -> Result {
    use std::io::Write;
    let mut sim = Builder::new().build();
    sim.client("test", async {
        create_dir("/data")?;
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .open("/data/log.txt")?;
        file.write_all_at(b"line1\n", 0)?;
        file.sync_all()?;
        drop(file);
        let mut file = OpenOptions::new().append(true).open("/data/log.txt")?;
        file.write_all(b"line2\n")?;
        drop(file);
        let file = OpenOptions::new().read(true).open("/data/log.txt")?;
        let mut buf = [0u8; 12];
        file.read_exact_at(&mut buf, 0)?;
        assert_eq!(&buf, b"line1\nline2\n");
        Ok(())
    });
    sim.run()
}

#[test]
fn file_try_clone() -> Result {
    let mut sim = Builder::new().build();
    sim.client("test", async {
        create_dir("/data")?;
        let file1 = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open("/data/file.txt")?;
        file1.write_all_at(b"hello world", 0)?;
        let file2 = file1.try_clone()?;
        let mut buf = [0u8; 5];
        file2.read_at(&mut buf, 6)?;
        assert_eq!(&buf, b"world");
        file2.write_all_at(b"HELLO", 0)?;
        let mut buf2 = [0u8; 5];
        file1.read_at(&mut buf2, 0)?;
        assert_eq!(&buf2, b"HELLO");
        Ok(())
    });
    sim.run()
}

#[test]
fn per_host_isolation() -> Result {
    use std::sync::Arc;
    use tokio::sync::Notify;
    let mut sim = Builder::new().build();
    let n1 = Arc::new(Notify::new());
    let n2 = Arc::new(Notify::new());
    let n1h = n1.clone();
    sim.host("host1", move || {
        let n = n1h.clone();
        async move {
            create_dir_all("/shared")?;
            write("/shared/data", b"from host1")?;
            n.notify_one();
            std::future::pending::<()>().await;
            Ok(())
        }
    });
    let n2h = n2.clone();
    sim.host("host2", move || {
        let n = n2h.clone();
        async move {
            create_dir_all("/shared")?;
            write("/shared/data", b"from host2")?;
            assert_eq!(read("/shared/data")?, b"from host2");
            n.notify_one();
            std::future::pending::<()>().await;
            Ok(())
        }
    });
    sim.client("test", async move {
        n1.notified().await;
        n2.notified().await;
        Ok(())
    });
    sim.run()
}

#[test]
fn as_raw_fd_yields_distinct_high_range_values() -> Result {
    use std::os::fd::AsRawFd;
    let mut sim = Builder::new().build();
    sim.client("test", async {
        create_dir_all("/test")?;
        let f1 = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open("/test/a")?;
        let f2 = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open("/test/b")?;
        // Distinct fds.
        assert_ne!(f1.as_raw_fd(), f2.as_raw_fd());
        // Both above SIM_FD_BASE, so they cannot collide with real OS fds.
        assert!(
            f1.as_raw_fd() >= 1 << 30,
            "fd {} below SIM_FD_BASE",
            f1.as_raw_fd()
        );
        assert!(
            f2.as_raw_fd() >= 1 << 30,
            "fd {} below SIM_FD_BASE",
            f2.as_raw_fd()
        );
        Ok(())
    });
    sim.run()
}

#[test]
fn as_raw_fd_isolated_per_host() -> Result {
    use std::os::fd::AsRawFd;
    use std::sync::Arc;
    use tokio::sync::Notify;
    let mut sim = Builder::new().build();
    // Each host opens one file; both expect the same fd integer because
    // the fd table is per-host (two independent counters starting at
    // SIM_FD_BASE).
    let n1 = Arc::new(Notify::new());
    let n2 = Arc::new(Notify::new());
    let n1h = n1.clone();
    sim.host("host1", move || {
        let n = n1h.clone();
        async move {
            create_dir_all("/h")?;
            let f = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .open("/h/f")?;
            assert_eq!(f.as_raw_fd(), 1 << 30);
            n.notify_one();
            std::future::pending::<()>().await;
            Ok(())
        }
    });
    let n2h = n2.clone();
    sim.host("host2", move || {
        let n = n2h.clone();
        async move {
            create_dir_all("/h")?;
            let f = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .open("/h/f")?;
            assert_eq!(f.as_raw_fd(), 1 << 30);
            n.notify_one();
            std::future::pending::<()>().await;
            Ok(())
        }
    });
    sim.client("test", async move {
        n1.notified().await;
        n2.notified().await;
        Ok(())
    });
    sim.run()
}

#[test]
fn truncate_then_extend_reads_zeros() -> Result {
    let mut sim = Builder::new().build();
    sim.client("test", async {
        let file = open_rw("/file")?;
        file.write_all_at(b"hello world", 0)?;
        file.set_len(0)?;
        file.set_len(11)?;
        let mut buf = [0xffu8; 11];
        file.read_exact_at(&mut buf, 0)?;
        assert_eq!(buf, [0u8; 11]);
        Ok(())
    });
    sim.run()
}

#[test]
fn open_file_follows_rename() -> Result {
    let mut sim = Builder::new().build();
    sim.client("test", async {
        let file = open_rw("/old")?;
        file.write_all_at(b"abc", 0)?;
        rename("/old", "/new")?;

        file.write_all_at(b"def", 3)?;
        file.sync_all()?;
        assert_eq!(read("/new")?, b"abcdef");

        let renamed = OpenOptions::new().write(true).open("/new")?;
        renamed.write_all_at(b"ghi", 6)?;
        let mut buf = [0u8; 9];
        file.read_exact_at(&mut buf, 0)?;
        assert_eq!(&buf, b"abcdefghi");
        Ok(())
    });
    sim.run()
}

#[test]
fn removed_file_lives_on_through_open_handle() -> Result {
    let mut sim = Builder::new().build();
    sim.client("test", async {
        let old = open_rw("/file")?;
        old.write_all_at(b"old", 0)?;
        remove_file("/file")?;

        let new = OpenOptions::new().write(true).create(true).open("/file")?;
        new.write_all_at(b"N", 0)?;

        let mut buf = [0u8; 3];
        old.read_exact_at(&mut buf, 0)?;
        assert_eq!(&buf, b"old");
        assert_eq!(old.metadata()?.nlink(), 0);
        assert_eq!(read("/file")?, b"N");
        Ok(())
    });
    sim.run()
}

#[test]
fn preallocated_space_reads_as_zeros() -> Result {
    let mut sim = Builder::new().build();
    sim.client("test", async {
        let file = open_rw("/wal")?;
        // Far more than fits in memory, were it stored.
        file.set_len(1 << 40)?;
        file.sync_all()?;
        assert_eq!(file.metadata()?.blocks(), 0);
        file.write_all_at(b"head", 0)?;
        file.write_all_at(b"x", 4096)?;
        assert_eq!(file.metadata()?.blocks(), 2);
        let mut tail = [0xffu8; 8];
        file.read_exact_at(&mut tail, (1 << 40) - 8)?;
        assert_eq!(tail, [0u8; 8]);
        assert_eq!(file.metadata()?.len(), 1 << 40);
        Ok(())
    });
    sim.run()
}

/// A write far into a file stores what was written, not the hole before it.
#[test]
fn write_at_the_end_of_a_sparse_file() -> Result {
    let mut sim = Builder::new().build();
    sim.client("test", async {
        let mut file = open_rw("/wal")?;
        file.set_len(1 << 40)?;
        file.write_all_at(b"tail", (1 << 40) - 4)?;
        file.sync_all()?;
        let mut tail = [0xffu8; 8];
        file.read_exact_at(&mut tail, (1 << 40) - 8)?;
        assert_eq!(&tail, b"\0\0\0\0tail");

        let max_len = i64::MAX as u64;
        file.write_all_at(b"x", max_len - 1)?;
        let mut byte = [0u8; 1];
        file.read_exact_at(&mut byte, max_len - 1)?;
        assert_eq!(&byte, b"x");
        assert_eq!(file.seek(SeekFrom::End(0))?, max_len);
        let invalid = file.seek(SeekFrom::Current(1)).unwrap_err();
        assert_eq!(invalid.kind(), ErrorKind::InvalidInput);
        for len in [max_len + 1, u64::MAX] {
            assert_eq!(
                file.set_len(len).unwrap_err().kind(),
                ErrorKind::FileTooLarge
            );
        }
        let too_big = file.write_at(b"x", max_len).unwrap_err();
        assert_eq!(too_big.kind(), ErrorKind::FileTooLarge);
        assert_eq!(file.stream_position()?, max_len);
        assert_eq!(file.metadata()?.len(), max_len);
        Ok(())
    });
    sim.run()
}

/// A file belongs to the filesystem it was opened on. On another one its
/// descriptor names nothing, rather than whatever has the same number there.
#[test]
fn file_does_not_reach_into_another_filesystem() -> std::io::Result<()> {
    let (ours, theirs) = (Fs::builder().build(), Fs::builder().build());
    let on_ours = ours.enter();
    let (mut file, other) = (open_rw("/file")?, open_rw("/other")?);
    file.write_all_at(b"ours", 0)?;
    drop(on_ours);

    // The first descriptor has the same number on either filesystem.
    let on_theirs = theirs.enter();
    let same_number = open_rw("/file")?;
    same_number.write_all_at(b"theirs", 0)?;
    let stale = [
        file.read_at(&mut [0; 4], 0).map(drop),
        file.write_at(b"lost", 0).map(drop),
        file.set_len(0),
        file.sync_all(),
        file.try_clone().map(drop),
        file.seek(SeekFrom::Start(2)).map(drop),
        file.seek(SeekFrom::Current(1)).map(drop),
    ];
    for result in stale {
        assert_eq!(result.unwrap_err().kind(), ErrorKind::NotFound);
    }
    assert_eq!(read("/file")?, b"theirs");
    drop(on_theirs);

    let _on_ours = ours.enter();
    assert_eq!(file.stream_position()?, 0);
    assert_eq!(read("/file")?, b"ours");
    assert_eq!(other.metadata()?.len(), 0);
    ours.close_all();
    for pos in [SeekFrom::Start(1), SeekFrom::Current(1), SeekFrom::End(0)] {
        assert_eq!(file.seek(pos).unwrap_err().kind(), ErrorKind::NotFound);
    }
    Ok(())
}

#[test]
fn file_handle_can_outlive_the_host() -> std::io::Result<()> {
    use std::sync::{Arc, Mutex};
    use turmoil::fs::{enter, FsState};

    let state = Arc::new(Mutex::new(FsState::default()));
    let owner = Arc::downgrade(&state);
    let file = {
        let _entered = enter(&state);
        File::create("/file")?
    };
    drop(state);
    assert!(owner.upgrade().is_none());
    drop(file);
    Ok(())
}
