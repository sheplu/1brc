//! Diagnostic: the cost of merely reading the file, with no parsing at all.
//!
//! Establishes the floor that any implementation is bounded by, and A/Bs the two ways of
//! getting at the bytes on Darwin: a shared mmap (cheap, but takes a minor fault per 16 KB
//! page, on a `vm_map` that all workers contend for) versus `pread` into a per-thread
//! buffer (an extra copy, but no faults).
//!
//! Usage: io_floor <mmap|pread> [path]   (OBRC_THREADS, default 8)

use std::env;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use obrc::sys::{set_thread_qos_user_interactive, Mapping};

const CHUNK: usize = 2 << 20;

/// XOR-reduce so the reads cannot be optimised away.
#[inline]
fn reduce(bytes: &[u8]) -> u64 {
    let mut acc = 0u64;
    let mut chunks = bytes.chunks_exact(8);
    for c in &mut chunks {
        acc ^= u64::from_le_bytes(c.try_into().unwrap());
    }
    for &b in chunks.remainder() {
        acc ^= b as u64;
    }
    acc
}

fn main() {
    let mode = env::args().nth(1).unwrap_or_else(|| "mmap".to_string());
    let path = env::args().nth(2).unwrap_or_else(|| "measurements.txt".to_string());
    let threads =
        env::var("OBRC_THREADS").ok().and_then(|s| s.parse().ok()).filter(|&n| n > 0).unwrap_or(8);

    let file = File::open(&path).expect("open");
    let len = file.metadata().expect("stat").len() as usize;
    let total = len.div_ceil(CHUNK);
    let cursor = AtomicUsize::new(0);
    let acc = AtomicU64::new(0);

    let start = Instant::now();
    match mode.as_str() {
        "mmap" => {
            let mapping = Mapping::open(&path).expect("mmap");
            let data = mapping.as_slice();
            std::thread::scope(|s| {
                for _ in 0..threads {
                    s.spawn(|| {
                        set_thread_qos_user_interactive();
                        let mut local = 0u64;
                        loop {
                            let i = cursor.fetch_add(1, Ordering::Relaxed);
                            if i >= total {
                                break;
                            }
                            let a = i * CHUNK;
                            let b = (a + CHUNK).min(len);
                            local ^= reduce(&data[a..b]);
                        }
                        acc.fetch_xor(local, Ordering::Relaxed);
                    });
                }
            });
        }
        "pread" => {
            std::thread::scope(|s| {
                for _ in 0..threads {
                    s.spawn(|| {
                        set_thread_qos_user_interactive();
                        let mut buf = vec![0u8; CHUNK];
                        let mut local = 0u64;
                        loop {
                            let i = cursor.fetch_add(1, Ordering::Relaxed);
                            if i >= total {
                                break;
                            }
                            let a = i * CHUNK;
                            let n = CHUNK.min(len - a);
                            file.read_exact_at(&mut buf[..n], a as u64).expect("pread");
                            local ^= reduce(&buf[..n]);
                        }
                        acc.fetch_xor(local, Ordering::Relaxed);
                    });
                }
            });
        }
        other => panic!("unknown mode {other:?}, expected mmap or pread"),
    }

    let secs = start.elapsed().as_secs_f64();
    eprintln!(
        "{mode:>5}  {threads} threads  {:.3} s  {:.1} GB/s  (checksum {:016x})",
        secs,
        len as f64 / secs / 1e9,
        acc.load(Ordering::Relaxed)
    );
}
