//! Diagnostic: the cost of merely reading the file, with no parsing at all.
//!
//! Establishes the floor that any implementation is bounded by, and A/Bs the ways of
//! getting at the bytes on Darwin: a shared mmap (cheap, but takes a minor fault per 16 KB
//! page, on a `vm_map` that all workers contend for) versus `pread` into a per-thread
//! buffer (an extra copy, but no faults).
//!
//! Modes:
//!   mmap         MAP_PRIVATE, XOR-reduce every byte
//!   mmap_shared  MAP_SHARED instead — no copy-on-write shadow object to set up
//!   mmap_seq     MAP_PRIVATE + MADV_SEQUENTIAL
//!   mmap_willneed  MAP_PRIVATE + MADV_WILLNEED
//!   mmap_touch   as `mmap`, but faults every page up front, one read per page
//!   mmap_mlock   as `mmap`, but faults every page up front by wiring it
//!   pread        one shared `File`, `pread` per chunk into a recycled buffer
//!   pread_fd     as `pread`, but each thread opens its own descriptor
//!   pread_nored  as `pread`, without the XOR reduce — copy cost alone
//!
//! The two prefault modes split the mmap number in two. A traversal of already-resident pages
//! runs at DRAM speed; `mmap` measures several times that, so nearly all of it is the 842k
//! minor faults rather than the reading. Both report the prefault separately and then the same
//! total as `mmap`, so the two are directly comparable and the split is visible.
//!
//! `pread` and `pread_fd` differ only in whether the workers share one file description.
//! If the second scales where the first does not, the ceiling is contention on that shared
//! structure rather than memory bandwidth, which is a fixable thing rather than a physical
//! limit.
//!
//! Usage: io_floor <mode> [path]   (OBRC_THREADS default 8; OBRC_CHUNK in bytes)

use std::env;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use obrc::sys::{
    set_thread_qos_user_interactive, Mapping, MADV_SEQUENTIAL, MADV_WILLNEED, MAP_PRIVATE,
    MAP_SHARED,
};

const DEFAULT_CHUNK: usize = 2 << 20;

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

/// Runs `worker` on `threads` threads and XOR-folds what they return.
fn spawn_all(threads: usize, worker: impl Fn() -> u64 + Sync) -> u64 {
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    set_thread_qos_user_interactive();
                    worker()
                })
            })
            .collect();
        handles.into_iter().fold(0u64, |a, h| a ^ h.join().unwrap())
    })
}

/// Hands out chunk indices until they run out, yielding `[a, b)` byte ranges.
struct Chunks<'a> {
    cursor: &'a AtomicUsize,
    total: usize,
    chunk: usize,
    len: usize,
}

impl Chunks<'_> {
    #[inline]
    fn next_range(&self) -> Option<(usize, usize)> {
        let i = self.cursor.fetch_add(1, Ordering::Relaxed);
        if i >= self.total {
            return None;
        }
        let a = i * self.chunk;
        Some((a, (a + self.chunk).min(self.len)))
    }
}

/// Page size on this platform. Not queried: Darwin arm64 is 16 KB everywhere, and a wrong
/// guess here would only change how many redundant touches the prefault does.
const PAGE: usize = 16384;

/// Installs the PTEs for every chunk, on `threads` threads, and returns the seconds it took.
///
/// `wire` uses `mlock`, which asks the kernel to do a whole range at once; the other path pays
/// an ordinary minor fault per page, forced by a volatile read that cannot be elided. Both
/// walk the same chunk cursor, so the two differ only in the mechanism.
fn prefault(data: &[u8], chunks: &Chunks, threads: usize, mapping: &Mapping, wire: bool) -> f64 {
    let start = Instant::now();
    spawn_all(threads, || {
        while let Some((a, b)) = chunks.next_range() {
            if wire {
                mapping.wire(a, b - a).expect("mlock");
            } else {
                let mut p = a;
                while p < b {
                    unsafe { core::ptr::read_volatile(data.as_ptr().add(p)) };
                    p += PAGE;
                }
            }
        }
        0
    });
    chunks.cursor.store(0, Ordering::Relaxed);
    start.elapsed().as_secs_f64()
}

fn main() {
    let mode = env::args().nth(1).unwrap_or_else(|| "mmap".to_string());
    let path = env::args().nth(2).unwrap_or_else(|| "measurements.txt".to_string());
    let threads =
        env::var("OBRC_THREADS").ok().and_then(|s| s.parse().ok()).filter(|&n| n > 0).unwrap_or(8);
    let chunk = env::var("OBRC_CHUNK")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_CHUNK);

    let file = File::open(&path).expect("open");
    let len = file.metadata().expect("stat").len() as usize;
    let cursor = AtomicUsize::new(0);
    let chunks = Chunks { cursor: &cursor, total: len.div_ceil(chunk), chunk, len };

    let mmap_flags = match mode.as_str() {
        "mmap_shared" => Some(MAP_SHARED),
        "mmap" | "mmap_seq" | "mmap_willneed" | "mmap_touch" | "mmap_mlock" => Some(MAP_PRIVATE),
        _ => None,
    };

    let start = Instant::now();
    let acc = match mmap_flags {
        // The mapping and any advice are inside the timed window on purpose: they are work
        // a real run would have to do, and the whole question with mmap is what the kernel
        // charges for setting it up and faulting it in.
        Some(flags) => {
            let mapping = Mapping::open_with(&path, flags).expect("mmap");
            match mode.as_str() {
                "mmap_seq" => mapping.advise(MADV_SEQUENTIAL),
                "mmap_willneed" => mapping.advise(MADV_WILLNEED),
                _ => {}
            }
            let data = mapping.as_slice();
            match mode.as_str() {
                "mmap_touch" | "mmap_mlock" => {
                    let wire = mode == "mmap_mlock";
                    let secs = prefault(data, &chunks, threads, &mapping, wire);
                    eprintln!("  prefault {secs:.3} s");
                }
                _ => {}
            }
            spawn_all(threads, || {
                let mut local = 0u64;
                while let Some((a, b)) = chunks.next_range() {
                    local ^= reduce(&data[a..b]);
                }
                local
            })
        }
        None => {
            let reduced = mode != "pread_nored";
            // `pread_fd` gives each worker its own descriptor; the others share one. The
            // shared `&File` is the arrangement v8 ships, so it is the one to beat.
            let per_thread_fd = mode == "pread_fd";
            match mode.as_str() {
                "pread" | "pread_fd" | "pread_nored" => {}
                other => panic!("unknown mode {other:?}"),
            }
            spawn_all(threads, || {
                let owned = per_thread_fd.then(|| File::open(&path).expect("open"));
                let f = owned.as_ref().unwrap_or(&file);
                let mut buf = vec![0u8; chunk];
                let mut local = 0u64;
                while let Some((a, b)) = chunks.next_range() {
                    let dst = &mut buf[..b - a];
                    f.read_exact_at(dst, a as u64).expect("pread");
                    if reduced {
                        local ^= reduce(dst);
                    } else {
                        // Keep the copy observable without walking the buffer.
                        local ^= dst[0] as u64;
                    }
                }
                local
            })
        }
    };

    let secs = start.elapsed().as_secs_f64();
    eprintln!(
        "{mode:>13}  {threads:>2} threads  {:>7} chunk  {secs:.3} s  {:.1} GB/s  \
         ({:.2} core-s, checksum {acc:016x})",
        format!("{}K", chunk >> 10),
        len as f64 / secs / 1e9,
        secs * threads as f64,
    );
}
