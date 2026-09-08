//! v9: v8, with the `;` scan's loop replaced by one fixed 16-byte window.
//!
//! Every version up to here scanned for the `;` a word at a time and stopped on the first
//! one. Station names average 8.07 bytes, which sounds like the loop almost always runs
//! once — but the average hides the shape. 50.4% of the official names are 7 bytes or
//! fewer and 46.7% are 8..=15, so the loop exits after one iteration or after two on a
//! near-even split, and the generator draws stations independently. The exit test is
//! therefore unpredictable *by construction*: no branch history can beat a coin flip on it,
//! and a billion rows each paid about half a misprediction.
//!
//! It stayed invisible because every ablation mode contained it. `parse`, `hash`, `table`
//! and `itable` all scan the same way, so the cost cancelled out of every difference the
//! harness reported and never appeared as a line item.
//!
//! [`find_semi_and_hash_flat`] reads both words up front, selects the delimiter without
//! branching, and folds the hash with two multiplies. More instructions, fewer cycles, and
//! the same hash bits as the loop it replaces. Measured against v8's engine in one batch:
//! 0.720 s → 0.621 s at 18 threads and 1.365 s → 1.128 s at 8. Over the 0.225 s read floor
//! that is 495 ms of compute down to 396 ms — a fifth of the whole budget, from one branch.
//!
//! Names of 16 bytes or more (2.9% of the list) take a cold re-scan through v8's loop. That
//! branch is biased ~33:1, so it predicts.
//!
//! Everything else is v8: `pread` into a recycled per-thread buffer, an [`InlineTable`] that
//! owns its keys because an offset into that buffer would dangle, and no scalar tail because
//! the buffer keeps slack past the last byte read.
//!
//! Usage: v9_flatscan [path]   (thread count via OBRC_THREADS, default 8)

use std::collections::BTreeMap;
use std::env;
use std::fs::File;
use std::io::Write;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicUsize, Ordering};

use obrc::chunk::{local_bounds, num_chunks, split_streams, CHUNK_SIZE};
use obrc::inline_table::{InlineTable, KEY_SLACK};
use obrc::output::{write_results, Stats};
use obrc::parse::parse_temp_branchless;
use obrc::swar::{find_semi_and_hash_flat, FLAT_SLACK};
use obrc::sys::set_thread_qos_user_interactive;
use obrc::MAX_LINE_LEN;

const TABLE_BITS: u32 = 16;

/// Same as v7 — see the note there on why 4 and above lose.
const STREAMS: usize = 3;

/// Read past the chunk so the line straddling its end can be finished locally. A worker
/// never sees another worker's bytes, so this overlap is the only way it can agree with its
/// neighbour on where the boundary line belongs.
const OVERLAP: usize = MAX_LINE_LEN;

/// Slack past the read that the wide path may touch: the flat scan reads a 16-byte window
/// from the name start, the parser loads 8 bytes, and the inline probe reads 32.
const TAIL_GUARD: usize = MAX_LINE_LEN + FLAT_SLACK;
const _: () = assert!(TAIL_GUARD >= KEY_SLACK);

/// Consumes the line at `pos` and returns the start of the next one.
#[inline(always)]
fn step(buf: &[u8], table: &mut InlineTable, pos: usize) -> usize {
    let (semi, hash) = find_semi_and_hash_flat(buf, pos);
    let (value, next) = parse_temp_branchless(buf, semi + 1);
    table.upsert(buf, pos, semi - pos, hash, value);
    next
}

fn worker(
    file: &File,
    len: usize,
    chunk: usize,
    cursor: &AtomicUsize,
    total: usize,
) -> InlineTable {
    set_thread_qos_user_interactive();
    let mut table = InlineTable::new(TABLE_BITS);
    // Allocated once and reused. Bytes past what `pread` returns are stale, never consumed:
    // every row ends at or before `end`, and the over-reads past a row are masked off.
    let mut buf = vec![0u8; chunk + OVERLAP + TAIL_GUARD];

    loop {
        let i = cursor.fetch_add(1, Ordering::Relaxed);
        if i >= total {
            break;
        }
        let base = i * chunk;
        if base >= len {
            continue;
        }
        let avail = (chunk + OVERLAP).min(len - base);
        file.read_exact_at(&mut buf[..avail], base as u64).expect("pread");
        let view = &buf[..avail];

        let Some((start, end)) = local_bounds(view, base, chunk) else {
            continue;
        };

        let mut s = split_streams::<STREAMS>(view, start, end);

        loop {
            let mut all = true;
            for k in 0..STREAMS {
                all &= s[k].0 < s[k].1;
            }
            if !all {
                break;
            }
            for k in 0..STREAMS {
                s[k].0 = step(&buf, &mut table, s[k].0);
            }
        }

        for k in 0..STREAMS {
            while s[k].0 < s[k].1 {
                s[k].0 = step(&buf, &mut table, s[k].0);
            }
        }
    }
    table
}

fn main() {
    let path = env::args().nth(1).unwrap_or_else(|| "measurements.txt".to_string());
    let threads =
        env::var("OBRC_THREADS").ok().and_then(|s| s.parse().ok()).filter(|&n| n > 0).unwrap_or(8);

    // Unlike the mapped versions, the chunk size here also sets the per-thread buffer that
    // is written then read straight back. Swept at 18 threads: 64 KB costs 877 ms and twice
    // the system time, 1 MiB is the floor at 676 ms, and it is flat past that. Syscall count
    // dominates; keeping the buffer inside L2 turns out not to matter. The default is left
    // at the shared [`CHUNK_SIZE`] — 1 MiB wins by ~1%, which is not worth diverging from
    // the rest of the ladder for.
    let chunk = env::var("OBRC_CHUNK")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n| n > MAX_LINE_LEN)
        .unwrap_or(CHUNK_SIZE);

    let file = File::open(&path).unwrap_or_else(|e| panic!("cannot open {path}: {e}"));
    let len = file.metadata().expect("stat").len() as usize;
    let total = num_chunks(len, chunk);
    let cursor = AtomicUsize::new(0);

    let tables = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| scope.spawn(|| worker(&file, len, chunk, &cursor, total)))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect::<Vec<_>>()
    });

    let mut merged: BTreeMap<&[u8], Stats> = BTreeMap::new();
    for t in &tables {
        for (name, stats) in t.iter() {
            merged.entry(name).and_modify(|s| s.merge(&stats)).or_insert(stats);
        }
    }

    let entries: Vec<(Vec<u8>, Stats)> = merged.into_iter().map(|(k, v)| (k.to_vec(), v)).collect();

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    write_results(&mut out, &entries).unwrap();
    out.flush().unwrap();
    std::process::exit(0);
}
