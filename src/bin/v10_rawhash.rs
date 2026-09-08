//! v10: v9, with the hash's final avalanche deleted. The smallest shipped win here, and the
//! one whose planned justification turned out to be wrong.
//!
//! **What works.** [`find_semi_and_hash_flat`](obrc::swar::find_semi_and_hash_flat) ends in
//! `h ^= h>>32; h *= K; h ^= h>>32` — five operations, strictly serial, sitting between the
//! last `mix` and the table load. Nothing in the row can proceed until the slot index is
//! known, so they cost their full latency. They exist to spread the *low* bits, because the
//! low bits of a multiply are barely mixed: bit 0 of a product is just the product of the
//! inputs' bit 0s. [`InlineTable`] now takes its slot off the *top* of the hash instead, and
//! the top of a multiply is already well mixed, so the avalanche buys nothing the table uses.
//!
//! Worth 2–4% of the hot loop, holding at every table size and thread count tested. Whole
//! binary: **1.098 s → 1.085 s at 8 threads, 590 ms → 575 ms at 18** (one batch, 9 reps).
//!
//! **What does not.** The plan going in was that 2^16 slots — 4 MiB per thread for 413 live
//! entries — thrashes the L1 dTLB, and that shrinking to 2^11 (128 KB, permanently mapped)
//! would be worth 40–70 ms. It is worth nothing. 2^11 is 6% *worse* and 2^12 is 2% worse,
//! because each extra probe is another dependent load and a low load factor buys more probes
//! than it saves TLB walks; from 2^13 to 2^16 the whole-binary time is flat inside 4 ms.
//! The default is 2^14 for the memory — 18 MiB of tables at 18 threads instead of 72 — and
//! not for the speed, which it does not change.
//!
//! An intermediate batch did show the size winning 2.3%, and a thread sweep showed the
//! v9→v10 gap widening from 0.1% at 6 threads to 6.1% at 18. Both evaporated at higher rep
//! counts. That is the house rule doing its job on the house: a 2% effect on this machine is
//! not distinguishable from drift, and a *trend* assembled from six such effects is not
//! either.
//!
//! Shrinking the table at all is only safe because [`InlineTable`] grows on demand. v9 would
//! have aborted on more distinct names than it had slots; this cannot, at any `bits`.
//!
//! Everything else is v9: the fixed-window `;` scan, `pread` into a recycled per-thread
//! buffer, three interleaved line streams.
//!
//! Usage: v10_rawhash [path]   (OBRC_THREADS, default 8; OBRC_TABLE_BITS, default 14)

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
use obrc::swar::{find_semi_and_hash_flat_raw, FLAT_SLACK};
use obrc::sys::set_thread_qos_user_interactive;
use obrc::MAX_LINE_LEN;

/// 2^14 slots, 1 MiB per thread. A floor, not a capacity — the table doubles on demand.
/// Chosen for footprint: 2^13..2^16 all measure the same, and below that probing costs.
const DEFAULT_TABLE_BITS: u32 = 14;

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
    let (semi, hash) = find_semi_and_hash_flat_raw(buf, pos);
    let (value, next) = parse_temp_branchless(buf, semi + 1);
    table.upsert(buf, pos, semi - pos, hash, value);
    next
}

fn worker(
    file: &File,
    len: usize,
    chunk: usize,
    bits: u32,
    cursor: &AtomicUsize,
    total: usize,
) -> InlineTable {
    set_thread_qos_user_interactive();
    let mut table = InlineTable::new(bits);
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

    // See v9 for the sweep behind this: syscall count dominates, and it is flat past 1 MiB.
    let chunk = env::var("OBRC_CHUNK")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n| n > MAX_LINE_LEN)
        .unwrap_or(CHUNK_SIZE);

    // Exposed because the size turned out to trade two costs against each other rather than
    // just one, and the knob is how that was found. See the module note.
    let bits = env::var("OBRC_TABLE_BITS")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&b| (1..=24).contains(&b))
        .unwrap_or(DEFAULT_TABLE_BITS);

    let file = File::open(&path).unwrap_or_else(|e| panic!("cannot open {path}: {e}"));
    let len = file.metadata().expect("stat").len() as usize;
    let total = num_chunks(len, chunk);
    let cursor = AtomicUsize::new(0);

    let tables = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| scope.spawn(|| worker(&file, len, chunk, bits, &cursor, total)))
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
