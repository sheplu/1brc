//! v13: fewer instructions per row, and slower for it. **This version does not ship.**
//!
//! It is kept because it is the cleanest counterexample in the project: the change did exactly
//! what the disassembly said it would — 116 instructions per row down to 96, four stack reloads
//! down to zero — and lost 15 ms anyway. Read it before believing an instruction count again.
//!
//! ```text
//! t18   v12 442 ms   v13 457 ms          t8   v12 792 ms   v13 827 ms
//! ```
//!
//! **The observation.** v12's sweep settled that the parse loop is not latency bound — more
//! interleaving made it slower, monotonically — so the ~313 ms of computation looked like
//! throughput, and the only thing that moves throughput is fewer instructions per row. `objdump
//! -d target/release/v12_pairs` said where they were, and neither was algorithmic.
//!
//! *Four stack loads of the table header, every row.* `upsert_words` re-reads `shift`,
//! `entries.len`, `entries.ptr` and `mask` from the frame each time it is called, and one of
//! them sits on the chain that ends at the entry's cache line:
//!
//! ```text
//! ldr  w12, [sp, #0xd8]   <- self.shift
//! lsr  x0,  x5, x12       <- slot()
//! ldr  x1,  [sp, #0xc0]   <- entries.len()
//! cmp  x0,  x1
//! b.hs <panic>
//! ...
//! ldr  x11, [sp, #0xb8]   <- entries.as_ptr()
//! ldr  x12, [sp, #0xc8]   <- self.mask
//! add  x8,  x11, x0, lsl #6
//! ldp  x13, x14, [x8]     <- key[0]
//! ```
//!
//! The cause is that `&mut InlineTable` stays live across `insert`, which may `grow` and replace
//! `entries`. LLVM cannot know it did not, so the header is dead after every call.
//!
//! *About eleven bounds-check instructions.* Four for the scan's two 8-byte name-window loads,
//! three for the parser's, and two apiece for the initial slot and the probe's back-edge. Nothing
//! in the types relates a row cursor to the buffer length, so each load re-proves it.
//!
//! **The change.** Two, and they are independent.
//!
//! [`InlineTable::hot_slots`] hands out `(&mut [Entry], u32)` and [`probe_words`] runs the probe
//! against the borrowed slice. The borrow checker proves what LLVM could not: while the slice is
//! live nothing can reallocate `entries`, so all four header values are loop-invariant and hoist
//! out of the chunk. The mask comes from `slots.len() - 1` rather than a field, which is what
//! removes the two remaining checks — `x & (len-1)` is provably `< len`, whereas two unrelated
//! arguments prove nothing. `InlineTable::grow` already compiled that way and `upsert_words` did
//! not, which is how the trick was found.
//!
//! The price is that a row which has to *insert* cannot run under the borrow. `probe_words`
//! returns `false` having written nothing, the run ends, the borrow drops, and that row is redone
//! through v12's [`step`] — which owns the table and may insert and grow. That is once per
//! distinct station, 413 times per thread over the whole file, so the repeated scan is free.
//!
//! [`scan_flat_keyed_win`] and [`parse_temp_branchless_win`] take `&[u8; 16]` and `&[u8; 8]`
//! instead of a slice and an index. The caller fetches each window once with `first_chunk`, and
//! the length travels in the type from there, so the arithmetic inside checks nothing. They are
//! copies of the slice forms rather than the slice forms rewritten, deliberately: `objdump` on
//! v12 had to stay byte-identical, or the comparison this version rests on is not a comparison.
//! Two tests named `win_agrees_with_the_slice_form` hold the copies equal.
//!
//! Everything else is v12: `pread` into a recycled per-thread buffer, two interleaved streams, the
//! fixed 16-byte `;` window, no final avalanche, a table that grows on demand.
//!
//! **What happened.** Both changes landed in the object code. In the hot loop the four
//! `ldr [sp, …]` are gone, `shift` lives in `w15`, `entries.ptr` in `x26` and `mask = len-1` in
//! `x28`, all set once when the borrow is taken; the slot index is `and x0, x10, x28` with no
//! compare, and the probe's back-edge is `and x0, x11, x28` with no compare. Per row that is 96
//! instructions against v12's 116.
//!
//! Both are also slower, separately and together, and the two costs add. From one interleaved
//! batch of `hot_floor` ablations, warm reader at 18 threads: `keyed2` 341 ms, `win2` 346,
//! `hdr2` 351, `hot2` 356. Same ordering at 8 threads and under `pread`, and the same 15 ms at
//! 18 threads in the end-to-end batch above — so it is not an artifact of the ablation harness.
//!
//! The mechanism is visible in the same disassembly. v12's eleven checks branch to *panic*
//! blocks: never taken, and nothing is live at the target, so the register allocator ignores
//! them. v13's replacements — `get(pos..)?`, `first_chunk()?`, and the probe returning `false` —
//! are real loop exits, and every loop-carried value has to be materialized at each one. The
//! preamble gains spills v12 never needed (`stur x16, [x29, #-0x98]`, `str x16, [sp, #0x10]`).
//! Trading a never-taken fused compare-and-branch for a live-range constraint is a bad trade,
//! and instruction count does not show it.
//!
//! Taken with v12's stream sweep and the NEON delimiter-bitmap result, the reading is that this
//! loop is not bound by anything the source can address: not latency, not byte scanning, and not
//! instruction count either.
//!
//! Usage: v13_hoist [path]   (OBRC_THREADS, default 8; OBRC_TABLE_BITS, default 14)

use std::collections::BTreeMap;
use std::env;
use std::fs::File;
use std::io::Write;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicUsize, Ordering};

use obrc::chunk::{local_bounds, num_chunks, split_streams, CHUNK_SIZE};
use obrc::inline_table::{probe_words, Entry, InlineTable, KEY_SLACK};
use obrc::output::{write_results, Stats};
use obrc::parse::{parse_temp_branchless, parse_temp_branchless_win};
use obrc::swar::{find_semi_and_hash_flat_raw, scan_flat_keyed, scan_flat_keyed_win, FLAT_SLACK};
use obrc::sys::set_thread_qos_user_interactive;
use obrc::MAX_LINE_LEN;

/// See v10: 2^13..2^16 all measure the same, and this is the smallest of them.
const DEFAULT_TABLE_BITS: u32 = 14;

/// See v12: measured, and a property of the core rather than of the program.
const STREAMS: usize = 2;

/// Read past the chunk so the line straddling its end can be finished locally.
const OVERLAP: usize = MAX_LINE_LEN;

/// Slack past the read that the wide path may touch: the flat scan reads a 16-byte window
/// from the name start, the parser loads 8 bytes, and the cold path's inline probe reads 32.
const TAIL_GUARD: usize = MAX_LINE_LEN + FLAT_SLACK;
const _: () = assert!(TAIL_GUARD >= KEY_SLACK);

/// The whole row against borrowed slots, or `None` with nothing written.
///
/// Declines on a name too long for the 16-byte window and on a probe that reached an empty slot.
/// Both are rare and both need the table itself, so both go back to [`step`].
#[inline(always)]
fn step_hot(buf: &[u8], slots: &mut [Entry], shift: u32, pos: usize) -> Option<usize> {
    let (len, hash, klo, khi) = scan_flat_keyed_win(buf.get(pos..)?.first_chunk()?)?;
    let val = pos + len + 1;
    let (value, adv) = parse_temp_branchless_win(buf.get(val..)?.first_chunk()?);

    let key_lo = (klo as u128) | ((khi as u128) << 64);
    probe_words(slots, shift, hash, key_lo, value).then_some(val + adv)
}

/// v12's row, kept whole: it owns the table, so it is the one that can insert and grow.
#[inline(always)]
fn step(buf: &[u8], table: &mut InlineTable, pos: usize) -> usize {
    let Some((semi, hash, klo, khi)) = scan_flat_keyed(buf, pos) else {
        return step_long(buf, table, pos);
    };
    let (value, next) = parse_temp_branchless(buf, semi + 1);
    table.upsert_words(buf, pos, semi - pos, hash, value, klo, khi);
    next
}

/// A name of 16 bytes or more: v10's row, out of line so it shares no registers with the hot
/// path. 2.9% of the official station list and none of this dataset's common names.
#[cold]
#[inline(never)]
fn step_long(buf: &[u8], table: &mut InlineTable, pos: usize) -> usize {
    let (semi, hash) = find_semi_and_hash_flat_raw(buf, pos);
    let (value, next) = parse_temp_branchless(buf, semi + 1);
    table.upsert(buf, pos, semi - pos, hash, value);
    next
}

/// Walks `[start, end)` as `STREAMS` interleaved line streams, holding the table header in
/// registers for as long as no row needs to insert.
///
/// The outer loop exists only to re-take the borrow after a row that declined. Streams that
/// already advanced within an interrupted batch just carry on from where they are — each stream
/// owns a disjoint range and moves through it monotonically, so an interrupted batch loses
/// nothing but the interleaving of that one iteration.
#[inline(always)]
fn run(buf: &[u8], table: &mut InlineTable, start: usize, end: usize) {
    let mut s = split_streams::<STREAMS>(buf, start, end);

    loop {
        // `STREAMS` when the interleaved loop ran out of rows, otherwise the stream that declined.
        let cold = {
            let (slots, shift) = table.hot_slots();
            'run: loop {
                let mut all = true;
                for k in 0..STREAMS {
                    all &= s[k].0 < s[k].1;
                }
                if !all {
                    break 'run STREAMS;
                }
                for k in 0..STREAMS {
                    let Some(next) = step_hot(buf, slots, shift, s[k].0) else {
                        break 'run k;
                    };
                    s[k].0 = next;
                }
            }
        };
        if cold == STREAMS {
            break;
        }
        s[cold].0 = step(buf, table, s[cold].0);
    }

    for k in 0..STREAMS {
        while s[k].0 < s[k].1 {
            s[k].0 = step(buf, table, s[k].0);
        }
    }
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

        let Some((start, end)) = local_bounds(&buf[..avail], base, chunk) else {
            continue;
        };
        run(&buf, &mut table, start, end);
    }
    table
}

fn main() {
    let path = env::args().nth(1).unwrap_or_else(|| "measurements.txt".to_string());
    let threads =
        env::var("OBRC_THREADS").ok().and_then(|s| s.parse().ok()).filter(|&n| n > 0).unwrap_or(8);

    let chunk = env::var("OBRC_CHUNK")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n| n > MAX_LINE_LEN)
        .unwrap_or(CHUNK_SIZE);

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

    // Deliberate: 18 tables, each with a key arena, would otherwise be walked and freed here
    // for no purpose. That teardown is part of the wall clock a benchmark sees.
    std::process::exit(0);
}
