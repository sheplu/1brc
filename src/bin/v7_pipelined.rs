//! v7: v6, but each worker walks several independent line streams at once.
//!
//! Rows form a serial dependency chain — a row's start address is only known once the
//! previous row's temperature has been parsed — so a single stream leaves most of the
//! core's out-of-order window idle. `hot_floor` measures the headroom directly: parsing
//! alone drops from 1.24 s to 0.74 s when split across four streams.
//!
//! Tried once before against v5 and it was a *loss*. The reason was the table: with keys
//! stored as offsets, each upsert ended in a load-modify-store to a shared array that the
//! streams had to serialise on, which cost more than the overlap won. v6 moved the key onto
//! the entry's own cache line, so the upsert is now a short L1-resident update and the
//! streams have much less to contend over.
//!
//! Usage: v7_pipelined [path]   (thread count via OBRC_THREADS, default 8)

use std::collections::BTreeMap;
use std::env;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};

use obrc::chunk::{chunk_bounds, num_chunks, split_streams, CHUNK_SIZE};
use obrc::inline_table::{InlineTable, KEY_SLACK};
use obrc::output::{write_results, Stats};
use obrc::parse::{parse_temp, parse_temp_branchless};
use obrc::swar::{find_semi_and_hash, find_semi_and_hash_scalar, SLACK};
use obrc::sys::{set_thread_qos_user_interactive, Mapping};
use obrc::table::Table;
use obrc::MAX_LINE_LEN;

const TABLE_BITS: u32 = 16;

/// Independent line streams per chunk. Swept against the 1B file: 2 and 3 are within noise
/// of each other, 4 and above lose — past that point the streams start competing for the
/// table's cache lines faster than they add overlap.
const STREAMS: usize = 3;

const TAIL_GUARD: usize = MAX_LINE_LEN + SLACK;
const _: () = assert!(TAIL_GUARD >= KEY_SLACK);

struct Tables {
    wide: InlineTable,
    tail: Table,
}

/// Consumes the line at `pos` and returns the start of the next one.
#[inline(always)]
fn step(data: &[u8], table: &mut InlineTable, pos: usize) -> usize {
    let (semi, hash) = find_semi_and_hash(data, pos);
    let (value, next) = parse_temp_branchless(data, semi + 1);
    table.upsert(data, pos, semi - pos, hash, value);
    next
}

fn worker(data: &[u8], cursor: &AtomicUsize, total: usize) -> Tables {
    set_thread_qos_user_interactive();
    let mut wide = InlineTable::new(TABLE_BITS);
    let mut tail = Table::new(8);
    let wide_limit = data.len().saturating_sub(TAIL_GUARD);

    loop {
        let i = cursor.fetch_add(1, Ordering::Relaxed);
        if i >= total {
            break;
        }
        let Some((start, end)) = chunk_bounds(data, i, CHUNK_SIZE) else {
            continue;
        };

        // Only the last chunk of the file ever clamps here.
        let wide_end = end.min(wide_limit).max(start);
        let mut s = split_streams::<STREAMS>(data, start, wide_end);

        // All streams live: one bounds test per STREAMS rows.
        loop {
            let mut all = true;
            for k in 0..STREAMS {
                all &= s[k].0 < s[k].1;
            }
            if !all {
                break;
            }
            for k in 0..STREAMS {
                s[k].0 = step(data, &mut wide, s[k].0);
            }
        }

        // Drain whichever streams still have rows.
        for k in 0..STREAMS {
            while s[k].0 < s[k].1 {
                s[k].0 = step(data, &mut wide, s[k].0);
            }
        }

        // Scalar tail. Resume from where the last stream stopped: `wide_end` is a guard
        // offset, not a line boundary, so that stream will have run past it to finish its
        // line.
        let mut pos = s[STREAMS - 1].0;
        while pos < end {
            let (semi, hash) = find_semi_and_hash_scalar(data, pos);
            let (value, next) = parse_temp(data, semi + 1);
            tail.upsert(data, pos, semi - pos, hash, value);
            pos = next;
        }
    }
    Tables { wide, tail }
}

fn main() {
    let path = env::args().nth(1).unwrap_or_else(|| "measurements.txt".to_string());
    let threads =
        env::var("OBRC_THREADS").ok().and_then(|s| s.parse().ok()).filter(|&n| n > 0).unwrap_or(8);

    let mapping = Mapping::open(&path).unwrap_or_else(|e| panic!("cannot map {path}: {e}"));
    let data = mapping.as_slice();
    let total = num_chunks(data.len(), CHUNK_SIZE);
    let cursor = AtomicUsize::new(0);

    let tables = std::thread::scope(|scope| {
        let handles: Vec<_> =
            (0..threads).map(|_| scope.spawn(|| worker(data, &cursor, total))).collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect::<Vec<_>>()
    });

    let mut merged: BTreeMap<&[u8], Stats> = BTreeMap::new();
    for t in &tables {
        for (name, stats) in t.wide.iter().chain(t.tail.iter(data)) {
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
