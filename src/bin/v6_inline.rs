//! v6: v5 with the station name stored inside the hash table entry.
//!
//! The ablation harness (`hot_floor`) shows that probing and updating a table slot is free
//! — `nocmp` and `hash` are within noise of each other — while adding the key comparison
//! costs ~0.27 s per billion rows. The comparison is expensive not because of the bytes it
//! compares but because of where they live: v5 stores only an offset, so verifying a key
//! means dereferencing into a scattered location in the 13 GB mapping. That is a second
//! random access on every row.
//!
//! [`InlineTable`] puts the name on the entry's own cache line, which the probe has already
//! loaded, so the common case touches nothing else.
//!
//! The very end of the file cannot use it: the inline probe reads a fixed 32 bytes from the
//! name start, which would run off the mapping. Those last few lines go through v5's
//! offset-keyed table instead, and both tables are merged at the end.
//!
//! Usage: v6_inline [path]   (thread count via OBRC_THREADS, default 8)

use std::collections::BTreeMap;
use std::env;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};

use obrc::chunk::{chunk_bounds, num_chunks, CHUNK_SIZE};
use obrc::inline_table::{InlineTable, KEY_SLACK};
use obrc::output::{write_results, Stats};
use obrc::parse::{parse_temp, parse_temp_branchless};
use obrc::swar::{find_semi_and_hash, find_semi_and_hash_scalar, SLACK};
use obrc::sys::{set_thread_qos_user_interactive, Mapping};
use obrc::table::Table;
use obrc::MAX_LINE_LEN;

const TABLE_BITS: u32 = 16;

/// Bytes at the end of the file that the wide path must not enter. Covers the fused scan's
/// overshoot past `;`, the parser's 8-byte load, and the inline key probe.
const TAIL_GUARD: usize = MAX_LINE_LEN + SLACK;
const _: () = assert!(TAIL_GUARD >= KEY_SLACK);

/// The wide table, plus a tiny offset-keyed one for the last lines of the file.
struct Tables {
    wide: InlineTable,
    tail: Table,
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

        let mut pos = start;

        let wide_end = end.min(wide_limit);
        while pos < wide_end {
            let (semi, hash) = find_semi_and_hash(data, pos);
            let (value, next) = parse_temp_branchless(data, semi + 1);
            wide.upsert(data, pos, semi - pos, hash, value);
            pos = next;
        }

        // Only the final chunk reaches this, and only for a line or two.
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
        let rows = t.wide.iter().chain(t.tail.iter(data));
        for (name, stats) in rows {
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
