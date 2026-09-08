//! v5: v4 with the two remaining per-row branch/call costs removed.
//!
//! v4 sits at ~12 ns/row, which is far more than the work it does. Two culprits, both in
//! the hot loop and both invisible in the source:
//!
//! 1. `parse_temp` branches on the sign and on the integer-digit count. Both are
//!    data-dependent and unpredictable, so a large fraction of rows pay a misprediction.
//! 2. The slice equality in `Table::upsert` lowers to a `bcmp` call.
//!
//! [`parse_temp_branchless`] and [`Table::upsert_wide_cmp`] replace them with straight-line
//! code. Everything else is identical to v4.
//!
//! Usage: v5_branchless [path]   (thread count via OBRC_THREADS, default 8)

use std::collections::BTreeMap;
use std::env;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};

use obrc::chunk::{chunk_bounds, num_chunks, CHUNK_SIZE};
use obrc::output::{write_results, Stats};
use obrc::parse::{parse_temp, parse_temp_branchless};
use obrc::swar::{find_semi_and_hash, find_semi_and_hash_scalar, SLACK};
use obrc::sys::{set_thread_qos_user_interactive, Mapping};
use obrc::table::Table;
use obrc::MAX_LINE_LEN;

const TABLE_BITS: u32 = 16;

/// Bytes at the end of the file that the wide loads must not enter. Covers both the fused
/// scan's overshoot past `;` and the parser's 8-byte load after it.
const TAIL_GUARD: usize = MAX_LINE_LEN + SLACK;

fn worker(data: &[u8], cursor: &AtomicUsize, total: usize) -> Table {
    set_thread_qos_user_interactive();
    let mut table = Table::new(TABLE_BITS);
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
            table.upsert_wide_cmp(data, pos, semi - pos, hash, value);
            pos = next;
        }

        // Scalar tail: only the final chunk reaches this, and only for a line or two.
        while pos < end {
            let (semi, hash) = find_semi_and_hash_scalar(data, pos);
            let (value, next) = parse_temp(data, semi + 1);
            table.upsert(data, pos, semi - pos, hash, value);
            pos = next;
        }
    }
    table
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
    for table in &tables {
        for (name, stats) in table.iter(data) {
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
