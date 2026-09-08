//! v3: v2 with std's `HashMap` replaced by a purpose-built open-addressed table.
//!
//! Usage: v3_hash [path]   (thread count via OBRC_THREADS, default 8)

use std::collections::BTreeMap;
use std::env;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};

use obrc::chunk::{chunk_bounds, num_chunks, CHUNK_SIZE};
use obrc::hash::fx_hash;
use obrc::output::{write_results, Stats};
use obrc::parse::parse_temp;
use obrc::sys::{set_thread_qos_user_interactive, Mapping};
use obrc::table::Table;

/// 65536 slots for at most 10,000 keys: a load factor low enough that linear probing
/// almost always hits on the first try.
const TABLE_BITS: u32 = 16;

fn worker(data: &[u8], cursor: &AtomicUsize, total: usize) -> Table {
    set_thread_qos_user_interactive();
    let mut table = Table::new(TABLE_BITS);

    loop {
        let i = cursor.fetch_add(1, Ordering::Relaxed);
        if i >= total {
            break;
        }
        let Some((start, end)) = chunk_bounds(data, i, CHUNK_SIZE) else {
            continue;
        };

        let mut pos = start;
        while pos < end {
            let semi =
                pos + data[pos..end].iter().position(|&b| b == b';').expect("line without ';'");
            let hash = fx_hash(&data[pos..semi]);
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
