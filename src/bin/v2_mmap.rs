//! v2: mmap + work-stealing threads + manual parsing, still using std's `HashMap`.
//!
//! Work stealing is here rather than in a later version because this machine's fast and
//! slow core tiers differ in throughput; a static N-way split of the file would finish
//! most chunks early and then wait on a straggler.
//!
//! Usage: v2_mmap [path]   (thread count via OBRC_THREADS, default 8)

use std::collections::{BTreeMap, HashMap};
use std::env;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};

use obrc::chunk::{chunk_bounds, num_chunks, CHUNK_SIZE};
use obrc::hash::FxBuildHasher;
use obrc::output::{write_results, Stats};
use obrc::parse::parse_temp;
use obrc::sys::{set_thread_qos_user_interactive, Mapping};
use obrc::MAX_LINE_LEN;

const _: () = assert!(CHUNK_SIZE > MAX_LINE_LEN);

type Table<'a> = HashMap<&'a [u8], Stats, FxBuildHasher>;

fn worker<'a>(data: &'a [u8], cursor: &AtomicUsize, total: usize) -> Table<'a> {
    set_thread_qos_user_interactive();
    let mut table: Table<'a> = HashMap::with_capacity_and_hasher(1024, FxBuildHasher::default());

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
            let (value, next) = parse_temp(data, semi + 1);
            table
                .entry(&data[pos..semi])
                .and_modify(|s| s.push(value))
                .or_insert_with(|| Stats::new(value));
            pos = next;
        }
    }
    table
}

pub fn threads_from_env() -> usize {
    env::var("OBRC_THREADS").ok().and_then(|s| s.parse().ok()).filter(|&n| n > 0).unwrap_or(8)
}

fn main() {
    let path = env::args().nth(1).unwrap_or_else(|| "measurements.txt".to_string());
    let threads = threads_from_env();

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
        for (name, stats) in table {
            merged.entry(name).and_modify(|s| s.merge(stats)).or_insert(*stats);
        }
    }

    let entries: Vec<(Vec<u8>, Stats)> = merged.into_iter().map(|(k, v)| (k.to_vec(), v)).collect();

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    write_results(&mut out, &entries).unwrap();
    out.flush().unwrap();

    // Skip teardown of a 13 GB mapping; it is pure cost after the answer is out.
    std::process::exit(0);
}
