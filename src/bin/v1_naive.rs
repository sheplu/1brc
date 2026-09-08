//! v1: the correctness oracle. Obvious, single-threaded, slow.
//!
//! This binary shares no arithmetic with v2+ on purpose. It accumulates in `f64` and
//! rounds in floating point, so diffing it against the integer-tenths implementations is
//! a real check rather than a comparison of one helper against itself.
//!
//! Usage: v1_naive [path]

use std::collections::BTreeMap;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

struct Acc {
    min: f64,
    max: f64,
    sum: f64,
    count: u64,
}

/// Java's `Math.round(v * 10.0) / 10.0` — floor(x + 0.5), so halves go toward +inf.
/// Note this is *not* `f64::round`, which rounds halves away from zero and would turn
/// -0.05 into -0.1 where Java gives 0.0.
fn round1(x: f64) -> f64 {
    let r = (x * 10.0 + 0.5).floor() / 10.0;
    // Collapse -0.0 to 0.0 so it never reaches the formatter.
    if r == 0.0 {
        0.0
    } else {
        r
    }
}

fn main() {
    let path = env::args().nth(1).unwrap_or_else(|| "measurements.txt".to_string());
    let file = File::open(&path).unwrap_or_else(|e| panic!("cannot open {path}: {e}"));
    let reader = BufReader::with_capacity(1 << 20, file);

    let mut map: BTreeMap<String, Acc> = BTreeMap::new();

    for line in reader.lines() {
        let line = line.expect("read error");
        if line.is_empty() {
            continue;
        }
        let (name, value) = line.split_once(';').expect("line has no ';'");
        let v: f64 = value.parse().expect("bad temperature");

        match map.get_mut(name) {
            Some(a) => {
                if v < a.min {
                    a.min = v;
                }
                if v > a.max {
                    a.max = v;
                }
                a.sum += v;
                a.count += 1;
            }
            None => {
                map.insert(name.to_string(), Acc { min: v, max: v, sum: v, count: 1 });
            }
        }
    }

    let stdout = std::io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    write!(out, "{{").unwrap();
    for (i, (name, a)) in map.iter().enumerate() {
        if i > 0 {
            write!(out, ", ").unwrap();
        }
        let mean = a.sum / a.count as f64;
        write!(out, "{}={:.1}/{:.1}/{:.1}", name, round1(a.min), round1(mean), round1(a.max))
            .unwrap();
    }
    writeln!(out, "}}").unwrap();
}
