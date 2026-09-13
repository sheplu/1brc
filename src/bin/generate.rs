//! Port of the official `CreateMeasurements.java`.
//!
//! Differences from the Java original, both deliberate:
//!   * the RNG is seeded, so a given (rows, seed) always produces a byte-identical file
//!     regardless of thread count or scheduling;
//!   * values are clamped to the spec range [-99.9, 99.9]. The Java generator doesn't
//!     clamp, and sigma=10 over 1e9 samples has a small but nonzero chance of emitting a
//!     three-integer-digit value, which would silently corrupt a branchless parser.
//!
//! `OBRC_GEN_STATIONS=n` keeps a stride-sampled n of the 413 names. It exists for one
//! experiment, and the experiment is worth recording here because it closed a line of attack.
//!
//! **Does the table's live set bound the loop?** 413 entries scattered over 16384 slots land on
//! ~408 distinct 128-byte lines, so the table holds ~52 KB of L1d resident while a 1 MiB chunk
//! streams past it, against 64 KB of L1d on twelve of this machine's eighteen cores. No sweep in
//! the project could see this: moving `OBRC_TABLE_BITS` from 2^13 to 2^16 changes the slot count
//! and leaves exactly 413 entries live. Only the data moves the live set.
//!
//! 400M rows at each of 50, 100, 200 and 413 stations (`scripts/v15-footprint.tsv`, endpoints
//! entered twice, REPS=9):
//!
//! ```text
//!            t18    (dup)      t8    (dup)     bytes    names >= 16B
//!   s50    0.164    0.164   0.280   0.279    5.405 GB      2.0%
//!   s100   0.158            0.270            5.498 GB      2.0%
//!   s200   0.160            0.274            5.439 GB      2.0%
//!   s413   0.164    0.164   0.282   0.280    5.518 GB      2.4%
//! ```
//!
//! Fifty stations is ~6 KB of entries, trivially resident, an eightfold cut in footprint. It is
//! not faster than 413 — it ties, at both thread counts, against a noise floor the duplicate
//! arms put at 0-2 ms. So the live set is not the bound, and the experiment was biased *toward*
//! finding an effect: fewer stations also means shorter probe chains and a more predictable
//! `step_long` branch, and even with those thrown in it came back flat.
//!
//! Two details worth keeping. The 4-6 ms by which 100 and 200 beat both endpoints is above the
//! noise floor but non-monotonic, so there is no lever in it; it is not name length (the >= 16B
//! fraction is flat, and s100 is the *larger* file) and it is not read (likewise). And netting
//! out the ~20 ms loader, these files run 0.360 ns/row against the real dataset's 0.361 — the
//! probe is a faithful miniature, which is what makes the flatness worth believing.
//!
//! Usage: generate <rows> <output-path> [seed]   (OBRC_GEN_STATIONS, default 413)

use std::collections::BTreeMap;
use std::env;
use std::fs::File;
use std::io::Write;
use std::process;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};

use obrc::stations::STATIONS;

const ROWS_PER_CHUNK: usize = 1_000_000;
/// Max chunks buffered ahead of the writer, to bound memory (~12 MB each).
const WINDOW: usize = 32;
const DEFAULT_SEED: u64 = 0x5EED_1B4C_0FFE_E123;
const CHUNK_CAPACITY: usize = ROWS_PER_CHUNK * 20;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

struct Rng {
    s: [u64; 4],
    spare_gaussian: Option<f64>,
}

impl Rng {
    fn seed(mut x: u64) -> Self {
        let s = [splitmix64(&mut x), splitmix64(&mut x), splitmix64(&mut x), splitmix64(&mut x)];
        Rng { s, spare_gaussian: None }
    }

    /// xoshiro256++
    #[inline]
    fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    #[inline]
    fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / (1u64 << 53) as f64)
    }

    /// Lemire's bounded reduction; the residual bias is ~2^-64.
    #[inline]
    fn below(&mut self, n: u32) -> u32 {
        (((self.next_u64() as u128) * (n as u128)) >> 64) as u32
    }

    /// Marsaglia polar method: no trig, and two deviates per accepted pair.
    #[inline]
    fn next_gaussian(&mut self) -> f64 {
        if let Some(v) = self.spare_gaussian.take() {
            return v;
        }
        loop {
            let u = 2.0 * self.next_f64() - 1.0;
            let v = 2.0 * self.next_f64() - 1.0;
            let s = u * u + v * v;
            if s > 0.0 && s < 1.0 {
                let f = (-2.0 * s.ln() / s).sqrt();
                self.spare_gaussian = Some(v * f);
                return u * f;
            }
        }
    }
}

#[inline]
fn push_tenths(out: &mut Vec<u8>, t: i32) {
    if t < 0 {
        out.push(b'-');
    }
    let v = t.unsigned_abs();
    let whole = v / 10;
    if whole >= 10 {
        out.push(b'0' + (whole / 10) as u8);
    }
    out.push(b'0' + (whole % 10) as u8);
    out.push(b'.');
    out.push(b'0' + (v % 10) as u8);
}

/// Generates one chunk into `out`. Deterministic in `chunk_index` alone.
fn generate_chunk(
    out: &mut Vec<u8>,
    prefixes: &[Box<[u8]>],
    means: &[f64],
    rows: usize,
    seed: u64,
    chunk_index: usize,
) {
    out.clear();
    let mut rng = Rng::seed(splitmix64(&mut (seed ^ chunk_index as u64)));
    let n = prefixes.len() as u32;
    for _ in 0..rows {
        let i = rng.below(n) as usize;
        let value = means[i] + rng.next_gaussian() * 10.0;
        // Java: Math.round(m * 10.0) / 10.0, i.e. floor(x + 0.5).
        let tenths = ((value * 10.0 + 0.5).floor() as i32).clamp(-999, 999);
        out.extend_from_slice(&prefixes[i]);
        push_tenths(out, tenths);
        out.push(b'\n');
    }
}

struct State {
    done: BTreeMap<usize, Vec<u8>>,
    next_to_write: usize,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: {} <rows> <output-path> [seed]", args[0]);
        process::exit(2);
    }
    let rows: usize = args[1].replace('_', "").parse().expect("rows must be an integer");
    let path = &args[2];
    let seed: u64 =
        args.get(3).map(|s| s.parse().expect("seed must be an integer")).unwrap_or(DEFAULT_SEED);

    // `OBRC_GEN_STATIONS=n` keeps only n of the 413, which is the only way to vary how many
    // table entries a run keeps live — the table-bits sweep moves the slot count but every
    // slot count holds the same 413. Stride, not prefix: the list is alphabetical, and its
    // first n would skew the name lengths the parser sees.
    let keep = env::var("OBRC_GEN_STATIONS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|n| (1..=STATIONS.len()).contains(n))
        .unwrap_or(STATIONS.len());
    let picked: Vec<(&str, f64)> =
        (0..keep).map(|i| STATIONS[i * STATIONS.len() / keep]).collect();

    // "name;" precomputed so the hot loop does one copy instead of two.
    let prefixes: Vec<Box<[u8]>> = picked
        .iter()
        .map(|(n, _)| {
            let mut v = n.as_bytes().to_vec();
            v.push(b';');
            v.into_boxed_slice()
        })
        .collect();
    let means: Vec<f64> = picked.iter().map(|(_, m)| *m).collect();

    let num_chunks = rows.div_ceil(ROWS_PER_CHUNK);
    let next_chunk = AtomicUsize::new(0);
    let state = Mutex::new(State { done: BTreeMap::new(), next_to_write: 0 });
    let cv = Condvar::new();

    let producers =
        std::thread::available_parallelism().map(|n| n.get().saturating_sub(1)).unwrap_or(7).max(1);
    let start = std::time::Instant::now();

    std::thread::scope(|scope| {
        for _ in 0..producers {
            scope.spawn(|| {
                let mut buf = Vec::with_capacity(CHUNK_CAPACITY);
                loop {
                    let i = next_chunk.fetch_add(1, Ordering::Relaxed);
                    if i >= num_chunks {
                        break;
                    }
                    // Don't run too far ahead of the writer.
                    {
                        let mut st = state.lock().unwrap();
                        while i >= st.next_to_write + WINDOW {
                            st = cv.wait(st).unwrap();
                        }
                    }
                    let rows_here = ROWS_PER_CHUNK.min(rows - i * ROWS_PER_CHUNK);
                    generate_chunk(&mut buf, &prefixes, &means, rows_here, seed, i);

                    let filled = std::mem::replace(&mut buf, Vec::with_capacity(CHUNK_CAPACITY));
                    let mut st = state.lock().unwrap();
                    st.done.insert(i, filled);
                    cv.notify_all();
                }
            });
        }

        // Single writer, emitting chunks strictly in order.
        let mut file = File::create(path).expect("cannot create output file");
        let mut written = 0usize;
        for i in 0..num_chunks {
            let bytes = {
                let mut st = state.lock().unwrap();
                loop {
                    if let Some(b) = st.done.remove(&i) {
                        break b;
                    }
                    st = cv.wait(st).unwrap();
                }
            };
            file.write_all(&bytes).expect("write failed");
            written += bytes.len();
            {
                let mut st = state.lock().unwrap();
                st.next_to_write = i + 1;
                cv.notify_all();
            }
            if i % 100 == 99 || i + 1 == num_chunks {
                eprint!(
                    "\r{}/{} chunks, {:.2} GB, {:.1}s",
                    i + 1,
                    num_chunks,
                    written as f64 / 1e9,
                    start.elapsed().as_secs_f64()
                );
            }
        }
        eprintln!();
    });
}
