//! Port of the official `CreateMeasurements.java`.
//!
//! Differences from the Java original, both deliberate:
//!   * the RNG is seeded, so a given (rows, seed) always produces a byte-identical file
//!     regardless of thread count or scheduling;
//!   * values are clamped to the spec range [-99.9, 99.9]. The Java generator doesn't
//!     clamp, and sigma=10 over 1e9 samples has a small but nonzero chance of emitting a
//!     three-integer-digit value, which would silently corrupt a branchless parser.
//!
//! Usage: generate <rows> <output-path> [seed]

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

    // "name;" precomputed so the hot loop does one copy instead of two.
    let prefixes: Vec<Box<[u8]>> = STATIONS
        .iter()
        .map(|(n, _)| {
            let mut v = n.as_bytes().to_vec();
            v.push(b';');
            v.into_boxed_slice()
        })
        .collect();
    let means: Vec<f64> = STATIONS.iter().map(|(_, m)| *m).collect();

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
