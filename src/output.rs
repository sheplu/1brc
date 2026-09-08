//! Aggregation state and result formatting, shared by v2 onward.
//!
//! All temperatures are held as integer tenths of a degree; the input format guarantees
//! exactly one fractional digit, so this is exact and avoids float accumulation entirely.

use std::io::{self, Write};

#[derive(Clone, Copy, Debug)]
pub struct Stats {
    pub min: i16,
    pub max: i16,
    pub sum: i64,
    pub count: u32,
}

impl Stats {
    #[inline]
    pub fn new(v: i16) -> Self {
        Stats { min: v, max: v, sum: v as i64, count: 1 }
    }

    #[inline]
    pub fn push(&mut self, v: i16) {
        self.min = self.min.min(v);
        self.max = self.max.max(v);
        self.sum += v as i64;
        self.count += 1;
    }

    #[inline]
    pub fn merge(&mut self, other: &Stats) {
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
        self.sum += other.sum;
        self.count += other.count;
    }
}

/// Mean in tenths, rounded half toward positive infinity.
///
/// The reference implementation computes `Math.round(mean * 10.0) / 10.0`, and Java's
/// `Math.round` is `floor(x + 0.5)` — so halves go toward +inf, including negative ones
/// (`Math.round(-2.5) == -2`). In integers that is `floor((2*sum + count) / (2*count))`.
/// `div_euclid` is required: Rust's `/` truncates toward zero and would round -2.5 to -3.
#[inline]
pub fn mean_tenths(sum: i64, count: u32) -> i64 {
    let count = count as i64;
    (2 * sum + count).div_euclid(2 * count)
}

/// Appends a tenths value as a decimal with exactly one fractional digit.
///
/// The sign comes from the already-rounded value, so a mean that rounds up to zero from
/// below prints `0.0` rather than `-0.0` — matching Java, where `Math.round` returns a
/// `long` and drops the sign before the division.
pub fn write_tenths(out: &mut Vec<u8>, tenths: i64) {
    if tenths < 0 {
        out.push(b'-');
    }
    let v = tenths.unsigned_abs();
    let whole = v / 10;
    let frac = (v % 10) as u8;

    let mut buf = [0u8; 20];
    let mut i = buf.len();
    let mut w = whole;
    loop {
        i -= 1;
        buf[i] = b'0' + (w % 10) as u8;
        w /= 10;
        if w == 0 {
            break;
        }
    }
    out.extend_from_slice(&buf[i..]);
    out.push(b'.');
    out.push(b'0' + frac);
}

/// Writes `{Name=min/mean/max, Name2=...}` for entries that are already sorted by name.
///
/// Sorting by raw UTF-8 bytes matches Java's `TreeMap` (UTF-16 code-unit) ordering for the
/// entire BMP; the two only diverge above U+FFFF, which no station name in the official set
/// reaches.
pub fn write_results<W: Write>(w: &mut W, entries: &[(Vec<u8>, Stats)]) -> io::Result<()> {
    let mut out = Vec::with_capacity(entries.len() * 48 + 2);
    out.push(b'{');
    for (i, (name, s)) in entries.iter().enumerate() {
        if i > 0 {
            out.extend_from_slice(b", ");
        }
        out.extend_from_slice(name);
        out.push(b'=');
        write_tenths(&mut out, s.min as i64);
        out.push(b'/');
        write_tenths(&mut out, mean_tenths(s.sum, s.count));
        out.push(b'/');
        write_tenths(&mut out, s.max as i64);
    }
    out.extend_from_slice(b"}\n");
    w.write_all(&out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fmt(tenths: i64) -> String {
        let mut v = Vec::new();
        write_tenths(&mut v, tenths);
        String::from_utf8(v).unwrap()
    }

    /// Cross-check against the reference semantics: floor(x + 0.5).
    fn reference_mean_tenths(sum: i64, count: u32) -> i64 {
        (sum as f64 / count as f64 + 0.5).floor() as i64
    }

    #[test]
    fn rounds_half_toward_positive_infinity() {
        // Exact .5 ties must go up, in both signs.
        assert_eq!(mean_tenths(5, 10), 1); // 0.5 -> 1
        assert_eq!(mean_tenths(-5, 10), 0); // -0.5 -> 0, not -1
        assert_eq!(mean_tenths(15, 10), 2); // 1.5 -> 2
        assert_eq!(mean_tenths(-15, 10), -1); // -1.5 -> -1
        assert_eq!(mean_tenths(-25, 10), -2); // -2.5 -> -2
    }

    #[test]
    fn rounds_non_ties() {
        assert_eq!(mean_tenths(14, 10), 1); // 1.4 -> 1
        assert_eq!(mean_tenths(16, 10), 2); // 1.6 -> 2
        assert_eq!(mean_tenths(-14, 10), -1); // -1.4 -> -1
        assert_eq!(mean_tenths(-16, 10), -2); // -1.6 -> -2
    }

    #[test]
    fn matches_reference_over_a_wide_sweep() {
        for count in [1u32, 2, 3, 7, 10, 99, 1000, 999_983] {
            for sum in -5000i64..=5000 {
                assert_eq!(
                    mean_tenths(sum, count),
                    reference_mean_tenths(sum, count),
                    "sum={sum} count={count}"
                );
            }
        }
    }

    #[test]
    fn never_prints_negative_zero() {
        // A mean that rounds up to zero from below.
        assert_eq!(mean_tenths(-4, 10), 0);
        assert_eq!(fmt(mean_tenths(-4, 10)), "0.0");
        assert_eq!(fmt(0), "0.0");
    }

    #[test]
    fn formats_tenths() {
        assert_eq!(fmt(-999), "-99.9");
        assert_eq!(fmt(999), "99.9");
        assert_eq!(fmt(-1), "-0.1");
        assert_eq!(fmt(1), "0.1");
        assert_eq!(fmt(180), "18.0");
        assert_eq!(fmt(-69), "-6.9");
    }

    /// 1e9 rows over the official 413 stations gives ~2.42M rows each; at the extreme
    /// value that is ~2.4e9 tenths, which overflows i32. Guard the choice of i64.
    #[test]
    fn sum_does_not_overflow_at_full_scale() {
        let per_station: i64 = 1_000_000_000 / 413;
        let worst = per_station * 999;
        assert!(worst > i32::MAX as i64, "i32 would have sufficed after all: {worst}");
        assert!(worst < i64::MAX / 4);
    }

    #[test]
    fn writes_java_style_map() {
        let entries = vec![
            (b"Abha".to_vec(), Stats { min: -230, max: 592, sum: 180, count: 1 }),
            (b"Z\xc3\xbcrich".to_vec(), Stats { min: -100, max: 100, sum: 0, count: 4 }),
        ];
        let mut out = Vec::new();
        write_results(&mut out, &entries).unwrap();
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "{Abha=-23.0/18.0/59.2, Zürich=-10.0/0.0/10.0}\n"
        );
    }

    #[test]
    fn stats_accumulate() {
        let mut s = Stats::new(50);
        s.push(-20);
        s.push(100);
        assert_eq!((s.min, s.max, s.sum, s.count), (-20, 100, 130, 3));

        let mut t = Stats::new(-300);
        t.merge(&s);
        assert_eq!((t.min, t.max, t.sum, t.count), (-300, 100, -170, 4));
    }
}
