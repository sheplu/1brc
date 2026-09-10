//! Splitting the input into line-aligned work units.
//!
//! The boundary rule is deliberately *self-consistent*: chunk `i`'s end is computed by the
//! same expression as chunk `i+1`'s start, so workers never need to communicate to agree
//! on where a line belongs.

/// 1 MiB, measured. The curve is nearly flat from 768 KiB to 1.5 MiB and worth ~9 ms of 436
/// against the 2 MiB this used to be; below 512 KiB it climbs steeply, reaching 557 ms at
/// 128 KiB, where 105k `pread`s of bookkeeping is the whole difference. The tilt at the top
/// does not appear at 8 threads, which points at the shared L2 — an E-cluster is 8 MB across
/// 6 cores, so six 1 MiB buffers fit between the read and the parse that reads them back, and
/// six 2 MiB ones do not.
///
/// The sweep saying 1 MiB had been in the README for several versions while this constant was
/// 2 MiB, because at the time it was taken the curve really was flat past 1 MiB and the value
/// looked like it did not matter. It stopped being flat once the parse got fast enough for the
/// difference to show. A swept constant is only swept for the program that swept it.
pub const CHUNK_SIZE: usize = 1 << 20;

/// Index of the first byte after the next `\n` at or after `from`, clamped to the end.
#[inline]
pub fn next_line_start(data: &[u8], from: usize) -> usize {
    if from >= data.len() {
        return data.len();
    }
    match data[from..].iter().position(|&b| b == b'\n') {
        Some(off) => from + off + 1,
        None => data.len(),
    }
}

/// Line-aligned `[start, end)` for raw chunk `i`, or `None` if the chunk contains no line
/// start (which happens when a single line spans the whole chunk — the preceding chunk
/// owns that line).
pub fn chunk_bounds(data: &[u8], i: usize, chunk_size: usize) -> Option<(usize, usize)> {
    let raw = i.checked_mul(chunk_size)?;
    if raw >= data.len() {
        return None;
    }
    let start = if raw == 0 { 0 } else { next_line_start(data, raw) };
    let raw_end = raw + chunk_size;
    let end = if raw_end >= data.len() { data.len() } else { next_line_start(data, raw_end) };
    if start >= end {
        None
    } else {
        Some((start, end))
    }
}

pub fn num_chunks(len: usize, chunk_size: usize) -> usize {
    len.div_ceil(chunk_size)
}

/// [`chunk_bounds`] for a reader that can only see its own chunk, in local coordinates.
///
/// `view` is the bytes at `base`: the chunk itself plus enough overlap to contain the line
/// crossing its end — one maximum line length is enough — truncated at EOF. `base` is only
/// consulted for whether this is the first chunk.
///
/// Both sides of a boundary evaluate the same expression over the same bytes, so they agree
/// on who owns the straddling line without communicating.
pub fn local_bounds(view: &[u8], base: usize, chunk_size: usize) -> Option<(usize, usize)> {
    let start = if base == 0 { 0 } else { next_line_start(view, 0) };
    // At EOF the view is shorter than the chunk and this yields its length, which is right:
    // the file's last line needs no newline after it.
    let end = next_line_start(view, chunk_size);
    if start >= end {
        None
    } else {
        Some((start, end))
    }
}

/// Divides the line-aligned range `[start, end)` into `N` line-aligned sub-ranges.
///
/// This is for instruction-level parallelism, not load balancing. Consecutive rows form a
/// serial dependency chain — a row's start is only known once the previous row's value has
/// been parsed — so one stream leaves most of the core's out-of-order window idle. Walking
/// `N` independent streams lets those chains overlap.
///
/// The sub-ranges are contiguous, ordered, and together cover `[start, end)` exactly. Any
/// of them may be empty when the range holds fewer than `N` lines. `start` must already be
/// a line start, as `chunk_bounds` guarantees.
pub fn split_streams<const N: usize>(data: &[u8], start: usize, end: usize) -> [(usize, usize); N] {
    let mut head = [start; N];
    let step = (end - start) / N;
    for k in 1..N {
        // Clamping to the previous head keeps the heads ordered when a line straddles
        // several split points.
        head[k] = next_line_start(data, start + step * k).min(end).max(head[k - 1]);
    }
    let mut out = [(end, end); N];
    for k in 0..N {
        out[k] = (head[k], if k + 1 < N { head[k + 1] } else { end });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every byte of the file must be covered exactly once, and every chunk must contain
    /// only whole lines.
    fn assert_partition(data: &[u8], chunk_size: usize) {
        let mut covered = 0usize;
        let mut expect_start = 0usize;
        for i in 0..num_chunks(data.len(), chunk_size) {
            if let Some((s, e)) = chunk_bounds(data, i, chunk_size) {
                assert_eq!(s, expect_start, "chunk {i} does not resume where the last left off");
                assert!(s < e);
                // A chunk must begin at a line start and end just past a '\n' (or at EOF).
                assert!(s == 0 || data[s - 1] == b'\n', "chunk {i} starts mid-line");
                assert!(e == data.len() || data[e - 1] == b'\n', "chunk {i} ends mid-line");
                covered += e - s;
                expect_start = e;
            }
        }
        assert_eq!(covered, data.len(), "chunks did not cover the whole input");
        assert_eq!(expect_start, data.len());
    }

    /// A worker that can only see its own chunk must reach exactly the same decision as one
    /// looking at the whole file — for every chunk, at every size.
    fn assert_local_matches_global(data: &[u8], chunk_size: usize, overlap: usize) {
        for i in 0..num_chunks(data.len(), chunk_size) {
            let base = i * chunk_size;
            if base >= data.len() {
                continue;
            }
            let avail = (chunk_size + overlap).min(data.len() - base);
            let view = &data[base..base + avail];
            let local = local_bounds(view, base, chunk_size).map(|(s, e)| (base + s, base + e));
            assert_eq!(
                local,
                chunk_bounds(data, i, chunk_size),
                "chunk {i} disagrees at chunk_size {chunk_size}"
            );
        }
    }

    #[test]
    fn local_bounds_agree_with_the_global_split() {
        let mut data = Vec::new();
        for i in 0..300 {
            // Lengths vary so boundaries land at every offset within a line.
            let name = "s".repeat(1 + i % 37);
            data.extend_from_slice(format!("{name}{i};{}.{}\n", i % 100, i % 10).as_bytes());
        }
        let max_line = 37 + 3 + 1 + 4 + 1 + 8;
        for chunk_size in [1usize, 2, 3, 7, 16, 41, 64, 100, 1000, 4096, data.len(), data.len() * 2]
        {
            assert_local_matches_global(&data, chunk_size, max_line);
        }

        // Missing trailing newline, and a single line longer than the chunk.
        assert_local_matches_global(b"a;1.0\nbb;2.0\nccc;3.0", 4, 64);
        assert_local_matches_global(b"averyveryverylongstationname;12.3\n", 4, 64);
        assert_local_matches_global(b"Abha;18.0\n", 4096, 64);
    }

    #[test]
    fn partitions_at_many_chunk_sizes() {
        let mut data = Vec::new();
        for i in 0..500 {
            data.extend_from_slice(format!("Station{i};{}.{}\n", i % 100, i % 10).as_bytes());
        }
        for chunk_size in
            [1usize, 2, 3, 7, 8, 13, 16, 64, 100, 1000, 4096, data.len(), data.len() * 2]
        {
            assert_partition(&data, chunk_size);
        }
    }

    /// The streams must tile the range exactly and every head must sit on a line start,
    /// including the degenerate cases where the range holds fewer lines than streams.
    fn assert_streams<const N: usize>(data: &[u8], start: usize, end: usize) {
        let r = split_streams::<N>(data, start, end);
        assert_eq!(r[0].0, start);
        assert_eq!(r[N - 1].1, end);
        for k in 0..N {
            assert!(r[k].0 <= r[k].1, "stream {k} is inverted: {:?}", r[k]);
            assert!(
                r[k].0 == 0 || r[k].0 == data.len() || data[r[k].0 - 1] == b'\n',
                "stream {k} starts mid-line"
            );
            if k + 1 < N {
                assert_eq!(r[k].1, r[k + 1].0, "gap or overlap between streams {k} and {}", k + 1);
            }
        }
        assert_eq!(r.iter().map(|(s, e)| e - s).sum::<usize>(), end - start);
    }

    #[test]
    fn splits_into_line_aligned_streams() {
        let mut data = Vec::new();
        for i in 0..200 {
            data.extend_from_slice(format!("Station{i};{}.{}\n", i % 100, i % 10).as_bytes());
        }
        for (i, _) in data.iter().enumerate().filter(|(_, &b)| b == b'\n') {
            // Every line start is a legal range start.
            assert_streams::<2>(&data, i + 1, data.len());
            assert_streams::<4>(&data, i + 1, data.len());
            assert_streams::<8>(&data, i + 1, data.len());
        }
    }

    /// Fewer lines than streams, and a single line spanning every split point.
    #[test]
    fn splits_degenerate_ranges() {
        let one = b"Abha;18.0\n".as_slice();
        assert_streams::<4>(one, 0, one.len());
        assert_eq!(split_streams::<4>(one, 0, one.len())[0], (0, one.len()));

        let two = b"a;1.0\nbb;2.0\n".as_slice();
        assert_streams::<4>(two, 0, two.len());

        let long = b"averyveryverylongstationname;12.3\n".as_slice();
        assert_streams::<8>(long, 0, long.len());

        assert_streams::<4>(b"", 0, 0);
    }

    #[test]
    fn handles_missing_trailing_newline() {
        let data = b"a;1.0\nbb;2.0\nccc;3.0".as_slice();
        for chunk_size in [1usize, 2, 5, 6, 7, 13, 100] {
            assert_partition(data, chunk_size);
        }
    }

    #[test]
    fn single_line_spanning_many_chunks_is_owned_once() {
        let data = b"averyveryverylongstationname;12.3\n".as_slice();
        // chunk_size 4 => the line spans 9 raw chunks; exactly one must claim it.
        let claimed: Vec<_> =
            (0..num_chunks(data.len(), 4)).filter_map(|i| chunk_bounds(data, i, 4)).collect();
        assert_eq!(claimed, vec![(0, data.len())]);
    }

    #[test]
    fn empty_input() {
        assert_eq!(chunk_bounds(&[], 0, 64), None);
        assert_eq!(num_chunks(0, 64), 0);
    }

    #[test]
    fn single_row() {
        let data = b"Abha;18.0\n".as_slice();
        assert_eq!(chunk_bounds(data, 0, CHUNK_SIZE), Some((0, 10)));
        assert_eq!(chunk_bounds(data, 1, CHUNK_SIZE), None);
    }
}
