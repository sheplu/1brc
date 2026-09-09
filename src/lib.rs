//! One Billion Row Challenge, in Rust, with no external crates.
//!
//! Binaries in `src/bin/` form an optimization ladder (`v1_naive` .. `v4_simd`); each one
//! must produce byte-identical output. `v1_naive` is the correctness oracle and shares no
//! arithmetic with the others.

pub mod block;
pub mod chunk;
pub mod hash;
pub mod inline_table;
pub mod output;
pub mod parse;
pub mod stations;
pub mod swar;
pub mod sys;
pub mod table;

/// Longest legal line: 100-byte name + ';' + "-99.9" + '\n'.
pub const MAX_LINE_LEN: usize = 100 + 1 + 5 + 1;

/// The spec caps unique station names at 10,000.
pub const MAX_STATIONS: usize = 10_000;

#[cfg(test)]
mod tests {
    use super::stations::STATIONS;

    #[test]
    fn station_list_is_well_formed() {
        assert_eq!(STATIONS.len(), 413);

        let mut names: Vec<&str> = STATIONS.iter().map(|(n, _)| *n).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(names.len(), before, "duplicate station names in the generated list");

        for (name, mean) in STATIONS {
            assert!(!name.is_empty());
            assert!(name.len() <= 100, "{name} exceeds the 100-byte limit");
            assert!(!name.contains(';') && !name.contains('\n'), "{name} contains a separator");
            assert!((-99.9..=99.9).contains(mean), "{name} has an out-of-range mean {mean}");
        }
    }
}
