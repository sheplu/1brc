//! `null` plus the threads, and nothing else.
//!
//! The second bracket on the ~24 ms `v12_pairs` spends outside `main`. This spawns
//! `OBRC_THREADS` scoped threads that set their QoS and return, joins them, and exits the same
//! way the real binary does. `spawn − null` is therefore the cost of creating, scheduling and
//! joining 18 threads with 2 MiB stacks apiece; `v12_pairs − spawn − null` is what is left for
//! the work itself.
//!
//! **Measured**: 18 ms at 18 threads, against `null`'s 18-21 ms in the same batch. Eighteen
//! threads with 2 MiB stacks apiece cost nothing measurable to create, schedule and join.
//!
//! That half of the bracket stands. The other half did not: see [`null`](../null/index.html) for
//! why the 18 ms is `batch.sh`'s shell rather than the loader, and why the whole budget outside
//! `main` turns out to be ~4 ms.
//!
//! Usage: spawn   (OBRC_THREADS, default 8)

use std::env;

use obrc::sys::set_thread_qos_user_interactive;

fn main() {
    let threads =
        env::var("OBRC_THREADS").ok().and_then(|s| s.parse().ok()).filter(|&n| n > 0).unwrap_or(8);

    std::thread::scope(|scope| {
        let handles: Vec<_> =
            (0..threads).map(|_| scope.spawn(set_thread_qos_user_interactive)).collect();
        for h in handles {
            h.join().unwrap();
        }
    });

    std::process::exit(0);
}
