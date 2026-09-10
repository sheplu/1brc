//! Hand-rolled bindings for the handful of syscalls we need.
//!
//! The challenge forbids external dependencies, so there is no `libc` crate here.
//! Anything std already covers (`File`, `FileExt::read_at` for `pread`) is used from
//! std rather than redeclared.

use core::ffi::{c_int, c_uint, c_void};
use std::fs::File;
use std::io;
use std::os::fd::AsRawFd;

pub const PROT_READ: c_int = 0x01;
pub const MAP_SHARED: c_int = 0x0001;
pub const MAP_PRIVATE: c_int = 0x0002;

pub const MADV_SEQUENTIAL: c_int = 2;
pub const MADV_WILLNEED: c_int = 3;

/// From `<sys/qos.h>`. Darwin exposes no thread-affinity API on arm64, so QoS is the
/// only lever for biasing work onto the faster core tiers.
pub const QOS_CLASS_USER_INTERACTIVE: c_uint = 0x21;

extern "C" {
    fn mmap(
        addr: *mut c_void,
        len: usize,
        prot: c_int,
        flags: c_int,
        fd: c_int,
        offset: i64,
    ) -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> c_int;
    fn madvise(addr: *mut c_void, len: usize, advice: c_int) -> c_int;
    fn mlock(addr: *const c_void, len: usize) -> c_int;
    fn pthread_set_qos_class_self_np(qos_class: c_uint, relative_priority: c_int) -> c_int;
}

pub fn set_thread_qos_user_interactive() {
    unsafe { pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) };
}

pub struct Mapping {
    ptr: *mut c_void,
    len: usize,
}

// The mapping is read-only and immutable for its whole lifetime.
unsafe impl Send for Mapping {}
unsafe impl Sync for Mapping {}

impl Mapping {
    pub fn open(path: &str) -> io::Result<Self> {
        Self::open_with(path, MAP_PRIVATE)
    }

    /// [`Mapping::open`] with the mmap flags chosen by the caller, so `io_floor` can A/B
    /// `MAP_PRIVATE` against `MAP_SHARED`. Read-only either way.
    pub fn open_with(path: &str, flags: c_int) -> io::Result<Self> {
        let file = File::open(path)?;
        let len = file.metadata()?.len() as usize;
        if len == 0 {
            return Ok(Mapping { ptr: core::ptr::null_mut(), len: 0 });
        }
        let ptr =
            unsafe { mmap(core::ptr::null_mut(), len, PROT_READ, flags, file.as_raw_fd(), 0) };
        // mmap reports failure as (void *)-1, not NULL.
        if ptr as isize == -1 {
            return Err(io::Error::last_os_error());
        }
        Ok(Mapping { ptr, len })
    }

    pub fn advise(&self, advice: c_int) {
        if self.len != 0 {
            unsafe { madvise(self.ptr, self.len, advice) };
        }
    }

    /// Wires `self[off..off + len]`, which installs its page table entries in one call instead
    /// of one minor fault per page as the bytes are first touched.
    ///
    /// Reading 13.8 GB through a mapping takes 842k faults at a 16 KB page, and those measure
    /// far above what an uncontended minor fault should cost — the interesting question is
    /// whether the kernel will do them in bulk under one lock acquisition instead. Failure is
    /// returned rather than panicked on: wiring is capped by `vm.user_wire_limit`, and being
    /// refused is a result.
    pub fn wire(&self, off: usize, len: usize) -> io::Result<()> {
        if len == 0 {
            return Ok(());
        }
        let addr = unsafe { (self.ptr as *const u8).add(off) } as *const c_void;
        if unsafe { mlock(addr, len) } != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    pub fn as_slice(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        unsafe { core::slice::from_raw_parts(self.ptr as *const u8, self.len) }
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        if self.len != 0 {
            unsafe { munmap(self.ptr, self.len) };
        }
    }
}
