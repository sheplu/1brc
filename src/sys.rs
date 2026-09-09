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
