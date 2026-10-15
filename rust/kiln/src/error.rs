use kiln_sys as sys;
use std::ffi::CStr;
use std::fmt;

/// What went wrong, as kiln reports it (`kiln_status` in `kiln.h`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    InvalidArg,
    /// No backend can run this model on this device.
    NoBackend,
    /// The file is missing, corrupt, or of an unsupported architecture.
    ModelLoad,
    OutOfMemory,
    ContextFull,
    /// Stopped by [`CancelHandle::cancel`](crate::CancelHandle::cancel).
    Cancelled,
    Backend,
    NotSupported,
    Internal,
    /// A status this version of the crate does not know.
    Other(u32),
}

impl Status {
    pub(crate) fn from_raw(s: sys::kiln_status) -> Status {
        match s {
            sys::kiln_status_KILN_ERR_INVALID_ARG => Status::InvalidArg,
            sys::kiln_status_KILN_ERR_NO_BACKEND => Status::NoBackend,
            sys::kiln_status_KILN_ERR_MODEL_LOAD => Status::ModelLoad,
            sys::kiln_status_KILN_ERR_OUT_OF_MEMORY => Status::OutOfMemory,
            sys::kiln_status_KILN_ERR_CONTEXT_FULL => Status::ContextFull,
            sys::kiln_status_KILN_ERR_CANCELLED => Status::Cancelled,
            sys::kiln_status_KILN_ERR_BACKEND => Status::Backend,
            sys::kiln_status_KILN_ERR_NOT_SUPPORTED => Status::NotSupported,
            sys::kiln_status_KILN_ERR_INTERNAL => Status::Internal,
            other => Status::Other(other),
        }
    }
}

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Status::InvalidArg => "invalid argument",
            Status::NoBackend => "no backend",
            Status::ModelLoad => "model load failed",
            Status::OutOfMemory => "out of memory",
            Status::ContextFull => "context full",
            Status::Cancelled => "cancelled",
            Status::Backend => "backend error",
            Status::NotSupported => "not supported",
            Status::Internal => "internal error",
            Status::Other(n) => return write!(f, "status {n}"),
        };
        f.write_str(s)
    }
}

/// A failed kiln call: the status and kiln's message for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    pub status: Status,
    pub message: String,
}

impl Error {
    pub(crate) fn new(status: Status, message: impl Into<String>) -> Error {
        Error {
            status,
            message: message.into(),
        }
    }

    /// Reads kiln's message for the call that just failed. kiln keeps it per thread, so this
    /// must run on the thread that made the call, right after it.
    pub(crate) fn last(status: sys::kiln_status) -> Error {
        let msg = unsafe { CStr::from_ptr(sys::kiln_last_error()) }.to_string_lossy().into_owned();
        Error::new(Status::from_raw(status), msg)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.message.is_empty() {
            write!(f, "{}", self.status)
        } else {
            write!(f, "{} ({})", self.message, self.status)
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn check(status: sys::kiln_status) -> Result<()> {
    if status == sys::kiln_status_KILN_OK {
        Ok(())
    } else {
        Err(Error::last(status))
    }
}
