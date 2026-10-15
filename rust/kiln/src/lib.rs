//! Safe Rust bindings to kiln, an on-device LLM runtime: chat, decision, embedding,
//! speech-to-text and text-to-speech models, through llama.cpp and whisper.cpp.
//!
//! ```no_run
//! use kiln::{ChatOptions, ContextParams, Message, Model, ModelParams, Sampling};
//!
//! let model = Model::load("models/Qwen3.5-0.8B-Q4_K_M.gguf", &ModelParams::default())?;
//! let mut ctx = model.context(&ContextParams::default())?;
//! let messages = [Message::user("What is the capital of France?")];
//! ctx.chat(&messages, &ChatOptions::default(), &Sampling::default(), |piece| {
//!     print!("{}", piece.text);
//!     true // false stops the reply
//! })?;
//! # Ok::<(), kiln::Error>(())
//! ```
//!
//! The C API this wraps is `include/kiln/kiln.h`; its comments are the full reference.

mod callback;
mod context;
mod error;
mod model;
mod types;

pub use context::{CancelHandle, Context};
pub use error::{Error, Result, Status};
pub use model::Model;
pub use types::*;

pub use kiln_sys as sys;

use std::ffi::{c_char, c_void, CStr};
use std::sync::RwLock;

/// kiln's version, e.g. "0.5.0".
pub fn version() -> &'static str {
    // a string literal inside libkiln, valid for the life of the program
    unsafe { CStr::from_ptr(sys::kiln_version()) }.to_str().unwrap_or("")
}

/// What kiln finds on this device: processor, fast cores, memory, GPU.
pub fn probe_device() -> Result<DeviceInfo> {
    let mut d: sys::kiln_device_info = unsafe { std::mem::zeroed() };
    error::check(unsafe { sys::kiln_probe_device(&mut d) })?;
    Ok(DeviceInfo::from(&d))
}

type LogFn = Box<dyn Fn(LogLevel, &str) + Send + Sync>;
static LOGGER: RwLock<Option<LogFn>> = RwLock::new(None);

unsafe extern "C" fn log_cb(level: sys::kiln_log_level, msg: *const c_char, _user: *mut c_void) {
    if msg.is_null() {
        return;
    }
    let msg = CStr::from_ptr(msg).to_string_lossy();
    let _ = std::panic::catch_unwind(|| {
        if let Ok(guard) = LOGGER.read() {
            if let Some(f) = guard.as_ref() {
                f(LogLevel::from_raw(level), msg.trim_end());
            }
        }
    });
}

/// Sends kiln's log lines to `f`, from whatever thread logs them. `None` turns it off.
pub fn set_log_callback(f: Option<impl Fn(LogLevel, &str) + Send + Sync + 'static>) {
    let on = f.is_some();
    *LOGGER.write().unwrap_or_else(|e| e.into_inner()) = f.map(|f| Box::new(f) as LogFn);
    unsafe { sys::kiln_set_log_callback(if on { Some(log_cb) } else { None }, std::ptr::null_mut()) }
}

/// The most detailed level that is logged (default: warnings).
pub fn set_log_level(level: LogLevel) {
    unsafe { sys::kiln_set_log_level(level.to_raw()) }
}

#[cfg(test)]
mod tests {
    #[test]
    fn version_matches_header() {
        let v = format!(
            "{}.{}.{}",
            super::sys::KILN_VERSION_MAJOR,
            super::sys::KILN_VERSION_MINOR,
            super::sys::KILN_VERSION_PATCH
        );
        assert_eq!(
            super::version(),
            v,
            "libkiln and kiln.h disagree: rebuild, or regenerate the bindings"
        );
        assert_eq!(env!("CARGO_PKG_VERSION"), v, "rust/Cargo.toml has another version than kiln.h");
    }
}
