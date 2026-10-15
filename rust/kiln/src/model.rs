use crate::context::Context;
use crate::error::{check, Error, Result, Status};
use crate::types::*;
use kiln_sys as sys;
use std::ffi::{c_char, CString};
use std::path::Path;
use std::ptr::NonNull;

/// A loaded model file (GGUF, or a whisper.cpp `.bin` for Whisper and Parakeet).
///
/// One model can serve several [`Context`]s, also from different threads. It is freed when
/// dropped, which the borrow checker only allows once every context made from it is gone.
pub struct Model {
    ptr: NonNull<sys::kiln_model>,
    info: ModelInfo,
}

// kiln shares a model between contexts and serialises the parts that are not thread-safe
// (speech engines, image / audio encoders) with its own locks.
unsafe impl Send for Model {}
unsafe impl Sync for Model {}

impl Model {
    pub fn load(path: impl AsRef<Path>, params: &ModelParams) -> Result<Model> {
        let path = path_cstring(path.as_ref())?;
        let raw = params.to_raw();
        let mut out: *mut sys::kiln_model = std::ptr::null_mut();
        check(unsafe { sys::kiln_model_load(path.as_ptr(), &raw, &mut out) })?;
        let ptr = NonNull::new(out).ok_or_else(|| Error::new(Status::Internal, "kiln_model_load returned no model"))?;
        let mut info: sys::kiln_model_info = unsafe { std::mem::zeroed() };
        let status = unsafe { sys::kiln_model_get_info(ptr.as_ptr(), &mut info) };
        let model = Model {
            ptr,
            info: ModelInfo::from(&info),
        };
        check(status)?;
        Ok(model)
    }

    /// What actually happened at load: backend, architecture, quantisation, and what kind of
    /// model this is.
    pub fn info(&self) -> &ModelInfo {
        &self.info
    }

    /// Loads the image / audio encoders that belong to this model (its "mmproj" GGUF): an
    /// embedding model that takes media, Qwen3-ASR, or a text-to-speech model. Call once,
    /// before creating contexts. The encoders are freed with the model.
    pub fn load_projector(&mut self, path: impl AsRef<Path>, params: &ProjectorParams) -> Result<ProjectorInfo> {
        let path = path_cstring(path.as_ref())?;
        let raw = sys::kiln_projector_params {
            image_max_tokens: params.image_max_tokens,
        };
        let mut out: sys::kiln_projector_info = unsafe { std::mem::zeroed() };
        check(unsafe { sys::kiln_model_load_projector(self.ptr.as_ptr(), path.as_ptr(), &raw, &mut out) })?;
        Ok(ProjectorInfo::from(&out))
    }

    /// A new context: one conversation with its own KV cache.
    pub fn context(&self, params: &ContextParams) -> Result<Context<'_>> {
        Context::new(self, params)
    }

    pub fn tokenize(&self, text: &str, add_special: bool) -> Result<Vec<i32>> {
        let c = cstring(text)?;
        let mut out = vec![0i32; text.len() + 16];
        loop {
            let n = unsafe { sys::kiln_tokenize(self.ptr.as_ptr(), c.as_ptr(), add_special, out.as_mut_ptr(), out.len() as i32) };
            if n >= 0 {
                out.truncate(n as usize);
                return Ok(out);
            }
            out.resize((-n) as usize, 0);
        }
    }

    /// Turns tokens back into text. Bytes that are not valid UTF-8 become U+FFFD.
    pub fn detokenize(&self, tokens: &[i32]) -> Result<String> {
        let bytes = fill_buffer(tokens.len() * 8 + 16, |buf, cap| unsafe {
            sys::kiln_detokenize(self.ptr.as_ptr(), tokens.as_ptr(), tokens.len() as i32, buf, cap)
        });
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// The prompt the model's own chat template makes of `messages`. For generation use
    /// [`Context::chat`], which also knows whether the prompt opened a reasoning block.
    pub fn apply_chat_template(&self, messages: &[Message], opts: &ChatOptions) -> Result<String> {
        let (_keep, raw) = raw_messages(messages)?;
        let o = opts.to_raw();
        let bytes = fill_buffer(4096, |buf, cap| unsafe {
            sys::kiln_apply_chat_template(self.ptr.as_ptr(), raw.as_ptr(), raw.len() as i32, &o, buf, cap)
        });
        if bytes.is_empty() && !messages.is_empty() {
            return Err(Error::new(Status::InvalidArg, "the chat template could not be applied"));
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    pub(crate) fn as_ptr(&self) -> *mut sys::kiln_model {
        self.ptr.as_ptr()
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe { sys::kiln_model_free(self.ptr.as_ptr()) }
    }
}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model").field("info", &self.info).finish()
    }
}

/// For the C functions that return the bytes written, or minus the size they need.
fn fill_buffer(first_cap: usize, mut call: impl FnMut(*mut c_char, i32) -> i32) -> Vec<u8> {
    let mut buf = vec![0u8; first_cap.max(1)];
    loop {
        let n = call(buf.as_mut_ptr() as *mut c_char, buf.len() as i32);
        if n >= 0 {
            buf.truncate(n as usize);
            return buf;
        }
        buf.resize((-n) as usize + 1, 0); // + 1 for the NUL
    }
}

/// C copies of `messages`; the returned structs point into the strings, which must outlive them.
pub(crate) fn raw_messages(messages: &[Message]) -> Result<(Vec<CString>, Vec<sys::kiln_chat_message>)> {
    let keep = messages.iter().map(|m| cstring(&m.content)).collect::<Result<Vec<_>>>()?;
    let raw = messages
        .iter()
        .zip(&keep)
        .map(|(m, c)| sys::kiln_chat_message {
            role: m.role.as_cstr().as_ptr(),
            content: c.as_ptr(),
        })
        .collect();
    Ok((keep, raw))
}
