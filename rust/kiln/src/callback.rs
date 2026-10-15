// The C callbacks (kiln_token_cb, kiln_audio_cb) and the Rust closures behind them.
//
// kiln calls the callback on the thread that made the call, with `user` pointing at one of the
// sinks below. A panic must not unwind into C: it is caught, generation is stopped, and the
// panic resumes once kiln has returned.
use crate::types::{Channel, Piece};
use kiln_sys as sys;
use std::any::Any;
use std::ffi::{c_char, c_void, CStr};
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};

/// Collects bytes and hands out whole UTF-8 characters. A token can end in the middle of a
/// character ("é" is two bytes; some models split it over two tokens).
#[derive(Default)]
pub(crate) struct Utf8Carry {
    buf: Vec<u8>,
}

impl Utf8Carry {
    /// Adds `bytes` and returns the text that is complete so far. Invalid bytes become U+FFFD.
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.buf.extend_from_slice(bytes);
        let mut out = String::new();
        let mut start = 0;
        loop {
            match std::str::from_utf8(&self.buf[start..]) {
                Ok(s) => {
                    out.push_str(s);
                    start = self.buf.len();
                    break;
                }
                Err(e) => {
                    let valid = start + e.valid_up_to();
                    out.push_str(std::str::from_utf8(&self.buf[start..valid]).unwrap());
                    match e.error_len() {
                        Some(n) => {
                            out.push('\u{FFFD}');
                            start = valid + n;
                        }
                        None => {
                            start = valid; // an unfinished character at the end: keep it
                            break;
                        }
                    }
                }
            }
        }
        self.buf.drain(..start);
        out
    }

    /// Whatever is left when generation ends.
    pub fn finish(&mut self) -> String {
        let s = String::from_utf8_lossy(&self.buf).into_owned();
        self.buf.clear();
        s
    }
}

pub(crate) struct TokenSink<'f> {
    f: &'f mut dyn FnMut(Piece<'_>) -> bool,
    carry: [Utf8Carry; 2],
    last_token: [i32; 2],
    stopped: bool,
    panic: Option<Box<dyn Any + Send>>,
}

impl<'f> TokenSink<'f> {
    pub fn new(f: &'f mut dyn FnMut(Piece<'_>) -> bool) -> Self {
        TokenSink {
            f,
            carry: Default::default(),
            last_token: [0; 2],
            stopped: false,
            panic: None,
        }
    }

    pub fn user(&mut self) -> *mut c_void {
        self as *mut Self as *mut c_void
    }

    fn feed(&mut self, channel: Channel, bytes: &[u8], token_id: i32) -> bool {
        let i = channel as usize;
        self.last_token[i] = token_id;
        let text = self.carry[i].push(bytes);
        if text.is_empty() {
            return true;
        }
        let go_on = (self.f)(Piece {
            channel,
            text: &text,
            token_id,
        });
        if !go_on {
            self.stopped = true;
        }
        go_on
    }

    /// Call after kiln returns: hands over any bytes still held, then re-raises a panic from
    /// the closure.
    pub fn finish(mut self) {
        if let Some(p) = self.panic.take() {
            resume_unwind(p);
        }
        if self.stopped {
            return;
        }
        for channel in [Channel::Reasoning, Channel::Answer] {
            let i = channel as usize;
            let text = self.carry[i].finish();
            if !text.is_empty() {
                let token_id = self.last_token[i];
                if let Err(p) = catch_unwind(AssertUnwindSafe(|| {
                    (self.f)(Piece {
                        channel,
                        text: &text,
                        token_id,
                    })
                })) {
                    resume_unwind(p);
                }
            }
        }
    }
}

pub(crate) unsafe extern "C" fn token_cb(channel: sys::kiln_channel, piece: *const c_char, token_id: i32, user: *mut c_void) -> bool {
    let sink = &mut *(user as *mut TokenSink<'_>);
    if sink.panic.is_some() || sink.stopped {
        return false;
    }
    let bytes = if piece.is_null() {
        &[][..]
    } else {
        CStr::from_ptr(piece).to_bytes()
    };
    match catch_unwind(AssertUnwindSafe(|| sink.feed(Channel::from_raw(channel), bytes, token_id))) {
        Ok(go_on) => go_on,
        Err(p) => {
            sink.panic = Some(p);
            false
        }
    }
}

pub(crate) struct AudioSink<'f> {
    f: &'f mut dyn FnMut(&[f32], u32) -> bool,
    panic: Option<Box<dyn Any + Send>>,
}

impl<'f> AudioSink<'f> {
    pub fn new(f: &'f mut dyn FnMut(&[f32], u32) -> bool) -> Self {
        AudioSink { f, panic: None }
    }

    pub fn user(&mut self) -> *mut c_void {
        self as *mut Self as *mut c_void
    }

    pub fn finish(mut self) {
        if let Some(p) = self.panic.take() {
            resume_unwind(p);
        }
    }
}

pub(crate) unsafe extern "C" fn audio_cb(samples: *const f32, n: i32, sample_rate: i32, user: *mut c_void) -> bool {
    let sink = &mut *(user as *mut AudioSink<'_>);
    if sink.panic.is_some() {
        return false;
    }
    let pcm = if samples.is_null() || n <= 0 {
        &[][..]
    } else {
        std::slice::from_raw_parts(samples, n as usize)
    };
    match catch_unwind(AssertUnwindSafe(|| (sink.f)(pcm, sample_rate.max(0) as u32))) {
        Ok(go_on) => go_on,
        Err(p) => {
            sink.panic = Some(p);
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Utf8Carry;

    #[test]
    fn whole_characters_only() {
        let mut c = Utf8Carry::default();
        let e = "é".as_bytes(); // [0xC3, 0xA9]
        assert_eq!(c.push(b"caf"), "caf");
        assert_eq!(c.push(&e[..1]), "");
        assert_eq!(c.push(&e[1..]), "é");
        assert_eq!(c.finish(), "");
    }

    #[test]
    fn four_byte_character_over_three_pieces() {
        let mut c = Utf8Carry::default();
        let s = "🙂".as_bytes();
        assert_eq!(c.push(&s[..1]), "");
        assert_eq!(c.push(&s[1..3]), "");
        assert_eq!(c.push(&s[3..]), "🙂");
    }

    #[test]
    fn invalid_bytes_are_replaced() {
        let mut c = Utf8Carry::default();
        assert_eq!(c.push(b"a\xFFb"), "a\u{FFFD}b");
        assert_eq!(c.push(b"\xC3"), "");
        assert_eq!(c.finish(), "\u{FFFD}");
    }
}
