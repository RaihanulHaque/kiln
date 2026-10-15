use crate::callback::{audio_cb, token_cb, AudioSink, TokenSink};
use crate::error::{check, Error, Result, Status};
use crate::model::{raw_messages, Model};
use crate::types::*;
use kiln_sys as sys;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

/// One conversation and its KV cache, made with [`Model::context`].
///
/// Every call blocks until it is done; run it off your UI thread. To stop a call from another
/// thread, take a [`CancelHandle`] first. The streaming closures run on the calling thread,
/// so they need not be `Send`.
pub struct Context<'m> {
    ptr: NonNull<sys::kiln_context>,
    model: &'m Model,
    cancel: Arc<CancelShared>,
}

// A context is used by one thread at a time (every call takes &mut self) but may move between
// threads.
unsafe impl Send for Context<'_> {}

struct CancelShared {
    // null once the context is freed
    ptr: Mutex<*mut sys::kiln_context>,
}

unsafe impl Send for CancelShared {}
unsafe impl Sync for CancelShared {}

/// Stops the call running on a context, from any thread. Safe to keep and use after the
/// context is gone; it then does nothing.
#[derive(Clone)]
pub struct CancelHandle(Arc<CancelShared>);

impl CancelHandle {
    /// The running call returns `Err` with [`Status::Cancelled`].
    pub fn cancel(&self) {
        let p = self.0.ptr.lock().unwrap_or_else(|e| e.into_inner());
        if !p.is_null() {
            unsafe { sys::kiln_cancel(*p) }
        }
    }
}

impl<'m> Context<'m> {
    pub(crate) fn new(model: &'m Model, params: &ContextParams) -> Result<Context<'m>> {
        let raw = params.to_raw();
        let mut out: *mut sys::kiln_context = std::ptr::null_mut();
        check(unsafe { sys::kiln_context_create(model.as_ptr(), &raw, &mut out) })?;
        let ptr = NonNull::new(out).ok_or_else(|| Error::new(Status::Internal, "kiln_context_create returned no context"))?;
        Ok(Context {
            ptr,
            model,
            cancel: Arc::new(CancelShared { ptr: Mutex::new(out) }),
        })
    }

    pub fn model(&self) -> &'m Model {
        self.model
    }

    pub fn cancel_handle(&self) -> CancelHandle {
        CancelHandle(self.cancel.clone())
    }

    /// Forgets the conversation (clears the KV cache).
    pub fn reset(&mut self) -> Result<()> {
        check(unsafe { sys::kiln_context_reset(self.ptr.as_ptr()) })
    }

    /// Answers the last message of `messages` and streams the reply to `on_piece`. Pass the
    /// whole conversation every turn; kiln reuses what is already in the cache. Return false
    /// from `on_piece` to stop (the call then returns `Ok` with `stopped_by_user`).
    pub fn chat(
        &mut self,
        messages: &[Message],
        opts: &ChatOptions,
        sampling: &Sampling,
        mut on_piece: impl FnMut(Piece<'_>) -> bool,
    ) -> Result<GenStats> {
        let (_keep, raw) = raw_messages(messages)?;
        let (o, p) = (opts.to_raw(), sampling.to_raw());
        let mut stats: sys::kiln_gen_stats = unsafe { std::mem::zeroed() };
        let mut sink = TokenSink::new(&mut on_piece);
        let status = unsafe {
            sys::kiln_chat(
                self.ptr.as_ptr(),
                raw.as_ptr(),
                raw.len() as i32,
                &o,
                &p,
                Some(token_cb),
                sink.user(),
                &mut stats,
            )
        };
        let result = check(status);
        sink.finish();
        result.map(|_| GenStats::from(&stats))
    }

    /// [`chat`](Self::chat), collecting the whole reply.
    pub fn chat_collect(&mut self, messages: &[Message], opts: &ChatOptions, sampling: &Sampling) -> Result<Reply> {
        let (mut answer, mut reasoning) = (String::new(), String::new());
        let stats = self.chat(messages, opts, sampling, |p| {
            match p.channel {
                Channel::Answer => answer.push_str(p.text),
                Channel::Reasoning => reasoning.push_str(p.text),
            }
            true
        })?;
        Ok(Reply { answer, reasoning, stats })
    }

    /// Continues a raw prompt that is already in the model's template.
    pub fn generate(&mut self, prompt: &str, sampling: &Sampling, mut on_piece: impl FnMut(Piece<'_>) -> bool) -> Result<GenStats> {
        let prompt = cstring(prompt)?;
        let p = sampling.to_raw();
        let mut stats: sys::kiln_gen_stats = unsafe { std::mem::zeroed() };
        let mut sink = TokenSink::new(&mut on_piece);
        let status = unsafe { sys::kiln_generate(self.ptr.as_ptr(), prompt.as_ptr(), &p, Some(token_cb), sink.user(), &mut stats) };
        let result = check(status);
        sink.finish();
        result.map(|_| GenStats::from(&stats))
    }

    /// Decision models: the probability of each answer to one typed question about `state`.
    pub fn decide(&mut self, state: Option<&str>, question: &Question) -> Result<Decision> {
        let state = opt_cstring(state)?;
        let id = cstring(&question.id)?;
        let instructions = cstring(&question.instructions)?;
        let keys = question.options.iter().map(|o| cstring(&o.key)).collect::<Result<Vec<_>>>()?;
        let descs = question
            .options
            .iter()
            .map(|o| opt_cstring(o.description.as_deref()))
            .collect::<Result<Vec<_>>>()?;
        let options: Vec<sys::kiln_decision_option> = keys
            .iter()
            .zip(&descs)
            .map(|(k, d)| sys::kiln_decision_option {
                key: k.as_ptr(),
                description: opt_ptr(d),
            })
            .collect();
        let q = sys::kiln_decision_question {
            id: id.as_ptr(),
            type_: question.kind.to_raw(),
            instructions: instructions.as_ptr(),
            options: if options.is_empty() { std::ptr::null() } else { options.as_ptr() },
            n_options: options.len() as i32,
        };
        let mut probs = vec![0f32; options.len().max(2)];
        let mut ans: sys::kiln_decision_answer = unsafe { std::mem::zeroed() };
        check(unsafe {
            sys::kiln_decide(
                self.ptr.as_ptr(),
                opt_ptr(&state),
                &q,
                probs.as_mut_ptr(),
                probs.len() as i32,
                &mut ans,
            )
        })?;
        probs.truncate(ans.n_probs.max(0) as usize);
        Ok(Decision {
            probs,
            best: ans.best.max(0) as usize,
            value: ans.value,
            confidence: ans.confidence,
            prompt_tokens: ans.prompt_tokens,
            ms: ans.ms,
        })
    }

    /// Embedding models: the vector for `text`.
    pub fn embed(&mut self, text: &str, opts: &EmbedOptions) -> Result<Embedding> {
        let text = cstring(text)?;
        let title = opt_cstring(opts.title.as_deref())?;
        let o = opts.to_raw(&title);
        let mut out = vec![0f32; self.out_len(opts)];
        let mut stats: sys::kiln_embed_stats = unsafe { std::mem::zeroed() };
        check(unsafe { sys::kiln_embed(self.ptr.as_ptr(), text.as_ptr(), &o, out.as_mut_ptr(), out.len() as i32, &mut stats) })?;
        out.truncate(stats.dimensions.max(0) as usize);
        Ok(Embedding {
            vector: out,
            stats: EmbedStats::from(&stats),
        })
    }

    /// Embedding models with a projector: one vector for a text with images and / or audio,
    /// or for media alone (`text` None). Each [`MEDIA_MARKER`] in the text is replaced by the
    /// next item; with no marker the items follow the text.
    pub fn embed_media(&mut self, text: Option<&str>, media: &[Media<'_>], opts: &EmbedOptions) -> Result<Embedding> {
        let text = opt_cstring(text)?;
        let title = opt_cstring(opts.title.as_deref())?;
        let o = opts.to_raw(&title);
        let keep = media.iter().map(RawMedia::new).collect::<Result<Vec<_>>>()?;
        let raw: Vec<sys::kiln_media> = keep.iter().map(|m| m.raw).collect();
        let mut out = vec![0f32; self.out_len(opts)];
        let mut stats: sys::kiln_embed_stats = unsafe { std::mem::zeroed() };
        check(unsafe {
            sys::kiln_embed_media(
                self.ptr.as_ptr(),
                opt_ptr(&text),
                raw.as_ptr(),
                raw.len() as i32,
                &o,
                out.as_mut_ptr(),
                out.len() as i32,
                &mut stats,
            )
        })?;
        out.truncate(stats.dimensions.max(0) as usize);
        Ok(Embedding {
            vector: out,
            stats: EmbedStats::from(&stats),
        })
    }

    fn out_len(&self, opts: &EmbedOptions) -> usize {
        let full = self.model.info().n_embd.max(1) as usize;
        if opts.dimensions > 0 {
            (opts.dimensions as usize).min(full)
        } else {
            full
        }
    }

    /// Speech models: writes down one audio clip and streams the words to `on_text`.
    /// Each call stands alone and clears the context.
    pub fn transcribe(
        &mut self,
        audio: &Media<'_>,
        opts: &TranscribeOptions,
        mut on_text: impl FnMut(&str) -> bool,
    ) -> Result<TranscribeStats> {
        let clip = RawMedia::new(audio)?;
        let language = opt_cstring(opts.language.as_deref())?;
        let prompt = opt_cstring(opts.prompt.as_deref())?;
        let o = sys::kiln_transcribe_options {
            language: opt_ptr(&language),
            prompt: opt_ptr(&prompt),
            max_tokens: opts.max_tokens,
        };
        let mut stats: sys::kiln_transcribe_stats = unsafe { std::mem::zeroed() };
        let mut on_piece = |p: Piece<'_>| on_text(p.text);
        let mut sink = TokenSink::new(&mut on_piece);
        let status = unsafe { sys::kiln_transcribe(self.ptr.as_ptr(), &clip.raw, &o, Some(token_cb), sink.user(), &mut stats) };
        let result = check(status);
        sink.finish();
        result.map(|_| TranscribeStats::from(&stats))
    }

    /// [`transcribe`](Self::transcribe), collecting the whole text.
    pub fn transcribe_collect(&mut self, audio: &Media<'_>, opts: &TranscribeOptions) -> Result<Transcript> {
        let mut text = String::new();
        let stats = self.transcribe(audio, opts, |t| {
            text.push_str(t);
            true
        })?;
        Ok(Transcript { text, stats })
    }

    /// Text-to-speech models (projector loaded): speaks `text`, handing over the audio about
    /// once a second as mono floats at the given sample rate. The conversation is kept.
    pub fn speak(&mut self, text: &str, opts: &SpeakOptions<'_>, mut on_audio: impl FnMut(&[f32], u32) -> bool) -> Result<SpeakStats> {
        let text = cstring(text)?;
        let voice = opts.voice.as_ref().map(RawMedia::new).transpose()?;
        let language = opt_cstring(opts.language.as_deref())?;
        let o = sys::kiln_speak_options {
            voice: voice.as_ref().map_or(std::ptr::null(), |v| &v.raw),
            language: opt_ptr(&language),
            max_frames: opts.max_frames,
            top_k: opts.top_k,
            top_p: opts.top_p,
            seed: opts.seed.unwrap_or(u32::MAX),
        };
        let mut stats: sys::kiln_speak_stats = unsafe { std::mem::zeroed() };
        let mut sink = AudioSink::new(&mut on_audio);
        let status = unsafe { sys::kiln_speak(self.ptr.as_ptr(), text.as_ptr(), &o, Some(audio_cb), sink.user(), &mut stats) };
        let result = check(status);
        sink.finish();
        result.map(|_| SpeakStats::from(&stats))
    }

    /// [`speak`](Self::speak), collecting all the audio.
    pub fn speak_collect(&mut self, text: &str, opts: &SpeakOptions<'_>) -> Result<Speech> {
        let (mut samples, mut rate) = (Vec::new(), 0);
        let stats = self.speak(text, opts, |pcm, r| {
            samples.extend_from_slice(pcm);
            rate = r;
            true
        })?;
        let sample_rate = if rate > 0 { rate } else { stats.sample_rate.max(0) as u32 };
        Ok(Speech {
            samples,
            sample_rate,
            stats,
        })
    }
}

impl Drop for Context<'_> {
    fn drop(&mut self) {
        let mut p = self.cancel.ptr.lock().unwrap_or_else(|e| e.into_inner());
        *p = std::ptr::null_mut();
        unsafe { sys::kiln_context_free(self.ptr.as_ptr()) }
    }
}
