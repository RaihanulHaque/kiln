// Rust versions of the structs in kiln.h. Defaults come from kiln's own *_default() functions,
// so they cannot drift from the C side.
use crate::error::{Error, Result, Status};
use kiln_sys as sys;
use std::ffi::{c_char, CString};
use std::path::Path;

pub(crate) fn cstring(s: &str) -> Result<CString> {
    CString::new(s).map_err(|_| Error::new(Status::InvalidArg, "string contains a NUL byte"))
}

pub(crate) fn opt_cstring(s: Option<&str>) -> Result<Option<CString>> {
    s.map(cstring).transpose()
}

pub(crate) fn opt_ptr(s: &Option<CString>) -> *const c_char {
    s.as_ref().map_or(std::ptr::null(), |c| c.as_ptr())
}

pub(crate) fn path_cstring(p: &Path) -> Result<CString> {
    let s = p
        .to_str()
        .ok_or_else(|| Error::new(Status::InvalidArg, format!("path is not UTF-8: {}", p.display())))?;
    cstring(s)
}

/// A fixed-size `char[N]` field from kiln, up to its NUL.
pub(crate) fn fixed_str(buf: &[c_char]) -> String {
    let bytes: Vec<u8> = buf.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

// ---------------------------------------------------------------- device

/// Which engine and processor runs a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    /// Probe the device and pick the best one. Recommended.
    #[default]
    Auto,
    Cpu,
    Gpu,
    Npu,
}

impl Backend {
    fn to_raw(self) -> sys::kiln_backend_kind {
        match self {
            Backend::Auto => sys::kiln_backend_kind_KILN_BACKEND_AUTO,
            Backend::Cpu => sys::kiln_backend_kind_KILN_BACKEND_CPU,
            Backend::Gpu => sys::kiln_backend_kind_KILN_BACKEND_GPU,
            Backend::Npu => sys::kiln_backend_kind_KILN_BACKEND_NPU,
        }
    }
    fn from_raw(k: sys::kiln_backend_kind) -> Backend {
        match k {
            sys::kiln_backend_kind_KILN_BACKEND_CPU => Backend::Cpu,
            sys::kiln_backend_kind_KILN_BACKEND_GPU => Backend::Gpu,
            sys::kiln_backend_kind_KILN_BACKEND_NPU => Backend::Npu,
            _ => Backend::Auto,
        }
    }
}

/// What [`probe_device`](crate::probe_device) found.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub soc: String,
    pub abi: String,
    pub cpu_cores: i32,
    /// Cores kiln treats as fast: all but the little cores (= `cpu_cores` if unknown).
    pub cpu_big_cores: i32,
    /// Bit i set = cpu i is one of those cores; 0 = no affinity information.
    pub cpu_fast_mask: u64,
    pub ram_total_bytes: u64,
    pub ram_available_bytes: u64,
    pub has_fp16: bool,
    pub has_dotprod: bool,
    pub has_i8mm: bool,
    pub has_sve: bool,
    pub has_gpu: bool,
    pub gpu_name: String,
    pub has_npu: bool,
    pub npu_name: String,
}

impl From<&sys::kiln_device_info> for DeviceInfo {
    fn from(d: &sys::kiln_device_info) -> Self {
        DeviceInfo {
            soc: fixed_str(&d.soc),
            abi: fixed_str(&d.abi),
            cpu_cores: d.cpu_cores,
            cpu_big_cores: d.cpu_big_cores,
            cpu_fast_mask: d.cpu_fast_mask,
            ram_total_bytes: d.ram_total_bytes,
            ram_available_bytes: d.ram_available_bytes,
            has_fp16: d.has_fp16,
            has_dotprod: d.has_dotprod,
            has_i8mm: d.has_i8mm,
            has_sve: d.has_sve,
            has_gpu: d.has_gpu,
            gpu_name: fixed_str(&d.gpu_name),
            has_npu: d.has_npu,
            npu_name: fixed_str(&d.npu_name),
        }
    }
}

// ---------------------------------------------------------------- model

#[derive(Debug, Clone)]
pub struct ModelParams {
    pub backend: Backend,
    /// -1 = all layers if a GPU is chosen, 0 = none.
    pub n_gpu_layers: i32,
    /// 0 = from the device probe.
    pub n_threads: i32,
    pub use_mmap: bool,
    pub use_mlock: bool,
}

impl Default for ModelParams {
    fn default() -> Self {
        let p = unsafe { sys::kiln_model_params_default() };
        ModelParams {
            backend: Backend::from_raw(p.backend),
            n_gpu_layers: p.n_gpu_layers,
            n_threads: p.n_threads,
            use_mmap: p.use_mmap,
            use_mlock: p.use_mlock,
        }
    }
}

impl ModelParams {
    pub(crate) fn to_raw(&self) -> sys::kiln_model_params {
        sys::kiln_model_params {
            backend: self.backend.to_raw(),
            n_gpu_layers: self.n_gpu_layers,
            n_threads: self.n_threads,
            use_mmap: self.use_mmap,
            use_mlock: self.use_mlock,
        }
    }
}

/// What actually happened when the model was loaded.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub backend_used: Backend,
    /// e.g. "llama.cpp/cpu", "whisper.cpp/cpu"
    pub backend_name: String,
    /// e.g. "qwen35", "gemma3"
    pub arch: String,
    /// e.g. "Q4_K_M"
    pub quant: String,
    pub n_params: i64,
    pub n_ctx_train: i32,
    pub n_vocab: i32,
    pub n_gpu_layers_used: i32,
    /// Resident weight memory.
    pub model_bytes: u64,
    /// The chat template has a reasoning switch (Qwen3.x, Gemma 4, ...).
    pub supports_thinking: bool,
    /// Answers typed questions with [`Context::decide`](crate::Context::decide); cannot write text.
    pub is_decision_model: bool,
    /// Turns text into a vector with [`Context::embed`](crate::Context::embed); cannot write text.
    pub is_embedding_model: bool,
    /// Embedding models: length of the vector; else 0.
    pub n_embd: i32,
    /// Whisper or Parakeet: transcribes without a projector. False for Qwen3-ASR.
    pub is_speech_model: bool,
}

impl From<&sys::kiln_model_info> for ModelInfo {
    fn from(m: &sys::kiln_model_info) -> Self {
        ModelInfo {
            backend_used: Backend::from_raw(m.backend_used),
            backend_name: fixed_str(&m.backend_name),
            arch: fixed_str(&m.arch),
            quant: fixed_str(&m.quant),
            n_params: m.n_params,
            n_ctx_train: m.n_ctx_train,
            n_vocab: m.n_vocab,
            n_gpu_layers_used: m.n_gpu_layers_used,
            model_bytes: m.model_bytes,
            supports_thinking: m.supports_thinking,
            is_decision_model: m.is_decision_model,
            is_embedding_model: m.is_embedding_model,
            n_embd: m.n_embd,
            is_speech_model: m.is_speech_model,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProjectorParams {
    /// Most tokens one image may take; a large photo is scaled down to fit. 0 = 280,
    /// negative = the model's own limit. Time grows much faster than this number.
    pub image_max_tokens: i32,
}

impl Default for ProjectorParams {
    fn default() -> Self {
        let p = unsafe { sys::kiln_projector_params_default() };
        ProjectorParams {
            image_max_tokens: p.image_max_tokens,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ProjectorInfo {
    pub supports_images: bool,
    pub supports_audio: bool,
    /// A text-to-speech projector: the model can [`speak`](crate::Context::speak).
    pub generates_audio: bool,
    /// Of the generated audio, e.g. 24000; 0 when `generates_audio` is false.
    pub sample_rate: i32,
}

impl From<&sys::kiln_projector_info> for ProjectorInfo {
    fn from(p: &sys::kiln_projector_info) -> Self {
        ProjectorInfo {
            supports_images: p.supports_images,
            supports_audio: p.supports_audio,
            generates_audio: p.generates_audio,
            sample_rate: p.sample_rate,
        }
    }
}

// ---------------------------------------------------------------- context

#[derive(Debug, Clone)]
pub struct ContextParams {
    /// 0 = the model's default, capped by RAM. kiln may give a different size; see `kiln.h`.
    pub n_ctx: i32,
    pub n_batch: i32,
    pub flash_attn: bool,
}

impl Default for ContextParams {
    fn default() -> Self {
        let p = unsafe { sys::kiln_context_params_default() };
        ContextParams {
            n_ctx: p.n_ctx,
            n_batch: p.n_batch,
            flash_attn: p.flash_attn,
        }
    }
}

impl ContextParams {
    pub(crate) fn to_raw(&self) -> sys::kiln_context_params {
        sys::kiln_context_params {
            n_ctx: self.n_ctx,
            n_batch: self.n_batch,
            flash_attn: self.flash_attn,
        }
    }
}

// ---------------------------------------------------------------- chat

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    pub(crate) fn as_cstr(self) -> &'static std::ffi::CStr {
        match self {
            Role::System => c"system",
            Role::User => c"user",
            Role::Assistant => c"assistant",
        }
    }
}

/// One turn of a conversation.
#[derive(Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Message {
        Message {
            role: Role::System,
            content: content.into(),
        }
    }
    pub fn user(content: impl Into<String>) -> Message {
        Message {
            role: Role::User,
            content: content.into(),
        }
    }
    pub fn assistant(content: impl Into<String>) -> Message {
        Message {
            role: Role::Assistant,
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChatOptions {
    /// Append the assistant turn opener (true for generation).
    pub add_assistant_prefix: bool,
    /// Reasoning models only. false = answer directly. Ignored if unsupported.
    pub enable_thinking: bool,
    /// Thinking budget: after this many reasoning tokens kiln closes the think block and the
    /// model answers. <= 0 = unlimited. Small models often never stop thinking by themselves.
    pub max_reasoning_tokens: i32,
}

impl Default for ChatOptions {
    fn default() -> Self {
        let o = unsafe { sys::kiln_chat_options_default() };
        ChatOptions {
            add_assistant_prefix: o.add_assistant_prefix,
            enable_thinking: o.enable_thinking,
            max_reasoning_tokens: o.max_reasoning_tokens,
        }
    }
}

impl ChatOptions {
    pub(crate) fn to_raw(&self) -> sys::kiln_chat_options {
        sys::kiln_chat_options {
            add_assistant_prefix: self.add_assistant_prefix,
            enable_thinking: self.enable_thinking,
            max_reasoning_tokens: self.max_reasoning_tokens,
        }
    }
}

/// How the next token is picked.
#[derive(Debug, Clone)]
pub struct Sampling {
    /// <= 0 = greedy: always the most likely token.
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub repeat_last_n: i32,
    /// 0 = random.
    pub seed: u32,
    /// <= 0 = until the model stops or the context is full.
    pub max_tokens: i32,
}

impl Default for Sampling {
    fn default() -> Self {
        let p = unsafe { sys::kiln_sampling_params_default() };
        Sampling {
            temperature: p.temperature,
            top_p: p.top_p,
            top_k: p.top_k,
            min_p: p.min_p,
            repeat_penalty: p.repeat_penalty,
            repeat_last_n: p.repeat_last_n,
            seed: p.seed,
            max_tokens: p.max_tokens,
        }
    }
}

impl Sampling {
    /// Always the most likely token, no repeat penalty: the same prompt gives the same text.
    pub fn greedy() -> Sampling {
        Sampling {
            temperature: 0.0,
            repeat_penalty: 1.0,
            ..Sampling::default()
        }
    }

    pub(crate) fn to_raw(&self) -> sys::kiln_sampling_params {
        sys::kiln_sampling_params {
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            min_p: self.min_p,
            repeat_penalty: self.repeat_penalty,
            repeat_last_n: self.repeat_last_n,
            seed: self.seed,
            max_tokens: self.max_tokens,
        }
    }
}

/// Where a piece of generated text belongs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channel {
    Answer,
    /// Inside the model's think block; the tags themselves are removed.
    Reasoning,
}

impl Channel {
    pub(crate) fn from_raw(c: sys::kiln_channel) -> Channel {
        if c == sys::kiln_channel_KILN_CHANNEL_REASONING {
            Channel::Reasoning
        } else {
            Channel::Answer
        }
    }
}

/// A piece of streamed text. Unlike the C API, `text` is always whole UTF-8 characters: a
/// character split across two tokens arrives with the second one.
#[derive(Debug, Clone, Copy)]
pub struct Piece<'a> {
    pub channel: Channel,
    pub text: &'a str,
    /// The token that completed this text.
    pub token_id: i32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct GenStats {
    /// Tokens in the rendered prompt.
    pub prompt_tokens: i32,
    /// Of those, how many were already in the KV cache (earlier turns).
    pub cached_tokens: i32,
    pub generated_tokens: i32,
    /// Of `generated_tokens`, how many went to the reasoning channel.
    pub reasoning_tokens: i32,
    pub prompt_ms: f64,
    pub gen_ms: f64,
    pub prompt_tps: f64,
    pub gen_tps: f64,
    pub stopped_by_eos: bool,
    pub stopped_by_user: bool,
    /// `max_reasoning_tokens` was reached and kiln closed the think block.
    pub reasoning_truncated: bool,
}

impl From<&sys::kiln_gen_stats> for GenStats {
    fn from(s: &sys::kiln_gen_stats) -> Self {
        GenStats {
            prompt_tokens: s.prompt_tokens,
            cached_tokens: s.cached_tokens,
            generated_tokens: s.generated_tokens,
            reasoning_tokens: s.reasoning_tokens,
            prompt_ms: s.prompt_ms,
            gen_ms: s.gen_ms,
            prompt_tps: s.prompt_tps,
            gen_tps: s.gen_tps,
            stopped_by_eos: s.stopped_by_eos,
            stopped_by_user: s.stopped_by_user,
            reasoning_truncated: s.reasoning_truncated,
        }
    }
}

/// A whole reply, from [`Context::chat_collect`](crate::Context::chat_collect).
#[derive(Debug, Clone, Default)]
pub struct Reply {
    pub answer: String,
    pub reasoning: String,
    pub stats: GenStats,
}

// ---------------------------------------------------------------- decision models

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuestionType {
    /// Yes or no.
    YesNo,
    /// Pick one of the options.
    Choice,
    /// Rate on a scale; the options are the levels, lowest first (2 to 10).
    Score,
}

impl QuestionType {
    pub(crate) fn to_raw(self) -> sys::kiln_question_type {
        match self {
            QuestionType::YesNo => sys::kiln_question_type_KILN_QUESTION_NOUL,
            QuestionType::Choice => sys::kiln_question_type_KILN_QUESTION_CHOICE,
            QuestionType::Score => sys::kiln_question_type_KILN_QUESTION_SCORE,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DecisionOption {
    /// Choice: the label returned. Yes / no: "true" or "false". Score: ignored.
    pub key: String,
    /// What the option means.
    pub description: Option<String>,
}

impl DecisionOption {
    pub fn new(key: impl Into<String>, description: Option<&str>) -> DecisionOption {
        DecisionOption {
            key: key.into(),
            description: description.map(str::to_owned),
        }
    }
}

/// One typed question for a decision model.
#[derive(Debug, Clone)]
pub struct Question {
    /// Short name, e.g. "wants_refund".
    pub id: String,
    pub kind: QuestionType,
    /// The question itself.
    pub instructions: String,
    /// Yes / no: optional, only to describe what yes and no mean.
    pub options: Vec<DecisionOption>,
}

#[derive(Debug, Clone)]
pub struct Decision {
    /// One probability per option. Yes / no: `[no, yes]`.
    pub probs: Vec<f32>,
    /// Index of the most probable entry.
    pub best: usize,
    /// Yes / no: P(yes). Score: the expected level, 0 .. n-1. Choice: P(best).
    pub value: f32,
    pub confidence: f32,
    pub prompt_tokens: i32,
    pub ms: f64,
}

// ---------------------------------------------------------------- embeddings

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EmbedKind {
    /// The text exactly as given.
    #[default]
    Raw,
    /// A search query: kiln adds the model's query prefix.
    Query,
    /// A text to be searched: kiln adds the model's document prefix.
    Document,
}

#[derive(Debug, Clone)]
pub struct EmbedOptions {
    pub kind: EmbedKind,
    /// Documents only, optional: the document's title, for models that take one.
    pub title: Option<String>,
    /// Keep only the first N numbers. <= 0 = the full vector. Only models trained for it stay
    /// accurate (EmbeddingGemma: 512, 256, 128).
    pub dimensions: i32,
    /// Scale the vector to length 1, so similarity is the dot product.
    pub normalize: bool,
}

impl Default for EmbedOptions {
    fn default() -> Self {
        let o = unsafe { sys::kiln_embed_options_default() };
        EmbedOptions {
            kind: EmbedKind::Raw,
            title: None,
            dimensions: o.dimensions,
            normalize: o.normalize,
        }
    }
}

impl EmbedOptions {
    pub fn query() -> EmbedOptions {
        EmbedOptions {
            kind: EmbedKind::Query,
            ..EmbedOptions::default()
        }
    }
    pub fn document() -> EmbedOptions {
        EmbedOptions {
            kind: EmbedKind::Document,
            ..EmbedOptions::default()
        }
    }

    /// The returned struct points into `title`, which must outlive it.
    pub(crate) fn to_raw(&self, title: &Option<CString>) -> sys::kiln_embed_options {
        sys::kiln_embed_options {
            kind: match self.kind {
                EmbedKind::Raw => sys::kiln_embed_kind_KILN_EMBED_RAW,
                EmbedKind::Query => sys::kiln_embed_kind_KILN_EMBED_QUERY,
                EmbedKind::Document => sys::kiln_embed_kind_KILN_EMBED_DOCUMENT,
            },
            title: opt_ptr(title),
            dimensions: self.dimensions,
            normalize: self.normalize,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct EmbedStats {
    /// Tokens the model read: text, prefix and media together.
    pub tokens: i32,
    /// Of those, how many came from images and audio.
    pub media_tokens: i32,
    pub dimensions: i32,
    /// A task prefix was added.
    pub prefixed: bool,
    /// Time inside the image / audio encoders only.
    pub media_ms: f64,
    pub ms: f64,
}

impl From<&sys::kiln_embed_stats> for EmbedStats {
    fn from(s: &sys::kiln_embed_stats) -> Self {
        EmbedStats {
            tokens: s.tokens,
            media_tokens: s.media_tokens,
            dimensions: s.dimensions,
            prefixed: s.prefixed,
            media_ms: s.media_ms,
            ms: s.ms,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Embedding {
    pub vector: Vec<f32>,
    pub stats: EmbedStats,
}

/// Where a media item goes inside the text: "A red shoe <__media__> seen from the side".
pub const MEDIA_MARKER: &str = "<__media__>";

/// One image or audio clip: a file, or the bytes of such a file. Images: jpg, png, bmp, gif.
/// Audio: wav, mp3, flac. The kind is read from the content.
#[derive(Debug, Clone, Copy)]
pub enum Media<'a> {
    Path(&'a Path),
    Bytes(&'a [u8]),
}

impl<'a> Media<'a> {
    pub fn path<P: AsRef<Path> + ?Sized>(p: &'a P) -> Media<'a> {
        Media::Path(p.as_ref())
    }
}

/// A `kiln_media` and the C string its path points into.
pub(crate) struct RawMedia {
    _path: Option<CString>,
    pub raw: sys::kiln_media,
}

impl RawMedia {
    pub fn new(m: &Media<'_>) -> Result<RawMedia> {
        Ok(match m {
            Media::Path(p) => {
                let c = path_cstring(p)?;
                let raw = sys::kiln_media {
                    path: c.as_ptr(),
                    data: std::ptr::null(),
                    size: 0,
                };
                RawMedia { _path: Some(c), raw }
            }
            Media::Bytes(b) => RawMedia {
                _path: None,
                raw: sys::kiln_media {
                    path: std::ptr::null(),
                    data: b.as_ptr(),
                    size: b.len(),
                },
            },
        })
    }
}

// ---------------------------------------------------------------- speech to text

#[derive(Debug, Clone, Default)]
pub struct TranscribeOptions {
    /// The language spoken, e.g. "English", when known. None = the model works it out.
    pub language: Option<String>,
    /// Qwen3-ASR: replaces the instruction. Whisper: text the words may follow on from (names,
    /// spellings). Parakeet: ignored.
    pub prompt: Option<String>,
    /// <= 0 = until the model stops or the context is full.
    pub max_tokens: i32,
}

#[derive(Debug, Clone, Default)]
pub struct TranscribeStats {
    pub audio_seconds: f64,
    pub audio_tokens: i32,
    pub prompt_tokens: i32,
    pub generated_tokens: i32,
    pub encode_ms: f64,
    pub prompt_ms: f64,
    pub gen_ms: f64,
    /// The whole call. Faster than real time when `ms < audio_seconds * 1000`.
    pub ms: f64,
    /// The language as the model named it, e.g. "English"; empty if it did not say.
    pub language: String,
    pub stopped_by_eos: bool,
    pub stopped_by_user: bool,
}

impl From<&sys::kiln_transcribe_stats> for TranscribeStats {
    fn from(s: &sys::kiln_transcribe_stats) -> Self {
        TranscribeStats {
            audio_seconds: s.audio_seconds,
            audio_tokens: s.audio_tokens,
            prompt_tokens: s.prompt_tokens,
            generated_tokens: s.generated_tokens,
            encode_ms: s.encode_ms,
            prompt_ms: s.prompt_ms,
            gen_ms: s.gen_ms,
            ms: s.ms,
            language: fixed_str(&s.language),
            stopped_by_eos: s.stopped_by_eos,
            stopped_by_user: s.stopped_by_user,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Transcript {
    pub text: String,
    pub stats: TranscribeStats,
}

// ---------------------------------------------------------------- text to speech

#[derive(Debug, Clone)]
pub struct SpeakOptions<'a> {
    /// A few seconds of someone speaking; the output imitates that voice. Pocket TTS produces
    /// almost nothing without one.
    pub voice: Option<Media<'a>>,
    /// Qwen3-TTS: "en", "zh", "de", ... (None = "en"). Pocket TTS: ignored.
    pub language: Option<String>,
    /// Longest output in frames (80 ms each for Pocket TTS); <= 0 = 512.
    pub max_frames: i32,
    /// <= 0 = 40.
    pub top_k: i32,
    /// <= 0 = 0.95.
    pub top_p: f32,
    /// None = random; a value makes the same text give the same audio.
    pub seed: Option<u32>,
}

impl Default for SpeakOptions<'_> {
    fn default() -> Self {
        let o = unsafe { sys::kiln_speak_options_default() };
        SpeakOptions {
            voice: None,
            language: None,
            max_frames: o.max_frames,
            top_k: o.top_k,
            top_p: o.top_p,
            seed: if o.seed == u32::MAX { None } else { Some(o.seed) },
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SpeakStats {
    pub audio_seconds: f64,
    pub sample_rate: i32,
    pub frames: i32,
    /// Text and voice reference together.
    pub prompt_tokens: i32,
    pub prompt_ms: f64,
    pub gen_ms: f64,
    pub ms: f64,
    pub stopped_by_eos: bool,
    pub stopped_by_user: bool,
}

impl From<&sys::kiln_speak_stats> for SpeakStats {
    fn from(s: &sys::kiln_speak_stats) -> Self {
        SpeakStats {
            audio_seconds: s.audio_seconds,
            sample_rate: s.sample_rate,
            frames: s.frames,
            prompt_tokens: s.prompt_tokens,
            prompt_ms: s.prompt_ms,
            gen_ms: s.gen_ms,
            ms: s.ms,
            stopped_by_eos: s.stopped_by_eos,
            stopped_by_user: s.stopped_by_user,
        }
    }
}

/// Generated audio: mono floats in -1..1.
#[derive(Debug, Clone, Default)]
pub struct Speech {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub stats: SpeakStats,
}

// ---------------------------------------------------------------- logging

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
}

impl LogLevel {
    pub(crate) fn to_raw(self) -> sys::kiln_log_level {
        match self {
            LogLevel::Error => sys::kiln_log_level_KILN_LOG_ERROR,
            LogLevel::Warn => sys::kiln_log_level_KILN_LOG_WARN,
            LogLevel::Info => sys::kiln_log_level_KILN_LOG_INFO,
            LogLevel::Debug => sys::kiln_log_level_KILN_LOG_DEBUG,
        }
    }
    pub(crate) fn from_raw(l: sys::kiln_log_level) -> LogLevel {
        match l {
            sys::kiln_log_level_KILN_LOG_ERROR => LogLevel::Error,
            sys::kiln_log_level_KILN_LOG_WARN => LogLevel::Warn,
            sys::kiln_log_level_KILN_LOG_INFO => LogLevel::Info,
            _ => LogLevel::Debug,
        }
    }
}
