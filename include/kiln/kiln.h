/*
 * kiln - on-device LLM runtime
 *
 * Stable C ABI. Everything crossing this boundary is plain C:
 * opaque handles, integers, NUL-terminated UTF-8 strings, callbacks.
 * No exceptions, no C++ types. Every fallible call returns kiln_status.
 *
 * Bindings (Dart FFI, JNI) are generated from this header only.
 */
#ifndef KILN_H
#define KILN_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#  define KILN_API __declspec(dllexport)
#else
#  define KILN_API __attribute__((visibility("default")))
#endif

#define KILN_VERSION_MAJOR 0
#define KILN_VERSION_MINOR 3
#define KILN_VERSION_PATCH 0

/* ------------------------------------------------------------------ */
/* Status                                                              */
/* ------------------------------------------------------------------ */

typedef enum kiln_status {
    KILN_OK = 0,
    KILN_ERR_INVALID_ARG      = 1,
    KILN_ERR_NO_BACKEND       = 2,   /* no backend can run this model on this device */
    KILN_ERR_MODEL_LOAD       = 3,   /* file missing, corrupt, unsupported arch */
    KILN_ERR_OUT_OF_MEMORY    = 4,
    KILN_ERR_CONTEXT_FULL     = 5,
    KILN_ERR_CANCELLED        = 6,
    KILN_ERR_BACKEND          = 7,   /* backend-specific failure; see kiln_last_error */
    KILN_ERR_NOT_SUPPORTED    = 8,
    KILN_ERR_INTERNAL         = 99
} kiln_status;

/* Human-readable detail for the last error on the calling thread. */
KILN_API const char* kiln_last_error(void);
KILN_API const char* kiln_status_string(kiln_status s);
KILN_API const char* kiln_version(void);

/* ------------------------------------------------------------------ */
/* Backends                                                            */
/* ------------------------------------------------------------------ */

typedef enum kiln_backend_kind {
    KILN_BACKEND_AUTO    = 0,   /* probe device, pick best available */
    KILN_BACKEND_CPU     = 1,   /* llama.cpp, CPU only. Guaranteed tier. */
    KILN_BACKEND_GPU     = 2,   /* llama.cpp, Metal / Vulkan / OpenCL */
    KILN_BACKEND_NPU     = 3    /* vendor NPU (QNN, etc). Detected, never assumed. */
} kiln_backend_kind;

/* ------------------------------------------------------------------ */
/* Device probe                                                        */
/* ------------------------------------------------------------------ */

typedef struct kiln_device_info {
    char     soc[64];             /* e.g. "Qualcomm SM8650", "Apple M4 Pro" */
    char     abi[16];             /* e.g. "arm64-v8a" */
    int32_t  cpu_cores;
    int32_t  cpu_big_cores;       /* cores kiln treats as fast: all but the little cores, = cpu_cores if unknown */
    uint64_t cpu_fast_mask;       /* bitmask of those cores (bit i = cpu i); 0 = no affinity info */
    uint64_t ram_total_bytes;
    uint64_t ram_available_bytes;
    bool     has_fp16;
    bool     has_dotprod;         /* ARMv8.2 dot product (big win for Q4/Q8) */
    bool     has_i8mm;            /* ARMv8.6 int8 matmul */
    bool     has_sve;
    bool     has_gpu;             /* a usable GPU backend compiled in and present */
    char     gpu_name[64];
    bool     has_npu;
    char     npu_name[64];
} kiln_device_info;

KILN_API kiln_status kiln_probe_device(kiln_device_info* out);

/* ------------------------------------------------------------------ */
/* Model                                                               */
/* ------------------------------------------------------------------ */

typedef struct kiln_model kiln_model;

typedef struct kiln_model_params {
    kiln_backend_kind backend;    /* KILN_BACKEND_AUTO recommended */
    int32_t  n_gpu_layers;        /* -1 = all if GPU chosen, 0 = none */
    int32_t  n_threads;           /* 0 = auto from probe */
    bool     use_mmap;
    bool     use_mlock;
} kiln_model_params;

KILN_API kiln_model_params kiln_model_params_default(void);

/* path: UTF-8 path to a GGUF file. */
KILN_API kiln_status kiln_model_load(const char* path,
                                     const kiln_model_params* params,
                                     kiln_model** out);
KILN_API void        kiln_model_free(kiln_model* m);

/* What actually happened. This is the truth-telling API. */
typedef struct kiln_model_info {
    kiln_backend_kind backend_used;
    char     backend_name[64];    /* e.g. "llama.cpp/metal", "llama.cpp/cpu" */
    char     arch[32];            /* e.g. "llama", "qwen2", "gemma3" */
    char     quant[16];           /* e.g. "Q4_K_M" */
    int64_t  n_params;
    int32_t  n_ctx_train;
    int32_t  n_vocab;
    int32_t  n_gpu_layers_used;
    uint64_t model_bytes;         /* resident weight memory */
    bool     supports_thinking;   /* chat template has a reasoning switch (Qwen3.x, Gemma 4, ...) */
    bool     is_decision_model;   /* answers typed questions with kiln_decide; cannot generate text */
    bool     is_embedding_model;  /* turns text into a vector with kiln_embed; cannot generate text */
    int32_t  n_embd;              /* embedding models: length of the vector kiln_embed returns; else 0 */
} kiln_model_info;

KILN_API kiln_status kiln_model_get_info(const kiln_model* m, kiln_model_info* out);

/* ------------------------------------------------------------------ */
/* Context (one conversation / KV cache)                               */
/* ------------------------------------------------------------------ */

typedef struct kiln_context kiln_context;

typedef struct kiln_context_params {
    int32_t  n_ctx;               /* 0 = model default, capped by RAM */
    int32_t  n_batch;
    bool     flash_attn;
} kiln_context_params;

KILN_API kiln_context_params kiln_context_params_default(void);

KILN_API kiln_status kiln_context_create(kiln_model* m,
                                         const kiln_context_params* params,
                                         kiln_context** out);
KILN_API void        kiln_context_free(kiln_context* c);
KILN_API kiln_status kiln_context_reset(kiln_context* c);   /* clear KV cache */

/* ------------------------------------------------------------------ */
/* Tokenizer and chat template                                         */
/* ------------------------------------------------------------------ */

/* Returns token count written, or negative = required size if out is too small. */
KILN_API int32_t kiln_tokenize(const kiln_model* m, const char* text,
                               bool add_special, int32_t* out, int32_t out_cap);

/* Returns bytes written (excluding NUL), or negative = required size. */
KILN_API int32_t kiln_detokenize(const kiln_model* m, const int32_t* tokens,
                                 int32_t n_tokens, char* out, int32_t out_cap);

typedef struct kiln_chat_message {
    const char* role;             /* "system" | "user" | "assistant" */
    const char* content;
} kiln_chat_message;

typedef struct kiln_chat_options {
    bool add_assistant_prefix;    /* append the "assistant:" turn opener (true for generation) */
    bool enable_thinking;         /* reasoning models only. false = answer directly. Ignored if unsupported. */
    int32_t max_reasoning_tokens; /* thinking budget, kiln_chat only: after this many reasoning tokens kiln closes
                                     the think block itself and the model goes on to answer. <= 0 = unlimited.
                                     Small models often never stop thinking on their own. */
} kiln_chat_options;

KILN_API kiln_chat_options kiln_chat_options_default(void);

/* Renders the model's own chat template (full jinja via llama.cpp/minja).
 * Prefer kiln_chat() for generation: it also knows whether the prompt opened a
 * reasoning block, which this function cannot tell you.
 * Returns bytes written (excluding NUL), or negative = required size. */
KILN_API int32_t kiln_apply_chat_template(const kiln_model* m,
                                          const kiln_chat_message* msgs,
                                          int32_t n_msgs,
                                          const kiln_chat_options* opts,
                                          char* out, int32_t out_cap);

/* ------------------------------------------------------------------ */
/* Generation                                                          */
/* ------------------------------------------------------------------ */

typedef struct kiln_sampling_params {
    float    temperature;         /* <= 0 = greedy */
    float    top_p;
    int32_t  top_k;
    float    min_p;
    float    repeat_penalty;
    int32_t  repeat_last_n;
    uint32_t seed;                /* 0 = random */
    int32_t  max_tokens;          /* <= 0 = until EOS or context full */
} kiln_sampling_params;

KILN_API kiln_sampling_params kiln_sampling_params_default(void);

typedef enum kiln_channel {
    KILN_CHANNEL_ANSWER    = 0,
    KILN_CHANNEL_REASONING = 1   /* text inside the model's think block; tags themselves are stripped */
} kiln_channel;

/* Called once per generated token piece. Return false to stop generation.
 * Pieces are UTF-8 but a multi-byte character may be split across calls. */
typedef bool (*kiln_token_cb)(kiln_channel channel, const char* piece, int32_t token_id, void* user);

typedef struct kiln_gen_stats {
    int32_t  prompt_tokens;       /* total tokens in the rendered prompt */
    int32_t  cached_tokens;       /* of those, how many were already in the KV cache (multi-turn) */
    int32_t  generated_tokens;
    int32_t  reasoning_tokens;    /* subset of generated_tokens that went to the reasoning channel */
    double   prompt_ms;
    double   gen_ms;
    double   prompt_tps;
    double   gen_tps;
    bool     stopped_by_eos;
    bool     stopped_by_user;
    bool     reasoning_truncated; /* max_reasoning_tokens was reached and kiln closed the think block */
} kiln_gen_stats;

/* High-level, recommended. Renders msgs with the model's template, reuses whatever
 * prefix of the conversation is already in the KV cache, streams pieces to cb.
 * Pass the full history every turn; kiln figures out what changed.
 * Blocking. stats may be NULL. Safe to call kiln_cancel from another thread. */
KILN_API kiln_status kiln_chat(kiln_context* c,
                               const kiln_chat_message* msgs, int32_t n_msgs,
                               const kiln_chat_options* chat_opts,
                               const kiln_sampling_params* params,
                               kiln_token_cb cb, void* user,
                               kiln_gen_stats* stats);

/* Low-level. Feeds a raw, already-templated prompt. Same prefix reuse and streaming. */
KILN_API kiln_status kiln_generate(kiln_context* c,
                                   const char* prompt,
                                   const kiln_sampling_params* params,
                                   kiln_token_cb cb, void* user,
                                   kiln_gen_stats* stats);

/* Thread-safe. Requests the in-flight kiln_generate on this context to stop. */
KILN_API void kiln_cancel(kiln_context* c);

/* ------------------------------------------------------------------ */
/* Decision models                                                     */
/* ------------------------------------------------------------------ */

/* A decision model (kiln_model_info.is_decision_model) does not write text. It reads a
 * state and one typed question and returns a probability per option, in one forward pass.
 * Supported today: LFM2 d1-omni, text state only (no images or audio). */

typedef enum kiln_question_type {
    KILN_QUESTION_NOUL   = 0,   /* yes / no */
    KILN_QUESTION_CHOICE = 1,   /* pick one of the options */
    KILN_QUESTION_SCORE  = 2    /* rate on a scale; options are the levels, lowest first (2 to 10) */
} kiln_question_type;

typedef struct kiln_decision_option {
    const char* key;              /* choice: the label returned. noul: "true" or "false" (also "yes" / "no"). score: ignored */
    const char* description;      /* what the option means; may be NULL */
} kiln_decision_option;

typedef struct kiln_decision_question {
    const char* id;               /* short name of the question, e.g. "wants_refund" */
    kiln_question_type type;
    const char* instructions;     /* the question itself */
    const kiln_decision_option* options;   /* noul: optional, only to describe what yes and no mean */
    int32_t     n_options;
} kiln_decision_question;

typedef struct kiln_decision_answer {
    int32_t n_probs;              /* probabilities written to probs. noul: 2, [0] = no, [1] = yes */
    int32_t best;                 /* index of the most probable entry */
    float   value;                /* noul: P(yes). score: expected level, 0 .. n-1. choice: P(best) */
    float   confidence;           /* 0 .. 1 for choice and score; for noul how far P(yes) is from 0.5 */
    int32_t prompt_tokens;
    double  ms;
} kiln_decision_answer;

/* state: the text the question is about; may be NULL. probs receives one probability per
 * option (2 for noul) and must hold at least that many. Blocking.
 * Returns KILN_ERR_NOT_SUPPORTED on a model that is not a decision model. */
KILN_API kiln_status kiln_decide(kiln_context* c, const char* state,
                                 const kiln_decision_question* question,
                                 float* probs, int32_t probs_cap,
                                 kiln_decision_answer* answer);

/* ------------------------------------------------------------------ */
/* Embedding models                                                    */
/* ------------------------------------------------------------------ */

/* An embedding model (kiln_model_info.is_embedding_model) does not write text. It turns a
 * text into a list of numbers; texts with a similar meaning get similar lists. With
 * normalised vectors the similarity of two texts is their dot product (1 = same meaning).
 * Tested with EmbeddingGemma 2 and nomic-embed-text v1.5.
 * Some embedding models also take images and audio and put them in the same space as text,
 * so a text query can find a photo. Those need a second file, the projector (an "mmproj"
 * GGUF): see kiln_model_load_projector and kiln_embed_media. */

typedef enum kiln_embed_kind {
    KILN_EMBED_RAW      = 0,   /* embed the text exactly as given */
    KILN_EMBED_QUERY    = 1,   /* a search query: kiln adds the model's query prefix */
    KILN_EMBED_DOCUMENT = 2    /* a text to be searched: kiln adds the model's document prefix */
} kiln_embed_kind;

typedef struct kiln_embed_options {
    kiln_embed_kind kind;         /* use QUERY and DOCUMENT for search; a model with no known prefix treats all as RAW */
    const char* title;            /* DOCUMENT only, optional: the document's title, for models that take one */
    int32_t     dimensions;       /* keep only the first N numbers (smaller to store). <= 0 = the full vector.
                                     Only models trained for it stay accurate (EmbeddingGemma: 512, 256, 128) */
    bool        normalize;        /* scale the vector to length 1 */
} kiln_embed_options;

KILN_API kiln_embed_options kiln_embed_options_default(void);   /* RAW, full vector, normalised */

typedef struct kiln_embed_stats {
    int32_t tokens;               /* tokens the model read: text, prefix and media together */
    int32_t media_tokens;         /* of those, how many came from images and audio */
    int32_t dimensions;           /* numbers written to out */
    bool    prefixed;             /* a task prefix was added */
    double  media_ms;             /* time inside the image / audio encoders only */
    double  ms;                   /* total, media_ms included */
} kiln_embed_stats;

/* out must hold opts->dimensions floats, or kiln_model_info.n_embd for the full vector.
 * Each call is independent; the context keeps nothing between texts. Blocking.
 * A text longer than the context returns KILN_ERR_CONTEXT_FULL (nothing is cut silently):
 * split long documents into pieces. For an embedding model kiln_context_params.n_ctx is the
 * longest text in tokens (0 = 2048). The context is never larger than the model's training
 * context (kiln_model_info.n_ctx_train), whatever is asked for. opts and stats may be NULL.
 * Returns KILN_ERR_NOT_SUPPORTED on a model that is not an embedding model. */
KILN_API kiln_status kiln_embed(kiln_context* c, const char* text,
                                const kiln_embed_options* opts,
                                float* out, int32_t out_cap,
                                kiln_embed_stats* stats);

/* ---- images and audio ---- */

typedef struct kiln_projector_params {
    int32_t image_max_tokens;     /* most tokens one image may take. A large photo is scaled down to fit.
                                     Time grows much faster than this number. 0 = 280, < 0 = the model's own limit */
} kiln_projector_params;

KILN_API kiln_projector_params kiln_projector_params_default(void);

typedef struct kiln_projector_info {
    bool supports_images;
    bool supports_audio;
} kiln_projector_info;

/* Loads the image / audio encoders that belong to an embedding model (its mmproj GGUF).
 * Call once after kiln_model_load, before embedding media; freed with the model. The
 * encoders stay in memory, often more than the text model itself. params and out may be NULL.
 * Returns KILN_ERR_NOT_SUPPORTED on a model that is not an embedding model. */
KILN_API kiln_status kiln_model_load_projector(kiln_model* m, const char* path,
                                               const kiln_projector_params* params,
                                               kiln_projector_info* out);

/* One image or one audio clip: either a file path, or the bytes of such a file.
 * Images: jpg, png, bmp, gif. Audio: wav, mp3, flac. The kind is read from the content. */
typedef struct kiln_media {
    const char*    path;          /* file to read, or NULL to use data */
    const uint8_t* data;          /* contents of an image / audio file, when path is NULL */
    size_t         size;          /* bytes in data */
} kiln_media;

/* Where a media item goes inside the text. "A red shoe <__media__> seen from the side". */
#define KILN_MEDIA_MARKER "<__media__>"

/* Like kiln_embed, for a text with images and / or audio, or for media alone (text NULL).
 * The result is one vector in the same space as kiln_embed's, so the two can be compared.
 * Each KILN_MEDIA_MARKER in text is replaced by the next media item; with no marker in the
 * text the items follow it in order. No task prefix is added when media are present.
 * An image costs a few hundred tokens of the context, audio a few dozen per second.
 * Needs kiln_model_load_projector first, else KILN_ERR_NOT_SUPPORTED. */
KILN_API kiln_status kiln_embed_media(kiln_context* c, const char* text,
                                      const kiln_media* media, int32_t n_media,
                                      const kiln_embed_options* opts,
                                      float* out, int32_t out_cap,
                                      kiln_embed_stats* stats);

/* ------------------------------------------------------------------ */
/* Logging                                                             */
/* ------------------------------------------------------------------ */

typedef enum kiln_log_level {
    KILN_LOG_ERROR = 0, KILN_LOG_WARN = 1, KILN_LOG_INFO = 2, KILN_LOG_DEBUG = 3
} kiln_log_level;

typedef void (*kiln_log_cb)(kiln_log_level level, const char* msg, void* user);
KILN_API void kiln_set_log_callback(kiln_log_cb cb, void* user);
KILN_API void kiln_set_log_level(kiln_log_level level);

#ifdef __cplusplus
}
#endif
#endif /* KILN_H */
