// JNI glue between the Kotlin API (dev.kiln.Native) and kiln.h.
//
// Kept deliberately flat: Kotlin passes numbers, arrays and strings, never its own objects, so
// no field of a Kotlin class is looked up from here and code shrinkers cannot break it. Strings
// travel as UTF-8 byte arrays, not jstring: JNI's string functions use "modified UTF-8", which
// writes emoji as two 3-byte halves and would hand kiln text it cannot tokenize properly.
// Results come back in arrays Kotlin allocates (stats as double[]), errors as KilnException.
#include <jni.h>

#include "kiln/kiln.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

namespace {

// ---- strings and arrays ----

// A Kotlin ByteArray (UTF-8) as a NUL-terminated C string; a null array gives nullptr.
struct Utf8 {
    std::string s;
    bool null = true;
    Utf8(JNIEnv* env, jbyteArray a) {
        if (!a) return;
        null = false;
        const jsize n = env->GetArrayLength(a);
        s.resize((size_t)n);
        if (n) env->GetByteArrayRegion(a, 0, n, reinterpret_cast<jbyte*>(&s[0]));
    }
    const char* c() const { return null ? nullptr : s.c_str(); }
};

jbyteArray to_bytes(JNIEnv* env, const char* p, size_t n) {
    jbyteArray a = env->NewByteArray((jsize)n);
    if (a && n) env->SetByteArrayRegion(a, 0, (jsize)n, reinterpret_cast<const jbyte*>(p));
    return a;
}

jbyteArray to_bytes(JNIEnv* env, const char* s) { return to_bytes(env, s, strlen(s)); }

// An array of UTF-8 byte arrays, each element may be null.
std::vector<Utf8> utf8_list(JNIEnv* env, jobjectArray a) {
    std::vector<Utf8> out;
    if (!a) return out;
    const jsize n = env->GetArrayLength(a);
    out.reserve((size_t)n);
    for (jsize i = 0; i < n; ++i) {
        auto e = static_cast<jbyteArray>(env->GetObjectArrayElement(a, i));
        out.emplace_back(env, e);
        if (e) env->DeleteLocalRef(e);
    }
    return out;
}

void put(JNIEnv* env, jdoubleArray out, std::initializer_list<double> v) {
    if (!out) return;
    const jsize n = std::min<jsize>(env->GetArrayLength(out), (jsize)v.size());
    env->SetDoubleArrayRegion(out, 0, n, v.begin());
}

void put(JNIEnv* env, jlongArray out, std::initializer_list<jlong> v) {
    if (!out) return;
    const jsize n = std::min<jsize>(env->GetArrayLength(out), (jsize)v.size());
    env->SetLongArrayRegion(out, 0, n, v.begin());
}

// ---- errors ----

// Throws dev.kiln.KilnException(status, kiln_last_error()) unless an exception from a Kotlin
// callback is already on its way. kiln keeps the message per thread; this runs on the thread
// that made the call, right after it.
bool fail(JNIEnv* env, kiln_status s) {
    if (s == KILN_OK) return false;
    if (env->ExceptionCheck()) return true;
    jclass cls = env->FindClass("dev/kiln/KilnException");
    if (!cls) return true;
    jmethodID ctor = env->GetMethodID(cls, "<init>", "(I[B)V");
    jbyteArray msg = to_bytes(env, kiln_last_error());
    auto ex = static_cast<jthrowable>(env->NewObject(cls, ctor, (jint)s, msg));
    if (ex) env->Throw(ex);
    return true;
}

void throw_arg(JNIEnv* env, const char* what) {
    jclass cls = env->FindClass("java/lang/IllegalArgumentException");
    if (cls) env->ThrowNew(cls, what);
}

// ---- callbacks ----

// kiln calls back on the thread that made the call, so the JNIEnv stays valid. Each piece is a
// new local reference, deleted at once: Android allows only a few hundred at a time.
struct TokenSink {
    JNIEnv*   env;
    jobject   sink;
    jmethodID mid;
    bool      failed = false;   // the Kotlin callback threw; let kiln wind down, rethrow after
};

bool token_cb(kiln_channel ch, const char* piece, int32_t token, void* user) {
    auto* t = static_cast<TokenSink*>(user);
    if (t->failed || !t->sink) return !t->failed;
    jbyteArray b = to_bytes(t->env, piece ? piece : "");
    const jboolean go_on = t->env->CallBooleanMethod(t->sink, t->mid, (jint)ch, b, (jint)token);
    t->env->DeleteLocalRef(b);
    if (t->env->ExceptionCheck()) { t->failed = true; return false; }
    return go_on == JNI_TRUE;
}

TokenSink token_sink(JNIEnv* env, jobject sink) {
    TokenSink t{env, sink, nullptr};
    if (sink) t.mid = env->GetMethodID(env->GetObjectClass(sink), "onPiece", "(I[BI)Z");
    return t;
}

struct AudioSink {
    JNIEnv*   env;
    jobject   sink;
    jmethodID mid;
    bool      failed = false;
};

bool audio_cb(const float* samples, int32_t n, int32_t rate, void* user) {
    auto* a = static_cast<AudioSink*>(user);
    if (a->failed) return false;
    jfloatArray f = a->env->NewFloatArray(n);
    if (f && n > 0) a->env->SetFloatArrayRegion(f, 0, n, samples);
    const jboolean go_on = a->env->CallBooleanMethod(a->sink, a->mid, f, (jint)rate);
    a->env->DeleteLocalRef(f);
    if (a->env->ExceptionCheck()) { a->failed = true; return false; }
    return go_on == JNI_TRUE;
}

// ---- shared argument decoding ----

kiln_sampling_params sampling(JNIEnv* env, jfloatArray f, jlongArray i) {
    kiln_sampling_params p = kiln_sampling_params_default();
    jfloat fv[4];
    jlong  iv[4];
    env->GetFloatArrayRegion(f, 0, 4, fv);
    env->GetLongArrayRegion(i, 0, 4, iv);
    p.temperature = fv[0]; p.top_p = fv[1]; p.min_p = fv[2]; p.repeat_penalty = fv[3];
    p.top_k = (int32_t)iv[0]; p.repeat_last_n = (int32_t)iv[1]; p.seed = (uint32_t)iv[2]; p.max_tokens = (int32_t)iv[3];
    return p;
}

void put_gen_stats(JNIEnv* env, jdoubleArray out, const kiln_gen_stats& s) {
    put(env, out, {(double)s.prompt_tokens, (double)s.cached_tokens, (double)s.generated_tokens, (double)s.reasoning_tokens,
                   s.prompt_ms, s.gen_ms, s.prompt_tps, s.gen_tps, (double)s.stopped_by_eos, (double)s.stopped_by_user,
                   (double)s.reasoning_truncated});
}

void put_embed_stats(JNIEnv* env, jdoubleArray out, const kiln_embed_stats& s) {
    put(env, out, {(double)s.tokens, (double)s.media_tokens, (double)s.dimensions, (double)s.prefixed, s.media_ms, s.ms});
}

// Messages: roles 0 = system, 1 = user, 2 = assistant.
struct Messages {
    std::vector<Utf8> content;
    std::vector<kiln_chat_message> raw;
    Messages(JNIEnv* env, jintArray roles, jobjectArray contents) : content(utf8_list(env, contents)) {
        static const char* kRoles[] = {"system", "user", "assistant"};
        const jsize n = env->GetArrayLength(roles);
        std::vector<jint> r((size_t)n);
        if (n) env->GetIntArrayRegion(roles, 0, n, r.data());
        for (jsize i = 0; i < n && (size_t)i < content.size(); ++i)
            raw.push_back({kRoles[r[(size_t)i] >= 0 && r[(size_t)i] < 3 ? r[(size_t)i] : 1], content[(size_t)i].c()});
    }
};

kiln_chat_options chat_options(jboolean prefix, jboolean thinking, jint max_reasoning) {
    kiln_chat_options o = kiln_chat_options_default();
    o.add_assistant_prefix = prefix; o.enable_thinking = thinking; o.max_reasoning_tokens = max_reasoning;
    return o;
}

// A media item: a path, or the bytes of a file. Owns the bytes it points to.
struct MediaItem {
    Utf8 path;
    std::vector<uint8_t> data;
    kiln_media raw{};
    MediaItem(JNIEnv* env, jbyteArray p, jbyteArray d) : path(env, p) {
        if (d) {
            const jsize n = env->GetArrayLength(d);
            data.resize((size_t)n);
            if (n) env->GetByteArrayRegion(d, 0, n, reinterpret_cast<jbyte*>(data.data()));
        }
    }
    const kiln_media* get() {
        raw.path = path.c();
        raw.data = raw.path ? nullptr : data.data();
        raw.size = raw.path ? 0 : data.size();
        return &raw;
    }
};

kiln_embed_options embed_options(jint kind, const Utf8& title, jint dims, jboolean normalize) {
    kiln_embed_options o = kiln_embed_options_default();
    o.kind = (kiln_embed_kind)kind; o.title = title.c(); o.dimensions = dims; o.normalize = normalize;
    return o;
}

template <class T> T* handle(jlong h) { return reinterpret_cast<T*>(static_cast<intptr_t>(h)); }

} // namespace

#define JNI(ret, name) extern "C" JNIEXPORT ret JNICALL Java_dev_kiln_Native_##name

// ------------------------------------------------------------------ library

JNI(jbyteArray, version)(JNIEnv* env, jclass) { return to_bytes(env, kiln_version()); }

// kiln's defaults, for the test that keeps the Kotlin data classes in step: sampling (temperature,
// top_p, top_k, min_p, repeat_penalty, repeat_last_n, seed, max_tokens), context (n_ctx, n_batch,
// flash_attn), model (n_gpu_layers, n_threads, use_mmap, use_mlock), chat (prefix, thinking, budget).
JNI(jdoubleArray, defaults)(JNIEnv* env, jclass) {
    const kiln_sampling_params s = kiln_sampling_params_default();
    const kiln_context_params c = kiln_context_params_default();
    const kiln_model_params m = kiln_model_params_default();
    const kiln_chat_options o = kiln_chat_options_default();
    const double v[] = {s.temperature, s.top_p, (double)s.top_k, s.min_p, s.repeat_penalty, (double)s.repeat_last_n, (double)s.seed,
                        (double)s.max_tokens, (double)c.n_ctx, (double)c.n_batch, (double)c.flash_attn, (double)m.n_gpu_layers,
                        (double)m.n_threads, (double)m.use_mmap, (double)m.use_mlock, (double)o.add_assistant_prefix,
                        (double)o.enable_thinking, (double)o.max_reasoning_tokens};
    const jsize n = (jsize)(sizeof v / sizeof v[0]);
    jdoubleArray out = env->NewDoubleArray(n);
    env->SetDoubleArrayRegion(out, 0, n, v);
    return out;
}

JNI(void, setLogLevel)(JNIEnv*, jclass, jint level) { kiln_set_log_level((kiln_log_level)level); }

// nums: cores, big cores, fast mask, ram total, ram available, fp16, dotprod, i8mm, sve, gpu, npu.
// Returns soc, abi, gpu name, npu name.
JNI(jobjectArray, probe)(JNIEnv* env, jclass, jlongArray nums) {
    kiln_device_info d{};
    if (fail(env, kiln_probe_device(&d))) return nullptr;
    put(env, nums, {d.cpu_cores, d.cpu_big_cores, (jlong)d.cpu_fast_mask, (jlong)d.ram_total_bytes, (jlong)d.ram_available_bytes,
                    d.has_fp16, d.has_dotprod, d.has_i8mm, d.has_sve, d.has_gpu, d.has_npu});
    jobjectArray out = env->NewObjectArray(4, env->FindClass("[B"), nullptr);
    const char* s[] = {d.soc, d.abi, d.gpu_name, d.npu_name};
    for (int i = 0; i < 4; ++i) {
        jbyteArray b = to_bytes(env, s[i]);
        env->SetObjectArrayElement(out, i, b);
        env->DeleteLocalRef(b);
    }
    return out;
}

// ------------------------------------------------------------------ model

JNI(jlong, modelLoad)(JNIEnv* env, jclass, jbyteArray path, jint backend, jint n_gpu_layers, jint n_threads, jboolean mmap, jboolean mlock) {
    Utf8 p(env, path);
    kiln_model_params mp = kiln_model_params_default();
    mp.backend = (kiln_backend_kind)backend; mp.n_gpu_layers = n_gpu_layers; mp.n_threads = n_threads;
    mp.use_mmap = mmap; mp.use_mlock = mlock;
    kiln_model* m = nullptr;
    if (fail(env, kiln_model_load(p.c(), &mp, &m))) return 0;
    return static_cast<jlong>(reinterpret_cast<intptr_t>(m));
}

JNI(void, modelFree)(JNIEnv*, jclass, jlong m) { kiln_model_free(handle<kiln_model>(m)); }

// nums: backend used, n_params, n_ctx_train, n_vocab, gpu layers used, model bytes, thinking,
// decision, embedding, n_embd, speech. Returns backend name, arch, quant.
JNI(jobjectArray, modelInfo)(JNIEnv* env, jclass, jlong m, jlongArray nums) {
    kiln_model_info i{};
    if (fail(env, kiln_model_get_info(handle<kiln_model>(m), &i))) return nullptr;
    put(env, nums, {i.backend_used, i.n_params, i.n_ctx_train, i.n_vocab, i.n_gpu_layers_used, (jlong)i.model_bytes,
                    i.supports_thinking, i.is_decision_model, i.is_embedding_model, i.n_embd, i.is_speech_model});
    jobjectArray out = env->NewObjectArray(3, env->FindClass("[B"), nullptr);
    const char* s[] = {i.backend_name, i.arch, i.quant};
    for (int k = 0; k < 3; ++k) {
        jbyteArray b = to_bytes(env, s[k]);
        env->SetObjectArrayElement(out, k, b);
        env->DeleteLocalRef(b);
    }
    return out;
}

// Returns images, audio, generates audio, sample rate.
JNI(jintArray, modelLoadProjector)(JNIEnv* env, jclass, jlong m, jbyteArray path, jint image_max_tokens) {
    Utf8 p(env, path);
    kiln_projector_params pp = kiln_projector_params_default();
    pp.image_max_tokens = image_max_tokens;
    kiln_projector_info info{};
    if (fail(env, kiln_model_load_projector(handle<kiln_model>(m), p.c(), &pp, &info))) return nullptr;
    const jint v[] = {info.supports_images, info.supports_audio, info.generates_audio, info.sample_rate};
    jintArray out = env->NewIntArray(4);
    env->SetIntArrayRegion(out, 0, 4, v);
    return out;
}

JNI(jintArray, tokenize)(JNIEnv* env, jclass, jlong m, jbyteArray text, jboolean add_special) {
    Utf8 t(env, text);
    std::vector<int32_t> toks(t.s.size() + 16);
    int32_t n = kiln_tokenize(handle<kiln_model>(m), t.c(), add_special, toks.data(), (int32_t)toks.size());
    if (n < 0) {
        toks.resize((size_t)-n);
        n = kiln_tokenize(handle<kiln_model>(m), t.c(), add_special, toks.data(), (int32_t)toks.size());
    }
    n = std::max(n, 0);
    jintArray out = env->NewIntArray(n);
    if (n) env->SetIntArrayRegion(out, 0, n, reinterpret_cast<const jint*>(toks.data()));
    return out;
}

JNI(jbyteArray, detokenize)(JNIEnv* env, jclass, jlong m, jintArray tokens) {
    const jsize n = env->GetArrayLength(tokens);
    std::vector<int32_t> toks((size_t)n);
    if (n) env->GetIntArrayRegion(tokens, 0, n, reinterpret_cast<jint*>(toks.data()));
    std::vector<char> buf((size_t)n * 8 + 16);
    int32_t w = kiln_detokenize(handle<kiln_model>(m), toks.data(), n, buf.data(), (int32_t)buf.size());
    if (w < 0) {
        buf.resize((size_t)-w + 1);
        w = kiln_detokenize(handle<kiln_model>(m), toks.data(), n, buf.data(), (int32_t)buf.size());
    }
    return to_bytes(env, buf.data(), (size_t)std::max(w, 0));
}

JNI(jbyteArray, applyChatTemplate)(JNIEnv* env, jclass, jlong m, jintArray roles, jobjectArray contents, jboolean prefix, jboolean thinking) {
    Messages msgs(env, roles, contents);
    const kiln_chat_options o = chat_options(prefix, thinking, 0);
    std::vector<char> buf(4096);
    int32_t w = kiln_apply_chat_template(handle<kiln_model>(m), msgs.raw.data(), (int32_t)msgs.raw.size(), &o, buf.data(), (int32_t)buf.size());
    if (w < 0) {
        buf.resize((size_t)-w + 1);
        w = kiln_apply_chat_template(handle<kiln_model>(m), msgs.raw.data(), (int32_t)msgs.raw.size(), &o, buf.data(), (int32_t)buf.size());
    }
    return to_bytes(env, buf.data(), (size_t)std::max(w, 0));
}

// ------------------------------------------------------------------ context

JNI(jlong, contextCreate)(JNIEnv* env, jclass, jlong m, jint n_ctx, jint n_batch, jboolean flash_attn) {
    kiln_context_params cp = kiln_context_params_default();
    cp.n_ctx = n_ctx; cp.n_batch = n_batch; cp.flash_attn = flash_attn;
    kiln_context* c = nullptr;
    if (fail(env, kiln_context_create(handle<kiln_model>(m), &cp, &c))) return 0;
    return static_cast<jlong>(reinterpret_cast<intptr_t>(c));
}

JNI(void, contextFree)(JNIEnv*, jclass, jlong c) { kiln_context_free(handle<kiln_context>(c)); }
JNI(void, contextReset)(JNIEnv* env, jclass, jlong c) { fail(env, kiln_context_reset(handle<kiln_context>(c))); }
JNI(void, cancel)(JNIEnv*, jclass, jlong c) { kiln_cancel(handle<kiln_context>(c)); }

JNI(void, chat)(JNIEnv* env, jclass, jlong c, jintArray roles, jobjectArray contents, jboolean prefix, jboolean thinking,
                jint max_reasoning, jfloatArray sf, jlongArray si, jobject sink, jdoubleArray stats) {
    Messages msgs(env, roles, contents);
    const kiln_chat_options o = chat_options(prefix, thinking, max_reasoning);
    const kiln_sampling_params p = sampling(env, sf, si);
    TokenSink t = token_sink(env, sink);
    kiln_gen_stats st{};
    const kiln_status s = kiln_chat(handle<kiln_context>(c), msgs.raw.data(), (int32_t)msgs.raw.size(), &o, &p, token_cb, &t, &st);
    if (t.failed) return;   // the Kotlin exception is pending: no more JNI calls but returning
    put_gen_stats(env, stats, st);
    fail(env, s);
}

JNI(void, generate)(JNIEnv* env, jclass, jlong c, jbyteArray prompt, jfloatArray sf, jlongArray si, jobject sink, jdoubleArray stats) {
    Utf8 pr(env, prompt);
    const kiln_sampling_params p = sampling(env, sf, si);
    TokenSink t = token_sink(env, sink);
    kiln_gen_stats st{};
    const kiln_status s = kiln_generate(handle<kiln_context>(c), pr.c(), &p, token_cb, &t, &st);
    if (t.failed) return;
    put_gen_stats(env, stats, st);
    fail(env, s);
}

// answer: n_probs, best, value, confidence, prompt tokens, ms.
JNI(void, decide)(JNIEnv* env, jclass, jlong c, jbyteArray state, jbyteArray id, jint type, jbyteArray instructions,
                  jobjectArray keys, jobjectArray descriptions, jfloatArray probs, jdoubleArray answer) {
    Utf8 st(env, state), qid(env, id), ins(env, instructions);
    std::vector<Utf8> k = utf8_list(env, keys), d = utf8_list(env, descriptions);
    std::vector<kiln_decision_option> opts;
    for (size_t i = 0; i < k.size(); ++i) opts.push_back({k[i].c(), i < d.size() ? d[i].c() : nullptr});
    const kiln_decision_question q = {qid.c(), (kiln_question_type)type, ins.c(), opts.empty() ? nullptr : opts.data(), (int32_t)opts.size()};
    const jsize cap = env->GetArrayLength(probs);
    std::vector<float> p((size_t)cap);
    kiln_decision_answer a{};
    if (fail(env, kiln_decide(handle<kiln_context>(c), st.c(), &q, p.data(), cap, &a))) return;
    env->SetFloatArrayRegion(probs, 0, std::min<jsize>(cap, std::max(a.n_probs, 0)), p.data());
    put(env, answer, {(double)a.n_probs, (double)a.best, a.value, a.confidence, (double)a.prompt_tokens, a.ms});
}

// Writes the vector to out; returns how many numbers were written.
JNI(jint, embed)(JNIEnv* env, jclass, jlong c, jbyteArray text, jint kind, jbyteArray title, jint dims, jboolean normalize,
                 jfloatArray out, jdoubleArray stats) {
    Utf8 t(env, text), ti(env, title);
    const kiln_embed_options o = embed_options(kind, ti, dims, normalize);
    const jsize cap = env->GetArrayLength(out);
    std::vector<float> v((size_t)cap);
    kiln_embed_stats st{};
    if (fail(env, kiln_embed(handle<kiln_context>(c), t.c(), &o, v.data(), cap, &st))) return 0;
    const jsize n = std::min<jsize>(cap, std::max(st.dimensions, 0));
    env->SetFloatArrayRegion(out, 0, n, v.data());
    put_embed_stats(env, stats, st);
    return n;
}

JNI(jint, embedMedia)(JNIEnv* env, jclass, jlong c, jbyteArray text, jobjectArray paths, jobjectArray datas, jint kind,
                      jbyteArray title, jint dims, jboolean normalize, jfloatArray out, jdoubleArray stats) {
    Utf8 t(env, text), ti(env, title);
    const jsize n_media = env->GetArrayLength(paths);
    if (env->GetArrayLength(datas) != n_media) { throw_arg(env, "paths and datas differ in length"); return 0; }
    std::vector<MediaItem> items;
    items.reserve((size_t)n_media);
    for (jsize i = 0; i < n_media; ++i) {
        auto p = static_cast<jbyteArray>(env->GetObjectArrayElement(paths, i));
        auto d = static_cast<jbyteArray>(env->GetObjectArrayElement(datas, i));
        items.emplace_back(env, p, d);
        if (p) env->DeleteLocalRef(p);
        if (d) env->DeleteLocalRef(d);
    }
    std::vector<kiln_media> raw;
    for (auto& m : items) raw.push_back(*m.get());
    const kiln_embed_options o = embed_options(kind, ti, dims, normalize);
    const jsize cap = env->GetArrayLength(out);
    std::vector<float> v((size_t)cap);
    kiln_embed_stats st{};
    if (fail(env, kiln_embed_media(handle<kiln_context>(c), t.c(), raw.data(), (int32_t)raw.size(), &o, v.data(), cap, &st))) return 0;
    const jsize n = std::min<jsize>(cap, std::max(st.dimensions, 0));
    env->SetFloatArrayRegion(out, 0, n, v.data());
    put_embed_stats(env, stats, st);
    return n;
}

// stats: audio seconds, audio tokens, prompt tokens, generated tokens, encode ms, prompt ms,
// gen ms, ms, eos, user. Returns the language the model named.
JNI(jbyteArray, transcribe)(JNIEnv* env, jclass, jlong c, jbyteArray path, jbyteArray data, jbyteArray language, jbyteArray prompt,
                            jint max_tokens, jobject sink, jdoubleArray stats) {
    MediaItem clip(env, path, data);
    Utf8 lang(env, language), pr(env, prompt);
    kiln_transcribe_options o = kiln_transcribe_options_default();
    o.language = lang.c(); o.prompt = pr.c(); o.max_tokens = max_tokens;
    TokenSink t = token_sink(env, sink);
    kiln_transcribe_stats st{};
    const kiln_status s = kiln_transcribe(handle<kiln_context>(c), clip.get(), &o, token_cb, &t, &st);
    if (t.failed) return nullptr;
    put(env, stats, {st.audio_seconds, (double)st.audio_tokens, (double)st.prompt_tokens, (double)st.generated_tokens, st.encode_ms,
                     st.prompt_ms, st.gen_ms, st.ms, (double)st.stopped_by_eos, (double)st.stopped_by_user});
    if (fail(env, s)) return nullptr;
    return to_bytes(env, st.language);
}

// stats: audio seconds, sample rate, frames, prompt tokens, prompt ms, gen ms, ms, eos, user.
JNI(void, speak)(JNIEnv* env, jclass, jlong c, jbyteArray text, jbyteArray voice_path, jbyteArray voice_data, jbyteArray language,
                 jint max_frames, jint top_k, jfloat top_p, jlong seed, jobject sink, jdoubleArray stats) {
    Utf8 t(env, text), lang(env, language);
    const bool has_voice = voice_path || voice_data;
    MediaItem voice(env, voice_path, voice_data);
    kiln_speak_options o = kiln_speak_options_default();
    o.voice = has_voice ? voice.get() : nullptr;
    o.language = lang.c(); o.max_frames = max_frames; o.top_k = top_k; o.top_p = top_p;
    o.seed = seed < 0 ? UINT32_MAX : (uint32_t)seed;
    AudioSink a{env, sink, sink ? env->GetMethodID(env->GetObjectClass(sink), "onAudio", "([FI)Z") : nullptr};
    kiln_speak_stats st{};
    const kiln_status s = kiln_speak(handle<kiln_context>(c), t.c(), &o, sink ? audio_cb : nullptr, &a, &st);
    if (a.failed) return;
    put(env, stats, {st.audio_seconds, (double)st.sample_rate, (double)st.frames, (double)st.prompt_tokens, st.prompt_ms, st.gen_ms, st.ms,
                     (double)st.stopped_by_eos, (double)st.stopped_by_user});
    fail(env, s);
}
