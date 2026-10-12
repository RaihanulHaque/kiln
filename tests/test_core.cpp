// Core behaviour tests. Needs real models: pass a directory containing the files that
// scripts/fetch_models.sh downloads (three chat models, a decision model, two embedding models).
// Missing models are skipped, not failed. Runs on host and on Android via scripts/android-test.sh.
#include "kiln/kiln.h"
#include "../src/probe/fast_cores.h"      // header-only rules, tested with layouts of real chips
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

static int g_fail = 0, g_pass = 0;
#define CHECK(cond, ...) do { if (cond) { ++g_pass; } else { ++g_fail; fprintf(stderr, "  FAIL %s:%d: %s  ", __FILE__, __LINE__, #cond); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)
#define SECTION(name) fprintf(stderr, "[%s]\n", name)

static bool exists(const std::string& p) { std::ifstream f(p); return f.good(); }

struct Capture {
    std::string answer, reasoning;
    int pieces = 0, stop_after = 0;
    static bool cb(kiln_channel ch, const char* piece, int32_t, void* u) {
        auto* c = static_cast<Capture*>(u);
        (ch == KILN_CHANNEL_REASONING ? c->reasoning : c->answer) += piece;
        ++c->pieces;
        return !(c->stop_after > 0 && c->pieces >= c->stop_after);
    }
};

static kiln_sampling_params greedy(int max_tokens) {
    kiln_sampling_params sp = kiln_sampling_params_default();
    sp.temperature = 0; sp.max_tokens = max_tokens; sp.seed = 42;
    return sp;
}

static void test_probe() {
    SECTION("probe");
    kiln_device_info d;
    CHECK(kiln_probe_device(&d) == KILN_OK, "");
    CHECK(d.cpu_cores > 0, "cores=%d", d.cpu_cores);
    CHECK(d.ram_total_bytes > 1000000000ull, "ram=%llu", (unsigned long long)d.ram_total_bytes);
    CHECK(d.cpu_big_cores > 0 && d.cpu_big_cores <= d.cpu_cores, "big=%d", d.cpu_big_cores);
    CHECK(strlen(d.abi) > 0, "");
}

// The core-selection rules, fed the layouts real chips report. No device needed.
static void test_core_selection() {
    SECTION("core selection");
    using kiln::fast_core_mask; using kiln::default_thread_count;

    // Galaxy S26 Ultra, Snapdragon 8 Elite Gen 5: values read from the phone on 2026-10-12.
    // No little cores: the six slower ones have capacity 741 of 1024.
    const std::vector<long> s26_f = {3628800, 3628800, 3628800, 3628800, 3628800, 3628800, 4742400, 4742400};
    const std::vector<long> s26_c = {741, 741, 741, 741, 741, 741, 1024, 1024};
    CHECK(fast_core_mask(s26_f, s26_c) == 0xffull, "S26 Ultra mask=%llx", (unsigned long long)fast_core_mask(s26_f, s26_c));
    CHECK(default_thread_count(8, 8, true) == 6, "S26 Ultra threads=%d", default_thread_count(8, 8, true));
    // without cpu_capacity only the frequency rule is left: the two prime cores, as before the fix
    CHECK(fast_core_mask(s26_f, {}) == 0xc0ull, "S26 Ultra without capacity");

    // Pixel 8, Tensor G3: frequencies as measured. The capacities are typical values for these
    // cores (A510 / A715 / X3), not read from the phone; what matters is that a little core is
    // far below half of the strongest. The five fast cores must stay selected, with 5 threads.
    const std::vector<long> p8_f = {1704000, 1704000, 1704000, 1704000, 2367000, 2367000, 2367000, 2367000, 2914000};
    const std::vector<long> p8_c = {250, 250, 250, 250, 760, 760, 760, 760, 1024};
    CHECK(fast_core_mask(p8_f, p8_c) == 0x1f0ull, "Pixel 8 mask=%llx", (unsigned long long)fast_core_mask(p8_f, p8_c));
    CHECK(fast_core_mask(p8_f, {}) == 0x1f0ull, "Pixel 8 without capacity");
    CHECK(default_thread_count(9, 5, true) == 5, "Pixel 8 threads=%d", default_thread_count(9, 5, true));

    // a slowest group at exactly half is kept, just under half is dropped
    CHECK(fast_core_mask({1000, 1000, 2000, 2000}, {512, 512, 1024, 1024}) == 0xfull, "capacity at half");
    CHECK(fast_core_mask({1000, 1000, 2000, 2000}, {511, 511, 1024, 1024}) == 0xcull, "capacity under half");
    // an unreadable capacity for one core turns the capacity rule off
    CHECK(fast_core_mask(s26_f, {741, 741, 741, 0, 741, 741, 1024, 1024}) == 0xc0ull, "one capacity missing");
    // all cores alike (desktop, or frequencies unreadable): nothing to choose, no pinning
    CHECK(fast_core_mask({3000, 3000, 3000, 3000}, {1024, 1024, 1024, 1024}) == 0, "uniform cores");
    CHECK(fast_core_mask({0, 0, 0, 0}, {}) == 0, "no frequencies");
    // thread counts: unpinned desktops are not reduced, small chips are not reduced, never above 8
    CHECK(default_thread_count(12, 12, false) == 8, "desktop threads=%d", default_thread_count(12, 12, false));
    CHECK(default_thread_count(4, 4, true) == 4, "4-core threads");
    CHECK(default_thread_count(10, 10, true) == 8, "10 fast cores");
    CHECK(default_thread_count(0, 0, false) == 4, "unknown");
}

static void test_bad_loads(const std::string& dir) {
    SECTION("bad loads");
    kiln_model* m = nullptr;
    CHECK(kiln_model_load("/nonexistent/model.gguf", nullptr, &m) == KILN_ERR_MODEL_LOAD, "status=%s", kiln_last_error());
    CHECK(m == nullptr, "");
    std::string junk = dir + "/junk.bin";
    { std::ofstream f(junk, std::ios::binary); for (int i = 0; i < 4096; ++i) f.put((char)(i * 31)); }
    CHECK(kiln_model_load(junk.c_str(), nullptr, &m) == KILN_ERR_MODEL_LOAD, "status=%s", kiln_last_error());
    remove(junk.c_str());
    CHECK(kiln_model_load(nullptr, nullptr, &m) == KILN_ERR_INVALID_ARG, "");
    CHECK(strlen(kiln_last_error()) > 0, "");
}

static void test_model(const std::string& path, const char* label, bool expect_thinking) {
    SECTION(label);
    if (!exists(path)) { fprintf(stderr, "  SKIP (missing %s)\n", path.c_str()); return; }

    kiln_model* m = nullptr;
    CHECK(kiln_model_load(path.c_str(), nullptr, &m) == KILN_OK, "%s", kiln_last_error());
    if (!m) return;
    kiln_model_info mi; kiln_model_get_info(m, &mi);
    CHECK(strlen(mi.arch) > 0, "");
    CHECK(mi.n_params > 100000000, "");
    CHECK(mi.supports_thinking == expect_thinking, "supports_thinking=%d", mi.supports_thinking);
    fprintf(stderr, "  arch=%s quant=%s backend=%s thinking=%d\n", mi.arch, mi.quant, mi.backend_name, mi.supports_thinking);

    // tokenizer round trip
    const char* text = "Hello, kiln! 123 héllo";
    int32_t toks[64];
    int n = kiln_tokenize(m, text, false, toks, 64);
    CHECK(n > 0 && n < 64, "n=%d", n);
    char back[256];
    int b = kiln_detokenize(m, toks, n, back, sizeof back);
    CHECK(b > 0, "b=%d", b);
    back[b] = 0;
    CHECK(strstr(back, "kiln") != nullptr, "back='%s'", back);

    // template renders with role markers, thinking flag changes it only when supported
    kiln_chat_message msgs[] = { {"system", "You are terse."}, {"user", "Say hi."} };
    char p_off[4096], p_on[4096];
    kiln_chat_options co = kiln_chat_options_default();
    int l_off = kiln_apply_chat_template(m, msgs, 2, &co, p_off, sizeof p_off);
    co.enable_thinking = true;
    int l_on = kiln_apply_chat_template(m, msgs, 2, &co, p_on, sizeof p_on);
    CHECK(l_off > 0 && l_on > 0, "l_off=%d l_on=%d", l_off, l_on);
    CHECK(strstr(p_off, "Say hi.") != nullptr, "");
    CHECK(strstr(p_off, "terse") != nullptr, "system prompt dropped: %s", p_off);
    if (expect_thinking) CHECK(strcmp(p_off, p_on) != 0, "thinking flag had no effect on template");
    else                 CHECK(strcmp(p_off, p_on) == 0, "thinking flag changed template of non-thinking model");
    // required-size protocol
    CHECK(kiln_apply_chat_template(m, msgs, 2, &co, nullptr, 0) == -l_on, "");

    kiln_context_params cp = kiln_context_params_default();
    cp.n_ctx = 4096;
    kiln_context* c = nullptr;
    CHECK(kiln_context_create(m, &cp, &c) == KILN_OK, "%s", kiln_last_error());
    if (!c) { kiln_model_free(m); return; }

    // turn 1
    Capture cap1; kiln_gen_stats st1;
    kiln_chat_message t1[] = { {"user", "What is the capital of France? Answer in one word."} };
    co = kiln_chat_options_default();
    auto sp = greedy(32);
    kiln_status s = kiln_chat(c, t1, 1, &co, &sp, Capture::cb, &cap1, &st1);
    CHECK(s == KILN_OK, "%s", kiln_last_error());
    CHECK(!cap1.answer.empty(), "");
    CHECK(cap1.answer.find("<think>") == std::string::npos && cap1.answer.find("</think>") == std::string::npos, "tags leaked: %s", cap1.answer.c_str());
    CHECK(cap1.reasoning.empty(), "reasoning when thinking off: '%s'", cap1.reasoning.c_str());
    CHECK(st1.cached_tokens == 0, "cached=%d", st1.cached_tokens);
    CHECK(st1.generated_tokens > 0 && st1.generated_tokens <= 32, "");
    fprintf(stderr, "  t1: '%s' (%d tok, eos=%d)\n", cap1.answer.c_str(), st1.generated_tokens, st1.stopped_by_eos);
    CHECK(cap1.answer.find("Paris") != std::string::npos, "answer='%s'", cap1.answer.c_str());

    // turn 2: history reuse
    std::string a1 = cap1.answer;
    Capture cap2; kiln_gen_stats st2;
    kiln_chat_message t2[] = { t1[0], {"assistant", a1.c_str()}, {"user", "What is the capital of Germany? Answer in one word."} };
    s = kiln_chat(c, t2, 3, &co, &sp, Capture::cb, &cap2, &st2);
    CHECK(s == KILN_OK, "%s", kiln_last_error());
    CHECK(st2.cached_tokens > 0, "no prefix reuse on turn 2");
    CHECK(st2.cached_tokens < st2.prompt_tokens, "");
    CHECK(st2.prompt_tokens > st1.prompt_tokens, "");
    fprintf(stderr, "  t2: '%s' (prompt %d, cached %d)\n", cap2.answer.c_str(), st2.prompt_tokens, st2.cached_tokens);
    CHECK(cap2.answer.find("Berlin") != std::string::npos, "answer='%s'", cap2.answer.c_str());

    // identical turn re-run: still works (cache == prompt edge case)
    Capture cap2b; kiln_gen_stats st2b;
    s = kiln_chat(c, t2, 3, &co, &sp, Capture::cb, &cap2b, &st2b);
    CHECK(s == KILN_OK, "%s", kiln_last_error());
    CHECK(!cap2b.answer.empty(), "");
    CHECK(st2b.cached_tokens > 0, "no prefix reuse on regenerate");

    // stop from callback
    Capture cap3; cap3.stop_after = 3; kiln_gen_stats st3;
    kiln_chat_message t3[] = { {"user", "Count from 1 to 50 separated by spaces."} };
    s = kiln_chat(c, t3, 1, &co, &sp, Capture::cb, &cap3, &st3);
    CHECK(s == KILN_OK, "%s", kiln_last_error());
    CHECK(st3.stopped_by_user, "");
    CHECK(st3.generated_tokens <= 4, "gen=%d", st3.generated_tokens);

    // cancel from another thread
    {
        Capture cap4; kiln_gen_stats st4;
        kiln_chat_message t4[] = { {"user", "Write a very long story about a dragon."} };
        auto sp_long = greedy(0);
        std::thread killer([&] { std::this_thread::sleep_for(std::chrono::milliseconds(150)); kiln_cancel(c); });
        s = kiln_chat(c, t4, 1, &co, &sp_long, Capture::cb, &cap4, &st4);
        killer.join();
        CHECK(s == KILN_ERR_CANCELLED, "status=%d", (int)s);
        CHECK(st4.stopped_by_user, "");
    }

    // reset, then works again
    CHECK(kiln_context_reset(c) == KILN_OK, "");
    Capture cap5; kiln_gen_stats st5;
    s = kiln_chat(c, t1, 1, &co, &sp, Capture::cb, &cap5, &st5);
    CHECK(s == KILN_OK && st5.cached_tokens == 0 && !cap5.answer.empty(), "after reset: %s", kiln_last_error());

    // thinking on
    if (expect_thinking) {
        Capture cap6; kiln_gen_stats st6;
        kiln_chat_options think = kiln_chat_options_default(); think.enable_thinking = true;
        kiln_chat_message t6[] = { {"user", "What is 17 * 23? Think step by step, then give the number."} };
        // Qwen3.5 0.8B rarely closes its think block on its own (any sampling), so give it a budget:
        // kiln closes the block after 256 reasoning tokens and the model goes on to answer.
        // Repeat penalty off: it made the answer loop more often (13 vs 16 of 16 runs finished).
        think.max_reasoning_tokens = 256;
        auto sp6 = kiln_sampling_params_default(); sp6.max_tokens = 1500; sp6.seed = 7; sp6.repeat_penalty = 1.0f;
        s = kiln_chat(c, t6, 1, &think, &sp6, Capture::cb, &cap6, &st6);
        CHECK(s == KILN_OK, "%s", kiln_last_error());
        CHECK(st6.reasoning_tokens > 0, "thinking on but no reasoning tokens");
        CHECK(st6.reasoning_tokens <= 256, "budget 256 but %d reasoning tokens", st6.reasoning_tokens);
        CHECK(st6.reasoning_truncated == (st6.reasoning_tokens == 256), "truncated=%d at %d reasoning tokens", st6.reasoning_truncated, st6.reasoning_tokens);
        CHECK(cap6.answer.find("<think>") == std::string::npos && cap6.answer.find("</think>") == std::string::npos, "tags leaked into answer: %s", cap6.answer.c_str());
        CHECK(cap6.reasoning.find("<think>") == std::string::npos && cap6.reasoning.find("</think>") == std::string::npos, "tags leaked into reasoning");
        CHECK(st6.reasoning_tokens == 0 || !cap6.reasoning.empty(), "reasoning tokens counted but nothing emitted");
        fprintf(stderr, "  think: reasoning=%d tok%s, answer='%.60s'\n", st6.reasoning_tokens, st6.reasoning_truncated ? " (cut at budget)" : "", cap6.answer.c_str());
        CHECK(!cap6.answer.empty(), "empty answer channel");
        // sampling is not bit-identical across backends, so the final number is only checked on a clean finish
        if (st6.stopped_by_eos) CHECK(cap6.answer.find("391") != std::string::npos, "answer='%s'", cap6.answer.c_str());
        else fprintf(stderr, "  (answer did not finish within %d tokens; number check skipped)\n", sp6.max_tokens);

        // without a budget nothing is cut, and the flag stays off
        Capture cap6b; kiln_gen_stats st6b;
        think.max_reasoning_tokens = 0; sp6.max_tokens = 40;
        kiln_context_reset(c);
        s = kiln_chat(c, t6, 1, &think, &sp6, Capture::cb, &cap6b, &st6b);
        CHECK(s == KILN_OK && !st6b.reasoning_truncated && st6b.reasoning_tokens == st6b.generated_tokens, "no budget: truncated=%d reasoning=%d of %d",
              st6b.reasoning_truncated, st6b.reasoning_tokens, st6b.generated_tokens);
    }

    // a chat model is not a decision model
    {
        float p[2]; kiln_decision_answer a;
        kiln_decision_question q = {"q", KILN_QUESTION_NOUL, "Is this a question?", nullptr, 0};
        CHECK(!mi.is_decision_model && kiln_decide(c, "state", &q, p, 2, &a) == KILN_ERR_NOT_SUPPORTED, "kiln_decide on a chat model");
        CHECK(kiln_model_load_projector(m, "/nonexistent/mmproj.gguf", nullptr, nullptr) == KILN_ERR_NOT_SUPPORTED, "projector on a chat model");
        float v[8];
        CHECK(!mi.is_embedding_model && mi.n_embd == 0 && kiln_embed(c, "text", nullptr, v, 8, nullptr) == KILN_ERR_NOT_SUPPORTED, "kiln_embed on a chat model");
    }

    kiln_context_free(c);

    // context overflow: tiny context, prompt does not fit
    cp.n_ctx = 32;
    CHECK(kiln_context_create(m, &cp, &c) == KILN_OK, "%s", kiln_last_error());
    std::string big(2000, 'a');
    for (size_t i = 7; i < big.size(); i += 8) big[i] = ' ';
    kiln_chat_message t7[] = { {"user", big.c_str()} };
    Capture cap7; kiln_gen_stats st7;
    s = kiln_chat(c, t7, 1, &co, &sp, Capture::cb, &cap7, &st7);
    CHECK(s == KILN_ERR_CONTEXT_FULL, "status=%d", (int)s);
    // context overflow mid-generation: short prompt, unbounded output
    kiln_context_reset(c);
    kiln_chat_message t8[] = { {"user", "Count upward forever: 1 2 3"} };
    Capture cap8; kiln_gen_stats st8;
    auto sp8 = greedy(0);
    s = kiln_chat(c, t8, 1, &co, &sp8, Capture::cb, &cap8, &st8);
    CHECK(s == KILN_ERR_CONTEXT_FULL || st8.stopped_by_eos, "status=%d gen=%d", (int)s, st8.generated_tokens);
    if (s == KILN_ERR_CONTEXT_FULL) CHECK(st8.generated_tokens > 0, "");
    kiln_context_free(c);
    kiln_model_free(m);
}

// Decision model (LFM2 d1-omni): typed questions answered in one pass, no generated text.
static void test_decision(const std::string& path) {
    SECTION("d1-omni decision model");
    { std::ifstream f(path); if (!f.good()) { fprintf(stderr, "  SKIP (missing %s)\n", path.c_str()); return; } }
    kiln_model* m = nullptr;
    kiln_model_params mp = kiln_model_params_default();
    CHECK(kiln_model_load(path.c_str(), &mp, &m) == KILN_OK, "%s", kiln_last_error());
    if (!m) return;
    kiln_model_info mi; kiln_model_get_info(m, &mi);
    CHECK(mi.is_decision_model, "arch=%s", mi.arch);
    kiln_context_params cp = kiln_context_params_default(); cp.n_ctx = 4096;
    kiln_context* c = nullptr;
    CHECK(kiln_context_create(m, &cp, &c) == KILN_OK, "%s", kiln_last_error());
    if (!c) { kiln_model_free(m); return; }

    const char* state = "Hi, I ordered a blender two weeks ago and it arrived broken. I have emailed twice and nobody "
                        "answered. I want my money back today or I will dispute the charge with my bank.";
    float p[8]; kiln_decision_answer a;
    auto sums_to_one = [&](int n) { float s = 0; for (int i = 0; i < n; ++i) s += p[i]; return fabsf(s - 1.0f) < 1e-3f; };

    // yes / no
    kiln_decision_question refund = {"wants_refund", KILN_QUESTION_NOUL, "Is the customer asking for a refund?", nullptr, 0};
    kiln_status s = kiln_decide(c, state, &refund, p, 8, &a);
    CHECK(s == KILN_OK, "%s", kiln_last_error());
    CHECK(a.n_probs == 2 && sums_to_one(2), "n_probs=%d", a.n_probs);
    CHECK(a.value > 0.5f && a.best == 1, "P(yes)=%.3f", a.value);
    fprintf(stderr, "  refund? yes=%.3f (%d tok, %.0f ms)\n", a.value, a.prompt_tokens, a.ms);

    kiln_decision_question weather = {"about_weather", KILN_QUESTION_NOUL, "Is the customer asking about the weather?", nullptr, 0};
    s = kiln_decide(c, state, &weather, p, 8, &a);
    CHECK(s == KILN_OK && a.value < 0.5f && a.best == 0, "P(yes)=%.3f", a.value);

    // pick one
    kiln_decision_option teams[] = {{"billing", "Payments, refunds and invoices"}, {"shipping", "Delivery status and lost parcels"},
                                    {"technical", "The product does not work as expected"}};
    kiln_decision_question team = {"team", KILN_QUESTION_CHOICE, "Which team should handle this?", teams, 3};
    s = kiln_decide(c, state, &team, p, 8, &a);
    CHECK(s == KILN_OK, "%s", kiln_last_error());
    CHECK(a.n_probs == 3 && sums_to_one(3) && a.best >= 0 && a.best < 3, "n_probs=%d best=%d", a.n_probs, a.best);
    CHECK(a.confidence >= 0.0f && a.confidence <= 1.0f, "confidence=%.3f", a.confidence);
    fprintf(stderr, "  team: %s (billing %.3f, shipping %.3f, technical %.3f)\n", teams[a.best].key, p[0], p[1], p[2]);

    // rate on a scale
    kiln_decision_option levels[] = {{nullptr, "Can wait"}, {nullptr, "Today"}, {nullptr, "Blocking the customer now"}};
    kiln_decision_question urgency = {"urgency", KILN_QUESTION_SCORE, "How urgent is this?", levels, 3};
    s = kiln_decide(c, state, &urgency, p, 8, &a);
    CHECK(s == KILN_OK, "%s", kiln_last_error());
    CHECK(a.n_probs == 3 && sums_to_one(3) && a.value >= 0.0f && a.value <= 2.0f, "expected level=%.3f", a.value);
    fprintf(stderr, "  urgency: expected level %.2f of 0..2\n", a.value);

    // same question twice gives the same numbers
    float p2[8]; kiln_decision_answer a2;
    s = kiln_decide(c, state, &urgency, p2, 8, &a2);
    CHECK(s == KILN_OK && p2[0] == p[0] && p2[1] == p[1] && p2[2] == p[2], "not deterministic");

    // bad input and wrong model kind
    CHECK(kiln_decide(c, state, &team, p, 2, &a) == KILN_ERR_INVALID_ARG, "probs too small accepted");
    kiln_decision_question one_level = {"q", KILN_QUESTION_SCORE, "How urgent?", levels, 1};
    CHECK(kiln_decide(c, state, &one_level, p, 8, &a) == KILN_ERR_INVALID_ARG, "1-level score accepted");
    CHECK(kiln_decide(c, nullptr, &refund, p, 8, &a) == KILN_OK, "null state: %s", kiln_last_error());
    kiln_chat_message msg[] = { {"user", "hi"} }; kiln_gen_stats st;
    CHECK(kiln_chat(c, msg, 1, nullptr, nullptr, nullptr, nullptr, &st) == KILN_ERR_NOT_SUPPORTED, "chat on a decision model");

    kiln_context_free(c);
    kiln_model_free(m);
}

// Embedding model: text in, vector out. ref = the first numbers llama.cpp's own llama-embedding
// tool prints for "The cat sat on the mat." (same commit, CPU, normalised).
// titled_model: the model has a place for a document title in its prefix (EmbeddingGemma).
static void test_embedding(const std::string& path, const char* label, bool expect_prefix, bool matryoshka, bool titled_model, const float ref[3]) {
    SECTION(label);
    if (!exists(path)) { fprintf(stderr, "  SKIP (missing %s)\n", path.c_str()); return; }
    kiln_model* m = nullptr;
    kiln_model_params mp = kiln_model_params_default();
    CHECK(kiln_model_load(path.c_str(), &mp, &m) == KILN_OK, "%s", kiln_last_error());
    if (!m) return;
    kiln_model_info mi; kiln_model_get_info(m, &mi);
    CHECK(mi.is_embedding_model && !mi.is_decision_model && !mi.supports_thinking, "arch=%s", mi.arch);
    CHECK(mi.n_embd == 768, "n_embd=%d", mi.n_embd);
    kiln_context_params cp = kiln_context_params_default(); cp.n_ctx = 256;
    kiln_context* c = nullptr;
    CHECK(kiln_context_create(m, &cp, &c) == KILN_OK, "%s", kiln_last_error());
    if (!c) { kiln_model_free(m); return; }
    const int D = mi.n_embd;
    auto dot = [](const std::vector<float>& a, const std::vector<float>& b, int n) { double s = 0; for (int i = 0; i < n; ++i) s += (double)a[i] * b[i]; return s; };

    // raw text: the numbers llama.cpp gives, length 1, same again on a second call
    std::vector<float> v(D), v2(D);
    kiln_embed_stats st;
    kiln_status s = kiln_embed(c, "The cat sat on the mat.", nullptr, v.data(), D, &st);
    CHECK(s == KILN_OK, "%s", kiln_last_error());
    CHECK(st.dimensions == D && st.tokens > 4 && st.tokens < 16 && !st.prefixed, "dims=%d tokens=%d prefixed=%d", st.dimensions, st.tokens, st.prefixed);
    CHECK(fabs(dot(v, v, D) - 1.0) < 1e-4, "length^2=%.6f", dot(v, v, D));
    // ref was measured on x86. A 4-bit model is computed with different instructions on each
    // kind of processor and the rounding differs: a Pixel 8 is off by up to 0.009 on nomic, an
    // M4 GPU by 0.003. A wrong prefix or pooling moves these numbers by far more than 0.015.
    CHECK(fabsf(v[0] - ref[0]) < 1.5e-2f && fabsf(v[1] - ref[1]) < 1.5e-2f && fabsf(v[2] - ref[2]) < 1.5e-2f,
          "got %.6f %.6f %.6f, llama.cpp gives %.6f %.6f %.6f", v[0], v[1], v[2], ref[0], ref[1], ref[2]);
    s = kiln_embed(c, "The cat sat on the mat.", nullptr, v2.data(), D, nullptr);
    CHECK(s == KILN_OK && memcmp(v.data(), v2.data(), D * sizeof(float)) == 0, "not deterministic");
    fprintf(stderr, "  %d numbers, %d tok in %.1f ms\n", st.dimensions, st.tokens, st.ms);

    // search: the query must land nearer the text that answers it than the one sharing its words
    kiln_embed_options q = kiln_embed_options_default(); q.kind = KILN_EMBED_QUERY;
    kiln_embed_options d = kiln_embed_options_default(); d.kind = KILN_EMBED_DOCUMENT;
    std::vector<float> qv(D), right(D), trap(D), other(D);
    kiln_embed_stats qst;
    CHECK(kiln_embed(c, "how do I reset my password", &q, qv.data(), D, &qst) == KILN_OK, "%s", kiln_last_error());
    CHECK(qst.prefixed == expect_prefix, "prefixed=%d", qst.prefixed);
    CHECK(kiln_embed(c, "Steps to recover account access: open Settings, choose Security, and tap Forgot credentials.", &d, right.data(), D, nullptr) == KILN_OK, "%s", kiln_last_error());
    CHECK(kiln_embed(c, "Passwords should contain at least twelve characters including a symbol.", &d, trap.data(), D, nullptr) == KILN_OK, "%s", kiln_last_error());
    CHECK(kiln_embed(c, "Soda loaf uses baking soda and buttermilk as the raising agent.", &d, other.data(), D, nullptr) == KILN_OK, "%s", kiln_last_error());
    const double s_right = dot(qv, right, D), s_trap = dot(qv, trap, D), s_other = dot(qv, other, D);
    CHECK(s_right > s_trap && s_trap > s_other, "right=%.3f trap=%.3f unrelated=%.3f", s_right, s_trap, s_other);
    fprintf(stderr, "  password query: answer %.3f, trap %.3f, unrelated %.3f\n", s_right, s_trap, s_other);

    // the kind changes the vector only when the model has a prefix
    std::vector<float> raw(D);
    CHECK(kiln_embed(c, "how do I reset my password", nullptr, raw.data(), D, nullptr) == KILN_OK, "%s", kiln_last_error());
    CHECK((memcmp(raw.data(), qv.data(), D * sizeof(float)) != 0) == expect_prefix, "query prefix");
    kiln_embed_options titled = d; titled.title = "Account help";
    std::vector<float> tv(D);
    CHECK(kiln_embed(c, "Steps to recover account access: open Settings, choose Security, and tap Forgot credentials.", &titled, tv.data(), D, nullptr) == KILN_OK, "%s", kiln_last_error());
    CHECK(fabs(dot(tv, tv, D) - 1.0) < 1e-4, "titled document");
    // the title must reach the model where the model has a place for it, and only there
    CHECK((memcmp(tv.data(), right.data(), D * sizeof(float)) != 0) == titled_model, "title %s", titled_model ? "was ignored" : "changed the vector of a model without titles");

    // shortened vector: the leading numbers of the full one, scaled back to length 1
    kiln_embed_options sh = kiln_embed_options_default(); sh.dimensions = 256;
    std::vector<float> sv(256);
    s = kiln_embed(c, "The cat sat on the mat.", &sh, sv.data(), 256, &st);
    CHECK(s == KILN_OK && st.dimensions == 256, "%s", kiln_last_error());
    CHECK(fabs(dot(sv, sv, 256) - 1.0) < 1e-4, "length^2=%.6f", dot(sv, sv, 256));
    const double head = sqrt(dot(v, v, 256));
    CHECK(fabs(sv[0] * head - v[0]) < 1e-5 && fabs(sv[255] * head - v[255]) < 1e-5, "not the leading part of the full vector");
    if (matryoshka) {
        // A shortened vector is a coarser one. It must still tell the answer from an unrelated
        // text and score it close to the full vector. It is not required to beat the trap
        // sentence: at 256 numbers the two are 0.001 apart and the order depends on the hardware.
        std::vector<float> a(256), b(256), t2(256);
        kiln_embed_options q2 = q, d2 = d; q2.dimensions = d2.dimensions = 256;
        kiln_embed(c, "how do I reset my password", &q2, a.data(), 256, nullptr);
        kiln_embed(c, "Steps to recover account access: open Settings, choose Security, and tap Forgot credentials.", &d2, b.data(), 256, nullptr);
        kiln_embed(c, "Soda loaf uses baking soda and buttermilk as the raising agent.", &d2, t2.data(), 256, nullptr);
        const double r256 = dot(a, b, 256), o256 = dot(a, t2, 256);
        CHECK(r256 > o256 + 0.1 && fabs(r256 - s_right) < 0.05, "256 numbers: right=%.3f unrelated=%.3f, full right=%.3f", r256, o256, s_right);
    }
    kiln_embed_options un = kiln_embed_options_default(); un.normalize = false;
    CHECK(kiln_embed(c, "The cat sat on the mat.", &un, v2.data(), D, nullptr) == KILN_OK, "%s", kiln_last_error());
    const double len = sqrt(dot(v2, v2, D));
    // the raw vector points the same way but keeps the model's own length (about 1246 for
    // EmbeddingGemma 2 and 22 for nomic), so a result of length 1 means the option was ignored
    CHECK(len > 0 && fabs(v2[0] / len - v[0]) < 1e-5 && fabs(len - 1.0) > 0.5, "normalize=false, length=%.4f", len);

    // bad input, text too long, wrong kind of call
    CHECK(kiln_embed(c, "text", nullptr, v.data(), D - 1, nullptr) == KILN_ERR_INVALID_ARG, "out too small accepted");
    kiln_embed_options big = kiln_embed_options_default(); big.dimensions = D + 1;
    CHECK(kiln_embed(c, "text", &big, v.data(), D, nullptr) == KILN_ERR_INVALID_ARG, "dimensions above n_embd accepted");
    CHECK(kiln_embed(c, nullptr, nullptr, v.data(), D, nullptr) == KILN_ERR_INVALID_ARG, "null text accepted");
    std::string longtext;
    for (int i = 0; i < 200; ++i) longtext += "The quick brown fox jumps over the lazy dog. ";
    CHECK(kiln_embed(c, longtext.c_str(), nullptr, v.data(), D, &st) == KILN_ERR_CONTEXT_FULL && st.tokens > 256, "long text: tokens=%d", st.tokens);
    CHECK(kiln_embed(c, "The cat sat on the mat.", nullptr, v2.data(), D, nullptr) == KILN_OK && fabsf(v2[0] - ref[0]) < 5e-3f, "embed after an error");
    kiln_chat_message msg[] = { {"user", "hi"} }; kiln_gen_stats gst;
    CHECK(kiln_chat(c, msg, 1, nullptr, nullptr, nullptr, nullptr, &gst) == KILN_ERR_NOT_SUPPORTED, "chat on an embedding model");
    CHECK(kiln_generate(c, "hi", nullptr, nullptr, nullptr, &gst) == KILN_ERR_NOT_SUPPORTED, "generate on an embedding model");
    float p[2]; kiln_decision_answer da;
    kiln_decision_question dq = {"q", KILN_QUESTION_NOUL, "Is this a question?", nullptr, 0};
    CHECK(kiln_decide(c, "state", &dq, p, 2, &da) == KILN_ERR_NOT_SUPPORTED, "decide on an embedding model");

    // media without a projector; kiln_embed_media with no media is kiln_embed
    kiln_media none = {"/nonexistent.png", nullptr, 0};
    CHECK(kiln_embed_media(c, nullptr, &none, 1, nullptr, v2.data(), D, nullptr) == KILN_ERR_NOT_SUPPORTED, "media without a projector");
    CHECK(kiln_embed_media(c, "The cat sat on the mat.", nullptr, 0, nullptr, v2.data(), D, nullptr) == KILN_OK && fabsf(v2[0] - ref[0]) < 5e-3f, "embed_media with text only");
    CHECK(kiln_embed_media(c, nullptr, nullptr, 0, nullptr, v2.data(), D, nullptr) == KILN_ERR_INVALID_ARG, "no text and no media accepted");

    kiln_context_free(c);
    kiln_model_free(m);
}

// Images and audio through an embedding model's projector. The reference numbers are what
// llama.cpp's llama-server returns for the same files (same commit, CPU): ref[0] invoice.png,
// ref[1] "An invoice <marker> for a blender", ref[2] speech_train.wav, ref[3] "Listen: <marker>".
static void test_embedding_media(const std::string& path, const std::string& mmproj, const std::string& media,
                                 const char* label, bool audio, const float ref[4][3]) {
    SECTION(label);
    if (!exists(path) || !exists(mmproj)) { fprintf(stderr, "  SKIP (missing %s)\n", exists(path) ? mmproj.c_str() : path.c_str()); return; }
    if (!exists(media + "/invoice.png")) { fprintf(stderr, "  SKIP (missing %s/invoice.png)\n", media.c_str()); return; }
    kiln_model* m = nullptr;
    kiln_model_params mp = kiln_model_params_default();
    CHECK(kiln_model_load(path.c_str(), &mp, &m) == KILN_OK, "%s", kiln_last_error());
    if (!m) return;
    kiln_model_info mi; kiln_model_get_info(m, &mi);
    const int D = mi.n_embd;
    kiln_projector_info pi;
    CHECK(kiln_model_load_projector(m, "/nonexistent/mmproj.gguf", nullptr, &pi) == KILN_ERR_MODEL_LOAD, "missing projector accepted");
    CHECK(kiln_model_load_projector(m, mmproj.c_str(), nullptr, &pi) == KILN_OK, "%s", kiln_last_error());
    CHECK(pi.supports_images && pi.supports_audio == audio, "images=%d audio=%d", pi.supports_images, pi.supports_audio);
    CHECK(kiln_model_load_projector(m, mmproj.c_str(), nullptr, &pi) == KILN_ERR_INVALID_ARG, "second projector accepted");
    kiln_context_params cp = kiln_context_params_default(); cp.n_ctx = 1024;
    kiln_context* c = nullptr;
    CHECK(kiln_context_create(m, &cp, &c) == KILN_OK, "%s", kiln_last_error());
    if (!c) { kiln_model_free(m); return; }

    auto dot = [&](const std::vector<float>& a, const std::vector<float>& b) { double s = 0; for (int i = 0; i < D; ++i) s += (double)a[i] * b[i]; return s; };
    auto file = [&](const char* name, const char* text, kiln_embed_stats* st) {
        std::vector<float> v(D);
        const std::string p = media + "/" + name;
        kiln_media item = {p.c_str(), nullptr, 0};
        kiln_status s = kiln_embed_media(c, text, &item, 1, nullptr, v.data(), D, st);
        CHECK(s == KILN_OK, "%s: %s", name, kiln_last_error());
        return v;
    };
    auto query = [&](const char* text) {
        std::vector<float> v(D);
        kiln_embed_options q = kiln_embed_options_default(); q.kind = KILN_EMBED_QUERY;
        CHECK(kiln_embed(c, text, &q, v.data(), D, nullptr) == KILN_OK, "%s", kiln_last_error());
        return v;
    };
    auto near_ref = [](const std::vector<float>& v, float a, float b, float d) { return fabsf(v[0] - a) < 5e-3f && fabsf(v[1] - b) < 5e-3f && fabsf(v[2] - d) < 5e-3f; };

    // an image alone: llama.cpp's numbers, length 1, same again
    kiln_embed_stats st;
    std::vector<float> invoice = file("invoice.png", nullptr, &st);
    CHECK(st.media_tokens > 0 && st.tokens > st.media_tokens && st.dimensions == D && !st.prefixed && st.media_ms > 0, "tokens=%d media=%d", st.tokens, st.media_tokens);
    CHECK(fabs(dot(invoice, invoice) - 1.0) < 1e-4, "length^2=%.6f", dot(invoice, invoice));
    CHECK(near_ref(invoice, ref[0][0], ref[0][1], ref[0][2]), "got %.6f %.6f %.6f", invoice[0], invoice[1], invoice[2]);
    fprintf(stderr, "  image: %d tok (%d media) in %.0f ms, encoder %.0f ms\n", st.tokens, st.media_tokens, st.ms, st.media_ms);
    std::vector<float> again = file("invoice.png", nullptr, nullptr);
    CHECK(memcmp(invoice.data(), again.data(), D * sizeof(float)) == 0, "image not deterministic");

    // the same image from memory
    {
        std::ifstream f(media + "/invoice.png", std::ios::binary);
        std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        kiln_media item = {nullptr, bytes.data(), bytes.size()};
        std::vector<float> v(D);
        CHECK(kiln_embed_media(c, nullptr, &item, 1, nullptr, v.data(), D, nullptr) == KILN_OK, "%s", kiln_last_error());
        CHECK(memcmp(invoice.data(), v.data(), D * sizeof(float)) == 0, "bytes and file differ");
    }

    // a text query finds the right picture
    std::vector<float> red = file("red_circle.png", nullptr, nullptr), blue = file("blue_square.png", nullptr, nullptr);
    std::vector<float> q_inv = query("an invoice showing the total amount to pay"), q_red = query("a red circle"), q_blue = query("a blue square");
    CHECK(dot(q_inv, invoice) > dot(q_inv, red) && dot(q_inv, invoice) > dot(q_inv, blue), "invoice %.3f red %.3f blue %.3f", dot(q_inv, invoice), dot(q_inv, red), dot(q_inv, blue));
    CHECK(dot(q_red, red) > dot(q_red, blue) && dot(q_red, red) > dot(q_red, invoice), "red %.3f blue %.3f invoice %.3f", dot(q_red, red), dot(q_red, blue), dot(q_red, invoice));
    CHECK(dot(q_blue, blue) > dot(q_blue, red) && dot(q_blue, blue) > dot(q_blue, invoice), "blue %.3f red %.3f invoice %.3f", dot(q_blue, blue), dot(q_blue, red), dot(q_blue, invoice));
    fprintf(stderr, "  \"a red circle\": red circle %.3f, blue square %.3f, invoice %.3f\n", dot(q_red, red), dot(q_red, blue), dot(q_red, invoice));

    // text and an image in one input, with the marker saying where the image goes
    std::vector<float> mixed = file("invoice.png", "An invoice " KILN_MEDIA_MARKER " for a blender", &st);
    CHECK(near_ref(mixed, ref[1][0], ref[1][1], ref[1][2]), "got %.6f %.6f %.6f", mixed[0], mixed[1], mixed[2]);
    {
        std::vector<float> at_end = file("invoice.png", "An invoice " KILN_MEDIA_MARKER, nullptr), appended = file("invoice.png", "An invoice", nullptr);
        CHECK(memcmp(at_end.data(), appended.data(), D * sizeof(float)) == 0, "media after the text differs from an explicit marker at the end");
    }

    if (!audio) {
        const std::string p = media + "/speech_train.wav";
        kiln_media item = {p.c_str(), nullptr, 0};
        std::vector<float> v(D);
        CHECK(kiln_embed_media(c, nullptr, &item, 1, nullptr, v.data(), D, nullptr) != KILN_OK, "audio accepted by an image-only projector");
    } else {
    // audio: a text query finds the clip that says it
    std::vector<float> train = file("speech_train.wav", nullptr, &st), recipe = file("speech_recipe.wav", nullptr, nullptr), tone = file("tone.wav", nullptr, nullptr);
    CHECK(st.media_tokens > 0, "audio media tokens=%d", st.media_tokens);
    CHECK(near_ref(train, ref[2][0], ref[2][1], ref[2][2]), "got %.6f %.6f %.6f", train[0], train[1], train[2]);
    fprintf(stderr, "  audio: %d tok (%d media) in %.0f ms, encoder %.0f ms\n", st.tokens, st.media_tokens, st.ms, st.media_ms);
    std::vector<float> q_train = query("when does the train to London leave"), q_recipe = query("how much flour do I need for the recipe");
    CHECK(dot(q_train, train) > dot(q_train, recipe) && dot(q_train, train) > dot(q_train, tone), "train %.3f recipe %.3f tone %.3f", dot(q_train, train), dot(q_train, recipe), dot(q_train, tone));
    CHECK(dot(q_recipe, recipe) > dot(q_recipe, train) && dot(q_recipe, recipe) > dot(q_recipe, tone), "recipe %.3f train %.3f tone %.3f", dot(q_recipe, recipe), dot(q_recipe, train), dot(q_recipe, tone));
    fprintf(stderr, "  \"when does the train to London leave\": train clip %.3f, recipe clip %.3f, tone %.3f\n", dot(q_train, train), dot(q_train, recipe), dot(q_train, tone));

    std::vector<float> listen = file("speech_train.wav", "Listen: " KILN_MEDIA_MARKER, nullptr);
    CHECK(near_ref(listen, ref[3][0], ref[3][1], ref[3][2]), "got %.6f %.6f %.6f", listen[0], listen[1], listen[2]);
    }
    {
        const std::string a = media + "/red_circle.png", b = media + "/blue_square.png";
        kiln_media two[] = {{a.c_str(), nullptr, 0}, {b.c_str(), nullptr, 0}};
        std::vector<float> v(D);
        CHECK(kiln_embed_media(c, nullptr, two, 2, nullptr, v.data(), D, &st) == KILN_OK && st.media_tokens > 100, "two images: %s", kiln_last_error());
        CHECK(kiln_embed_media(c, "one " KILN_MEDIA_MARKER " marker", two, 2, nullptr, v.data(), D, nullptr) == KILN_ERR_INVALID_ARG, "marker count mismatch accepted");
    }

    // shortened vector, bad files, a context too small for an image
    {
        const std::string p = media + "/invoice.png";
        kiln_media item = {p.c_str(), nullptr, 0};
        kiln_embed_options sh = kiln_embed_options_default(); sh.dimensions = 256;
        std::vector<float> sv(256);
        CHECK(kiln_embed_media(c, nullptr, &item, 1, &sh, sv.data(), 256, &st) == KILN_OK && st.dimensions == 256, "%s", kiln_last_error());
        kiln_media missing = {"/nonexistent/picture.png", nullptr, 0};
        std::vector<float> v(D);
        CHECK(kiln_embed_media(c, nullptr, &missing, 1, nullptr, v.data(), D, nullptr) == KILN_ERR_INVALID_ARG, "missing file accepted");
        const uint8_t junk[64] = {1, 2, 3, 4, 5};
        kiln_media bad = {nullptr, junk, sizeof junk};
        CHECK(kiln_embed_media(c, nullptr, &bad, 1, nullptr, v.data(), D, nullptr) == KILN_ERR_INVALID_ARG, "junk bytes accepted");
        kiln_media empty = {nullptr, nullptr, 0};
        CHECK(kiln_embed_media(c, nullptr, &empty, 1, nullptr, v.data(), D, nullptr) == KILN_ERR_INVALID_ARG, "empty media item accepted");
        kiln_context_params small = kiln_context_params_default(); small.n_ctx = 32;
        kiln_context* c2 = nullptr;
        CHECK(kiln_context_create(m, &small, &c2) == KILN_OK, "%s", kiln_last_error());
        if (c2) {
            CHECK(kiln_embed_media(c2, nullptr, &item, 1, nullptr, v.data(), D, &st) == KILN_ERR_CONTEXT_FULL && st.tokens > 32, "image in a 32-token context: tokens=%d", st.tokens);
            kiln_context_free(c2);
        }
        std::vector<float> after = file("invoice.png", nullptr, nullptr);
        CHECK(memcmp(invoice.data(), after.data(), D * sizeof(float)) == 0, "embed after errors differs");
    }

    kiln_context_free(c);
    kiln_model_free(m);
}

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : "models";
    kiln_set_log_level(KILN_LOG_ERROR);
    test_probe();
    test_core_selection();
    test_bad_loads(dir);
    //                                                       label           thinking
    test_model(dir + "/LiquidAI_LFM2.5-350M_Q4_K_M.gguf", "LFM2.5-350M",  false);
    test_model(dir + "/Qwen3.5-0.8B-Q4_K_M.gguf",         "Qwen3.5-0.8B", true);
    test_model(dir + "/gemma-3-270m-it-Q8_0.gguf",         "Gemma3-270M",  false);
    test_decision(dir + "/d1-omni-600M-Q8_0.gguf");
    const float gemma_ref[3] = {-0.001069f, -0.025002f, 0.020770f};
    const float nomic_ref[3] = {0.047576f, 0.061704f, -0.118842f};
    test_embedding(dir + "/embeddinggemma-2-UD-Q4_K_XL.gguf", "EmbeddingGemma 2 (embedding model)", true, true, true, gemma_ref);
    test_embedding(dir + "/nomic-embed-text-v1.5.Q4_K_M.gguf", "nomic-embed-text v1.5 (embedding model)", true, false, false, nomic_ref);
    // test images and audio: tests/media in the repo, or media/ next to the models (Android)
    const char* media_env = getenv("KILN_TEST_MEDIA");
    std::string media = media_env ? media_env : exists(dir + "/media/invoice.png") ? dir + "/media" : dir + "/../tests/media";
    const float gemma_media_ref[4][3] = {{-0.048288f, -0.022876f, 0.024317f}, {-0.028795f, -0.009980f, 0.006978f},
                                         {-0.031794f, -0.024362f, 0.002251f}, {-0.033693f, -0.033323f, -0.007274f}};
    test_embedding_media(dir + "/embeddinggemma-2-UD-Q4_K_XL.gguf", dir + "/mmproj-embeddinggemma-2-Q8_0.gguf", media,
                         "EmbeddingGemma 2 (text, images and audio)", true, gemma_media_ref);
    const float jina_media_ref[4][3] = {{0.040660f, -0.008720f, 0.047007f}, {0.042589f, -0.011429f, 0.048737f}, {0, 0, 0}, {0, 0, 0}};
    test_embedding_media(dir + "/jina-embeddings-v5-omni-nano-retrieval-Q4_K_M.gguf", dir + "/jina-embeddings-v5-omni-nano-retrieval-vision-mmproj-F16.gguf", media,
                         "jina-embeddings-v5-omni-nano (text and images)", false, jina_media_ref);
    fprintf(stderr, "\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
