// Core behaviour tests. Needs real models: pass a directory containing the files that
// scripts/fetch_models.sh downloads (three chat models, a decision model, two embedding models).
// Missing models are skipped, not failed. Runs on host and on Android via scripts/android-test.sh.
#include "kiln/kiln.h"
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
static void test_embedding(const std::string& path, const char* label, bool expect_prefix, bool matryoshka, const float ref[3]) {
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
    CHECK(fabsf(v[0] - ref[0]) < 5e-3f && fabsf(v[1] - ref[1]) < 5e-3f && fabsf(v[2] - ref[2]) < 5e-3f,
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

    // shortened vector: the leading numbers of the full one, scaled back to length 1
    kiln_embed_options sh = kiln_embed_options_default(); sh.dimensions = 256;
    std::vector<float> sv(256);
    s = kiln_embed(c, "The cat sat on the mat.", &sh, sv.data(), 256, &st);
    CHECK(s == KILN_OK && st.dimensions == 256, "%s", kiln_last_error());
    CHECK(fabs(dot(sv, sv, 256) - 1.0) < 1e-4, "length^2=%.6f", dot(sv, sv, 256));
    const double head = sqrt(dot(v, v, 256));
    CHECK(fabs(sv[0] * head - v[0]) < 1e-5 && fabs(sv[255] * head - v[255]) < 1e-5, "not the leading part of the full vector");
    if (matryoshka) {
        std::vector<float> a(256), b(256), t2(256);
        kiln_embed_options q2 = q, d2 = d; q2.dimensions = d2.dimensions = 256;
        kiln_embed(c, "how do I reset my password", &q2, a.data(), 256, nullptr);
        kiln_embed(c, "Steps to recover account access: open Settings, choose Security, and tap Forgot credentials.", &d2, b.data(), 256, nullptr);
        kiln_embed(c, "Passwords should contain at least twelve characters including a symbol.", &d2, t2.data(), 256, nullptr);
        CHECK(dot(a, b, 256) > dot(a, t2, 256), "256 numbers: right=%.3f trap=%.3f", dot(a, b, 256), dot(a, t2, 256));
    }
    kiln_embed_options un = kiln_embed_options_default(); un.normalize = false;
    CHECK(kiln_embed(c, "The cat sat on the mat.", &un, v2.data(), D, nullptr) == KILN_OK, "%s", kiln_last_error());
    const double len = sqrt(dot(v2, v2, D));
    CHECK(len > 0 && fabs(v2[0] / len - v[0]) < 1e-5, "normalize=false, length=%.4f", len);

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

    kiln_context_free(c);
    kiln_model_free(m);
}

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : "models";
    kiln_set_log_level(KILN_LOG_ERROR);
    test_probe();
    test_bad_loads(dir);
    //                                                       label           thinking
    test_model(dir + "/LiquidAI_LFM2.5-350M_Q4_K_M.gguf", "LFM2.5-350M",  false);
    test_model(dir + "/Qwen3.5-0.8B-Q4_K_M.gguf",         "Qwen3.5-0.8B", true);
    test_model(dir + "/gemma-3-270m-it-Q8_0.gguf",         "Gemma3-270M",  false);
    test_decision(dir + "/d1-omni-600M-Q8_0.gguf");
    const float gemma_ref[3] = {-0.001069f, -0.025002f, 0.020770f};
    const float nomic_ref[3] = {0.047576f, 0.061704f, -0.118842f};
    test_embedding(dir + "/embeddinggemma-2-UD-Q4_K_XL.gguf", "EmbeddingGemma 2 (embedding model)", true, true, gemma_ref);
    test_embedding(dir + "/nomic-embed-text-v1.5.Q4_K_M.gguf", "nomic-embed-text v1.5 (embedding model)", true, false, nomic_ref);
    fprintf(stderr, "\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
