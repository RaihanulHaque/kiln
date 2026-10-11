// Core behaviour tests. Needs real models: pass a directory containing
//   LiquidAI_LFM2.5-350M_Q4_K_M.gguf, Qwen3.5-0.8B-Q4_K_M.gguf, gemma-3-270m-it-Q8_0.gguf
// Missing models are skipped, not failed. Runs on host and on Android via scripts/android-test.sh.
#include "kiln/kiln.h"
#include <atomic>
#include <chrono>
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
        auto sp6 = kiln_sampling_params_default(); sp6.max_tokens = 3000; sp6.seed = 7;   // greedy loops forever in think mode on 0.8B
        s = kiln_chat(c, t6, 1, &think, &sp6, Capture::cb, &cap6, &st6);
        CHECK(s == KILN_OK, "%s", kiln_last_error());
        CHECK(st6.reasoning_tokens > 0, "thinking on but no reasoning tokens");
        CHECK(cap6.answer.find("<think>") == std::string::npos && cap6.answer.find("</think>") == std::string::npos, "tags leaked into answer: %s", cap6.answer.c_str());
        CHECK(cap6.reasoning.find("<think>") == std::string::npos && cap6.reasoning.find("</think>") == std::string::npos, "tags leaked into reasoning");
        CHECK(st6.reasoning_tokens == 0 || !cap6.reasoning.empty(), "reasoning tokens counted but nothing emitted");
        fprintf(stderr, "  think: reasoning=%d tok, answer='%.60s'\n", st6.reasoning_tokens, cap6.answer.c_str());
        if (st6.stopped_by_eos) {
            CHECK(!cap6.answer.empty(), "finished but empty answer channel");
            CHECK(cap6.answer.find("391") != std::string::npos, "answer='%s'", cap6.answer.c_str());
        } else fprintf(stderr, "  (did not finish thinking within budget; answer checks skipped)\n");
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

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : "models";
    kiln_set_log_level(KILN_LOG_ERROR);
    test_probe();
    test_bad_loads(dir);
    //                                                       label           thinking
    test_model(dir + "/LiquidAI_LFM2.5-350M_Q4_K_M.gguf", "LFM2.5-350M",  false);
    test_model(dir + "/Qwen3.5-0.8B-Q4_K_M.gguf",         "Qwen3.5-0.8B", true);
    test_model(dir + "/gemma-3-270m-it-Q8_0.gguf",         "Gemma3-270M",  false);
    fprintf(stderr, "\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
