// kiln-cli: the truth-telling harness. Loads a GGUF, runs a prompt, prints what actually happened.
#include "kiln/kiln.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

static void usage() {
    fprintf(stderr,
        "usage: kiln-cli --probe\n"
        "       kiln-cli -m model.gguf [-p \"prompt\"] [-n max_tokens] [--cpu|--gpu] [-t threads] [-c n_ctx] [--think] [--think-budget N] [--raw] [--chat] [-v]\n"
        "  --chat   interactive multi-turn REPL (shows KV prefix reuse per turn)\n"
        "  --think  enable reasoning on models that support it; reasoning is printed dimmed\n"
        "  --think-budget N  like --think, but close the reasoning block after N tokens so the model answers\n"
        "decision models:\n"
        "       kiln-cli -m model.gguf --state \"text\" --ask noul|choice|score \"question\" [-o \"key=description\"]...\n"
        "  -o       one option: a choice (key=description), a score level (description, lowest first),\n"
        "           or for noul what yes / no mean (yes=..., no=...)\n"
        "embedding models:\n"
        "       kiln-cli -m model.gguf --embed \"text\" [--embed \"text\"]... [--query \"text\"] [--dim N]\n"
        "       kiln-cli -m model.gguf --mmproj projector.gguf --media file [--media file]... [--query \"text\"]\n"
        "  --embed  a text to turn into a vector; prints its first numbers\n"
        "  --media  an image (jpg, png, bmp, gif) or audio file (wav, mp3, flac) to turn into a vector; needs --mmproj\n"
        "  --mmproj the model's image / audio encoders\n"
        "  --image-tokens N  most tokens one image may take (default 280; -1 = the model's own limit)\n"
        "  --query  rank the --embed texts and --media files by how well they match this query, best first\n"
        "  --dim    keep only the first N numbers of each vector\n"
        "speech models:\n"
        "       kiln-cli -m model.gguf --mmproj projector.gguf --transcribe clip.wav [--transcribe clip2.mp3]... [--language English] [-n max_tokens]\n"
        "       kiln-cli -m ggml-base-q5_1.bin --transcribe clip.wav            (Whisper and Parakeet files: no --mmproj)\n"
        "  --transcribe  an audio file (wav, mp3, flac) to write down; Qwen3-ASR needs --mmproj, Whisper and Parakeet do not\n"
        "       kiln-cli -m pocket-tts-en.gguf --mmproj mmproj-pocket-tts-en.gguf --speak \"text\" --voice clip.wav [--wav out.wav] [--language en]\n"
        "  --speak  text to say with a text-to-speech model; --voice is a few seconds of the voice to imitate; writes a 16-bit wav\n"
        "  --language    the language spoken, when known; without it the model works it out\n");
}

static bool on_piece(kiln_channel ch, const char* piece, int32_t, void*) {
    static kiln_channel last = KILN_CHANNEL_ANSWER;
    if (ch != last) { fputs(ch == KILN_CHANNEL_REASONING ? "\033[2m" : "\033[0m\n\n", stdout); last = ch; }
    fputs(piece, stdout); fflush(stdout);
    return true;
}

static void print_probe(const kiln_device_info& d) {
    printf("soc:        %s\n", d.soc);
    printf("abi:        %s\n", d.abi);
    printf("cores:      %d (fast: %d, mask: 0x%llx)\n", d.cpu_cores, d.cpu_big_cores, (unsigned long long)d.cpu_fast_mask);
    printf("ram:        %.1f GB total, %.1f GB available\n", d.ram_total_bytes / 1e9, d.ram_available_bytes / 1e9);
    printf("cpu feats:  fp16=%d dotprod=%d i8mm=%d sve=%d\n", d.has_fp16, d.has_dotprod, d.has_i8mm, d.has_sve);
    printf("gpu:        %s%s\n", d.has_gpu ? "yes - " : "no", d.has_gpu ? d.gpu_name : "");
    printf("npu:        %s%s\n", d.has_npu ? "yes - " : "no", d.has_npu ? d.npu_name : "");
}

static void print_stats(const kiln_gen_stats& st) {
    fprintf(stderr, "\033[0m\nprompt:   %d tok (%d cached) in %.0f ms (%.1f tok/s)\n", st.prompt_tokens, st.cached_tokens, st.prompt_ms, st.prompt_tps);
    fprintf(stderr, "generate: %d tok (%d reasoning%s) in %.0f ms (%.1f tok/s)%s\n", st.generated_tokens, st.reasoning_tokens,
            st.reasoning_truncated ? ", cut at budget" : "", st.gen_ms, st.gen_tps,
            st.stopped_by_eos ? " [eos]" : st.stopped_by_user ? " [stopped]" : "");
}

int main(int argc, char** argv) {
    std::string model, prompt = "Explain in one sentence why the sky is blue.";
    int max_tokens = 128, threads = 0, n_ctx = 0;
    bool probe = false, raw = false, think = false, chat = false;
    int think_budget = 0;
    std::string state, ask_type, ask;
    std::vector<std::string> opt_keys, opt_descs;
    std::vector<std::string> embed_texts, embed_media;
    std::string mmproj;
    std::vector<std::string> clips;
    std::string language;
    std::vector<std::string> speak;
    std::string voice, wav_out = "out.wav";
    bool max_tokens_set = false;
    int image_tokens = 0;
    std::string embed_query;
    int embed_dim = 0;
    kiln_backend_kind backend = KILN_BACKEND_AUTO;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* { if (i + 1 >= argc) { usage(); exit(2); } return argv[++i]; };
        if      (a == "--probe") probe = true;
        else if (a == "-m")      model = next();
        else if (a == "-p")      prompt = next();
        else if (a == "-n")      { max_tokens = atoi(next()); max_tokens_set = true; }
        else if (a == "-t")      threads = atoi(next());
        else if (a == "-c")      n_ctx = atoi(next());
        else if (a == "--cpu")   backend = KILN_BACKEND_CPU;
        else if (a == "--gpu")   backend = KILN_BACKEND_GPU;
        else if (a == "--raw")   raw = true;
        else if (a == "--think") think = true;
        else if (a == "--think-budget") { think = true; think_budget = atoi(next()); }
        else if (a == "--chat")  chat = true;
        else if (a == "--state") state = next();
        else if (a == "--ask")   { ask_type = next(); ask = next(); }
        else if (a == "--embed") embed_texts.push_back(next());
        else if (a == "--query") embed_query = next();
        else if (a == "--media") embed_media.push_back(next());
        else if (a == "--mmproj") mmproj = next();
        else if (a == "--transcribe") clips.push_back(next());
        else if (a == "--language") language = next();
        else if (a == "--speak") speak.push_back(next());
        else if (a == "--voice") voice = next();
        else if (a == "--wav") wav_out = next();
        else if (a == "--image-tokens") image_tokens = atoi(next());
        else if (a == "--dim")   embed_dim = atoi(next());
        else if (a == "-o") {
            std::string o = next(); size_t eq = o.find('=');
            opt_keys.push_back(eq == std::string::npos ? "" : o.substr(0, eq));
            opt_descs.push_back(eq == std::string::npos ? o : o.substr(eq + 1));
        }
        else if (a == "-v")      kiln_set_log_level(KILN_LOG_INFO);
        else if (a == "-vv")     kiln_set_log_level(KILN_LOG_DEBUG);
        else { usage(); return 2; }
    }

    kiln_device_info dev;
    kiln_probe_device(&dev);
    if (probe) { print_probe(dev); return 0; }
    if (model.empty()) { usage(); return 2; }

    kiln_model_params mp = kiln_model_params_default();
    mp.backend = backend; mp.n_threads = threads;
    kiln_model* m = nullptr;
    if (kiln_model_load(model.c_str(), &mp, &m) != KILN_OK) { fprintf(stderr, "load failed: %s\n", kiln_last_error()); return 1; }
    kiln_model_info mi; kiln_model_get_info(m, &mi);

    kiln_context_params cp = kiln_context_params_default();
    cp.n_ctx = n_ctx;
    kiln_context* c = nullptr;
    if (kiln_context_create(m, &cp, &c) != KILN_OK) { fprintf(stderr, "context failed: %s\n", kiln_last_error()); kiln_model_free(m); return 1; }
    // Metal aborts at exit if the model or context is still alive, so every exit from here frees both.
    auto quit = [&](int code) { kiln_context_free(c); kiln_model_free(m); return code; };

    fprintf(stderr, "--- kiln %s ---\n", kiln_version());
    fprintf(stderr, "device:   %s | %s | %d cores | %.1f GB RAM\n", dev.soc, dev.abi, dev.cpu_cores, dev.ram_total_bytes / 1e9);
    fprintf(stderr, "backend:  %s (gpu layers: %d)\n", mi.backend_name, mi.n_gpu_layers_used);
    fprintf(stderr, "model:    %s %s, %.2fB params, %.2f GB resident, ctx_train=%d, thinking=%s\n",
            mi.arch, mi.quant, mi.n_params / 1e9, mi.model_bytes / 1e9, mi.n_ctx_train, mi.is_decision_model ? "n/a (decision model)" : mi.is_embedding_model ? "n/a (embedding model)" : mi.is_speech_model ? "n/a (speech model)" : mi.supports_thinking ? "yes" : "no");

    if (!mmproj.empty()) {
        kiln_projector_info pi;
        kiln_projector_params pp = kiln_projector_params_default();
        pp.image_max_tokens = image_tokens;
        if (kiln_model_load_projector(m, mmproj.c_str(), &pp, &pi) != KILN_OK) { fprintf(stderr, "projector failed: %s\n", kiln_last_error()); return quit(1); }
        fprintf(stderr, "projector: images=%s audio=%s\n", pi.supports_images ? "yes" : "no", pi.supports_audio ? "yes" : "no");
    }

    if (!speak.empty()) {
        struct Wav { std::vector<float> pcm; int32_t rate = 0; int pieces = 0; };
        auto on_audio = [](const float* s, int32_t n, int32_t rate, void* u) {
            auto* w_ = static_cast<Wav*>(u); w_->pcm.insert(w_->pcm.end(), s, s + n); w_->rate = rate; ++w_->pieces;
            fprintf(stderr, "."); return true;
        };
        kiln_speak_options so = kiln_speak_options_default();
        kiln_media ref = {voice.c_str(), nullptr, 0};
        if (!voice.empty()) so.voice = &ref;
        if (!language.empty()) so.language = language.c_str();
        int failed = 0;
        for (size_t si = 0; si < speak.size(); ++si) {
        Wav w;
        kiln_speak_stats ss;
        const kiln_status s = kiln_speak(c, speak[si].c_str(), &so, on_audio, &w, &ss);
        fprintf(stderr, "\n");
        if (s != KILN_OK) { fprintf(stderr, "speak failed: %s\n", kiln_last_error()); ++failed; continue; }
        // 16-bit mono wav; a second --speak writes <name>.2.wav and so on
        std::string out_name = wav_out;
        if (si > 0) { const size_t dot = out_name.rfind('.'); out_name.insert(dot == std::string::npos ? out_name.size() : dot, "." + std::to_string(si + 1)); }
        FILE* f = fopen(out_name.c_str(), "wb");
        if (!f) { fprintf(stderr, "cannot write %s\n", out_name.c_str()); ++failed; continue; }
        const uint32_t n = (uint32_t)w.pcm.size(), bytes = n * 2, rate = (uint32_t)w.rate;
        auto u32 = [&](uint32_t v) { fwrite(&v, 4, 1, f); }; auto u16 = [&](uint16_t v) { fwrite(&v, 2, 1, f); };
        fwrite("RIFF", 1, 4, f); u32(36 + bytes); fwrite("WAVEfmt ", 1, 8, f); u32(16); u16(1); u16(1); u32(rate); u32(rate * 2); u16(2); u16(16);
        fwrite("data", 1, 4, f); u32(bytes);
        for (float v : w.pcm) { int16_t s16 = (int16_t)(std::max(-1.0f, std::min(1.0f, v)) * 32767.0f); fwrite(&s16, 2, 1, f); }
        fclose(f);
        fprintf(stderr, "          %.2f s of audio in %d pieces, %d frames, written to %s | %.0f ms = %.1fx real time | prompt %.0f ms (%d tok), generation %.0f ms%s\n",
                ss.audio_seconds, w.pieces, ss.frames, out_name.c_str(), ss.ms, ss.ms > 0 ? ss.audio_seconds * 1000.0 / ss.ms : 0.0,
                ss.prompt_ms, ss.prompt_tokens, ss.gen_ms, ss.stopped_by_eos ? "" : " [cut]");
        }
        kiln_context_free(c); kiln_model_free(m);
        return failed ? 1 : 0;
    }

    if (!clips.empty()) {
        kiln_transcribe_options to = kiln_transcribe_options_default();
        to.language   = language.empty() ? nullptr : language.c_str();
        to.max_tokens = max_tokens_set ? max_tokens : 0;     // a transcript is as long as the clip; -n only when asked
        int failed = 0;
        for (const std::string& f : clips) {
            kiln_media clip = {f.c_str(), nullptr, 0};
            kiln_transcribe_stats ts;
            const kiln_status s = kiln_transcribe(c, &clip, &to, on_piece, nullptr, &ts);
            printf("\n"); fflush(stdout);
            if (s != KILN_OK) { fprintf(stderr, "transcribe failed: %s\n", kiln_last_error()); ++failed; continue; }
            fprintf(stderr, "          %.1f s of audio (%d of %d prompt tok) in %.0f ms = %.1fx real time | encoder %.0f ms, prompt %.0f ms, %d tok written in %.0f ms%s%s%s\n",
                    ts.audio_seconds, ts.audio_tokens, ts.prompt_tokens, ts.ms, ts.ms > 0 ? ts.audio_seconds * 1000.0 / ts.ms : 0.0,
                    ts.encode_ms, ts.prompt_ms, ts.generated_tokens, ts.gen_ms,
                    ts.language[0] ? " | language: " : "", ts.language, ts.stopped_by_eos ? "" : " [cut]");
        }
        kiln_context_free(c); kiln_model_free(m);
        return failed ? 1 : 0;
    }

    if (!embed_texts.empty() || !embed_media.empty() || !embed_query.empty()) {
        kiln_embed_options eo = kiln_embed_options_default();
        eo.dimensions = embed_dim;
        const int dim = embed_dim > 0 ? embed_dim : mi.n_embd;
        auto run = [&](const std::string& text, kiln_embed_kind kind, std::vector<float>& v, kiln_embed_stats& es) {
            v.assign(dim > 0 ? dim : 1, 0.0f);
            eo.kind = kind;
            if (kiln_embed(c, text.c_str(), &eo, v.data(), (int32_t)v.size(), &es) == KILN_OK) return true;
            fprintf(stderr, "embed failed: %s\n", kiln_last_error());
            return false;
        };
        std::vector<float> qv, v;
        kiln_embed_stats es;
        const bool ranking = !embed_query.empty();
        if (ranking && embed_texts.empty() && embed_media.empty()) {
            fprintf(stderr, "--query needs something to rank: add --embed texts or --media files\n");
            return quit(2);
        }
        std::vector<std::pair<double, std::string>> ranked;     // printed best first at the end
        if (ranking) {
            if (!run(embed_query, KILN_EMBED_QUERY, qv, es)) return quit(1);
            fprintf(stderr, "query:    %d tok in %.1f ms, %d numbers%s\n", es.tokens, es.ms, es.dimensions, es.prefixed ? ", prefix added" : "");
        }
        for (const std::string& t : embed_texts) {
            if (!run(t, ranking ? KILN_EMBED_DOCUMENT : KILN_EMBED_RAW, v, es)) return quit(1);
            if (ranking) {
                double sim = 0;
                for (int i = 0; i < es.dimensions; ++i) sim += (double)qv[i] * v[i];
                ranked.emplace_back(sim, t);
            } else {
                printf("[");
                for (int i = 0; i < es.dimensions && i < 6; ++i) printf("%s%.6f", i ? ", " : "", v[i]);
                printf("%s]  %s\n", es.dimensions > 6 ? ", ..." : "", t.c_str());
            }
            fprintf(stderr, "          %d tok in %.1f ms, %d numbers%s\n", es.tokens, es.ms, es.dimensions, es.prefixed ? ", prefix added" : "");
        }
        for (const std::string& f : embed_media) {
            v.assign(dim > 0 ? dim : 1, 0.0f);
            kiln_media item = {f.c_str(), nullptr, 0};
            if (kiln_embed_media(c, nullptr, &item, 1, &eo, v.data(), (int32_t)v.size(), &es) != KILN_OK) { fprintf(stderr, "embed failed: %s\n", kiln_last_error()); return quit(1); }
            if (ranking) {
                double sim = 0;
                for (int i = 0; i < es.dimensions; ++i) sim += (double)qv[i] * v[i];
                ranked.emplace_back(sim, f);
            } else {
                printf("[");
                for (int i = 0; i < es.dimensions && i < 6; ++i) printf("%s%.6f", i ? ", " : "", v[i]);
                printf("%s]  %s\n", es.dimensions > 6 ? ", ..." : "", f.c_str());
            }
            fprintf(stderr, "          %d tok (%d media) in %.1f ms (encoder %.1f ms), %d numbers\n", es.tokens, es.media_tokens, es.ms, es.media_ms, es.dimensions);
        }
        std::stable_sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        for (const auto& r : ranked) printf("%.3f  %s\n", r.first, r.second.c_str());
        kiln_context_free(c); kiln_model_free(m);
        return 0;
    }

    if (!ask.empty()) {
        kiln_question_type qt = ask_type == "noul" ? KILN_QUESTION_NOUL : ask_type == "choice" ? KILN_QUESTION_CHOICE : KILN_QUESTION_SCORE;
        if (ask_type != "noul" && ask_type != "choice" && ask_type != "score") { usage(); return quit(2); }
        std::vector<kiln_decision_option> opts;
        for (size_t i = 0; i < opt_keys.size(); ++i)
            opts.push_back({opt_keys[i].empty() ? nullptr : opt_keys[i].c_str(), opt_descs[i].c_str()});
        kiln_decision_question q = {"question", qt, ask.c_str(), opts.data(), (int32_t)opts.size()};
        std::vector<float> probs(opts.size() + 2);
        kiln_decision_answer ans;
        if (kiln_decide(c, state.empty() ? nullptr : state.c_str(), &q, probs.data(), (int32_t)probs.size(), &ans) != KILN_OK) {
            fprintf(stderr, "decide failed: %s\n", kiln_last_error()); return quit(1);
        }
        for (int32_t i = 0; i < ans.n_probs; ++i) {
            std::string label = qt == KILN_QUESTION_NOUL ? (i ? "yes" : "no") : qt == KILN_QUESTION_SCORE ? std::to_string(i) + " " + opt_descs[i] : opt_keys[i];
            printf("%s %6.2f%%  %s\n", i == ans.best ? "*" : " ", probs[i] * 100, label.c_str());
        }
        if (qt == KILN_QUESTION_SCORE) printf("expected level: %.2f\n", ans.value);
        fprintf(stderr, "confidence %.2f, %d tok in %.0f ms\n", ans.confidence, ans.prompt_tokens, ans.ms);
        kiln_context_free(c); kiln_model_free(m);
        return 0;
    }

    kiln_sampling_params sp = kiln_sampling_params_default();
    sp.max_tokens = max_tokens;
    kiln_chat_options co = kiln_chat_options_default();
    co.enable_thinking = think;
    co.max_reasoning_tokens = think_budget;
    kiln_gen_stats st;
    kiln_status s = KILN_OK;

    if (raw) {
        s = kiln_generate(c, prompt.c_str(), &sp, on_piece, nullptr, &st);
        print_stats(st);
    } else if (chat) {
        std::vector<std::string> store;            // owns message text
        std::vector<kiln_chat_message> msgs;
        std::string line;
        fprintf(stderr, "(chat mode: type a message, empty line to quit)\n");
        while (true) {
            fprintf(stderr, "\n> "); 
            if (!std::getline(std::cin, line) || line.empty()) break;
            store.push_back(line);
            msgs.push_back({"user", store.back().c_str()});
            // rebuild pointers in case store reallocated
            for (size_t i = 0; i < msgs.size(); ++i) msgs[i].content = store[i].c_str();
            std::string reply;
            auto collect = [](kiln_channel ch, const char* p, int32_t t, void* u) -> bool {
                if (ch == KILN_CHANNEL_ANSWER) static_cast<std::string*>(u)->append(p);
                return on_piece(ch, p, t, nullptr);
            };
            s = kiln_chat(c, msgs.data(), (int)msgs.size(), &co, &sp, collect, &reply, &st);
            print_stats(st);
            if (s != KILN_OK) { fprintf(stderr, "error: %s\n", kiln_last_error()); break; }
            store.push_back(reply);
            msgs.push_back({"assistant", store.back().c_str()});
            for (size_t i = 0; i < msgs.size(); ++i) msgs[i].content = store[i].c_str();
        }
    } else {
        kiln_chat_message msgs[] = { {"user", prompt.c_str()} };
        s = kiln_chat(c, msgs, 1, &co, &sp, on_piece, nullptr, &st);
        print_stats(st);
    }
    if (s != KILN_OK && s != KILN_ERR_CANCELLED) fprintf(stderr, "failed: %s\n", kiln_last_error());

    kiln_context_free(c);
    kiln_model_free(m);
    return s == KILN_OK ? 0 : 1;
}
