// Runs the Rust API against the test models of the C tests (scripts/fetch_models.sh --media
// --speech). A test whose model is missing prints SKIP and passes. KILN_MODELS points
// elsewhere; the default is the repository's models/ folder.
use kiln::*;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

// kiln pins its threads to the fast cores; two models running at once only slow each other.
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn model_path(file: &str) -> Option<PathBuf> {
    let dir = std::env::var("KILN_MODELS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| repo().join("models"));
    let p = dir.join(file);
    if p.exists() {
        Some(p)
    } else {
        eprintln!("SKIP (missing {})", p.display());
        None
    }
}

fn media(file: &str) -> PathBuf {
    repo().join("tests/media").join(file)
}

/// Lowercase letters and digits, one space between words: compares words, not punctuation.
fn words_only(s: &str) -> String {
    s.to_lowercase()
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[test]
fn probe_and_version() {
    let d = probe_device().unwrap();
    assert!(d.cpu_cores > 0 && d.cpu_big_cores > 0 && d.ram_total_bytes > 0, "{d:?}");
    assert!(!version().is_empty());
    eprintln!("{} ({} cores, {} fast), kiln {}", d.soc, d.cpu_cores, d.cpu_big_cores, version());
}

#[test]
fn errors_carry_kilns_message() {
    let e = Model::load("/nonexistent/model.gguf", &ModelParams::default()).unwrap_err();
    assert_eq!(e.status, Status::ModelLoad);
    assert!(!e.message.is_empty(), "{e}");
    let e = Model::load("bad\0path", &ModelParams::default()).unwrap_err();
    assert_eq!(e.status, Status::InvalidArg);
}

#[test]
fn chat_streams_reuses_cache_and_stops() {
    let Some(path) = model_path("gemma-3-270m-it-Q8_0.gguf") else {
        return;
    };
    let _g = serial();
    let model = Model::load(&path, &ModelParams::default()).unwrap();
    let info = model.info();
    assert!(!info.is_decision_model && !info.is_embedding_model && !info.is_speech_model);
    assert_eq!(info.arch, "gemma3");

    // tokenizer and template
    let toks = model.tokenize("Hello, world", false).unwrap();
    assert!(!toks.is_empty());
    assert_eq!(model.detokenize(&toks).unwrap(), "Hello, world");
    let prompt = model
        .apply_chat_template(&[Message::user("Name a fruit.")], &ChatOptions::default())
        .unwrap();
    assert!(prompt.contains("Name a fruit."), "{prompt}");

    let mut ctx = model.context(&ContextParams::default()).unwrap();
    let greedy = Sampling {
        max_tokens: 24,
        ..Sampling::greedy()
    };
    let mut turn = vec![Message::user("Name three colours, separated by commas.")];

    let mut pieces = 0;
    let mut streamed = String::new();
    let stats = ctx
        .chat(&turn, &ChatOptions::default(), &greedy, |p| {
            assert_eq!(p.channel, Channel::Answer);
            streamed.push_str(p.text);
            pieces += 1;
            true
        })
        .unwrap();
    assert!(!streamed.trim().is_empty() && pieces > 1);
    assert!(
        stats.generated_tokens > 0 && stats.prompt_tokens > 5 && stats.gen_tps > 0.0,
        "{stats:?}"
    );
    eprintln!("reply: {streamed:?} ({} tok at {:.0} tok/s)", stats.generated_tokens, stats.gen_tps);

    // the same question again, from a fresh cache, gives the same greedy text
    ctx.reset().unwrap();
    let again = ctx.chat_collect(&turn, &ChatOptions::default(), &greedy).unwrap();
    assert_eq!(again.answer, streamed);

    // the next turn reuses the cached conversation
    turn.push(Message::assistant(again.answer));
    turn.push(Message::user("And one more?"));
    let next = ctx.chat_collect(&turn, &ChatOptions::default(), &greedy).unwrap();
    assert!(next.stats.cached_tokens > 0, "{:?}", next.stats);

    // returning false stops: Ok, with stopped_by_user
    let mut n = 0;
    let stats = ctx
        .chat(
            &[Message::user("Count from 1 to 50.")],
            &ChatOptions::default(),
            &Sampling::greedy(),
            |_| {
                n += 1;
                n < 3
            },
        )
        .unwrap();
    assert!(stats.stopped_by_user && n == 3, "{stats:?} n={n}");
}

#[test]
fn cancel_from_another_thread_and_panics() {
    let Some(path) = model_path("gemma-3-270m-it-Q8_0.gguf") else {
        return;
    };
    let _g = serial();
    let model = Model::load(&path, &ModelParams::default()).unwrap();
    let mut ctx = model.context(&ContextParams::default()).unwrap();
    let long = [Message::user("Write a long story about a lighthouse keeper.")];
    let sampling = Sampling {
        max_tokens: 2000,
        ..Sampling::greedy()
    };

    let handle = ctx.cancel_handle();
    let mut first = true;
    let err = ctx
        .chat(&long, &ChatOptions::default(), &sampling, |_| {
            if first {
                first = false;
                let h = handle.clone();
                std::thread::spawn(move || h.cancel()).join().unwrap();
            }
            true
        })
        .unwrap_err();
    assert_eq!(err.status, Status::Cancelled);

    // a panic in the closure reaches the caller, and the context still works
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = ctx.chat(&long, &ChatOptions::default(), &sampling, |_| panic!("from the closure"));
    }));
    assert!(caught.is_err());
    let ok = ctx.chat_collect(
        &[Message::user("Say hi.")],
        &ChatOptions::default(),
        &Sampling {
            max_tokens: 8,
            ..Sampling::greedy()
        },
    );
    assert!(ok.is_ok(), "{ok:?}");

    // a handle kept past the context does nothing
    drop(ctx);
    handle.cancel();
}

#[test]
fn thinking_goes_to_its_own_channel() {
    let Some(path) = model_path("Qwen3.5-0.8B-Q4_K_M.gguf") else {
        return;
    };
    let _g = serial();
    let model = Model::load(&path, &ModelParams::default()).unwrap();
    assert!(model.info().supports_thinking);
    let mut ctx = model.context(&ContextParams::default()).unwrap();
    let opts = ChatOptions {
        enable_thinking: true,
        max_reasoning_tokens: 32,
        ..ChatOptions::default()
    };
    let r = ctx
        .chat_collect(
            &[Message::user("What is 12 times 7?")],
            &opts,
            &Sampling {
                max_tokens: 128,
                ..Sampling::greedy()
            },
        )
        .unwrap();
    assert!(!r.reasoning.trim().is_empty() && r.stats.reasoning_tokens > 0, "{r:?}");
    assert!(r.stats.reasoning_tokens <= 33, "{:?}", r.stats);
    assert!(!r.answer.contains("<think>") && !r.reasoning.contains("<think>"));
    eprintln!("reasoning {} tok, answer {:?}", r.stats.reasoning_tokens, r.answer);
}

#[test]
fn decision_model() {
    let Some(path) = model_path("d1-omni-600M-Q8_0.gguf") else { return };
    let _g = serial();
    let model = Model::load(&path, &ModelParams::default()).unwrap();
    assert!(model.info().is_decision_model);
    let mut ctx = model
        .context(&ContextParams {
            n_ctx: 4096,
            ..ContextParams::default()
        })
        .unwrap();
    let state = "Hi, I ordered a blender two weeks ago and it arrived broken. I have emailed twice and nobody \
                 answered. I want my money back today or I will dispute the charge with my bank.";
    let yes_no = |id: &str, q: &str| Question {
        id: id.into(),
        kind: QuestionType::YesNo,
        instructions: q.into(),
        options: vec![],
    };

    let refund = ctx
        .decide(Some(state), &yes_no("wants_refund", "Is the customer asking for a refund?"))
        .unwrap();
    assert_eq!(refund.probs.len(), 2);
    assert!((refund.probs.iter().sum::<f32>() - 1.0).abs() < 1e-3);
    assert!(refund.value > 0.5 && refund.best == 1, "{refund:?}");
    let weather = ctx
        .decide(Some(state), &yes_no("about_weather", "Is the customer asking about the weather?"))
        .unwrap();
    assert!(weather.value < 0.5 && weather.best == 0, "{weather:?}");

    let team = Question {
        id: "team".into(),
        kind: QuestionType::Choice,
        instructions: "Which team should handle this?".into(),
        options: vec![
            DecisionOption::new("billing", Some("Payments, refunds and invoices")),
            DecisionOption::new("shipping", Some("Delivery status and lost parcels")),
            DecisionOption::new("technical", Some("The product does not work as expected")),
        ],
    };
    let t = ctx.decide(Some(state), &team).unwrap();
    assert!(t.probs.len() == 3 && t.best < 3 && (0.0..=1.0).contains(&t.confidence), "{t:?}");
    assert_eq!(ctx.decide(Some(state), &team).unwrap().probs, t.probs, "not deterministic");
    eprintln!("refund? yes={:.3}; team: {}", refund.value, team.options[t.best].key);
}

#[test]
fn embeddings_rank_and_truncate() {
    let Some(path) = model_path("embeddinggemma-2-UD-Q4_K_XL.gguf") else {
        return;
    };
    let _g = serial();
    let model = Model::load(&path, &ModelParams::default()).unwrap();
    let n = model.info().n_embd as usize;
    assert!(model.info().is_embedding_model && n > 0);
    let mut ctx = model.context(&ContextParams::default()).unwrap();

    let q = ctx.embed("when does the train to London leave", &EmbedOptions::query()).unwrap();
    assert_eq!(q.vector.len(), n);
    assert!(q.stats.prefixed);
    assert!((dot(&q.vector, &q.vector) - 1.0).abs() < 1e-4);
    let train = ctx
        .embed("The train to London leaves at nine in the morning.", &EmbedOptions::document())
        .unwrap();
    let flour = ctx
        .embed("Add two cups of flour and one egg to the bowl.", &EmbedOptions::document())
        .unwrap();
    assert!(dot(&q.vector, &train.vector) > dot(&q.vector, &flour.vector));

    let short = ctx
        .embed(
            "hello",
            &EmbedOptions {
                dimensions: 256,
                ..EmbedOptions::default()
            },
        )
        .unwrap();
    assert_eq!(short.vector.len(), 256);
    assert_eq!(short.stats.dimensions, 256);
    eprintln!(
        "train {:.3}, flour {:.3}",
        dot(&q.vector, &train.vector),
        dot(&q.vector, &flour.vector)
    );
}

#[test]
fn image_embeddings_from_file_and_bytes() {
    let (Some(path), Some(proj)) = (
        model_path("embeddinggemma-2-UD-Q4_K_XL.gguf"),
        model_path("mmproj-embeddinggemma-2-Q8_0.gguf"),
    ) else {
        return;
    };
    let _g = serial();
    let mut model = Model::load(&path, &ModelParams::default()).unwrap();
    let pi = model.load_projector(&proj, &ProjectorParams::default()).unwrap();
    assert!(pi.supports_images && !pi.generates_audio);
    let mut ctx = model.context(&ContextParams::default()).unwrap();

    let red_path = media("red_circle.png");
    let blue_path = media("blue_square.png");
    let red = ctx.embed_media(None, &[Media::path(&red_path)], &EmbedOptions::default()).unwrap();
    assert!(red.stats.media_tokens > 0 && red.stats.media_ms > 0.0, "{:?}", red.stats);
    let bytes = std::fs::read(&red_path).unwrap();
    let red2 = ctx.embed_media(None, &[Media::Bytes(&bytes)], &EmbedOptions::default()).unwrap();
    assert_eq!(red.vector, red2.vector, "bytes and file differ");
    let blue = ctx.embed_media(None, &[Media::path(&blue_path)], &EmbedOptions::default()).unwrap();
    let q = ctx.embed("a red circle", &EmbedOptions::query()).unwrap();
    assert!(dot(&q.vector, &red.vector) > dot(&q.vector, &blue.vector));

    let mixed = ctx
        .embed_media(
            Some(&format!("A shape {MEDIA_MARKER} on white")),
            &[Media::path(&red_path)],
            &EmbedOptions::default(),
        )
        .unwrap();
    assert!(mixed.stats.tokens > red.stats.tokens);
}

fn transcribe_train_clip(file: &str, expect_lang: &str) {
    let Some(path) = model_path(file) else { return };
    let _g = serial();
    let model = Model::load(&path, &ModelParams::default()).unwrap();
    assert!(
        model.info().is_speech_model && model.info().backend_name.starts_with("whisper.cpp/"),
        "{:?}",
        model.info()
    );
    let mut ctx = model.context(&ContextParams::default()).unwrap();
    let clip = media("speech_train.wav");

    let t = ctx.transcribe_collect(&Media::path(&clip), &TranscribeOptions::default()).unwrap();
    let w = words_only(&t.text);
    assert!(
        w == "the train to london leaves at nine in the morning" || w == "the train to london leaves at 9 in the morning",
        "got {:?}",
        t.text
    );
    assert_eq!(t.stats.language, expect_lang);
    assert!(
        t.stats.audio_seconds > 3.0 && t.stats.audio_seconds < 3.3 && t.stats.stopped_by_eos,
        "{:?}",
        t.stats
    );

    let bytes = std::fs::read(&clip).unwrap();
    let from_bytes = ctx
        .transcribe_collect(&Media::Bytes(&bytes), &TranscribeOptions::default())
        .unwrap();
    assert_eq!(from_bytes.text, t.text);
    eprintln!(
        "{file}: {:?}, {:.1} s of audio in {:.0} ms",
        t.text, t.stats.audio_seconds, t.stats.ms
    );
}

#[test]
fn transcribe_parakeet() {
    transcribe_train_clip("ggml-parakeet-tdt-0.6b-v3-q8_0.bin", "");
}

#[test]
fn transcribe_whisper() {
    transcribe_train_clip("ggml-base-q5_1.bin", "English");
}

#[test]
fn speak_and_read_back() {
    let (Some(path), Some(proj)) = (model_path("pocket-tts-en.gguf"), model_path("mmproj-pocket-tts-en.gguf")) else {
        return;
    };
    let _g = serial();
    let mut model = Model::load(&path, &ModelParams::default()).unwrap();
    let pi = model.load_projector(&proj, &ProjectorParams::default()).unwrap();
    assert!(pi.generates_audio && pi.sample_rate == 24000, "{pi:?}");
    let mut ctx = model.context(&ContextParams::default()).unwrap();
    let voice = media("speech_train.wav");
    let opts = SpeakOptions {
        voice: Some(Media::path(&voice)),
        seed: Some(7),
        ..SpeakOptions::default()
    };
    let text = "The train to London leaves at nine in the morning.";

    let mut pieces = 0;
    let speech = ctx.speak_collect(text, &opts).unwrap();
    assert_eq!(speech.sample_rate, 24000);
    let secs = speech.samples.len() as f64 / 24000.0;
    assert!(
        secs > 1.5 && secs < 8.0 && speech.stats.stopped_by_eos,
        "{secs:.2} s, {:?}",
        speech.stats
    );
    let peak = speech.samples.iter().fold(0f32, |m, v| m.max(v.abs()));
    assert!(peak > 0.05 && peak <= 1.0, "peak {peak}");

    // stop from the closure after the first piece
    let stats = ctx
        .speak(text, &opts, |_, _| {
            pieces += 1;
            false
        })
        .unwrap();
    assert!(stats.stopped_by_user && pieces == 1, "{stats:?}");
    eprintln!("{secs:.2} s of audio in {:.0} ms", speech.stats.ms);

    // the honest check: Parakeet writes the words down again
    let Some(asr) = model_path("ggml-parakeet-tdt-0.6b-v3-q8_0.bin") else {
        return;
    };
    let wav = write_wav(&speech.samples, speech.sample_rate);
    let asr = Model::load(&asr, &ModelParams::default()).unwrap();
    let heard = asr
        .context(&ContextParams::default())
        .unwrap()
        .transcribe_collect(&Media::Bytes(&wav), &TranscribeOptions::default())
        .unwrap();
    let w = words_only(&heard.text);
    assert!(w.contains("train") && w.contains("london"), "Parakeet heard {:?}", heard.text);
    eprintln!("read back: {:?}", heard.text);
}

/// 16-bit mono PCM wav in memory.
fn write_wav(samples: &[f32], rate: u32) -> Vec<u8> {
    let data_len = (samples.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&(rate * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for s in samples {
        out.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
    }
    out
}
