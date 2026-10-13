# Speech models compared: Whisper, Moonshine, Parakeet, Qwen3-ASR

One run on one machine, to see how the speech model kiln runs today (Qwen3-ASR 0.6B) compares
with the small models people usually reach for. It answers two questions: which model writes
down English best, and why kiln cannot simply load Whisper or Parakeet.

**Where this was run:** Linux x86_64, Ryzen 5 8500G, CPU only, 8 threads, 2026-10-13. Not run
on macOS, Android or iOS. Section 6 has the steps to repeat it on a Mac.

**How far to trust it:** the numbers below were written down from the run's output. The folder
with the models, the audio, the scripts and every transcript was a temporary one and was lost
when the machine restarted, so the raw data cannot be checked any more. Treat the table as a
first look, and repeat it before deciding anything on it.

## 1. What was compared

| model | what it is | file used | run with |
|---|---|---|---|
| Whisper tiny, base, small | OpenAI's models, 39 to 244 million weights, 99 languages | `ggml-tiny.bin`, `ggml-base.bin`, `ggml-base-q5_1.bin`, `ggml-small.bin`, `ggml-small-q5_1.bin` from `ggerganov/whisper.cpp` | whisper.cpp (commit 0c7be69), `whisper-cli` |
| Moonshine base | a 62-million-weight model built for short commands, English | `csukuangfj/sherpa-onnx-moonshine-base-en-int8` | sherpa-onnx 1.13.8 (ONNX Runtime) |
| Parakeet TDT 0.6B v2 | NVIDIA's 600-million-weight model, English | `csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8` | sherpa-onnx 1.13.8 (ONNX Runtime) |
| Qwen3-ASR 0.6B | an audio encoder plus a small Qwen3 text model, many languages | `Qwen3-ASR-0.6B-Q8_0.gguf` and its `mmproj` from `ggml-org/Qwen3-ASR-0.6B-GGUF` | **kiln 0.4.0**, `kiln-cli --transcribe` |

"q5_1" and "int8" mean the weights are stored with fewer bits (5 and 8) to make the file smaller.

Only Qwen3-ASR ran through kiln. The others ran in their own programs, because kiln cannot
load them (section 5).

## 2. The recordings

All real speech, none of it generated.

| set | what | size |
|---|---|---|
| clean | LibriSpeech, read audiobooks, one speaker (`hf-internal-testing/librispeech_asr_dummy`, config `clean`) | 73 clips, 481 s, 1150 words |
| other | LibriSpeech `test-other`, the harder half: 12 speakers, 5 clips from each of 12 places in `openslr/librispeech_asr` | 60 clips, 305 s, 853 words |
| long | the three chapters of the clean set, each chapter's clips joined in order | 145 s, 135 s and 201 s; 335, 343 and 472 words |
| long, no correct text | a speech by George W. Bush on the Columbia disaster (Wikimedia Commons); a toy advert spoken very fast ("Micro Machines", from OpenAI's Whisper samples); the JFK clip from whisper.cpp | 199 s, 30 s, 11 s |

Every file was converted to 16 kHz, one channel, 16-bit wav first.

## 3. Results

**WER** (word error rate) is the share of words a model got wrong: 5% means one word in
twenty. Lower is better. Before counting, both texts go through the same clean-up
(`whisper-normalizer`: lower case, no punctuation, "nine" and "9" made equal), so a missing
comma is not an error.

**Speed** is seconds of audio divided by seconds of work: 6x means a 6-second clip takes one
second. Model loading is not counted.

| model | size on disk | WER clean | WER other | WER long chapters | speed, short clips | speed, long clips | peak memory |
|---|---|---|---|---|---|---|---|
| Whisper tiny | 78 MB | 10.2% | 12.8% | 9.2% | 20x | 40x | 0.3 GB |
| Whisper base | 148 MB | 7.8% | 8.9% | 7.2% | 9x | 22x | 0.4 GB |
| Whisper base, 5-bit | 60 MB | 8.3% | 8.2% | 7.3% | 10x | 26x | 0.3 GB |
| Whisper small, 5-bit | 190 MB | 5.7% | 6.2% | 7.4% | 3.0x | 9x | 0.6 GB |
| Whisper small | 488 MB | 5.8% | 6.0% | 6.5% | 2.7x | 8x | 0.9 GB |
| Moonshine base, 8-bit | 286 MB | 6.4% | 5.7% | fails (169%) | 26x | 2.4x | 8.2 GB |
| Parakeet TDT 0.6B v2, 8-bit | 661 MB | **2.4%** | **2.1%** | **2.3%** | 20x | 12x | 2.9 GB |
| Qwen3-ASR 0.6B, 8-bit, in kiln | 1019 MB | 3.8% | 3.9% | 3.5% | 5.9x | 5.5x | 1.7 GB |

Peak memory is the highest of the three sets, which is the long one for every model. Parakeet
needed 1.1 GB on the short clips and 2.9 GB on a 3-minute clip given in one piece.

Qwen3-ASR in kiln on the long files, one call each: the 199-second speech in 37 s, the three
chapters in 24, 25 and 40 s. All of them fit in the default context of 4096 tokens.

The Bush speech has no correct text, so the models can only be compared with each other.
Against Parakeet's transcript: Qwen3-ASR differs in 0.8% of the words, Whisper tiny in 3.2%,
Whisper small 5-bit in 4.5%, Whisper base in 5.1%, and Whisper base 5-bit in 19%, because it
left out 63 words in a row.

## 4. What the results say

- **Parakeet is the best for English**: fewest errors on every set, and three to four times
  faster than Qwen3-ASR on short clips.
- **Qwen3-ASR is second.** About one and a half times Parakeet's errors. It is the largest
  file and the slowest for its quality, but it is the only one of the four families that
  kiln runs, and it took 3-minute recordings in one call without losing text.
- **Whisper is slow on short clips.** It always works on 30 seconds of sound, so a 3-second
  command is padded and costs as much as a 30-second one. That is why its "short clips" speed
  is two to three times lower than its "long clips" speed.
- **Whisper loses text on long audio.** Whisper small 5-bit left out two whole sentences of
  one chapter, and Whisper base 5-bit left out 63 words of the Bush speech.
- **Moonshine is for short clips only.** It was fast and decent on clips of a few seconds. On
  the long files it wrote one sentence over and over and used 8 GB of memory.

Examples. A clip from the "other" set; the correct text is *if you're clean they say honest
poverty, and if you're dirty they say serve you right*:

| model | wrote |
|---|---|
| Whisper tiny | If you are clean they say "Unspoverty" and if you are dirty they say "Serve you right". |
| Whisper base | If you're clean, they say, "On his poverty." And if you're dirty, they say, "Serve you right." |
| Whisper small, Moonshine, Parakeet, Qwen3-ASR | honest poverty (all four correct) |

The fast toy advert; the words are *dramatic details, terrific trim, precision paint jobs*:

| model | wrote |
|---|---|
| Whisper tiny | dramatic-to-tails for effect-driven position-page-out |
| Whisper base | dramatic details for a fixed room position, page art |
| Whisper small, 5-bit | dramatic details, terrific trims, precision paint jobs |
| Moonshine base | dramatic details for a friction, precision page on |
| Parakeet | dramatic details, perfect trim, precision, paint jobs |
| Qwen3-ASR | dramatic details, terrific trim, precision paint jobs |

**One real failure in kiln.** A 2.9-second clip that says "Truly, truly, Fan" came back from
Qwen3-ASR as Chinese: 朱莉，朱莉，凡。 The model guesses the language from the sound, and three
short words were not enough. With `--language English` (`kiln_transcribe_options.language`)
it answered in English ("Julie, Julie Fan.", still the wrong words, as Whisper tiny also
wrote). An app that knows which language is spoken should always pass it.

Limits of this run:

- The sets are small. With 1150 and 853 words, one or two errors move a score by 0.1 to 0.2
  points. The order of the models is clear; the decimals are not.
- English only. Parakeet v2 and Moonshine are English models; Whisper and Qwen3-ASR are built
  for many languages, and that was not tested.
- The desktop was in use during the run. Each model was run twice and the faster time is
  shown. Most agreed within 5%; Whisper base varied up to three times between the two runs,
  so its speed is approximate.
- Whisper ran with whisper.cpp's default search (it keeps the 5 best candidates while
  writing); the others always take the single most likely word.
- No phone. The phone speed is what decides whether a model is usable for voice input.

## 5. Why kiln cannot load Whisper or Parakeet

kiln has one engine, llama.cpp. llama.cpp runs one kind of model: a text model that reads
tokens and writes the next one. Sound gets in only through an encoder (the `mmproj` file) that
turns it into vectors the text model reads as if they were words.

**Qwen3-ASR has exactly that shape**: an audio encoder, then a small Qwen3 chat model that
writes the sentence. llama.cpp runs both halves, so `kiln_transcribe` works.

**Whisper** is an encoder followed by its own decoder, which is not a text model llama.cpp
knows. llama.cpp contains only the first half (`tools/mtmd/models/whisper-enc.cpp`), because
other models (Ultravox, Voxtral, Qwen2-Audio) use Whisper's encoder as their ears. Nothing in
llama.cpp turns that encoder's output into words. Whisper files are also not GGUF: whisper.cpp
has its own format (`ggml-base.bin`), and `kiln_model_load` refuses a file that is not GGUF
(`KILN_ERR_MODEL_LOAD`).

**Parakeet** has no text model at all. It uses another method, called a transducer: for each
slice of sound a tiny network decides whether to write a piece of a word or to move on to the
next slice. llama.cpp has no code for that loop. The file `tools/mtmd/models/parakeet.cpp` in
llama.cpp is the encoder of one large NVIDIA chat model (Nemotron 3 Nano Omni), not a way to
run Parakeet. Parakeet's own files are ONNX, not GGUF.

What adding them would take. kiln was built for more than one engine: `src/core/internal.h`
has the `Backend` interface and `src/core/registry.cpp` the list.

| model | engine to add | what it costs |
|---|---|---|
| Whisper | whisper.cpp | The smaller step: it is built on ggml, the same maths library as llama.cpp. But two copies of ggml in one library have to be kept apart, and Whisper was the weakest family here. |
| Parakeet | ONNX Runtime, through sherpa-onnx | A second, unrelated engine: a larger app (the Android library alone is in the tens of megabytes; not measured here), its own build for Android and iOS, and model files that are not GGUF. It was the best model here. |

Neither was tried. The sensible order is to measure Qwen3-ASR on the Pixel 8 first
(`KILN_BENCH_ONLY=asr scripts/bench.sh`). If it is fast enough there, no second engine is
needed. If it is not, Parakeet is the one worth the work.

## 6. Repeating it on a Mac

Not run on macOS: these are the steps used on Linux, with the Mac differences noted. The
helper scripts of the Linux run were lost with the temporary folder, so this is by hand.

One thing to keep in mind when reading Mac numbers: kiln and whisper.cpp use the GPU (Metal)
on a Mac by default, and the sherpa-onnx package from pip runs on the CPU. So speeds on the
Mac compare engines on different hardware. Add `--cpu` to `kiln-cli` and `-ng` to
`whisper-cli` for a CPU against CPU comparison. Close other apps first.

**kiln and Qwen3-ASR** (this is also the first macOS run of 0.4.0):

```bash
git fetch && git checkout feat/transcribe
cmake --preset host && cmake --build --preset host -j
scripts/fetch_models.sh --media --speech
./build/host/kiln-tests models                    # 398 passed on Linux x86
./build/host/kiln-cli -m models/Qwen3-ASR-0.6B-Q8_0.gguf --mmproj models/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf \
    --transcribe tests/media/speech_train.wav --transcribe third_party/llama.cpp/tools/mtmd/test-2.mp3
```

Each clip prints its text, then a line with the time and "x real time".

**Whisper**, in a folder outside the repo:

```bash
git clone --depth 1 https://github.com/ggml-org/whisper.cpp && cd whisper.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target whisper-cli -j
curl -L -o ggml-base.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin
./build/bin/whisper-cli -m ggml-base.bin -l en -nt -otxt clip1.wav clip2.wav    # writes clip1.wav.txt, ...
```

It wants 16 kHz wav: `ffmpeg -i in.mp3 -ar 16000 -ac 1 -c:a pcm_s16le out.wav`. The last lines
it prints give `load time` and `total time`; work time is the difference.

**Parakeet** (and Moonshine), in the same kind of folder:

```bash
python3 -m venv venv && ./venv/bin/pip install sherpa-onnx numpy
R=https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/resolve/main
mkdir parakeet && for f in tokens.txt joiner.int8.onnx decoder.int8.onnx encoder.int8.onnx; do curl -L -o parakeet/$f $R/$f; done
./venv/bin/python - clip1.wav clip2.wav <<'EOF'
import sys, time, wave, numpy as np, sherpa_onnx
rec = sherpa_onnx.OfflineRecognizer.from_transducer(
    encoder="parakeet/encoder.int8.onnx", decoder="parakeet/decoder.int8.onnx", joiner="parakeet/joiner.int8.onnx",
    tokens="parakeet/tokens.txt", num_threads=8, model_type="nemo_transducer")
for f in sys.argv[1:]:
    w = wave.open(f)                                   # 16 kHz, one channel, 16-bit
    x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    t = time.time(); s = rec.create_stream(); s.accept_waveform(16000, x); rec.decode_stream(s)
    print(f"{len(x) / 16000:.1f} s in {time.time() - t:.2f} s: {s.result.text}")
EOF
```

**Recordings with correct texts.** The clean set comes from the Hugging Face rows API, which
returns each clip's text and a link to its audio:
`https://datasets-server.huggingface.co/rows?dataset=hf-internal-testing/librispeech_asr_dummy&config=clean&split=validation&offset=0&length=100`.
The "other" set is the same call with `dataset=openslr/librispeech_asr&config=other&split=test`.
The clips of one chapter (the middle number of the id, as in `1272-128104-0003`) joined in
order make a long recording with a known text.

**Scoring**: `pip install jiwer whisper-normalizer`, pass both texts through
`whisper_normalizer.english.EnglishTextNormalizer()`, then `jiwer.wer(correct, written)`.

If this is going to be repeated more than once, it should become a script in `scripts/` that
is run and checked on the machine it is written on.
