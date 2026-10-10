# Codebase walkthrough

What each file does and what happens during one chat call. Pair with
[CONCEPTS.md](CONCEPTS.md) for the ideas and [ARCHITECTURE.md](ARCHITECTURE.md) for the rules.

## Layout

```
include/kiln/kiln.h                 the public C API. The only header bindings see.
src/core/internal.h                 C++ interfaces Backend / Model / Context that backends implement
src/core/kiln.cpp                   C API functions: validate args, pick a backend, forward
src/core/error.cpp                  thread-local last-error string, logging
src/core/registry.cpp               ordered list of backends (today: llama.cpp only)
src/probe/probe.cpp                 device probe: SoC, cores, clocks -> fast-core mask, RAM, ISA bits
src/backends/llamacpp/
    llamacpp_backend.cpp            the one backend: load, template, generate, threadpool pinning
    reasoning_splitter.h            splits streamed text into answer / reasoning by think tags
tools/kiln-cli/main.cpp             command-line harness; --chat for multi-turn, --think
tools/kiln-bench/main.cpp           resource sampler + repeated-run benchmark
tests/test_core.cpp                 the C test suite (host and device)
flutter/kiln_flutter/
    src/CMakeLists.txt              Gradle entry: builds the core from the repo root
    ffigen.yaml                     generates lib/src/kiln_bindings_generated.dart from kiln.h
    lib/src/native.dart             loads libkiln for the platform
    lib/src/types.dart              Dart mirrors of the C structs, map <-> native
    lib/src/worker.dart             background isolate that owns handles and runs blocking calls
    lib/src/kiln.dart               public API: Kiln, KilnModel, KilnContext, KilnGeneration
    example/lib/main.dart           the demo chat app
scripts/                            android-run / android-test / bench / bench_plot / bench_compare / brand / make_promo
third_party/llama.cpp               git submodule, pinned to a tag
```

## The three objects

- **Model** (`kiln_model`): the weights, tokenizer, chat template, and the device decision
  (backend, thread count, pin mask). Created by `kiln_model_load`. Expensive. Share it.
- **Context** (`kiln_context`): one conversation's KV cache plus a pinned ggml threadpool.
  Created from a model. Cheap-ish. One per chat; a context runs one generation at a time.
- **Generation**: not an object in C (a blocking call), but in Dart it is `KilnGeneration`:
  a stream of pieces, a future of stats, and `cancel()`.

## One `kiln_chat` call, top to bottom

```mermaid
sequenceDiagram
    participant UI as Flutter UI (main isolate)
    participant W as worker isolate
    participant C as kiln.cpp (C API)
    participant M as LlamaModel
    participant X as LlamaContext
    participant G as llama.cpp / ggml

    UI->>W: chat {messages, enableThinking, sampling, piecePort}
    Note over UI: returns KilnGeneration immediately
    W->>C: kiln_chat(ctx, msgs, opts, params, cb, stats)
    C->>M: render_chat(msgs, opts)
    M->>G: common_chat_templates_apply (minja jinja)
    G-->>M: prompt text (+ think block open?)
    M-->>C: RenderedPrompt
    C->>X: generate(rendered, params, cb)
    X->>G: llama_tokenize
    X->>X: longest common prefix vs KV cache
    X->>G: llama_memory_seq_rm (or clear if refused)
    loop prompt eval, batches of n_batch
        X->>G: llama_decode(batch)
    end
    loop until EOS / max_tokens / cancel / context full
        X->>G: llama_sampler_sample
        X->>G: llama_token_to_piece
        X->>X: ReasoningSplitter.feed(piece)
        X-->>W: cb(channel, text)  (same thread)
        W-->>UI: piecePort.send([channel, text])
        X->>G: llama_decode(token)
    end
    X-->>C: status + kiln_gen_stats
    C-->>W: return
    W-->>UI: reply {stats}
    Note over UI: stream closes, stats future completes
    UI->>C: kiln_cancel(ctx) (any time, any thread)
```

1. **Dart** `KilnContext.chat(history)` sends a message to the worker isolate with the
   message list, options, sampling params, and a port for pieces. Returns immediately with a
   `KilnGeneration` wrapping that port's stream.
2. **Worker isolate** (`worker.dart`) marshals the messages into C structs, creates a
   `NativeCallable.isolateLocal` for the token callback, and calls `kiln_chat`. The isolate
   is now blocked inside C until generation ends.
3. **C API** (`kiln.cpp`) validates handles and asks the model to **render** the chat:
   `Model::render_chat` in the backend.
4. **Template** (`llamacpp_backend.cpp`, `render_chat`): builds a `common_chat_templates_inputs`
   with the messages and `enable_thinking`, runs the model's jinja template through
   llama.cpp's minja. Output: the prompt string, plus whether the prompt already opened a
   think block (so the splitter starts in reasoning mode).
5. **Prefix reuse** (`LlamaContext::generate`): tokenizes the prompt, compares with the token
   list currently in the KV cache, keeps the common prefix. Tries `llama_memory_seq_rm` to drop
   the rest; if the model refuses (recurrent state), clears everything and starts from zero.
6. **Prompt eval**: decodes the non-cached tokens in batches of `n_batch` on the pinned
   threadpool. The abort callback checks the cancel flag between graph nodes.
7. **Sampling loop**: sample a token (penalties, top-k, top-p, min-p, temperature, or greedy),
   stop on end-of-generation, otherwise turn the token into text and feed it to the
   `ReasoningSplitter`, which emits answer/reasoning pieces to the callback, holding back any
   partial think tag until it can decide. Decode the token into the KV cache and repeat.
8. **Callback** fires on the worker isolate's thread; it posts `[channel, text]` to the Dart
   port. Returning `false` from the callback stops generation.
9. **End**: the EOG token is decoded too (so next turn's history is a pure prefix), stats are
   filled, the sampler is freed, `kiln_chat` returns. The worker sends the stats back; Dart
   completes the future and closes the stream.

`kiln_cancel` can be called from any thread at any point in 6 or 7; it flips an atomic that
both the abort callback and the loop check.

## The generation loop as a state machine

```mermaid
stateDiagram-v2
    [*] --> Tokenize
    Tokenize --> PrefixMatch
    PrefixMatch --> PromptEval: evaluate only the non-cached suffix
    PromptEval --> Sample
    Sample --> Done_EOS: token is end-of-generation
    Sample --> Split: otherwise
    Split --> Emit_Answer: not inside think tags
    Split --> Emit_Reasoning: inside think tags
    Split --> Hold: piece may be a partial tag
    Emit_Answer --> Decode
    Emit_Reasoning --> Decode
    Hold --> Decode
    Decode --> Sample: n_past < n_ctx
    Decode --> Done_Full: context full
    Sample --> Done_Max: max_tokens reached
    Sample --> Done_Cancel: cancel flag set
    Emit_Answer --> Done_Stop: callback returned false
    Done_EOS --> [*]
    Done_Full --> [*]
    Done_Max --> [*]
    Done_Cancel --> [*]
    Done_Stop --> [*]
```

## Two Dart pitfalls that bit us

- `StreamController.close()` returns a future that completes only after a listener has
  drained the stream. Awaiting it in `KilnContext.chat`'s cleanup deadlocked any caller who
  ignored `pieces` and awaited `stats` alone; on Android that surfaced as an ANR. Never await
  that close.
- Mirroring a failure onto the pieces stream must happen *before* the controller is closed,
  so the error handler sits before `whenComplete` in the chain and checks `isClosed`.

## Threading model in one paragraph

Everything blocking runs on the worker isolate (Dart) or on whatever thread calls the C API.
Inside the C call, ggml's threadpool does the math on N pinned workers. The only cross-thread
interaction the library allows is `kiln_cancel`. Pieces are delivered synchronously on the
calling thread; the Dart layer turns that into an async stream.

## Where to change things

| want to | touch |
|---|---|
| add an API call | `kiln.h`, `kiln.cpp`, then the backend; re-run ffigen; add to `kiln.dart` |
| change thread/pin policy | `probe.cpp` (which cores are "fast") and `create_context` in the backend |
| support a new engine | implement `Backend/Model/Context` from `internal.h`, register in `registry.cpp` |
| change how thinking is detected | `LlamaModel` constructor (empirical on/off render) |
| change sampling defaults | `kiln_sampling_params_default` in `kiln.cpp` |
| change the example app | `flutter/kiln_flutter/example/lib/main.dart` |

## Build matrix

| preset | output | used by |
|---|---|---|
| `host` | `build/host/libkiln.dylib`, `kiln-cli`, `kiln-tests`, `kiln-bench` | dev loop, host tests, desktop Flutter |
| `android-arm64` | `build/android-arm64/libkiln.so` + the same tools | adb scripts, benchmarks |
| Gradle via plugin | `libkiln.so` inside the APK | the Flutter app |

The three builds compile the same sources; the Gradle path just points CMake at the repo
root. ISA baseline for Android is `armv8.2-a+dotprod+fp16`.
