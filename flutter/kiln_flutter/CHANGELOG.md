## 0.2.0

- `chat(maxReasoningTokens:)`: thinking budget. After that many reasoning tokens kiln closes the
  think block and the model answers. `KilnGenStats.reasoningTruncated` reports when it was used.
- `KilnContext.decide`: ask a decision model (LFM2 d1-omni) a yes / no, choice or score
  question about a text state and get a probability per option. `KilnModelInfo.isDecisionModel`
  tells the two kinds of model apart.
- Hybrid models (Qwen3.5, LFM2) reuse the KV cache across turns through state checkpoints.

## 0.1.0

- First release. Android arm64, llama.cpp backend, streaming chat with reasoning channel,
  KV prefix reuse, cancel, device probe, model info and generation stats.
