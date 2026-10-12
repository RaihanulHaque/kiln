Speed is first run → last run. "work" is while the model is computing: generating, deciding or embedding.

| run | speed, first → last run | baseline RSS | peak RSS | work CPU mean | work CPU max | work freq | CPU temp start → max | OS throttle | min mem avail |
|---|---|---|---|---|---|---|---|---|---|
| m4pro_LiquidAI_LFM2.5-350M_Q4_K_M | – | 87 MB | 365 MB | 26% | 41% | – | – → – | – | – |
| m4pro_Qwen3.5-0.8B-Q4_K_M | – | 87 MB | 761 MB | 25% | 33% | – | – → – | – | – |
| m4pro_d1-omni-600M-Q8_0 | question: 20 ms → 20 ms | 89 MB | 543 MB | 15% | 18% | – | – → – | 0 | – |
| m4pro_embeddinggemma-2-UD-Q4_K_XL | text: 12 ms → 17 ms<br>red_circle.png: 98 ms → 95 ms<br>speech_train.wav: 50 ms → 55 ms | 89 MB | 1068 MB | 21% | 33% | – | – → – | 0 | – |
| m4pro_gemma-3-270m-it-Q8_0 | – | 89 MB | 468 MB | 25% | 34% | – | – → – | – | – |
| m4pro_jina-embeddings-v5-omni-nano-retrieval-Q4_K_M | text: 8 ms → 9 ms<br>red_circle.png: 87 ms → 89 ms | 86 MB | 706 MB | 19% | 33% | – | – → – | 0 | – |
| m4pro_nomic-embed-text-v1.5.Q4_K_M | text: 6 ms → 6 ms | 88 MB | 189 MB | 23% | 38% | – | – → – | 0 | – |
| pixel8_LiquidAI_LFM2.5-350M_Q4_K_M | 111 → 98 tok/s | 8 MB | 514 MB | 494% | 503% | 2476 MHz | 42.0 → 91.0 °C | 0 | 1488 MB |
| pixel8_Qwen3.5-0.8B-Q4_K_M | 37 → 26 tok/s | 8 MB | 1218 MB | 482% | 503% | 2088 MHz | 42.0 → 83.0 °C | 1 | 1356 MB |
| pixel8_d1-omni-600M-Q8_0 | question: 212 ms → 379 ms | 8 MB | 803 MB | 485% | 504% | 1588 MHz | 45.0 → 81.0 °C | 1 | 1438 MB |
| pixel8_embeddinggemma-2-UD-Q4_K_XL | text: 39 ms → 54 ms<br>red_circle.png: 5.7 s → 6.0 s<br>speech_train.wav: 3.1 s → 1.6 s | 9 MB | 1064 MB | 434% | 504% | 1177 MHz | 45.0 → 84.0 °C | 1 | 943 MB |
| pixel8_gemma-3-270m-it-Q8_0 | 80 → 65 tok/s | 8 MB | 664 MB | 485% | 503% | 2066 MHz | 45.0 → 81.0 °C | 1 | 1358 MB |
| pixel8_jina-embeddings-v5-omni-nano-retrieval-Q4_K_M | text: 25 ms → 27 ms<br>red_circle.png: 4.6 s → 6.0 s | 8 MB | 677 MB | 427% | 505% | 1255 MHz | 46.0 → 84.0 °C | 1 | 1228 MB |
| pixel8_nomic-embed-text-v1.5.Q4_K_M | text: 27 ms → 33 ms | 8 MB | 177 MB | 478% | 505% | 2294 MHz | 47.0 → 81.0 °C | 1 | 1563 MB |
