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
| s26ultra-2t_LiquidAI_LFM2.5-350M_Q4_K_M | 208 → 164 tok/s | 8 MB | 513 MB | 192% | 201% | 3676 MHz | 105.0 → 105.0 °C | 0 | 4060 MB |
| s26ultra-2t_Qwen3.5-0.8B-Q4_K_M | 71 → 45 tok/s | 8 MB | 1218 MB | 200% | 203% | 2809 MHz | 105.0 → 105.0 °C | 0 | 3697 MB |
| s26ultra-2t_d1-omni-600M-Q8_0 | question: 130 ms → 201 ms | 7 MB | 802 MB | 197% | 202% | 2940 MHz | 105.0 → 105.0 °C | 0 | 3968 MB |
| s26ultra-2t_embeddinggemma-2-UD-Q4_K_XL | text: 26 ms → 37 ms<br>red_circle.png: 2.5 s → 3.6 s<br>speech_train.wav: 800 ms → 796 ms | 7 MB | 1062 MB | 196% | 203% | 2183 MHz | 105.0 → 105.0 °C | 0 | 3186 MB |
| s26ultra-2t_gemma-3-270m-it-Q8_0 | 185 → 152 tok/s | 7 MB | 666 MB | 191% | 201% | 3737 MHz | 105.0 → 105.1 °C | 0 | 3778 MB |
| s26ultra-2t_jina-embeddings-v5-omni-nano-retrieval-Q4_K_M | text: 15 ms → 19 ms<br>red_circle.png: 2.8 s → 4.3 s | 7 MB | 677 MB | 197% | 202% | 2212 MHz | 105.0 → 105.0 °C | 0 | 3652 MB |
| s26ultra-2t_nomic-embed-text-v1.5.Q4_K_M | text: 17 ms → 24 ms | 7 MB | 177 MB | 194% | 201% | 3477 MHz | 105.0 → 105.0 °C | 0 | 4121 MB |
| s26ultra-8t_LiquidAI_LFM2.5-350M_Q4_K_M | 208 → 164 tok/s | 8 MB | 513 MB | 745% | 800% | 3182 MHz | 39.7 → 51.3 °C | 1 | 3542 MB |
| s26ultra-8t_Qwen3.5-0.8B-Q4_K_M | 61 → 49 tok/s | 8 MB | 1218 MB | 768% | 802% | 2307 MHz | 39.6 → 53.2 °C | 2 | 3435 MB |
| s26ultra-8t_gemma-3-270m-it-Q8_0 | 170 → 133 tok/s | 8 MB | 666 MB | 722% | 799% | 3148 MHz | 40.2 → 52.3 °C | 1 | 3719 MB |
| s26ultra_LiquidAI_LFM2.5-350M_Q4_K_M | 224 → 194 tok/s | 8 MB | 513 MB | 569% | 603% | 3278 MHz | 39.7 → 50.9 °C | 1 | 3659 MB |
| s26ultra_Qwen3.5-0.8B-Q4_K_M | 76 → 56 tok/s | 8 MB | 1218 MB | 598% | 603% | 2432 MHz | 39.7 → 52.8 °C | 2 | 3354 MB |
| s26ultra_d1-omni-600M-Q8_0 | question: 77 ms → 122 ms | 7 MB | 802 MB | 573% | 606% | 2654 MHz | 40.2 → 53.3 °C | 2 | 3639 MB |
| s26ultra_embeddinggemma-2-UD-Q4_K_XL | text: 17 ms → 24 ms<br>red_circle.png: 1.5 s → 2.2 s<br>speech_train.wav: 1.4 s → 552 ms | 8 MB | 1075 MB | 572% | 608% | 1997 MHz | 40.9 → 52.9 °C | 2 | 3043 MB |
| s26ultra_gemma-3-270m-it-Q8_0 | 173 → 146 tok/s | 7 MB | 666 MB | 554% | 602% | 3312 MHz | 40.2 → 51.1 °C | 1 | 3568 MB |
| s26ultra_jina-embeddings-v5-omni-nano-retrieval-Q4_K_M | text: 9 ms → 11 ms<br>red_circle.png: 1.5 s → 2.1 s | 8 MB | 677 MB | 580% | 605% | 2174 MHz | 39.5 → 52.2 °C | 2 | 3386 MB |
| s26ultra_nomic-embed-text-v1.5.Q4_K_M | text: 10 ms → 13 ms | 8 MB | 177 MB | 554% | 607% | 3178 MHz | 40.1 → 51.8 °C | 1 | 3745 MB |
