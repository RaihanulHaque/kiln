# Parakeet 110M in kiln: work in progress

Not part of the library yet. This folder holds what is needed to carry on: the whisper.cpp patch,
the scripts and the numbers. Models, audio and outputs are not in git; the steps below make them
again in `build/asr/` (git-ignored). `RESULTS.md` has every measurement so far.

## Where it stands (2026-11-26)

1. **NVIDIA `parakeet-tdt_ctc-110m` (English, CC-BY-4.0) runs through kiln.** whisper.cpp needed
   optional linear biases (the 0.6B has none). LibriSpeech WER: test-clean 2.62%, test-other 5.45%
   (0.6B: 2.20%, 3.86%), at 3.8x the speed and a fifth of the size.
2. **Encoder profiled per node.** Weight matmuls 53%, the subsampling front end 23%, attention 12%
   (32% on a 35 s clip).
3. **Front end rewritten** (channels first, a custom op for the first conv): encoder 11-18% faster,
   front end 3.3x closer to PyTorch float64, WER unchanged (2.617%).
4. **Threads**: 6 instead of 8 on the 8500G, +20% for Parakeet. That one is a kiln change, in its own
   PR (`feat/smt-one-thread-per-core`, PR #52), not here.
5. **Old front end removed** (2026-11-27): the `PARAKEET_OLD_FRONTEND` switch is gone from the patch.
   The old graph is in the patch at commit e8b2ee1, for anyone who needs to compare again.
6. **Batch norm folded** into the depthwise conv at load time, with NeMo's eps = 1e-5 (missing
   before): 4 graph ops fewer per layer. F16 encoder output moves by 3e-5 rel rms; LibriSpeech WER
   unchanged (M4 Pro, Metal: 2.61% / 5.45%); no measurable speed change on the M4 Pro CPU. `RESULTS.md`, phase 3b.
7. **macOS works** for every step below (2026-11-27, M4 Pro): `prof/build.sh` links the Mac way.

## Next

- On the M4 Pro the TDT decoder (LSTM and joint, one small graph per step, on the CPU) is now more
  time than the encoder: 54 s against 45 s for test-clean. On the Ryzen the encoder was 88%.
- Attention for long clips (32% at 35 s), the GLU sigmoid, the scheduler rebuild per clip length.
- CTC head (phase 2): the 110M's `ctc_decoder.*` tensors are skipped today; a CTC path would also
  run the Bangla FastConformer-CTC models.
- Decide where the whisper.cpp change lives (upstream PR, a fork, or a patch kiln applies).

## Set up again (Linux or macOS; the conda env `ML` has torch, numpy, pyyaml, jiwer, soundfile, pyarrow)

```bash
git switch wip/parakeet-110m
git submodule update --init --recursive
git -C third_party/whisper.cpp apply ../../experiments/parakeet/parakeet-bias-frontend.patch
cmake --preset host && cmake --build --preset host -j
scripts/fetch_models.sh --speech              # the 0.6B, for comparison

mkdir -p build/asr/{dl,conv,data,res,ref/clips,ref/base,prof} && cd build/asr
curl -L -o dl/parakeet-tdt_ctc-110m.nemo https://huggingface.co/nvidia/parakeet-tdt_ctc-110m/resolve/main/parakeet-tdt_ctc-110m.nemo
mkdir -p nemo110 && tar -xf dl/parakeet-tdt_ctc-110m.nemo -C nemo110
conda run -n ML python ../../experiments/parakeet/convert_tdt.py --model dl/parakeet-tdt_ctc-110m.nemo --out-dir conv --out-name ggml-parakeet-tdt_ctc-110m-f16.bin
# whisper.cpp's quantizer, built on its own
cmake -S ../../third_party/whisper.cpp -B wcpp -DCMAKE_BUILD_TYPE=Release && cmake --build wcpp -j --target parakeet-quantize
LD_LIBRARY_PATH=wcpp/bin wcpp/bin/parakeet-quantize conv/ggml-parakeet-tdt_ctc-110m-f16.bin conv/ggml-parakeet-tdt_ctc-110m-q8_0.bin q8_0

# LibriSpeech from Hugging Face (openslr.org was 10-30 KB/s)
curl -L -o dl/clean.parquet https://huggingface.co/api/datasets/openslr/librispeech_asr/parquet/clean/test/0.parquet
curl -L -o dl/other.parquet https://huggingface.co/api/datasets/openslr/librispeech_asr/parquet/other/test/0.parquet
conda run -n ML python -I ../../experiments/parakeet/unpack.py dl/clean.parquet data/test-clean
conda run -n ML python -I ../../experiments/parakeet/unpack.py dl/other.parquet data/test-other
```

The 20 reference clips (`RESULTS.md`, phase 2) are 20 test-clean files spread evenly over length:

```bash
conda run -n ML python -I -c "
import soundfile as sf, pathlib
fs = sorted(pathlib.Path('data/test-clean').glob('*.flac'))
items = sorted((sf.info(str(f)).duration, f) for f in fs)
for d, f in [items[int(i * (len(items) - 1) / 19)] for i in range(20)]:
    a, sr = sf.read(str(f), dtype='float32'); a.tofile('ref/clips/%s.f32' % f.stem)"
```

`ref/base` was made with the old front end, which is no longer in the patch: apply the patch from
commit e8b2ee1 and run `PARAKEET_OLD_FRONTEND=1 prof/encprof ... --dump ref/base`.

On macOS use `DYLD_LIBRARY_PATH` in place of `LD_LIBRARY_PATH` for `parakeet-quantize`, and the
Python in the `ML` env needs `jiwer` (`pip install jiwer`).

## Tools

| file | what it does |
|---|---|
| `parakeet-bias-frontend.patch` | the whisper.cpp change: optional biases, the new front end |
| `convert_tdt.py` | whisper.cpp's converter, skipping the CTC head |
| `unpack.py` | LibriSpeech parquet to flac files and a trans.txt |
| `eval.py` | WER and speed through `kiln-cli` (`-t`, `--limit`) |
| `prof/encprof.cpp` | encoder outputs (`--dump`), timing (`--reps`), per-node profile (`--prof`); `PROF_DUMP_NODE=name PROF_DUMP_DIR=dir` writes one node and the mel |
| `prof/build.sh` | builds encprof into `build/asr/prof/` |
| `prof/cmp.py` | compares two `--dump` folders |
| `prof/fe_ref.py` | the front end in PyTorch float64 from `nemo110/`, against dumped nodes (run in `build/asr`) |
| `prof/cats.py` | groups a `--prof` listing into categories |

Examples, from `build/asr`:

```bash
prof/encprof conv/ggml-parakeet-tdt_ctc-110m-q8_0.bin -t 6 --reps 3 ref/clips/*.f32
prof/encprof conv/ggml-parakeet-tdt_ctc-110m-q8_0.bin -t 6 --prof --reps 3 ref/clips/2961-960-0013.f32
conda run -n ML python -I ../../experiments/parakeet/eval.py conv/ggml-parakeet-tdt_ctc-110m-q8_0.bin data/test-clean res/clean-110m
```

With Q8_0 weights any change moves the encoder output by 1 to 4%, so judge Q8_0 by WER and check
exact numbers with the F16 file.
