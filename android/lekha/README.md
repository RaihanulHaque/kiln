# Lekha

A small Android app for trying kiln's OCR (`kiln_ocr`, OCRia-nano-1) by hand. Lekha means "writing"
in Bengali. Take a photo, pick one from the gallery, share a picture into the app, or tap Sample
for the test pages. The app shows the text, a box around each line (coloured by the recognizer
that read it), each line's confidence, and how long finding and reading the lines took.

Plain Android views and no library except kiln (`:kiln` in this folder), so it builds offline.

## Build and install

```bash
cd android
./gradlew :lekha:assembleRelease      # lekha/build/outputs/apk/release/lekha-release.apk (debug-signed)
adb install -r lekha/build/outputs/apk/release/lekha-release.apk
```

The model is not in the APK (its license is not open; kiln ships no weights). Make it with
`scripts/fetch_models.sh --ocr`, then push it:

```bash
adb push models/ocria-nano-1.gguf /sdcard/Android/data/dev.kiln.lekha/files/
```

Or tap **Model file** in the app and choose the `.gguf`; it is copied into the app's own folder.

## Buttons

- **Camera / Gallery / Sample**: where the picture comes from. Sample cycles through the English,
  Bengali and upside-down test pages from `tests/media`.
- **Read again**: the same picture once more. The first read after starting is a little slower.
- **Size**: the long side the page is shrunk to before lines are found (1600 is kiln's default).
  Smaller is faster and misses small print.
- **Boxes**: draw the line boxes or not. Blue Latin, purple World (Bengali, Hindi, ...),
  red CJK, green Arabic / Hebrew.

kiln does not apply a photo's EXIF orientation, so a camera photo stored sideways is turned
upright by the app and handed over as a JPEG. Any other picture goes to kiln byte for byte.

## Where it was run

Galaxy A17 5G, 2026-11-25: the samples read as in `kiln-tests` (English page 9 lines in 309 ms,
Bengali page 4 lines in 152 ms, the first call after start). Camera and gallery not yet tried.
