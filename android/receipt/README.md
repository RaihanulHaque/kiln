# Hishab Receipt (test app)

A temporary app for trying receipt photos with Hishab on a phone:

```
photo --kiln OCR--> lines --rules (Receipt.kt)--> "Meghna Mart grocery 1530" --Hishab model--> Meghna Mart grocery|-1530|grocery
```

The rules find the shop, the kind of shop and the final total, and check the total against
subtotal + VAT + service − discount and cash paid − change. A total that does not add up is
shown as "Please check". They are a port of `run.py` in the Hishab repo
(`experiments/receipt_ocr`), which also has the synthetic test set and its results. Built from
Lekha (camera, gallery, share), plain Android views, no library except kiln.

## Build, install, models

```bash
cd android
ANDROID_HOME=~/Library/Android/sdk ./gradlew :receipt:assembleRelease
adb install -r receipt/build/outputs/apk/release/receipt-release.apk
adb shell am start -n dev.kiln.receipt/.MainActivity     # makes the app's folder
D=/sdcard/Android/data/dev.kiln.receipt/files
adb push ../models/ocria-nano-1.gguf $D/
adb push ../models/hishab/qwen3.5-0.8b__pipe_signed__full__v13__trim38k__L20-Q4_K_M-imat.gguf $D/
```

Then restart the app. Any `.gguf` in that folder with `hishab` or `pipe_signed` in its name is
taken as the Hishab model. The first install wipes the folder, so push after installing.

## Where it was run

Galaxy A17 5G, 2026-11-25, the 10 sample receipts in `assets/` (synthetic, from the Hishab
repo's generator): 9 right (shop, total and category), 1 faded Bangla slip with a wrong total
shown as "Please check". OCR 0.45 to 0.9 s and Hishab 0.6 to 1.0 s per receipt (2.1 s for the
first Hishab call after start). Real receipt photos not tried yet.
