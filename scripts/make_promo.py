#!/usr/bin/env python3
"""Render kiln's promo animation (GIF + MP4) with Pillow. Deterministic, no browser.

usage: python scripts/make_promo.py docs/img/kiln_promo
"""
import math
import os
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1200, 676
FPS = 24
BG = (16, 12, 11)
INK = (236, 228, 222)
DIM = (140, 128, 120)
EMBER = (217, 72, 15)
EMBER_HI = (255, 150, 70)
GLOW = (255, 110, 30)

F = "/System/Library/Fonts/"
def font(name, size):
    try:
        return ImageFont.truetype(F + name, size)
    except OSError:
        return ImageFont.load_default()

LOGO = font("Avenir Next.ttc", 150)
TAG = font("Avenir Next.ttc", 34)
MONO = font("SFNSMono.ttf", 26)
MONO_S = font("SFNSMono.ttf", 20)
CHIP = font("Avenir Next.ttc", 22)

frames = []

def ease(t):  # smoothstep
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)

def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))

def base():
    img = Image.new("RGB", (W, H), BG)
    # subtle vignette / ember floor glow
    glow = Image.new("RGB", (W, H), BG)
    gd = ImageDraw.Draw(glow)
    gd.ellipse((W * 0.2, H * 0.75, W * 0.8, H * 1.45), fill=(60, 22, 8))
    glow = glow.filter(ImageFilter.GaussianBlur(90))
    return Image.blend(img, glow, 1.0)

def flame_mark(d, cx, cy, s, t, alpha=1.0):
    """A kiln arch with a breathing flame. s = scale (px for the arch half-width)."""
    # arch
    col = lerp(BG, EMBER, alpha)
    d.arc((cx - s, cy - s * 1.1, cx + s, cy + s * 0.9), 180, 360, fill=col, width=int(s * 0.16))
    d.line((cx - s, cy - 0.1 * s, cx - s, cy + s * 0.55), fill=col, width=int(s * 0.16))
    d.line((cx + s, cy - 0.1 * s, cx + s, cy + s * 0.55), fill=col, width=int(s * 0.16))
    d.line((cx - s * 1.25, cy + s * 0.62, cx + s * 1.25, cy + s * 0.62), fill=col, width=int(s * 0.16))
    # flame: three teardrops breathing
    b = 1 + 0.08 * math.sin(t * 5.0)
    for k, (dx, hs, c) in enumerate([(0, 0.62, EMBER_HI), (-0.22, 0.42, GLOW), (0.22, 0.38, GLOW)]):
        fx = cx + dx * s
        hh = hs * s * b * (1 + 0.05 * math.sin(t * 7 + k))
        cc = lerp(BG, c, alpha)
        d.polygon([(fx, cy + 0.45 * s - hh * 1.6), (fx - 0.17 * s, cy + 0.45 * s), (fx + 0.17 * s, cy + 0.45 * s)], fill=cc)
        d.ellipse((fx - 0.17 * s, cy + 0.28 * s, fx + 0.17 * s, cy + 0.62 * s), fill=cc)

def text_center(d, y, s, f, fill):
    w = d.textlength(s, font=f)
    d.text(((W - w) / 2, y), s, font=f, fill=fill)

# ------------------------------------------------------------------ beat 1: logo (0-3.2 s)
def beat_logo(t, T=3.2):
    img = base()
    d = ImageDraw.Draw(img)
    a = ease(t / 0.8)
    flame_mark(d, W / 2 - 150, H / 2 - 20, 70, t, alpha=a)
    col = lerp(BG, INK, a)
    d.text((W / 2 - 40, H / 2 - 115), "kiln", font=LOGO, fill=col)
    a2 = ease((t - 1.1) / 0.7)
    text_center(d, H / 2 + 90, "on-device LLM runtime", TAG, lerp(BG, INK, a2))
    a3 = ease((t - 1.7) / 0.7)
    text_center(d, H / 2 + 136, "honest about what it runs on", TAG, lerp(BG, EMBER_HI, a3))
    fade = 1 - ease((t - (T - 0.4)) / 0.4)
    return Image.blend(Image.new("RGB", (W, H), BG), img, fade)

# ------------------------------------------------------------------ beat 2: terminal (0-8 s)
CMD = "$ kiln-cli -m qwen3.5-0.8b.gguf --think -p \"Why is the sky blue?\""
REASON = "The user wants a one-sentence answer. Rayleigh scattering, shorter wavelengths scatter more. Keep it tight."
ANSWER = ("Sunlight is scattered by air molecules, and because blue light has a shorter wavelength "
          "it scatters far more than red, so the sky looks blue from every direction.")
STATS = ["device:   Google Tensor G3 | arm64-v8a | 9 cores (5 fast, pinned) | 7.9 GB",
         "backend:  llama.cpp/cpu    model: qwen35 Q4_K_M 0.75B  0.52 GB resident"]

def wrap(d, s, f, maxw):
    words, lines, cur = s.split(" "), [], ""
    for w_ in words:
        trial = (cur + " " + w_).strip()
        if d.textlength(trial, font=f) > maxw and cur:
            lines.append(cur); cur = w_
        else:
            cur = trial
    if cur: lines.append(cur)
    return lines

def beat_terminal(t, T=8.0):
    img = base()
    d = ImageDraw.Draw(img)
    # window
    x0, y0, x1, y1 = 70, 60, W - 70, H - 60
    d.rounded_rectangle((x0, y0, x1, y1), 18, fill=(24, 19, 17), outline=(50, 40, 36), width=2)
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse((x0 + 22 + i * 26, y0 + 18, x0 + 38 + i * 26, y0 + 34), fill=c)
    flame_mark(d, x1 - 60, y0 + 30, 14, t)
    y = y0 + 64
    # command types in 0-1.2 s
    n = int(len(CMD) * ease(t / 1.2))
    d.text((x0 + 28, y), CMD[:n], font=MONO, fill=INK); y += 46
    if t > 1.4:
        for s in STATS:
            d.text((x0 + 28, y), s, font=MONO_S, fill=DIM); y += 30
        y += 12
    # reasoning streams 1.8-3.6 s, dimmed italic-ish
    if t > 1.8:
        rn = int(len(REASON) * ease((t - 1.8) / 1.8))
        for ln in wrap(d, REASON[:rn], MONO_S, x1 - x0 - 56):
            d.text((x0 + 28, y), ln, font=MONO_S, fill=(120, 100, 90)); y += 28
        y += 10
    # answer streams 3.8-6.6 s
    gen_tok = 0
    if t > 3.8:
        an = int(len(ANSWER) * ease((t - 3.8) / 2.8))
        gen_tok = int(41 * (t - 3.8)) if t < 6.6 else 115
        for ln in wrap(d, ANSWER[:an], MONO, x1 - x0 - 56):
            d.text((x0 + 28, y), ln, font=MONO, fill=INK); y += 36
    # live stats bar
    tps = 0 if t < 3.9 else 41.2 + 0.6 * math.sin(t * 9)
    bar = f"{'reasoning' if 1.8 < t < 3.8 else 'generate':>9}   {tps:5.1f} tok/s   {gen_tok:3d} tok   {'' if t < 6.6 else '[eos]'}"
    d.rounded_rectangle((x0 + 20, y1 - 58, x1 - 20, y1 - 18), 10, fill=(34, 26, 22))
    d.text((x0 + 36, y1 - 50), bar, font=MONO_S, fill=EMBER_HI if t > 3.8 else DIM)
    # cursor
    if int(t * 3) % 2 == 0 and t < 6.6:
        d.rectangle((x0 + 28 + 2, y, x0 + 28 + 14, y + 26), fill=EMBER)
    fade_in = ease(t / 0.4)
    fade_out = 1 - ease((t - (T - 0.4)) / 0.4)
    return Image.blend(Image.new("RGB", (W, H), BG), img, min(fade_in, fade_out))

# ------------------------------------------------------------------ beat 3: end card (0-4 s)
CHIPS = ["C API", "Flutter", "Android", "llama.cpp core", "Qwen · Gemma · LFM", "think on/off"]
NUMS = [("21 tok/s", "sustained, Pixel 8, 0.8B"), ("6 MB", "libkiln.so"), ("127", "checks green on device")]

def beat_end(t, T=4.2):
    img = base()
    d = ImageDraw.Draw(img)
    a = ease(t / 0.6)
    flame_mark(d, 120, 120, 44, t, alpha=a)
    d.text((180, 62), "kiln", font=font("Avenir Next.ttc", 96), fill=lerp(BG, INK, a))
    # numbers
    for i, (big, small) in enumerate(NUMS):
        ai = ease((t - 0.5 - i * 0.25) / 0.5)
        x = 100 + i * 360
        d.text((x, 250), big, font=font("Avenir Next.ttc", 72), fill=lerp(BG, EMBER_HI, ai))
        d.text((x + 4, 335), small, font=CHIP, fill=lerp(BG, DIM, ai))
    # chips
    x = 100
    for i, c in enumerate(CHIPS):
        ai = ease((t - 1.4 - i * 0.12) / 0.4)
        w = d.textlength(c, font=CHIP) + 30
        d.rounded_rectangle((x, 440, x + w, 486), 23, outline=lerp(BG, EMBER, ai), width=2)
        d.text((x + 18, 449), c, font=CHIP, fill=lerp(BG, INK, ai))
        x += w + 10
    a4 = ease((t - 2.2) / 0.6)
    d.text((100, 560), "open source · MIT · github.com/<you>/kiln", font=TAG, fill=lerp(BG, DIM, a4))
    fade_out = 1 - ease((t - (T - 0.5)) / 0.5)
    return Image.blend(Image.new("RGB", (W, H), BG), img, fade_out)

def render(beat, T):
    n = int(T * FPS)
    for i in range(n):
        frames.append(beat(i / FPS, T))

render(beat_logo, 3.2)
render(beat_terminal, 8.0)
render(beat_end, 4.2)

out = sys.argv[1] if len(sys.argv) > 1 else "kiln_promo"
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

# MP4 (LinkedIn prefers video): image sequence -> ffmpeg
import tempfile
mp4 = out + ".mp4"
with tempfile.TemporaryDirectory() as td:
    for i, f in enumerate(frames):
        f.save(os.path.join(td, f"f{i:05d}.png"))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", os.path.join(td, "f%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", mp4], check=True)

# GIF: half size, 12 fps, palette per clip
gif = out + ".gif"
small = [f.resize((W // 2, H // 2), Image.LANCZOS) for f in frames[::2]]
small[0].save(gif, save_all=True, append_images=small[1:], duration=int(1000 / (FPS / 2)), loop=0, optimize=True)
print(mp4, os.path.getsize(mp4) // 1024, "KB;", gif, os.path.getsize(gif) // 1024, "KB;", len(frames), "frames")
