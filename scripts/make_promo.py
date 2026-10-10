#!/usr/bin/env python3
"""Render kiln's promo animation (MP4 + GIF). usage: python scripts/make_promo.py docs/img/kiln_promo"""
import math
import os
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(__file__))
from brand import BG, INK, DIM, EMBER, EMBER_HI, ease, lerp, lockup, render_mark, ui_font, font  # noqa: E402

W, H = 1200, 676
FPS = 24
MONO = font("SFNSMono.ttf", 25)
MONO_S = font("SFNSMono.ttf", 19)
TAG = ui_font(32, "Medium")
CHIP = ui_font(21, "Demi")

frames = []

def base():
    img = Image.new("RGB", (W, H), BG)
    glow = Image.new("RGB", (W, H), BG)
    ImageDraw.Draw(glow).ellipse((W * 0.25, H * 0.8, W * 0.75, H * 1.5), fill=(58, 22, 9))
    return Image.blend(img, glow.filter(ImageFilter.GaussianBlur(110)), 1.0)

def put(img, layer, x, y):
    img.paste(layer, (int(x), int(y)), layer)

def text_center(d, y, s, f, fill):
    w = d.textlength(s, font=f)
    d.text(((W - w) / 2, y), s, font=f, fill=fill)

def text_center_a(img, y, s, f, color, alpha):
    """Centered text composited with real alpha (no ghosting over gradients)."""
    if alpha <= 0:
        return
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    w = d.textlength(s, font=f)
    d.text(((W - w) / 2, y), s, font=f, fill=color + (int(255 * min(1.0, alpha)),))
    img.paste(layer, (0, 0), layer)

def fade(img, a):
    return Image.blend(Image.new("RGB", (W, H), BG), img, max(0.0, min(1.0, a)))

# --------------------------------------------------------------- beat 1: logo (5.2 s)
def ease_out(t, p=4):          # ease-out-quart: fast start, long soft settle
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** p

def letters_layer(size, t, start, stagger=0.16, dur=1.1):
    """'kiln' drawn letter by letter: each letter fades in from a soft blur and drifts
    2 px into place. Returns an RGBA layer sized to the word."""
    from brand import wordmark_font
    f = wordmark_font(size)
    tmp = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    bb = tmp.textbbox((0, 0), "kiln", font=f)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    pad = 24
    out = Image.new("RGBA", (tw + 2 * pad, th + 2 * pad), (0, 0, 0, 0))
    x = pad - bb[0]
    for i, ch in enumerate("kiln"):
        a = ease_out((t - start - i * stagger) / dur)
        if a > 0:
            layer = Image.new("RGBA", out.size, (0, 0, 0, 0))
            ImageDraw.Draw(layer).text((x + (1 - a) * 6, pad - bb[1]), ch, font=f, fill=INK + (int(255 * a),))
            blur = (1 - a) * 5
            if blur > 0.2:
                layer = layer.filter(ImageFilter.GaussianBlur(blur))
            out.alpha_composite(layer)
        x += tmp.textlength(ch, font=f)
    return out, bb

def beat_logo(t, T=5.2):
    img = base()
    # floor glow warms up slowly over the whole beat
    warm = Image.new("RGB", (W, H), BG)
    ImageDraw.Draw(warm).ellipse((W * 0.3, H * 0.55, W * 0.7, H * 1.3), fill=(70, 26, 10))
    img = Image.blend(img, warm.filter(ImageFilter.GaussianBlur(120)), 0.6 * ease_out(t / 3.0, 2))

    size_text = 168
    from brand import wordmark_font
    f = wordmark_font(size_text)
    tmp = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    bb = tmp.textbbox((0, 0), "kiln", font=f)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    mark_size = int(round(th / 0.80))
    gap = int(th * 0.30)
    total_w = mark_size + gap + tw
    x0 = (W - total_w) / 2
    y_text = H / 2 - th / 2 - 48
    y_mark = y_text - (mark_size - th) / 2

    # 1. ember ignites alone (0 - 1.4 s), 2. arch breathes in around it (0.6 - 2.2 s)
    glow = ease_out(t / 1.4, 3)
    arch_a = ease_out((t - 0.6) / 1.6, 4)
    scale = 0.965 + 0.035 * arch_a
    from brand import render_mark, EMBER
    mark = render_mark(mark_size, t=t, glow=glow, stroke_color=lerp(BG, EMBER, arch_a))
    if scale < 1:
        ms = int(mark_size * scale)
        mark = mark.resize((ms, ms), Image.LANCZOS)
        put(img, mark, x0 + (mark_size - ms) / 2, y_mark + (mark_size - ms) / 2)
    else:
        put(img, mark, x0, y_mark)

    # 3. letters settle in (1.5 s onward)
    layer, lbb = letters_layer(size_text, t, start=1.5)
    put(img, layer, x0 + mark_size + gap - 24, y_text - 24)

    # 4. taglines, pure fades, long tail
    text_center_a(img, H / 2 + 92, "on-device LLM runtime", TAG, INK, ease_out((t - 2.6) / 1.2, 3))
    text_center_a(img, H / 2 + 136, "honest about what it runs on", TAG, EMBER_HI, ease_out((t - 3.1) / 1.2, 3))
    return fade(img, 1 - ease((t - (T - 0.6)) / 0.6))

# --------------------------------------------------------------- beat 2: terminal (8 s)
CMD = "$ kiln-cli -m qwen3.5-0.8b.gguf --think -p \"Why is the sky blue?\""
REASON = "The user wants one sentence. Rayleigh scattering: shorter wavelengths scatter more. Keep it tight."
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
    x0, y0, x1, y1 = 70, 60, W - 70, H - 60
    d.rounded_rectangle((x0, y0, x1, y1), 18, fill=(24, 19, 17), outline=(52, 42, 37), width=2)
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse((x0 + 22 + i * 26, y0 + 18, x0 + 38 + i * 26, y0 + 34), fill=c)
    put(img, render_mark(30, t=t), x1 - 52, y0 + 11)
    y = y0 + 64
    n = int(len(CMD) * ease(t / 1.2))
    d.text((x0 + 28, y), CMD[:n], font=MONO, fill=INK); y += 46
    if t > 1.4:
        for s in STATS:
            d.text((x0 + 28, y), s, font=MONO_S, fill=DIM); y += 29
        y += 12
    if t > 1.8:
        rn = int(len(REASON) * ease((t - 1.8) / 1.8))
        for ln in wrap(d, REASON[:rn], MONO_S, x1 - x0 - 56):
            d.text((x0 + 28, y), ln, font=MONO_S, fill=(124, 104, 94)); y += 27
        y += 10
    gen_tok = 0
    if t > 3.8:
        an = int(len(ANSWER) * ease((t - 3.8) / 2.8))
        gen_tok = int(41 * (t - 3.8)) if t < 6.6 else 115
        for ln in wrap(d, ANSWER[:an], MONO, x1 - x0 - 56):
            d.text((x0 + 28, y), ln, font=MONO, fill=INK); y += 35
    tps = 0 if t < 3.9 else 41.2 + 0.6 * math.sin(t * 9)
    bar = f"{'reasoning' if 1.8 < t < 3.8 else 'generate':>9}   {tps:5.1f} tok/s   {gen_tok:3d} tok   {'' if t < 6.6 else '[eos]'}"
    d.rounded_rectangle((x0 + 20, y1 - 58, x1 - 20, y1 - 18), 10, fill=(34, 26, 22))
    d.text((x0 + 36, y1 - 49), bar, font=MONO_S, fill=EMBER_HI if t > 3.8 else DIM)
    if int(t * 3) % 2 == 0 and t < 6.6:
        d.rectangle((x0 + 30, y, x0 + 42, y + 25), fill=EMBER)
    return fade(img, min(ease(t / 0.4), 1 - ease((t - (T - 0.4)) / 0.4)))

# --------------------------------------------------------------- beat 3: end card (4.2 s)
CHIPS = ["C API", "Flutter", "Android", "llama.cpp core", "Qwen · Gemma · LFM", "think on/off"]
NUMS = [("21 tok/s", "sustained, Pixel 8, 0.8B"), ("6 MB", "libkiln.so"), ("127", "checks green on device")]

def beat_end(t, T=4.2):
    img = base()
    a = ease(t / 0.6)
    lk = lockup(88, 98, t=t, mark_alpha=a, text_alpha=a)
    put(img, lk, 100, 70)
    d = ImageDraw.Draw(img)
    big = ui_font(70, "Demi")
    for i, (bigs, small) in enumerate(NUMS):
        ai = ease((t - 0.5 - i * 0.25) / 0.5)
        x = 100 + i * 360
        d.text((x, 250), bigs, font=big, fill=lerp(BG, EMBER_HI, ai))
        d.text((x + 3, 335), small, font=ui_font(22, "Medium"), fill=lerp(BG, DIM, ai))
    x = 100
    for i, c in enumerate(CHIPS):
        ai = ease((t - 1.4 - i * 0.12) / 0.4)
        w = d.textlength(c, font=CHIP) + 30
        d.rounded_rectangle((x, 440, x + w, 484), 22, outline=lerp(BG, EMBER, ai), width=2)
        d.text((x + 15, 449), c, font=CHIP, fill=lerp(BG, INK, ai))
        x += w + 10
    d.text((100, 560), "open source · MIT · github.com/RaihanulHaque/kiln", font=TAG, fill=lerp(BG, DIM, ease((t - 2.2) / 0.6)))
    return fade(img, 1 - ease((t - (T - 0.5)) / 0.5))

def render(beat, T):
    for i in range(int(T * FPS)):
        frames.append(beat(i / FPS, T))

render(beat_logo, 5.2)
render(beat_terminal, 8.0)
render(beat_end, 4.2)

out = sys.argv[1] if len(sys.argv) > 1 else "kiln_promo"
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
mp4 = out + ".mp4"
with tempfile.TemporaryDirectory() as td:
    for i, f in enumerate(frames):
        f.save(os.path.join(td, f"f{i:05d}.png"))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", os.path.join(td, "f%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", mp4], check=True)
gif = out + ".gif"
small = [f.resize((W // 2, H // 2), Image.LANCZOS) for f in frames[::2]]
small[0].save(gif, save_all=True, append_images=small[1:], duration=int(1000 / (FPS / 2)), loop=0, optimize=True)
print(mp4, os.path.getsize(mp4) // 1024, "KB;", gif, os.path.getsize(gif) // 1024, "KB;", len(frames), "frames")
