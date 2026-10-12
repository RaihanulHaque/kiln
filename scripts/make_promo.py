#!/usr/bin/env python3
"""Render kiln's promo animation (MP4 for LinkedIn, GIF for the README).

usage: python scripts/make_promo.py docs/img/kiln_promo

Three beats: the logo ignites; three cards show one chat, one decision and one embedding
search running at once; the end card lists what ships and what is next. Every number
on the end card comes from README.md's "Measured" tables (Pixel 8, v0.3.0).
"""
import math
import os
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(__file__))
from brand import BG, INK, DIM, EMBER, EMBER_HI, ease, lerp, lockup, render_mark, ui_font, font, wordmark_font  # noqa: E402

W, H = 1200, 676
FPS = 24
MONO = font("SFNSMono.ttf", 19)
TAG = ui_font(32, "Medium")
CHIP = ui_font(21, "Demi")
CARD_TITLE = ui_font(24, "Demi")
CARD_SUB = ui_font(16, "Medium")
BODY = ui_font(19, "Regular")
BODY_B = ui_font(19, "Demi")
PANEL = (24, 19, 17)
PANEL_HI = (34, 27, 23)
PANEL_EDGE = (52, 42, 37)
TRACK = (46, 38, 33)

frames = []

def base():
    img = Image.new("RGB", (W, H), BG)
    glow = Image.new("RGB", (W, H), BG)
    ImageDraw.Draw(glow).ellipse((W * 0.25, H * 0.8, W * 0.75, H * 1.5), fill=(58, 22, 9))
    return Image.blend(img, glow.filter(ImageFilter.GaussianBlur(110)), 1.0)

def put(img, layer, x, y):
    img.paste(layer, (int(x), int(y)), layer)

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

def ease_out(t, p=4):          # ease-out-quart: fast start, long soft settle
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** p

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

# --------------------------------------------------------------- beat 1: logo (5.2 s)
def letters_layer(size, t, start, stagger=0.16, dur=1.1):
    """'kiln' drawn letter by letter: each letter fades in from a soft blur and drifts
    into place. Returns an RGBA layer sized to the word."""
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
    warm = Image.new("RGB", (W, H), BG)
    ImageDraw.Draw(warm).ellipse((W * 0.3, H * 0.55, W * 0.7, H * 1.3), fill=(70, 26, 10))
    img = Image.blend(img, warm.filter(ImageFilter.GaussianBlur(120)), 0.6 * ease_out(t / 3.0, 2))

    size_text = 168
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

    glow = ease_out(t / 1.4, 3)
    arch_a = ease_out((t - 0.6) / 1.6, 4)
    scale = 0.965 + 0.035 * arch_a
    mark = render_mark(mark_size, t=t, glow=glow, stroke_color=lerp(BG, EMBER, arch_a))
    if scale < 1:
        ms = int(mark_size * scale)
        mark = mark.resize((ms, ms), Image.LANCZOS)
        put(img, mark, x0 + (mark_size - ms) / 2, y_mark + (mark_size - ms) / 2)
    else:
        put(img, mark, x0, y_mark)

    layer, lbb = letters_layer(size_text, t, start=1.5)
    put(img, layer, x0 + mark_size + gap - 24, y_text - 24)

    text_center_a(img, H / 2 + 92, "run AI models on the phone itself", TAG, INK, ease_out((t - 2.6) / 1.2, 3))
    text_center_a(img, H / 2 + 136, "chat · decide · embed, no cloud, no account", TAG, EMBER_HI, ease_out((t - 3.1) / 1.2, 3))
    return fade(img, 1 - ease((t - (T - 0.6)) / 0.6))

# --------------------------------------------------------------- beat 2: three cards (10 s)
CARD_W, CARD_H, CARD_Y = 350, 470, 72
CARD_X = [60, 425, 790]

CHAT_Q = "Why is the sky blue?"
CHAT_A = ("Air scatters blue light far more than red, because blue has a shorter "
          "wavelength, so blue reaches your eyes from every direction.")
DECIDE_MSG = "My refund never arrived and I was charged twice."
DECIDE_OPTS = [("billing", 0.779), ("shipping", 0.018), ("technical", 0.202)]
EMBED_Q = "when does the train leave?"
EMBED_HITS = [("speech_train.wav", 0.836, "audio"),
              ("The train to Leeds leaves from platform 4 at 9:15.", 0.790, None),
              ("The dentist moved my appointment to Thursday.", 0.620, None),
              ("red_circle.png", 0.518, "image")]

def card(img, i, title, sub, alpha):
    """Rounded card with a title row; returns (draw, x, y) of the content origin."""
    x, y = CARD_X[i], CARD_Y
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle((x, y, x + CARD_W, y + CARD_H), 20, fill=PANEL + (255,), outline=PANEL_EDGE + (255,), width=2)
    d.text((x + 22, y + 18), title, font=CARD_TITLE, fill=INK + (255,))
    d.text((x + 22, y + 50), sub, font=CARD_SUB, fill=DIM + (255,))
    d.line((x + 22, y + 80, x + CARD_W - 22, y + 80), fill=PANEL_EDGE + (255,), width=1)
    layer.putalpha(layer.split()[3].point(lambda v: int(v * max(0.0, min(1.0, alpha)))))
    img.alpha_composite(layer)
    return x + 22, y + 96

def bar(d, x, y, w, frac, label, right, color=EMBER, lf=BODY):
    d.text((x, y), label, font=lf, fill=INK)
    d.text((x + w - d.textlength(right, font=MONO), y + 2), right, font=MONO, fill=DIM)
    d.rounded_rectangle((x, y + 30, x + w, y + 38), 4, fill=TRACK)
    if frac > 0:
        d.rounded_rectangle((x, y + 30, x + max(8, w * frac), y + 38), 4, fill=color)

def chat_card(img, t):
    a = ease(t / 0.5)
    if a <= 0:
        return
    x, y = card(img, 0, "Chat", "Qwen3.5 0.8B · thinking off", a)
    if t < 0.5:
        return
    d = ImageDraw.Draw(img)
    cw = CARD_W - 44
    # user bubble, right aligned
    qw = d.textlength(CHAT_Q, font=BODY) + 28
    d.rounded_rectangle((x + cw - qw, y, x + cw, y + 40), 14, fill=PANEL_HI)
    d.text((x + cw - qw + 14, y + 9), CHAT_Q, font=BODY, fill=INK)
    y += 62
    n = int(len(CHAT_A) * ease((t - 0.8) / 3.2))
    done = t > 4.0
    for ln in wrap(d, CHAT_A[:n], BODY, cw):
        d.text((x, y), ln, font=BODY, fill=INK); y += 28
    if n and not done and int(t * 3) % 2 == 0:
        cx = x + d.textlength(wrap(d, CHAT_A[:n], BODY, cw)[-1], font=BODY) + 3
        d.rectangle((cx, y - 26, cx + 8, y - 6), fill=EMBER)
    if t > 0.9:
        tok = int(33 * n / len(CHAT_A))
        tps = 37.0 + 0.4 * math.sin(t * 9)
        d.text((x, CARD_Y + CARD_H - 44), f"{tps:4.1f} tok/s · {tok} tokens · cache reused",
               font=CARD_SUB, fill=EMBER_HI if not done else DIM)

def decide_card(img, t):
    a = ease(t / 0.5)
    if a <= 0:
        return
    x, y = card(img, 1, "Decide", "LFM2 d1-omni 600M · one forward pass", a)
    if t < 0.5:
        return
    d = ImageDraw.Draw(img)
    cw = CARD_W - 44
    d.rounded_rectangle((x, y, x + cw, y + 68), 14, fill=PANEL_HI)
    yy = y + 10
    for ln in wrap(d, DECIDE_MSG, BODY, cw - 28):
        d.text((x + 14, yy), ln, font=BODY, fill=INK); yy += 26
    y += 86
    d.text((x, y), "Which team should handle this?", font=BODY_B, fill=INK); y += 44
    grow = ease((t - 1.2) / 0.8)
    for i, (name, p) in enumerate(DECIDE_OPTS):
        bar(d, x, y, cw, p * grow, name, f"{p * 100:4.1f}%" if grow > 0 else "",
            color=EMBER if i == 0 else (120, 60, 30))
        y += 58
    if t > 2.0:
        d.text((x, CARD_Y + CARD_H - 44), "billing · 77.9% sure · 87 tokens read · 0.2 s",
               font=CARD_SUB, fill=EMBER_HI)

def icon(d, x, y, kind):
    d.rounded_rectangle((x, y, x + 28, y + 28), 7, fill=PANEL_HI)
    if kind == "audio":
        for i, h in enumerate((6, 12, 18, 10, 14)):
            d.line((x + 6 + i * 4, y + 14 - h / 2, x + 6 + i * 4, y + 14 + h / 2), fill=INK, width=2)
    elif kind == "image":
        d.ellipse((x + 7, y + 7, x + 21, y + 21), fill=(230, 60, 50))

def embed_card(img, t):
    a = ease(t / 0.5)
    if a <= 0:
        return
    x, y = card(img, 2, "Embed and search", "EmbeddingGemma 2 · text, images, audio", a)
    if t < 0.5:
        return
    d = ImageDraw.Draw(img)
    cw = CARD_W - 44
    d.rounded_rectangle((x, y, x + cw, y + 42), 14, fill=PANEL_HI, outline=lerp(PANEL_EDGE, EMBER, ease((t - 0.6) / 0.6)))
    n = int(len(EMBED_Q) * ease((t - 0.6) / 1.0))
    d.text((x + 14, y + 10), EMBED_Q[:n], font=BODY, fill=INK)
    y += 60
    show = ease((t - 1.8) / 0.7)
    for i, (label, score, kind) in enumerate(EMBED_HITS):
        ai = ease((t - 1.8 - i * 0.15) / 0.5)
        if ai <= 0:
            break
        lx = x
        if kind:
            icon(d, x, y - 2, kind); lx = x + 38
        lbl = label if d.textlength(label, font=BODY) < cw - 70 else label[:int(len(label) * (cw - 90) / d.textlength(label, font=BODY))] + "…"
        bar(d, x, y, cw, score * ai, "", f"{score:.3f}", color=EMBER if i == 0 else (120, 60, 30))
        d.text((lx, y), lbl, font=BODY, fill=lerp(PANEL, INK, ai))
        y += 60
    if show >= 1:
        d.text((x, CARD_Y + CARD_H - 44), "query 59 ms · 768 numbers each · 12 notes",
               font=CARD_SUB, fill=EMBER_HI)

def beat_cards(t, T=10.0):
    img = base().convert("RGBA")
    chat_card(img, t - 0.2)
    decide_card(img, t - 1.6)
    embed_card(img, t - 3.0)
    img = img.convert("RGB")
    text_center_a(img, CARD_Y + CARD_H + 36, "three kinds of model, one library, one phone",
                  TAG, INK, ease_out((t - 6.2) / 1.0, 3))
    text_center_a(img, CARD_Y + CARD_H + 80, "the numbers are real: Pixel 8, measured, in the README",
                  ui_font(22, "Medium"), DIM, ease_out((t - 6.8) / 1.0, 3))
    return fade(img, min(ease(t / 0.4), 1 - ease((t - (T - 0.4)) / 0.4)))

# --------------------------------------------------------------- beat 3: end card (6 s)
CHIPS = ["C API", "kiln_flutter on pub.dev", "any GGUF", "Android", "llama.cpp core", "MIT"]
NUMS = [("111 tok/s", "LFM2.5 350M chat, Pixel 8"),
        ("0.21 s", "one decision, 79 tokens, Pixel 8"),
        ("39 ms", "one text embedding, Pixel 8")]
NEXT = "next: speech to text, text to speech · Kotlin, React Native and Swift bindings · iOS"

def beat_end(t, T=6.0):
    img = base()
    a = ease(t / 0.6)
    lk = lockup(88, 98, t=t, mark_alpha=a, text_alpha=a)
    put(img, lk, 100, 60)
    d = ImageDraw.Draw(img)
    big = ui_font(66, "Demi")
    for i, (bigs, small) in enumerate(NUMS):
        ai = ease((t - 0.5 - i * 0.25) / 0.5)
        x = 100 + i * 360
        d.text((x, 225), bigs, font=big, fill=lerp(BG, EMBER_HI, ai))
        d.text((x + 3, 308), small, font=ui_font(21, "Medium"), fill=lerp(BG, DIM, ai))
    x = 100
    for i, c in enumerate(CHIPS):
        ai = ease((t - 1.4 - i * 0.12) / 0.4)
        w = d.textlength(c, font=CHIP) + 30
        d.rounded_rectangle((x, 400, x + w, 444), 22, outline=lerp(BG, EMBER, ai), width=2)
        d.text((x + 15, 409), c, font=CHIP, fill=lerp(BG, INK, ai))
        x += w + 10
    d.text((100, 478), NEXT, font=ui_font(22, "Medium"), fill=lerp(BG, DIM, ease((t - 2.4) / 0.6)))
    d.text((100, 548), "github.com/RaihanulHaque/kiln", font=TAG, fill=lerp(BG, INK, ease((t - 2.9) / 0.6)))
    d.text((100, 592), "built at CortexTech · cortextech.dev", font=ui_font(24, "Medium"), fill=lerp(BG, EMBER_HI, ease((t - 3.2) / 0.6)))
    return fade(img, 1 - ease((t - (T - 0.5)) / 0.5))

def render(beat, T):
    for i in range(int(T * FPS)):
        frames.append(beat(i / FPS, T))

BEATS = [(5.2, beat_logo), (10.0, beat_cards), (6.0, beat_end)]
if __name__ == "__main__":
    for T, beat in BEATS:
        render(beat, T)

    out = sys.argv[1] if len(sys.argv) > 1 else "kiln_promo"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    mp4 = out + ".mp4"
    with tempfile.TemporaryDirectory() as td:
        for i, f in enumerate(frames):
            f.save(os.path.join(td, f"f{i:05d}.png"))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", os.path.join(td, "f%05d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", mp4], check=True)
    # GIF for the README: 720 px wide, 12 fps, one 128-colour palette for the whole clip and
    # only the changed rectangle stored per frame. PIL's writer made the same clip 3.5x larger.
    gif = out + ".gif"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-vf",
                    "fps=12,scale=720:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];"
                    "[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle", gif], check=True)
    print(mp4, os.path.getsize(mp4) // 1024, "KB;", gif, os.path.getsize(gif) // 1024, "KB;", len(frames), "frames")
