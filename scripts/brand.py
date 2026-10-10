"""kiln brand mark: an arch (the kiln door) with a single soft ember inside.

Shared by make_promo.py and the static asset exporter. Everything is drawn at
SS x supersampling and downscaled, so edges are smooth at any size.
"""
import math

from PIL import Image, ImageDraw, ImageFilter, ImageFont

F = "/System/Library/Fonts/"
BG = (16, 12, 11)
INK = (238, 231, 225)
DIM = (142, 130, 122)
EMBER = (222, 86, 24)
EMBER_HI = (255, 160, 80)
CORE = (255, 232, 200)

def font(name, size, index=0):
    return ImageFont.truetype(F + name, size, index=index)

def wordmark_font(size):      # Avenir Next Demi Bold
    return font("Avenir Next.ttc", size, index=2)

def ui_font(size, weight="Medium"):
    idx = {"Regular": 7, "Medium": 5, "Demi": 2, "Bold": 0}[weight]
    return font("Avenir Next.ttc", size, index=idx)

def lerp(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(len(a)))

def ease(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)

def render_mark(size, t=0.0, draw_progress=1.0, glow=1.0, stroke_color=EMBER, ss=4):
    """RGBA image of the mark on a square canvas `size` px. Arch is built from filled masks
    (outer minus inner) so there are no polyline joints; feet get round caps.
    draw_progress: 0..1 reveals the arch bottom-up (used by the intro). glow: ember 0..1."""
    S = size * ss
    s = S * 0.80                         # arch outer height
    w = S * 0.62                         # arch outer width
    sw = S * 0.085                       # stroke
    cx, top = S / 2, (S - s) / 2
    left, right = cx - w / 2, cx + w / 2
    bottom = top + s
    r = w / 2

    def arch(mask, l, tp, rt, bt, rad, fill):
        md = ImageDraw.Draw(mask)
        md.rectangle((l, tp + rad, rt, bt), fill=fill)
        md.ellipse((l, tp, rt, tp + 2 * rad), fill=fill)

    m = Image.new("L", (S, S), 0)
    arch(m, left, top, right, bottom - sw / 2, r, 255)
    arch(m, left + sw, top + sw, right - sw, bottom + 1, r - sw, 0)
    md = ImageDraw.Draw(m)
    for fx in (left + sw / 2, right - sw / 2):   # round caps on the feet
        md.ellipse((fx - sw / 2, bottom - sw, fx + sw / 2, bottom), fill=255)
    if draw_progress < 1.0:                       # reveal: wipe from the feet upward
        cut = bottom - (bottom - top + sw) * ease(draw_progress)
        md.rectangle((0, 0, S, cut), fill=0)

    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    col = Image.new("RGBA", (S, S), stroke_color + (255,))
    img.paste(col, (0, 0), m)

    if glow > 0:
        b = 1 + 0.06 * math.sin(t * 2.2)
        ex, ey = cx, bottom - s * 0.30
        for rad, c, al in ((0.30 * b, EMBER, 110), (0.19 * b, EMBER_HI, 170), (0.085, CORE, 255)):
            layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
            rr = s * rad
            ImageDraw.Draw(layer).ellipse((ex - rr, ey - rr, ex + rr, ey + rr), fill=c + (int(al * glow),))
            layer = layer.filter(ImageFilter.GaussianBlur(rr * (0.55 if rad > 0.1 else 0.25)))
            img = Image.alpha_composite(img, layer)
    return img.resize((size, size), Image.LANCZOS)

def lockup(mark_px, text_px, gap=None, t=0.0, mark_alpha=1.0, text_alpha=1.0, draw_progress=1.0, bg=BG):
    """Mark + 'kiln' wordmark. The arch spans exactly ascender-top to baseline of the
    wordmark; mark_px is ignored for height and kept for API compatibility. Returns RGBA."""
    f = wordmark_font(text_px)
    tmp = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    bb = tmp.textbbox((0, 0), "kiln", font=f)           # l, t, r, b  (b = baseline, no descenders)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    arch_h = th                                          # arch outer height == ascender height
    size = int(round(arch_h / 0.80))                     # render_mark draws the arch at 0.80 of canvas
    gap = gap if gap is not None else int(th * 0.30)
    W, H = size + gap + tw, max(size, th)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    mark = render_mark(size, t=t, draw_progress=draw_progress, glow=mark_alpha)
    if mark_alpha < 1:
        mark.putalpha(mark.split()[3].point(lambda v: int(v * mark_alpha)))
    top_pad = (size - arch_h) / 2                        # arch top inside the mark canvas
    text_y = (H - th) / 2
    img.alpha_composite(mark, (0, int(round(text_y - top_pad))))
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(layer).text((size + gap - bb[0], text_y - bb[1]), "kiln", font=f, fill=INK + (int(255 * text_alpha),))
    img.alpha_composite(layer)
    return img

def export_static(out_dir):
    import os
    os.makedirs(out_dir, exist_ok=True)
    # mark alone, transparent + on dark
    for px in (256, 1024):
        render_mark(px).save(f"{out_dir}/kiln-mark-{px}.png")
    m = render_mark(1024)
    dark = Image.new("RGBA", (1024, 1024), BG + (255,)); dark.alpha_composite(m); dark.save(f"{out_dir}/kiln-mark-dark-1024.png")
    # lockup on dark, padded
    lk = lockup(0, 200)
    pad = 80
    card = Image.new("RGBA", (lk.width + 2 * pad, lk.height + 2 * pad), BG + (255,))
    card.alpha_composite(lk, (pad, pad)); card.save(f"{out_dir}/kiln-lockup-dark.png")
    lk.save(f"{out_dir}/kiln-lockup.png")
    # SVG of the mark (vector, same geometry)
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <defs>
    <radialGradient id="g" cx="50" cy="66" r="24" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#ffe8c8"/>
      <stop offset="0.28" stop-color="#ffa050" stop-opacity="0.9"/>
      <stop offset="1" stop-color="#de5618" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <circle cx="50" cy="66" r="24" fill="url(#g)"/>
  <path d="M23.25 85.75 V41 A26.75 26.75 0 0 1 76.75 41 V85.75" fill="none" stroke="#de5618" stroke-width="8.5" stroke-linecap="round"/>
</svg>
"""
    open(f"{out_dir}/kiln-mark.svg", "w").write(svg)
