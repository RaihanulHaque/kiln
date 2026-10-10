"""Procedural ambient bed for the promo video: a slow sea-like swell of filtered noise over
a warm low drone, with soft crests at the story beats. No samples, nothing to license."""
import numpy as np
from scipy.signal import butter, sosfilt


def _bump(t, at, width):
    return np.exp(-0.5 * ((t - at) / width) ** 2)


def render(duration, beats, sr=48000, seed=7):
    """beats: list of (time_s, strength) where the swell should crest."""
    rng = np.random.default_rng(seed)
    n = int(duration * sr)
    t = np.arange(n) / sr

    # --- sea: brown noise, band-limited, breathing slowly
    white = rng.standard_normal(n)
    brown = np.cumsum(white)
    brown -= sosfilt(butter(1, 15, "hp", fs=sr, output="sos"), brown)          # leak DC
    brown = sosfilt(butter(2, 700, "lp", fs=sr, output="sos"), brown)
    brown /= np.max(np.abs(brown)) + 1e-9
    breathe = 0.55 + 0.45 * (0.5 * np.sin(2 * np.pi * 0.07 * t - 1.2) + 0.5 * np.sin(2 * np.pi * 0.11 * t + 0.4))
    crest = np.zeros(n)
    for at, strength in beats:
        crest += strength * _bump(t, at, 1.1)
    sea = brown * (0.30 * breathe + 0.45 * np.clip(crest, 0, 1))

    # --- drone: root, fifth, octave, slightly detuned, slow tremolo
    def voice(f, amp, det=0.0, trem=0.0):
        return amp * np.sin(2 * np.pi * (f + det) * t) * (1 - trem + trem * 0.5 * (1 + np.sin(2 * np.pi * 0.19 * t)))
    drone = (voice(55.0, 0.22, 0, 0.3) + voice(55.0, 0.10, 0.4) + voice(82.41, 0.10, 0, 0.5)
             + voice(110.0, 0.06, -0.3) + voice(164.8, 0.03, 0.2, 0.6))
    drone *= 0.6 + 0.4 * np.clip(crest, 0, 1)

    # --- ember ignition: a brief warm bloom (short low sine tail) at the first beat
    ember = np.zeros(n)
    if beats:
        at = beats[0][0]
        env = np.clip(t - at, 0, None)
        ember = 0.18 * np.sin(2 * np.pi * 41.2 * t) * np.exp(-env / 1.4) * (t >= at)

    mix = sea + drone + ember
    fade_in = np.clip(t / 2.0, 0, 1) ** 2
    fade_out = np.clip((duration - t) / 1.8, 0, 1) ** 1.5
    mix *= fade_in * fade_out
    mix /= np.max(np.abs(mix)) + 1e-9
    return (0.32 * mix).astype(np.float32)          # about -10 dBFS peak: present, never loud


def write_wav(path, audio, sr=48000):
    from scipy.io import wavfile
    wavfile.write(path, sr, audio)
