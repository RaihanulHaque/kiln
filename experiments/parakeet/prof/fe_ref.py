# Front end (NeMo dw_striding subsampling) in float64 from the .nemo weights, against kiln's dumps.
import sys, numpy as np, torch, torch.nn.functional as F
S = sys.argv[1]
def rd(p):
    b = open(p, 'rb').read(); h = np.frombuffer(b[:16], np.int32)
    return h, np.frombuffer(b[16:], np.float32)
h, mel = rd(f'{S}/mel.bin'); n_mel, n_len = h[0], h[1]
mel = torch.tensor(mel.reshape(n_len, n_mel), dtype=torch.float64)   # (T, F)
sd = {k: v.double() for k, v in torch.load('nemo110/model_weights.ckpt', map_location='cpu').items() if k.startswith('encoder.pre_encode')}
p = 'encoder.pre_encode.'
x = mel[None, None]                                                    # (1, 1, T, F)
x = F.relu(F.conv2d(x, sd[p+'conv.0.weight'], sd[p+'conv.0.bias'], stride=2, padding=1))
x = F.conv2d(x, sd[p+'conv.2.weight'], sd[p+'conv.2.bias'], stride=2, padding=1, groups=x.shape[1])
x = F.relu(F.conv2d(x, sd[p+'conv.3.weight'], sd[p+'conv.3.bias']))
x = F.conv2d(x, sd[p+'conv.5.weight'], sd[p+'conv.5.bias'], stride=2, padding=1, groups=x.shape[1])
x = F.relu(F.conv2d(x, sd[p+'conv.6.weight'], sd[p+'conv.6.bias']))
b, c, t, f = x.shape
x = x.transpose(1, 2).reshape(b, t, c * f)
ref = F.linear(x, sd[p+'out.weight'], sd[p+'out.bias'])[0].numpy()   # (T', 512)
for name in sys.argv[2:]:
    h, v = rd(f'{S}/{name}.bin'); v = v.reshape(h[1], h[0]).astype(np.float64)
    n = min(len(v), len(ref)); d = np.abs(v[:n] - ref[:n])
    print(f'{name}: frames {len(v)} vs {len(ref)}, max|d| {d.max():.2e}, rel rms {np.sqrt((d**2).mean()/(ref[:n]**2).mean()):.2e}')
for name in sys.argv[2:]:
    h, v = rd(f'{S}/{name}.bin'); v = v.reshape(h[1], h[0]).astype(np.float64)
    per = np.sqrt(((v - ref) ** 2).mean(1) / (ref ** 2).mean(1))
    print(name, 'per-frame rel rms:', ' '.join(f'{x:.0e}' for x in per))
