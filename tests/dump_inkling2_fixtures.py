# dump_inkling2_fixtures.py <pack dir> <out.json>: the oracle tests/test_inkling2.sh compares a server against.
# Needs mlx-lm with the inkling2 model (PR #2086 until it merges). Per prompt: the ids, the last row's top-10 logits
# after a whole-prompt prefill, and a greedy continuation fed one token at a time. The prompts cover a short one, the
# chat template, one past the sliding window (1024), one past the log-scaling floor (8192, two prefill chunks) and
# one with an image.
import json, sys
import mlx.core as mx
from mlx_lm import load

path, out = sys.argv[1], sys.argv[2]
model, tok = load(path)
cases = {
    "short": "The capital of France is",
    "chat": tok.apply_chat_template([{"role": "user", "content": "Write a haiku about rivers."}], tokenize=False, add_generation_prompt=True),
    "long": "The quick brown fox jumps over the lazy dog. " * 150,
    "xlong": "".join(f"{i}. Entry {i * 7919 % 1000} of the ledger was checked against line {i * 31 % 977} and filed.\n" for i in range(520)),
}
res = {}
for name, text in cases.items():
    ids = tok.encode(text, add_special_tokens=False)
    cache = model.make_cache()
    logits = model(mx.array([ids]), cache=cache)[0, -1].astype(mx.float32)
    mx.eval(logits)
    top = mx.argsort(-logits)[:10].tolist()
    entry = {"text": text, "ids": ids, "top_ids": top, "top_logits": [logits[i].item() for i in top]}
    gen = []
    nxt = int(mx.argmax(logits).item())
    for _ in range(24):
        gen.append(nxt)
        logits = model(mx.array([[nxt]]), cache=cache)[0, -1]
        nxt = int(mx.argmax(logits).item())
    entry["greedy_ids"] = gen
    entry["greedy_text"] = tok.decode(gen)
    res[name] = entry
    print(name, len(ids), repr(entry["greedy_text"]), top[:3], entry["top_logits"][:3], flush=True)

# One image (220 x 300, off the 40-pixel block grid, so the processor pads it): the tower's rows spliced into a chat.
import base64, struct, zlib
import numpy as np
from mlx_lm.models.inkling2.processor import Processor

def png(a):
    h, w, _ = a.shape
    raw = b"".join(b"\x00" + a[y].tobytes() for y in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")

img = np.full((220, 300, 3), 255, np.uint8)
img[20:120, 30:130] = (220, 30, 30)
img[90:200, 160:280] = (30, 60, 220)
yy, xx = np.ogrid[:220, :300]
img[(yy - 60) ** 2 + (xx - 220) ** 2 < 40 ** 2] = (30, 170, 60)
question = "What shapes and colors are in this image? Answer in one sentence."
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
text, extra = Processor()(text, images=[img])
ids = tok.encode(text, add_special_tokens=False)
cache = model.make_cache()
logits = model(mx.array([ids]), cache=cache, pixel_values=extra["pixel_values"])[0, -1].astype(mx.float32)
mx.eval(logits)
top = mx.argsort(-logits)[:10].tolist()
entry = {"image_png_b64": base64.b64encode(png(img)).decode(), "question": question, "ids": ids, "top_ids": top, "top_logits": [logits[i].item() for i in top]}
gen, nxt = [], int(mx.argmax(logits).item())
for _ in range(60):
    gen.append(nxt)
    nxt = int(mx.argmax(model(mx.array([[nxt]]), cache=cache)[0, -1]).item())
entry["greedy_ids"], entry["greedy_text"] = gen, tok.decode(gen)
res["image"] = entry
print("image", len(ids), repr(entry["greedy_text"]), flush=True)
json.dump(res, open(out, "w"))
