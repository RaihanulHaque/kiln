#!/usr/bin/env python3
"""Turn the OCRia-nano-1 ONNX files into one GGUF that kiln reads (kiln_ocr).

    pip install onnx numpy
    python scripts/convert_ocria.py <dir with det.onnx, router.onnx, rec_*.onnx, *.json> ocria-nano-1.gguf

The models come from https://huggingface.co/llamaindex/OCRia-nano-1 under the OCRia-nano-1 Model
License (free for research, personal use and small organisations; read it before a product
uses it). The converted file is a derivative of the weights under the same license: convert
it yourself, do not publish it without those terms.

What goes in the file, for each of the six models (det, router, rec_latin, rec_cjk, rec_rtl,
rec_world):
  - its weights, as F32 tensors named "<model>.<n>" (ONNX names are longer than ggml allows);
  - its graph, as JSON in "ocria.<model>.graph": the nodes in order, each with its op, input
    and output names and attributes. kiln builds the same computation in ggml from it.
Two rewrites make the graph smaller and easier to run:
  - DequantizeLinear on an int8 weight is applied here, so the weight is stored as F32;
  - a QuantizeLinear followed by a DequantizeLinear on an activation (rec_latin is quantized
    to 8 bits) becomes one "FakeQuant" node: round to the 8-bit grid and back. Without it the
    Latin recognizer gives different numbers, because the rounding is part of what it learned
    (it also stands in for the ReLUs the quantizer removed).
  - The recognizers end in a CTC decode (ArgMax, ReduceMax, ReduceLogSumExp). The graph is cut
    before it: kiln reads the per-step letter scores ("logits") and decodes them itself.
Character sets and the detector threshold go in as metadata; LayoutNet (tables) is not
converted.
"""
import json
import os
import sys

import numpy as np
import onnx
from onnx import numpy_helper

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party", "llama.cpp", "gguf-py"))
import gguf  # noqa: E402

MODELS = ["det", "router", "rec_latin", "rec_cjk", "rec_rtl", "rec_world"]
GROUPS = ["latin", "cjk", "rtl", "world"]
# ops kiln implements (src/backends/ocr/ocr_graph.cpp); anything else stops the conversion
SUPPORTED = {"Conv", "Relu", "Sigmoid", "Add", "Sub", "Mul", "Resize", "Concat", "ReduceSum",
             "ReduceMean", "ReduceMax", "Squeeze", "Gemm", "FakeQuant", "Exp"}
CTC_TAIL = {"ArgMax", "ReduceMax", "ReduceLogSumExp", "Sub", "Exp"}


def attrs_of(node):
    out = {}
    for a in node.attribute:
        v = onnx.helper.get_attribute_value(a)
        if isinstance(v, bytes):
            v = v.decode()
        elif isinstance(v, (list, tuple)):
            v = [x.decode() if isinstance(x, bytes) else x for x in v]
        elif hasattr(v, "dims"):          # a tensor attribute (Constant)
            v = numpy_helper.to_array(v).tolist()
        out[a.name] = v
    return out


def convert(path, name, writer):
    m = onnx.load(path)
    g = m.graph
    init = {t.name: numpy_helper.to_array(t) for t in g.initializer}
    consts = {}
    for n in g.node:
        if n.op_type == "Constant":
            consts[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)

    def value(x):
        return init[x] if x in init else consts.get(x)

    producers = {o: n for n in g.node for o in n.output}
    consumers = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)

    # where the CTC tail starts: the input of ArgMax
    logits = None
    for n in g.node:
        if n.op_type == "ArgMax":
            logits = n.input[0]
    tail_ops = set()
    if logits is not None:
        # every node that (transitively) reads the logits belongs to the decode
        todo = [logits]
        while todo:
            t = todo.pop()
            for c in consumers.get(t, []):
                if id(c) not in tail_ops:
                    tail_ops.add(id(c))
                    todo.extend(c.output)

    tensors = {}       # ONNX name -> stored tensor name
    weights = {}       # ONNX name -> F32 array: weights after dequantization
    for k, v in init.items():
        if v.dtype == np.float32 and v.ndim >= 1:
            weights[k] = v

    nodes = []
    alias = {}         # an activation's DequantizeLinear output -> the FakeQuant output it equals

    def resolve(x):
        return alias.get(x, x)

    for n in g.node:
        op = n.op_type
        if op == "Constant" or id(n) in tail_ops:
            continue
        if op == "DequantizeLinear":
            x, s = n.input[0], value(n.input[1])
            zp = value(n.input[2]) if len(n.input) > 2 and n.input[2] else np.array(0)
            if x in init:                       # a weight: dequantize now
                q = init[x].astype(np.float32)
                a = attrs_of(n).get("axis", 1)
                if s.ndim == 1 and s.size > 1:  # per channel
                    shape = [1] * q.ndim
                    shape[a] = -1
                    w = (q - zp.astype(np.float32).reshape(shape)) * s.astype(np.float32).reshape(shape)
                else:
                    w = (q - float(zp)) * float(s)
                weights[n.output[0]] = w.astype(np.float32)
                continue
            # an activation: the matching QuantizeLinear made the FakeQuant
            p = producers.get(x)
            if p is None or p.op_type != "QuantizeLinear":
                raise SystemExit(f"{name}: DequantizeLinear on {x} without a QuantizeLinear")
            alias[n.output[0]] = p.output[0]
            continue
        if op == "QuantizeLinear":
            s = float(value(n.input[1]))
            zp_arr = value(n.input[2])
            zp = int(zp_arr)
            lo, hi = (0, 255) if zp_arr.dtype == np.uint8 else (-128, 127)
            nodes.append({"op": "FakeQuant", "in": [resolve(n.input[0])], "out": [n.output[0]],
                          "attrs": {"scale": s, "zero_point": zp, "qmin": lo, "qmax": hi}})
            continue
        if op not in SUPPORTED:
            raise SystemExit(f"{name}: op {op} is not supported by kiln")
        ins, attrs = [], attrs_of(n)
        for i in n.input:
            if not i:
                ins.append("")
                continue
            i = resolve(i)
            if i in consts:                     # small constants become attributes
                attrs.setdefault("const", {})[i] = consts[i].tolist()
                ins.append("const:" + i)
            else:
                ins.append(i)
        if op == "Conv":
            w = weights[ins[1]]
            if w.ndim == 3:                     # 1-D conv: store as 2-D with height 1
                weights[ins[1]] = w.reshape(w.shape[0], w.shape[1], 1, w.shape[2])
        nodes.append({"op": op, "in": ins, "out": list(n.output), "attrs": attrs})

    # store every weight a node reads, under a short name
    used = {i for nd in nodes for i in nd["in"] if i in weights}
    for k in sorted(used):
        t = f"{name}.{len(tensors)}"
        tensors[k] = t
        writer.add_tensor(t, np.ascontiguousarray(weights[k], dtype=np.float32))
    for nd in nodes:
        nd["in"] = [("w:" + tensors[i]) if i in tensors else i for i in nd["in"]]

    inp = [i.name for i in g.input if i.name not in init][0]
    outs = [resolve(logits)] if logits is not None else [o.name for o in g.output]
    graph = {"input": inp, "outputs": outs, "nodes": nodes}
    writer.add_string(f"ocria.{name}.graph", json.dumps(graph, separators=(",", ":")))
    ops = sorted({nd["op"] for nd in nodes})
    print(f"  {name}: {len(nodes)} nodes, {len(tensors)} tensors, "
          f"{sum(weights[k].size for k in used)} weights, ops {ops}, outputs {outs}")


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]
    w = gguf.GGUFWriter(dst, "ocria")
    w.add_name("OCRia-nano-1")
    w.add_string("ocria.license", "OCRia-nano-1 Model License (https://huggingface.co/llamaindex/OCRia-nano-1/blob/main/LICENSE)")
    for name in MODELS:
        convert(os.path.join(src, name + ".onnx"), name, w)
    for gname in GROUPS:
        with open(os.path.join(src, f"rec_{gname}.charset.json"), encoding="utf-8") as f:
            chars = json.load(f)
        w.add_array(f"ocria.rec_{gname}.charset", chars)
        meta = os.path.join(src, f"rec_{gname}.meta.json")
        rtl = False
        if os.path.exists(meta):
            with open(meta, encoding="utf-8") as f:
                rtl = bool(json.load(f).get("visual_rtl", False))
        w.add_bool(f"ocria.rec_{gname}.visual_rtl", rtl)
    det_meta = {}
    p = os.path.join(src, "det.meta.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            det_meta = json.load(f)
    w.add_float32("ocria.det.thresh", float(det_meta.get("thresh", 0.5)))
    w.add_float32("ocria.det.merge", float(det_meta.get("merge", 0.6)))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {dst} ({os.path.getsize(dst) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
