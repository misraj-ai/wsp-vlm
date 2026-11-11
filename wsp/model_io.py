"""Model I/O for Hugging Face vision-language models: generic loading, module discovery
(vision tower / connector / language model), batched generation, teacher-forced reads with
activation capture, and reversible weight swapping for "original" vs "edited" runs.

Only this module and the forward server touch the live model. The agent's Python kernel never
imports it: the kernel sees exported weight files, not a runnable network.
"""
from __future__ import annotations

import contextlib
import re

import torch
import torch.nn as nn

from .data import open_image

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
VISION_KEYS = ("vision", "visual", "image_encoder", "vit", "siglip", "clip", "img_encoder")
CONNECTOR_KEYS = ("projector", "connector", "merger", "aligner", "mm_proj", "resampler", "adapter")
MLP_KEYS = ("mlp", "feed_forward", "ffn", "fc")
ATTN_KEYS = ("attn", "attention", "mixer")


# --------------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------------
def load_model(mcfg: dict):
    import transformers as tf

    kw = {"trust_remote_code": mcfg.get("trust_remote_code", False)}
    if mcfg.get("revision"):
        kw["revision"] = mcfg["revision"]
    processor = tf.AutoProcessor.from_pretrained(mcfg["path"], **kw)
    tok = getattr(processor, "tokenizer", processor)
    tok.padding_side = "left"
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    mkw = dict(kw, dtype=DTYPES[mcfg.get("dtype", "bfloat16")])
    if mcfg.get("attn_implementation"):
        mkw["attn_implementation"] = mcfg["attn_implementation"]
    names = [mcfg["auto_class"]] if mcfg.get("auto_class") else [
        "AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModelForCausalLM"]
    err = None
    for n in names:
        cls = getattr(tf, n, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(mcfg["path"], **mkw)
            break
        except (ValueError, KeyError, OSError) as e:  # wrong auto class for this architecture
            err = e
    else:
        raise RuntimeError(f"could not load {mcfg['path']} with {names}: {err}")
    model.to(mcfg.get("device", "cuda")).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, processor


# --------------------------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------------------------
def _linears(mod: nn.Module, prefix: str):
    for n, m in mod.named_modules():
        if isinstance(m, nn.Linear):
            yield f"{prefix}.{n}" if n else prefix, m


def _pick_out_linear(layer: nn.Module, layer_name: str, keys, d_model: int | None, prefer: tuple):
    """Inside one block, the Linear that writes back to the residual stream from the sub-block
    whose name contains one of `keys` (MLP or attention)."""
    cands = []
    for n, m in _linears(layer, layer_name):
        local = n[len(layer_name) + 1:]
        if not any(k in local.split(".")[0] or k in local for k in keys):
            continue
        if d_model is not None and m.out_features != d_model:
            continue
        cands.append((n, m))
    if not cands:
        return None
    for p in prefer:
        for n, m in cands:
            if n.split(".")[-1] == p:
                return n
    return cands[-1][0]


def _norm_width(m):
    if isinstance(m, nn.LayerNorm) and m.normalized_shape:
        return int(m.normalized_shape[0])
    w = getattr(m, "weight", None)
    if "norm" in type(m).__name__.lower() and w is not None and getattr(w, "ndim", 0) == 1:
        return int(w.shape[0])
    return None


def _layer_d_model(layer: nn.Module) -> int | None:
    """Residual width of a block: its own (direct-child) norms; per-head q/k norms inside attention
    must not be mistaken for it. Fallback: the widest norm in the block."""
    direct = [w for _, c in layer.named_children() if (w := _norm_width(c))]
    if direct:
        return max(set(direct), key=direct.count)
    allw = [w for m in layer.modules() if (w := _norm_width(m))]
    return max(allw) if allw else None


def discover(model: nn.Module, overrides: dict | None = None) -> dict:
    o = overrides or {}
    mods = dict(model.named_modules())
    lists = [(n, m) for n, m in mods.items() if isinstance(m, nn.ModuleList) and len(m) > 0
             and all(isinstance(c, nn.Module) for c in m)]

    def is_vision(n):
        return any(k in n.lower() for k in VISION_KEYS)

    def choose(pattern, want_vision):
        c = [(n, m) for n, m in lists if (re.search(pattern, n) if pattern else
                                          (is_vision(n) == want_vision and not any(k in n.split(".")[-1].lower() for k in CONNECTOR_KEYS)))]
        c = [(n, m) for n, m in c if len(m) >= 2]
        return max(c, key=lambda x: len(x[1]))[0] if c else None

    lm = choose(o.get("lm_layers"), False)
    vis = choose(o.get("vision_layers"), True)
    if lm is None:
        raise RuntimeError("could not find the language-model layer list; set model.discovery.lm_layers")

    if o.get("connector"):
        conn = [n for n in mods if re.search(o["connector"], n)]
    else:
        conn = [n for n in mods if n and any(k in n.split(".")[-1].lower() for k in CONNECTOR_KEYS)
                and not n.startswith(lm + ".")]
    conn = [n for n in conn if not any(n.startswith(c + ".") for c in conn)]

    def per_layer(list_name):
        out = []
        for i, layer in enumerate(mods[list_name]):
            ln = f"{list_name}.{i}"
            d = _layer_d_model(layer)
            w = (_pick_out_linear(layer, ln, MLP_KEYS, d, ("down_proj", "fc2", "w2", "c_proj", "dense_4h_to_h", "wo"))
                 if not o.get("writer") else next((n for n in mods if n.startswith(ln + ".") and re.search(o["writer"], n)), None))
            a = (_pick_out_linear(layer, ln, ATTN_KEYS, d, ("o_proj", "out_proj", "dense", "c_proj", "proj"))
                 if not o.get("attn_out") else next((n for n in mods if n.startswith(ln + ".") and re.search(o["attn_out"], n)), None))
            out.append({"name": ln, "d_model": d, "mlp_writer": w, "attn_writer": a,
                        "mlp_writer_weight": w + ".weight" if w else None, "attn_writer_weight": a + ".weight" if a else None,
                        "kind": type(layer).__name__,
                        "mixer": next((k for k in ("linear_attn", "self_attn", "attn", "mixer")
                                       if hasattr(layer, k)), None)})
        return out

    pname = {id(p): n for n, p in model.named_parameters()}
    ie = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    oe = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    frozen = set()
    for m in (ie, oe):
        if m is not None and getattr(m, "weight", None) is not None and id(m.weight) in pname:
            frozen.add(pname[id(m.weight)])
    lm_parent = lm.rsplit(".", 1)[0] if "." in lm else ""
    for n, _ in model.named_parameters():
        rest = n[len(lm_parent) + 1:] if lm_parent and n.startswith(lm_parent + ".") else None
        if rest is not None and "." in rest and rest.count(".") == 1 and rest.split(".")[0] != lm.rsplit(".", 1)[-1] \
                and ("norm" in rest.split(".")[0] or rest.split(".")[0] == "ln_f"):
            frozen.add(n)  # the language model's final norm (norm, final_layernorm, embedding_norm, ln_f, ...)

    cfg = getattr(model, "config", None)
    img_tok = None
    for k in ("image_token_id", "image_token_index"):
        if cfg is not None and getattr(cfg, k, None) is not None:
            img_tok = int(getattr(cfg, k))
            break
    vision_root = None
    if vis:
        parts = vis.split(".")
        for j in range(len(parts)):
            if any(k in parts[j].lower() for k in VISION_KEYS):
                vision_root = ".".join(parts[: j + 1])
                break
    n_params = {"total": sum(p.numel() for p in model.parameters())}
    for key, prefix in (("language_layers", lm), ("vision_layers", vis)):
        if prefix:
            n_params[key] = sum(p.numel() for n, p in model.named_parameters() if n.startswith(prefix + "."))
    n_params["connector"] = sum(p.numel() for n, p in model.named_parameters() if any(n.startswith(c + ".") for c in conn))
    return {
        "architecture": type(model).__name__,
        "lm_layers": lm, "lm": per_layer(lm),
        "vision_layers": vis, "vision_root": vision_root, "vision": per_layer(vis) if vis else [],
        "connector": conn,
        "frozen": sorted(frozen),
        "image_token_id": img_tok,
        "n_params": n_params,
    }


def describe(disc: dict, model: nn.Module, max_names: int = 60) -> str:
    """Plain-text model description for the agent's first message."""
    lines = [f"Architecture: {disc['architecture']}; parameters: {disc['n_params']}"]
    lm = disc["lm"]
    lines.append(f"Language model: {len(lm)} blocks under '{disc['lm_layers']}', width {lm[0]['d_model']}; "
                 f"block kinds: {sorted(set(b['kind'] + ('/' + b['mixer'] if b['mixer'] else '') for b in lm))}")
    lines.append(f"  example tensors that write the residual stream: MLP {lm[0]['mlp_writer_weight']}, attention {lm[0]['attn_writer_weight']} "
                 "(MODEL['lm'][i] lists them for every block; nn.Linear weights are [out, in])")
    if disc["vision"]:
        v = disc["vision"]
        lines.append(f"Vision tower: {len(v)} blocks under '{disc['vision_layers']}', width {v[0]['d_model']}; "
                     f"example writers {v[0]['mlp_writer_weight']}, {v[0]['attn_writer_weight']} (MODEL['vision'][i])")
    lines.append(f"Connector module(s): {disc['connector']}")
    lines.append(f"Frozen (edits refused): {disc['frozen']}")
    lines.append(f"Image placeholder token id: {disc['image_token_id']}")
    lines.append("Tensor names outside the repeated blocks:")
    rep = [disc["lm_layers"]] + ([disc["vision_layers"]] if disc["vision_layers"] else [])
    shown = 0
    for n, p in model.named_parameters():
        if any(re.match(re.escape(r) + r"\.\d+\.", n) for r in rep):
            continue
        lines.append(f"  {n} {tuple(p.shape)}")
        shown += 1
        if shown >= max_names:
            lines.append("  ...")
            break
    for key in ("lm", "vision"):
        if disc[key]:
            first = disc[key][0]["name"]
            lines.append(f"Tensor names of one {key} block ({first}):")
            for n, p in model.named_parameters():
                if n.startswith(first + "."):
                    lines.append(f"  {n} {tuple(p.shape)}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# Weight swap
# --------------------------------------------------------------------------------------------
class WeightSwapper:
    """Applies an edit (name -> tensor) in place and restores the originals afterwards."""

    def __init__(self, model: nn.Module):
        self.params = dict(model.named_parameters())
        self._orig: dict[str, torch.Tensor] = {}

    @contextlib.contextmanager
    def applied(self, edits: dict[str, torch.Tensor] | None):
        if not edits:
            yield
            return
        try:
            for n, t in edits.items():
                p = self.params[n]
                if n not in self._orig:
                    self._orig[n] = p.detach().to("cpu", copy=True)
                p.data.copy_(t.to(device=p.device, dtype=p.dtype))
            yield
        finally:
            for n in edits:
                if n in self._orig:
                    self.params[n].data.copy_(self._orig[n].to(self.params[n].device))


# --------------------------------------------------------------------------------------------
# Prompting
# --------------------------------------------------------------------------------------------
def _messages(prompt: str, system: str | None):
    m = []
    if system:
        m.append({"role": "system", "content": [{"type": "text", "text": system}]})
    m.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]})
    return m


def prompt_text(processor, prompt: str, protocol: dict, chat_kwargs: dict) -> str:
    return processor.apply_chat_template(_messages(prompt, protocol.get("system_prompt")),
                                         add_generation_prompt=True, tokenize=False, **(chat_kwargs or {}))


def _eos_ids(model, processor) -> set:
    ids = set()
    g = getattr(model, "generation_config", None)
    e = getattr(g, "eos_token_id", None)
    if isinstance(e, int):
        ids.add(e)
    elif e:
        ids.update(e)
    tok = getattr(processor, "tokenizer", processor)
    if tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    return ids


# --------------------------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------------------------
@torch.inference_mode()
def generate(model, processor, items: list[dict], tasks: dict, protocol: dict, chat_kwargs: dict | None = None) -> list[dict]:
    tok = getattr(processor, "tokenizer", processor)
    eos = _eos_ids(model, processor)
    bs = max(1, int(protocol.get("batch_size", 8)))
    gk = {"max_new_tokens": int(protocol["max_new_tokens"]), "do_sample": bool(protocol.get("do_sample")),
          "pad_token_id": tok.pad_token_id}
    if gk["do_sample"]:
        gk.update({"temperature": protocol.get("temperature", 1.0), "top_p": protocol.get("top_p", 1.0)})
        if protocol.get("top_k"):
            gk["top_k"] = protocol["top_k"]
    if protocol.get("repetition_penalty", 1.0) != 1.0:
        gk["repetition_penalty"] = protocol["repetition_penalty"]
    if protocol.get("no_repeat_ngram_size"):
        gk["no_repeat_ngram_size"] = protocol["no_repeat_ngram_size"]
    out = []
    dev = next(model.parameters()).device
    for b in range(0, len(items), bs):
        batch = items[b:b + bs]
        texts = [prompt_text(processor, it.get("prompt") or tasks[it["task"]]["prompt"], protocol, chat_kwargs) for it in batch]
        imgs = [open_image(it["image"], protocol.get("image_max_side")) for it in batch]
        enc = processor(text=texts, images=imgs, return_tensors="pt", padding=True).to(dev)
        if "pixel_values" in enc and enc["pixel_values"].is_floating_point():
            enc["pixel_values"] = enc["pixel_values"].to(next(model.parameters()).dtype)
        torch.manual_seed(int(protocol.get("seed", 0)) * 1_000_003 + b)
        gen = model.generate(**enc, **gk)
        new = gen[:, enc["input_ids"].shape[1]:]
        for row in new.tolist():
            n = len(row)
            for j, t in enumerate(row):
                if t in eos:
                    n = j + 1
                    break
            finished = any(t in eos for t in row[:n])
            text = tok.decode(row[:n], skip_special_tokens=True)
            out.append({"text": text, "n_tokens": n, "hit_max": (not finished) and n >= gk["max_new_tokens"]})
    return out


# --------------------------------------------------------------------------------------------
# Teacher-forced reads with activation capture
# --------------------------------------------------------------------------------------------
def _first_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (tuple, list)):
        for y in x:
            t = _first_tensor(y)
            if t is not None:
                return t
    if hasattr(x, "last_hidden_state"):
        return x.last_hidden_state
    if hasattr(x, "pooler_output") and x.pooler_output is not None:
        return x.pooler_output
    return None


def _layers(spec, n):
    if spec in (None, False):
        return []
    if spec in ("all", True):
        return list(range(n))
    return [i if i >= 0 else n + i for i in spec]


class Capture:
    """Registers forward hooks for a capture spec; tensors land in self.store[key] (detached, CPU)."""

    def __init__(self, model, disc, spec):
        self.mods = dict(model.named_modules())
        self.disc, self.spec, self.store, self.handles = disc, spec or {}, {}, []

    def __enter__(self):
        lm, vis = self.disc["lm"], self.disc["vision"]
        plan = []
        for i in _layers(self.spec.get("lm_resid"), len(lm)):
            plan.append((f"lm.resid.{i}", lm[i]["name"], "out"))
        for i in _layers(self.spec.get("lm_mlp_out"), len(lm)):
            plan.append((f"lm.mlp_out.{i}", lm[i]["mlp_writer"], "out"))
        for i in _layers(self.spec.get("lm_mlp_act"), len(lm)):
            plan.append((f"lm.mlp_act.{i}", lm[i]["mlp_writer"], "in"))
        for i in _layers(self.spec.get("lm_attn_out"), len(lm)):
            plan.append((f"lm.attn_out.{i}", lm[i]["attn_writer"], "out"))
        for i in _layers(self.spec.get("vision_resid"), len(vis)):
            plan.append((f"vision.resid.{i}", vis[i]["name"], "out"))
        for i in _layers(self.spec.get("vision_mlp_act"), len(vis)):
            plan.append((f"vision.mlp_act.{i}", vis[i]["mlp_writer"], "in"))
        if self.spec.get("connector"):
            for c in self.disc["connector"]:
                plan.append((f"connector.{c}", c, "out"))
        for key, name, where in plan:
            if not name or name not in self.mods:
                raise ValueError(f"capture {key}: module not found ({name})")
            m = self.mods[name]
            if where == "out":
                self.handles.append(m.register_forward_hook(self._hook_out(key)))
            else:
                self.handles.append(m.register_forward_pre_hook(self._hook_in(key)))
        return self

    def _hook_out(self, key):
        def f(mod, inp, out):
            t = _first_tensor(out)
            if t is not None:
                self.store[key] = t.detach().float().cpu()
        return f

    def _hook_in(self, key):
        def f(mod, inp):
            t = _first_tensor(inp)
            if t is not None:
                self.store[key] = t.detach().float().cpu()
        return f

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()


@torch.inference_mode()
def teacher_forced(model, processor, disc, item: dict, prompt: str, target: str, protocol: dict,
                   spec: dict, chat_kwargs: dict | None = None, append_eos: bool = True) -> dict:
    """One item: score `target` as the assistant reply to (image, prompt). Returns token ids,
    per-token log-probs, top-k alternatives and the captured activations at the chosen positions."""
    tok = getattr(processor, "tokenizer", processor)
    dev = next(model.parameters()).device
    ptxt = prompt_text(processor, prompt, protocol, chat_kwargs)
    full = ptxt + target + (tok.eos_token if append_eos and tok.eos_token else "")
    img = open_image(item["image"], protocol.get("image_max_side"))
    enc_p = processor(text=[ptxt], images=[img], return_tensors="pt")
    enc = processor(text=[full], images=[img], return_tensors="pt").to(dev)
    if "pixel_values" in enc and enc["pixel_values"].is_floating_point():
        enc["pixel_values"] = enc["pixel_values"].to(next(model.parameters()).dtype)
    ids = enc["input_ids"][0]
    pl = enc_p["input_ids"].shape[1]
    if not torch.equal(ids[:pl].cpu(), enc_p["input_ids"][0]):
        common = 0
        for a, b in zip(ids.tolist(), enc_p["input_ids"][0].tolist()):
            if a != b:
                break
            common += 1
        pl = common
    T = ids.shape[0]
    want_attn = spec.get("lm_attn")
    with Capture(model, disc, spec) as cap:
        out = model(**enc, output_attentions=bool(want_attn), use_cache=False)
    logits = out.logits[0].float()
    logp = torch.log_softmax(logits[:-1], -1)
    tgt = ids[1:]
    tok_lp = logp.gather(-1, tgt[:, None])[:, 0]
    k = int(spec.get("topk", 0) or 0)
    res = {"input_ids": ids.cpu(), "prompt_len": pl, "target_logprobs": tok_lp[pl - 1:].cpu()}
    if k:
        v, i = logp[pl - 1:].topk(k, -1)
        res["topk_logprobs"], res["topk_ids"] = v.cpu(), i.cpu()
    pos_kind = spec.get("positions", "target")
    if pos_kind == "target":
        pos = torch.arange(pl, T)
    elif pos_kind == "prompt_last":
        pos = torch.tensor([pl - 1])
    elif pos_kind == "image":
        if disc["image_token_id"] is None:
            raise ValueError("positions=image: model has no image placeholder token id")
        pos = (ids.cpu() == disc["image_token_id"]).nonzero()[:, 0]
    elif pos_kind == "all":
        pos = torch.arange(T)
    else:
        raise ValueError(f"unknown positions '{pos_kind}'")
    res["positions"] = pos
    reduce = spec.get("reduce", "none")
    acts = {}
    for key, t in cap.store.items():
        if key.startswith("lm."):
            t = t[0] if t.dim() == 3 else t
            if t.shape[0] == T:
                t = t[pos]
        else:  # vision / connector outputs: their own token axis (patches)
            t = t.reshape(-1, t.shape[-1])
        acts[key] = t.mean(0, keepdim=True) if reduce == "mean" else t
    res["acts"] = acts
    if want_attn:
        if out.attentions is None:
            raise ValueError("attention patterns need model.attn_implementation: eager in the config")
        res["attn"] = {i: out.attentions[i][0].half().cpu() for i in _layers(want_attn, len(out.attentions))}
    return res
