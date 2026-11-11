"""Build a tiny, random Llava-style VLM (vision tower -> projector -> Llama LM) and synthetic task
data, entirely offline. Used only to test the harness plumbing; its outputs are meaningless.

    python scripts/make_toy.py --out toy
"""
import argparse
import json
import os
import random

import torch
from PIL import Image, ImageDraw
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import (CLIPImageProcessor, CLIPVisionConfig, LlamaConfig, LlavaConfig,
                          LlavaForConditionalGeneration, LlavaProcessor, PreTrainedTokenizerFast)

AR = "ابتثجحخدذرسشصضطظعغفقكلمنهويىةءأإآ"
TEMPLATE = ("{% for m in messages %}{{ '<|user|>' if m['role']=='user' else ('<|system|>' if m['role']=='system' else '<|assistant|>') }}"
            "{% if m['content'] is string %}{{ m['content'] }}{% else %}{% for c in m['content'] %}"
            "{% if c['type']=='image' %}<image>{% else %}{{ c['text'] }}{% endif %}{% endfor %}{% endif %}"
            "{% if m['role']=='assistant' %}</s>{% endif %}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}")


def build_model(out):
    corpus = ["abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789 .,:;|-<>/=_()#\n" * 3,
              "<table><tr><th>Name</th><th>Age</th></tr><tr><td>Ali</td><td>30</td></tr></table>",
              "| Name | Age |\n|---|---|\n| Ali | 30 |", AR * 3, "مرحبا بالعالم الاسم العمر جدول"]
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(corpus, trainers.BpeTrainer(
        vocab_size=700, special_tokens=["<unk>", "<s>", "</s>", "<pad>", "<image>", "<|user|>", "<|assistant|>", "<|system|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    ft = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", bos_token="<s>", eos_token="</s>", pad_token="<pad>")
    ft.add_special_tokens({"additional_special_tokens": ["<image>"]})
    ft.chat_template = TEMPLATE
    ip = CLIPImageProcessor(size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32})
    vc = CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                          image_size=32, patch_size=8, projection_dim=32)
    tc = LlamaConfig(vocab_size=len(ft), hidden_size=48, intermediate_size=96, num_hidden_layers=4, num_attention_heads=4,
                     num_key_value_heads=2, max_position_embeddings=1024, pad_token_id=ft.pad_token_id,
                     bos_token_id=ft.bos_token_id, eos_token_id=ft.eos_token_id, tie_word_embeddings=False)
    cfg = LlavaConfig(vision_config=vc, text_config=tc, image_token_index=ft.convert_tokens_to_ids("<image>"),
                      vision_feature_layer=-1, vision_feature_select_strategy="full")
    torch.manual_seed(0)
    model = LlavaForConditionalGeneration(cfg)
    proc = LlavaProcessor(image_processor=ip, tokenizer=ft, patch_size=8, vision_feature_select_strategy="full",
                          chat_template=TEMPLATE, num_additional_image_tokens=1)
    d = os.path.join(out, "model")
    model.save_pretrained(d)
    proc.save_pretrained(d)
    return d


def build_qwen35(out):
    """Tiny Qwen3.5 (hybrid linear/full attention LM, Qwen vision tower + merger). Needs torchvision."""
    import transformers as tf
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

    specials = ["<unk>", "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>",
                "<|image_pad|>", "<|video_pad|>", "<think>", "</think>"]
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(["abc def <table><tr><th>A</th></tr></table> " + AR + " 0123456789 user assistant\n"] * 5,
                            trainers.BpeTrainer(vocab_size=500, special_tokens=specials,
                                                initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    ft = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", eos_token="<|im_end|>", pad_token="<|endoftext|>")
    tmpl = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{% if m['content'] is string %}{{ m['content'] }}{% else %}"
            "{% for c in m['content'] %}{% if c['type']=='image' %}<|vision_start|><|image_pad|><|vision_end|>{% else %}"
            "{{ c['text'] }}{% endif %}{% endfor %}{% endif %}<|im_end|>\n{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
    ft.chat_template = tmpl
    ip = tf.Qwen2VLImageProcessor(min_pixels=32 * 32, max_pixels=64 * 64, patch_size=16, merge_size=2, temporal_patch_size=2)
    vp = Qwen3VLVideoProcessor(patch_size=16, merge_size=2, temporal_patch_size=2)
    proc = tf.Qwen3VLProcessor(image_processor=ip, tokenizer=ft, video_processor=vp, chat_template=tmpl)
    cfg = tf.AutoConfig.for_model("qwen3_5")
    tc, vc = cfg.text_config, cfg.vision_config
    for k, v in dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                     head_dim=32, vocab_size=len(ft), linear_num_value_heads=2, linear_num_key_heads=2, linear_key_head_dim=16,
                     linear_value_head_dim=16, eos_token_id=ft.eos_token_id, pad_token_id=ft.pad_token_id).items():
        setattr(tc, k, v)
    tc.layer_types = ["linear_attention", "full_attention"]
    for k, v in dict(hidden_size=32, intermediate_size=64, depth=2, num_heads=2, out_hidden_size=64, patch_size=16,
                     spatial_merge_size=2, temporal_patch_size=2, num_position_embeddings=16).items():
        setattr(vc, k, v)
    for k, t in (("image_token_id", "<|image_pad|>"), ("video_token_id", "<|video_pad|>"),
                 ("vision_start_token_id", "<|vision_start|>"), ("vision_end_token_id", "<|vision_end|>")):
        setattr(cfg, k, ft.convert_tokens_to_ids(t))
    cfg.eos_token_id = ft.eos_token_id
    torch.manual_seed(0)
    m = tf.Qwen3_5ForConditionalGeneration(cfg)
    m.generation_config.eos_token_id, m.generation_config.pad_token_id = ft.eos_token_id, ft.pad_token_id
    d = os.path.join(out, "model")
    m.save_pretrained(d)
    proc.save_pretrained(d)
    return d


def _img(path, text, table=None):
    im = Image.new("RGB", (160, 96), "white")
    dr = ImageDraw.Draw(im)
    if table:
        y = 4
        for row in table:
            dr.text((4, y), " | ".join(row), fill="black")
            y += 14
    else:
        dr.text((4, 4), text, fill="black")
    im.save(path)


def build_data(out, n_public=24, n_private=16):
    rnd = random.Random(0)
    words = ["alpha", "beta", "gamma", "delta", "table", "page", "line", "text"]
    for split, n in (("public", n_public), ("private", n_private)):
        root = os.path.join(out, "data", split)
        os.makedirs(os.path.join(root, "images"), exist_ok=True)
        for task in ("ocr", "arabic", "table", "handwriting"):
            rows = []
            for i in range(n):
                iid = f"{task}_{split}_{i:04d}"
                p = os.path.join("images", iid + ".png")
                if task == "table":
                    head = ["Name", "Age", "City"] if i % 2 else ["الاسم", "العمر", "المدينة"]
                    body = [[rnd.choice(words), str(rnd.randint(10, 90)), rnd.choice(words)] for _ in range(3)]
                    ref = "<table><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr>" + "".join(
                        "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in body) + "</table>"
                    _img(os.path.join(root, p), None, [head] + body)
                elif task in ("arabic", "handwriting"):
                    ref = " ".join("".join(rnd.choice(AR) for _ in range(rnd.randint(2, 6))) for _ in range(5))
                    _img(os.path.join(root, p), ref)
                else:
                    ref = " ".join(rnd.choice(words) for _ in range(6))
                    _img(os.path.join(root, p), ref)
                rows.append({"id": iid, "image": p, "reference": ref})
            with open(os.path.join(root, f"{task}.jsonl"), "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="toy")
    ap.add_argument("--arch", choices=["llava", "qwen35"], default="llava")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print("model:", build_model(a.out) if a.arch == "llava" else build_qwen35(a.out))
    build_data(a.out)
    print("data:", os.path.join(a.out, "data"))
