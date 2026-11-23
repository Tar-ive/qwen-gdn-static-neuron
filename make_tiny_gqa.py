"""Random-weight Qwen3.5 text model with the Qwen3.8-27B head layout (48 value heads per 16 key heads in GDN,
24/4 attention heads, head_dim 256, partial rotary 0.25), shrunk in width/depth, to test the grouped-value path."""
import shutil, sys, torch
from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
out = sys.argv[1] if len(sys.argv) > 1 else "ckpt/tiny-gqa"
torch.manual_seed(0)
cfg = Qwen3_5TextConfig(vocab_size=248320, hidden_size=256, intermediate_size=512, num_hidden_layers=8,
    num_attention_heads=24, num_key_value_heads=4, head_dim=256, linear_num_key_heads=16, linear_num_value_heads=48,
    linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4, tie_word_embeddings=False,
    layer_types=["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3 + ["full_attention"],
    rope_parameters={"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25,
                     "mrope_interleaved": True, "mrope_section": [11, 11, 10]})
m = Qwen3_5ForCausalLM(cfg)
with torch.no_grad():  # larger-than-init weights so the GDN state and the UT systems are non-trivial
    for n, p in m.named_parameters():
        if p.ndim == 2 and "embed" not in n: p.mul_(4.0)
m.save_pretrained(out)
for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
    shutil.copy(f"ckpt/Qwen3.5-0.8B/{f}", out)
print("saved", out)
