"""
Compatibility patch for loading Pythia models into TransformerLens.

Bug: transformer_lens 3.5.1's GPT-NeoX weight-conversion code
(transformer_lens/pretrained/weight_conversions/neox.py) reads
`neox.embed_out.weight` to get the unembedding matrix. But `transformers`
>=5.x renamed that attribute to `lm_head` for GPTNeoXForCausalLM (verified
directly: a fresh GPTNeoXForCausalLM has children ['gpt_neox', 'lm_head'],
no `embed_out`) -- so `HookedTransformer.from_pretrained("EleutherAI/pythia-*")`
crashes with `AttributeError: 'GPTNeoXForCausalLM' object has no attribute
'embed_out'` on the exact transformer_lens/transformers version pair this
project already depends on for GPT-2 (transformer_lens>=5.4.0 is a hard
requirement of transformer_lens 3.5.1, so downgrading transformers to an
older version that still has `embed_out` is not viable -- it breaks other
model families transformer_lens also imports, e.g. olmo2).

Fix: monkeypatch convert_neox_weights to alias `lm_head` onto `embed_out`
on the loaded HF model before calling the original conversion function --
a one-line compatibility shim, not a reimplementation. Import this module
BEFORE calling HookedTransformer.from_pretrained on any Pythia model:

    import pythia_compat  # noqa: F401 -- side-effecting import, applies the patch
    model = HookedTransformer.from_pretrained("EleutherAI/pythia-410m")

Verified locally: Pythia-410M loads correctly with this patch (24 layers,
d_model=1024, 16 heads, matching the model's published config), and every
5%-rounded percentage 0-100 still tokenizes as a single token under
Pythia's tokenizer (same property GPT-2's BPE has, which the whole
single-forward-pass evaluation methodology in Phases B/C/D/E depends on).
"""
import transformer_lens.loading_from_pretrained as _loading_module
import transformer_lens.pretrained.weight_conversions.neox as _neox_module

_original_convert_neox_weights = _neox_module.convert_neox_weights


def _convert_neox_weights_patched(neox, cfg):
    if not hasattr(neox, "embed_out") and hasattr(neox, "lm_head"):
        neox.embed_out = neox.lm_head
    return _original_convert_neox_weights(neox, cfg)


# Patch BOTH the defining module (in case anything imports the module and
# calls it via attribute access) AND loading_from_pretrained's own
# `from .weight_conversions import convert_neox_weights` local binding
# (which is the actual call site HookedTransformer.from_pretrained uses --
# reassigning the origin module's attribute alone does NOT reach an
# already-bound `from X import Y` reference in another module).
_neox_module.convert_neox_weights = _convert_neox_weights_patched
_loading_module.convert_neox_weights = _convert_neox_weights_patched
