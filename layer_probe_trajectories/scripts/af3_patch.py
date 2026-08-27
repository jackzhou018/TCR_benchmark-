"""In-memory source patches that make AlphaFold 3 return its intermediates.

AF3 runs its 48 Pairformer blocks through ``hk.experimental.layer_stack``,
a ``jax.lax.scan`` over stacked parameters, so there is nothing to attach an
ordinary hook to. Instead we take the *installed* source of the three
functions involved, apply a small textual edit at a unique anchor, and exec
the result back into the owning module's namespace. Every edit is either

  * plumbing an already-computed value out (the diffusion trajectory is
    literally the scan's discarded ``y`` output), or
  * switching ``layer_stack`` to its ``with_per_layer_inputs=True`` form so
    the same scan also stacks the per-block pair state.

No arithmetic is touched. If an anchor is missing -- upstream changed -- the
patch raises rather than silently instrumenting the wrong thing. Parity with
the unpatched model under the same seed is checked by tests/test_af3_parity.py.
"""
from __future__ import annotations

import hashlib
import inspect
import textwrap

PATCHES: dict[str, str] = {}


def _rewrite(fn, replacements, module, name=None):
    """Rewrite `fn`'s installed source at fixed anchors and re-exec it.

    The source keeps its original indentation -- a method still carries its
    class indent -- so the anchors below can be copied verbatim out of the
    upstream file. A method is re-exec'd inside a throwaway class to make that
    indentation legal; dedenting instead would silently shift every anchor.
    """
    src = inspect.getsource(fn)
    digest = hashlib.sha256(src.encode()).hexdigest()
    for old, new in replacements:
        if old not in src:
            raise RuntimeError(
                f"anchor not found while patching {fn.__qualname__} "
                f"(installed source sha256={digest}):\n---\n{old}\n---"
            )
        src = src.replace(old, new, 1)
    ns = dict(module.__dict__)
    key = name or fn.__name__
    indented = src[:1].isspace()
    if indented:
        src = "class _LPTHolder:\n" + src
    exec(compile(src, f"<lpt_patch:{fn.__qualname__}>", "exec"), ns)
    PATCHES[fn.__qualname__] = digest
    return ns["_LPTHolder"].__dict__[key] if indented else ns[key]


# --------------------------------------------------------------------------
# 1. Evoformer: stack the per-block pair state and return it.
# --------------------------------------------------------------------------
_EVO_OLD_STACK = """      pairformer_stack = hk.experimental.layer_stack(
          self.config.pairformer.num_layer
      )(pairformer_fn)

      pair_activations, single_activations = pairformer_stack(
          (pair_activations, single_activations)
      )
"""
_EVO_NEW_STACK = """      _lpt_pre = pair_activations

      def _lpt_pairformer_fn(x):
        _out = pairformer_fn(x)
        return _out, _out[0]

      # hk.layer_stack derives its Haiku scope name from with_per_layer_inputs
      # ("__layer_stack_no_per_layer" vs "__layer_stack_with_per_layer"), so the
      # original name has to be passed explicitly or every stacked Pairformer
      # weight is looked up at a path the checkpoint does not have.
      pairformer_stack = hk.experimental.layer_stack(
          self.config.pairformer.num_layer, with_per_layer_inputs=True,
          name="__layer_stack_no_per_layer"
      )(_lpt_pairformer_fn)

      (pair_activations, single_activations), _lpt_layers = pairformer_stack(
          (pair_activations, single_activations)
      )
"""
_EVO_OLD_OUT = """      output = {
          'single': single_activations,
          'pair': pair_activations,
          'target_feat': target_feat,
      }"""
_EVO_NEW_OUT = """      output = {
          'single': single_activations,
          'pair': pair_activations,
          'target_feat': target_feat,
          'lpt_pair_pre': _lpt_pre.astype(jnp.bfloat16),
          'lpt_pair_layers': _lpt_layers.astype(jnp.bfloat16),
      }"""

# --------------------------------------------------------------------------
# 2. Model: carry the captured tensors through the recycle fori_loop so the
#    values that survive are the final recycle's, and emit them.
# --------------------------------------------------------------------------
_MODEL_OLD_INIT = """    embeddings = {
        'pair': jnp.zeros(
            [num_res, num_res, self.config.evoformer.pair_channel],
            dtype=jnp.float32,
        ),
        'single': jnp.zeros(
            [num_res, self.config.evoformer.seq_channel], dtype=jnp.float32
        ),
        'target_feat': target_feat,
    }"""
_MODEL_NEW_INIT = """    _lpt_nl = self.config.evoformer.pairformer.num_layer
    _lpt_pc = self.config.evoformer.pair_channel
    embeddings = {
        'pair': jnp.zeros(
            [num_res, num_res, self.config.evoformer.pair_channel],
            dtype=jnp.float32,
        ),
        'single': jnp.zeros(
            [num_res, self.config.evoformer.seq_channel], dtype=jnp.float32
        ),
        'target_feat': target_feat,
        'lpt_pair_pre': jnp.zeros(
            [num_res, num_res, _lpt_pc], dtype=jnp.bfloat16
        ),
        'lpt_pair_layers': jnp.zeros(
            [_lpt_nl, num_res, num_res, _lpt_pc], dtype=jnp.bfloat16
        ),
    }"""
_MODEL_OLD_OUT = """    output = {
        'diffusion_samples': samples,"""
_MODEL_NEW_OUT = """    output = {
        'lpt_pair_pre': embeddings['lpt_pair_pre'],
        'lpt_pair_layers': embeddings['lpt_pair_layers'],
        'diffusion_samples': samples,"""

# --------------------------------------------------------------------------
# 3. diffusion_head.sample: the scan already produces the per-step positions
#    as its stacked `y`; upstream throws them away into `_`.
# --------------------------------------------------------------------------
_DIFF_OLD = """  result, _ = hk.scan(apply_denoising_step, init, noise_levels[1:], unroll=4)
  _, positions_out, _ = result"""
_DIFF_NEW = """  result, _lpt_traj = hk.scan(
      apply_denoising_step, init, noise_levels[1:], unroll=4
  )
  _, positions_out, _ = result"""
_DIFF_OLD_RET = (
    "  return {'atom_positions': positions_out, 'mask': final_dense_atom_mask}"
)
_DIFF_NEW_RET = """  return {
      'atom_positions': positions_out,
      'mask': final_dense_atom_mask,
      'lpt_trajectory': _lpt_traj,
      'lpt_noise_levels': noise_levels,
  }"""


def _rebind_hk_method(cls, name: str, fn):
    """Install `fn` as `cls.name` the way Haiku's metaclass would.

    hk.Module's metaclass wraps every method at class-creation time so that
    parameters are created inside the module's name scope. Assigning a bare
    function afterwards skips that wrapping, and Haiku then looks the module's
    weights up at the wrong scope -- which surfaces as a confusing
    "Unable to retrieve parameter ... All parameters must be created as part
    of `init`". Re-applying wrap_method restores the scope.
    """
    from haiku._src.module import wrap_method

    setattr(cls, name, wrap_method(name, fn, lambda: cls))


def apply_patches() -> dict[str, str]:
    """Patch AF3 in place (module attributes only) and return source hashes."""
    from alphafold3.model import model as model_mod
    from alphafold3.model.network import diffusion_head, evoformer

    _rebind_hk_method(evoformer.Evoformer, "__call__", _rewrite(
        evoformer.Evoformer.__call__,
        [(_EVO_OLD_STACK, _EVO_NEW_STACK), (_EVO_OLD_OUT, _EVO_NEW_OUT)],
        evoformer,
        name="__call__",
    ))
    _rebind_hk_method(model_mod.Model, "__call__", _rewrite(
        model_mod.Model.__call__,
        [(_MODEL_OLD_INIT, _MODEL_NEW_INIT), (_MODEL_OLD_OUT, _MODEL_NEW_OUT)],
        model_mod,
        name="__call__",
    ))
    diffusion_head.sample = _rewrite(
        diffusion_head.sample,
        [(_DIFF_OLD, _DIFF_NEW), (_DIFF_OLD_RET, _DIFF_NEW_RET)],
        diffusion_head,
        name="sample",
    )
    return dict(PATCHES)
