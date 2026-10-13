# Server context checkpoints

A model with sliding-window attention (SWA), recurrent layers, or a hybrid of both cannot drop the tail of its memory the way a plain KV cache can. When a new prompt shares a prefix with the slot's cached prompt but then differs, the server needs a saved copy of that state at (or before) the point where the prompts differ. These copies are the context checkpoints. Each slot keeps up to `--ctx-checkpoints N` of them (default 32), and the server restores the newest one that is usable for the new prompt. Without a usable checkpoint, the server processes the whole prompt again.

This page describes the delphi.cpp options that decide which checkpoints a slot keeps. The upstream options `--ctx-checkpoints` and `--checkpoint-min-step` are unchanged.

## Eviction

When a slot already holds `--ctx-checkpoints` checkpoints and takes a new one, it drops one.

| option | default | effect |
| --- | --- | --- |
| `--ctx-checkpoints-eviction {fifo,variance}` | `variance` | `fifo` drops the oldest checkpoint, as upstream does. `variance` drops the interior checkpoint whose removal leaves the most uniform spacing. |

Upstream first drops the checkpoints that are closer than `--checkpoint-min-step` to the checkpoint before them (when the slot is nearly full, and never one that the current request made). Both policies keep that step. The policy applies only when the slot is still full afterwards.

`variance` looks at the positions of the stored checkpoints and at the position of the new one. Removing a checkpoint merges the two gaps next to it. The number of gaps and their sum do not change, so the variance of the gaps is smallest for the checkpoint with the smallest product of its two gaps. That checkpoint goes. The oldest checkpoint is never interior, so it stays, and the new checkpoint is the last point, so the newest stored checkpoint can go when it sits close to the new one. Ties go to the older checkpoint.

Why `variance` is the default: after a long session, a fifo slot holds only the most recent positions. A request that rewinds (an edited message, a regenerated reply, a client that rewrites old tool output) then finds no checkpoint before the change and the server processes the whole prompt again. A uniform spread keeps a checkpoint near every part of the prompt. The cost of the rule is a loop over the checkpoints of one slot, once per new checkpoint, which is negligible.

Prior art: ik_llama.cpp PR 2020 (`--ctx-checkpoints-eviction`). The delphi.cpp version differs in that it counts the new checkpoint as the last point of the sequence, and it works on `n_tokens` of the checkpoint instead of `pos_max`.
