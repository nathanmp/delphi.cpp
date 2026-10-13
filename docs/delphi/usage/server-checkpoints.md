# Server context checkpoints

A model with sliding-window attention (SWA), recurrent layers, or a hybrid of both cannot drop the tail of its memory the way a plain KV cache can. When a new prompt shares a prefix with the slot's cached prompt but then differs, the server needs a saved copy of that state at (or before) the point where the prompts differ. These copies are the context checkpoints. Each slot keeps up to `--ctx-checkpoints N` of them (default 32), and the server restores the newest one that is usable for the new prompt. Without a usable checkpoint, the server processes the whole prompt again.

This page describes the delphi.cpp options that decide which checkpoints a slot keeps. The upstream options `--ctx-checkpoints` and `--checkpoint-min-step` are unchanged.

## Eviction

When a slot already holds `--ctx-checkpoints` checkpoints and takes a new one, it drops one.

| option | default | effect |
| --- | --- | --- |
| `--ctx-checkpoints-eviction {fifo,variance}` | `variance` | `fifo` drops the oldest checkpoint, as upstream does (but see the pin below). `variance` drops the interior checkpoint whose removal leaves the most uniform spacing. |

Upstream first drops the checkpoints that are closer than `--checkpoint-min-step` to the checkpoint before them (when the slot is nearly full, and never one that the current request made). Both policies keep that step. The policy applies only when the slot is still full afterwards.

`variance` looks at the positions of the stored checkpoints and at the position of the new one. Removing a checkpoint merges the two gaps next to it. The number of gaps and their sum do not change, so the variance of the gaps is smallest for the checkpoint with the smallest product of its two gaps. That checkpoint goes. The oldest checkpoint is not interior, so it stays, and the new checkpoint is the last point, so the newest stored checkpoint can go when it sits close to the new one. Ties go to the older checkpoint.

Why `variance` is the default: after a long session, a fifo slot holds only the most recent positions. A request that rewinds (an edited message, a regenerated reply, a client that rewrites old tool output) then finds no checkpoint before the change and the server processes the whole prompt again. A uniform spread keeps a checkpoint near every part of the prompt. The cost of the rule is a loop over the checkpoints of one slot, once per new checkpoint, which is negligible.

Prior art: ik_llama.cpp PR 2020 (`--ctx-checkpoints-eviction`). The delphi.cpp version differs in that it counts the new checkpoint as the last point of the sequence, and it works on `n_tokens` of the checkpoint instead of `pos_max`.

## Pin the first checkpoint

| option | default | effect |
| --- | --- | --- |
| `--ctx-checkpoints-pin-first`, `--no-ctx-checkpoints-pin-first` | enabled | A full slot never drops its oldest checkpoint. |

The oldest checkpoint of a slot is usually the end of the system prompt. That prefix is the only part that different conversations share, for example the instructions and tool definitions of an agent harness. When a new conversation arrives in a slot that holds an old one, the prompts differ right after the system prompt, and the server can restart from the pinned checkpoint instead of processing the system prompt again. Every other checkpoint belongs to one conversation only. Without the pin, a long conversation pushes its oldest checkpoint out first, and the next conversation pays for the whole system prompt.

How the pin works with each policy:

- `fifo` drops the second oldest checkpoint instead of the oldest.
- `variance` already never drops the oldest checkpoint, because it removes interior checkpoints only. `--no-ctx-checkpoints-pin-first` lets it consider the oldest one too, by comparing the variance that remains after removing it.
- The upstream spacing rule never removes the oldest checkpoint either.
- With a limit of one checkpoint (`--ctx-checkpoints 1`), the pin cannot hold: the single checkpoint is dropped for the new one.

The pin does not keep a checkpoint that no longer matches the prompt. When a new prompt differs from the cached one before a checkpoint, the server erases that checkpoint, as it always did. A checkpoint that the server replaces at the same position takes the place of the old one in the list, so it stays the oldest.

Cost: one of the `--ctx-checkpoints` places holds the pinned checkpoint. Its memory is the size of one checkpoint.

## Interaction with the host prompt cache (`--cache-ram`)

The prompt cache stores whole prompts of slots that other requests displaced: the prompt tokens, the full state of the sequence, and a copy of the slot's list of checkpoints. The copies count against `--cache-ram`, so a larger `--ctx-checkpoints` means that fewer prompts fit in the cache. A prompt that returns from the cache brings back its list with it, including the pinned checkpoint. The cache picks the stored prompt by the length of the shared token prefix only; it does not check that a checkpoint exists at or before that prefix. For SWA and recurrent models, a cached prompt without such a checkpoint is processed again from the start, so the choice of the checkpoints matters for how much of a cache hit is really used.
