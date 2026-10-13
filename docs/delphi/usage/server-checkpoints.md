# Server context checkpoints

A model with sliding-window attention (SWA), recurrent layers, or a hybrid of both cannot drop the tail of its memory the way a plain KV cache can. When a new prompt shares a prefix with the slot's cached prompt but then differs, the server needs a saved copy of that state at (or before) the point where the prompts differ. These copies are the context checkpoints. Each slot keeps up to `--ctx-checkpoints N` of them (default 32), and the server restores the newest one that is usable for the new prompt. Without a usable checkpoint, the server processes the whole prompt again.

This page describes the delphi.cpp options that decide which checkpoints a slot takes and keeps. The upstream options `--ctx-checkpoints` and `--checkpoint-min-step` are unchanged.

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

## Checkpoints at chat message ends

| option | default | effect |
| --- | --- | --- |
| `--ctx-checkpoints-turn-step N` | `4096` | A chat request gets checkpoints at the ends of its messages, at least N tokens apart. `0` turns it off. |

Upstream takes checkpoints a few tokens before the end of the prompt, at the start of the last user message, and at the starts of earlier user messages that are `--checkpoint-min-step` apart. The next request of a chat shares its prompt with this one up to where its new messages start, and a client that edits, regenerates or branches shares it up to the message before. The end of each message is therefore where a checkpoint is useful, for any role, and for any template.

Which message ends: the end of the leading system messages, the end of the last assistant message, and up to three more about N tokens apart, the latest first. The last message is not one of them, as its end is the end of the prompt. The first two are required: they are taken even when a checkpoint is already within N tokens, unless it is at the same position. A required end less than N / 4 tokens from the start is skipped, because a checkpoint there saves little (and the oldest checkpoint of a slot is pinned, so it should be worth it).

How the ends are found: the server renders the chat template for the messages up to the message in question, without the generation prompt, and tokenizes the text the way the prompt is tokenized. The end counts only if that text is the start of the prompt text, and those tokens are the first tokens of the prompt tokens. The prompt is processed in batches that stop at these positions, and the checkpoint is taken there. This needs no knowledge of the template, but some templates change an earlier message when a later one is added, for example by dropping the reasoning of earlier assistant turns. The end of such a message does not pass the test and is left out. Upstream's checkpoints at the starts of user messages still apply there, for templates where the server knows the start of a user message.

When it applies: chat endpoints (the Responses and Anthropic endpoints too) with a jinja template, when the model makes checkpoints at all (SWA, recurrent or hybrid models, `--ctx-checkpoints` above 0). Not for requests that carry images or audio, not for `/completion`, and not for prompts shorter than N bytes.

Limit and eviction: these are ordinary checkpoints. They count against `--ctx-checkpoints`, they are dropped by `--ctx-checkpoints-eviction`, the oldest one is the pinned one, and upstream's spacing rule can drop the ones of earlier requests that are close to an earlier checkpoint. Only the spacing rule for taking one differs: a message end needs N tokens, not `--checkpoint-min-step`, from the newest checkpoint.

Why this default: N is half of upstream's spacing, so any message end of a long chat is at most a few thousand tokens of reprocessing away from a checkpoint. A smaller N takes more checkpoints, and each one is a copy of the SWA window or recurrent state on the host. A larger N leaves long turns without one. It is on by default because the cost is bounded: per request at most five template renders and tokenizations of a prefix (in the HTTP thread, together a few times the cost of tokenizing the prompt), at most five extra batch splits, and no work at all for a short prompt or a model without checkpoints. Use `--ctx-checkpoints-turn-step 0` to remove all of it.

## Interaction with the host prompt cache (`--cache-ram`)

The prompt cache stores whole prompts of slots that other requests displaced: the prompt tokens, the full state of the sequence, and a copy of the slot's list of checkpoints. The copies count against `--cache-ram`, so a larger `--ctx-checkpoints` means that fewer prompts fit in the cache. A prompt that returns from the cache brings back its list with it, including the pinned checkpoint. The cache picks the stored prompt by the length of the shared token prefix only; it does not check that a checkpoint exists at or before that prefix. For SWA and recurrent models, a cached prompt without such a checkpoint is processed again from the start, so the choice of the checkpoints matters for how much of a cache hit is really used.
