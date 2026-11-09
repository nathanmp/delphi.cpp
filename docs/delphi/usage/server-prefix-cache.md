# Server prefix cache

Agent harnesses start each conversation with the same long prefix: the instructions, the tool definitions, often project notes. The prefix cache keeps the model state at the end of such a shared prefix, so a new conversation processes only the tokens after it. The state stays valid across server restarts when you give the cache a directory.

It is on by default with a small RAM budget, and it does nothing until two conversations share a prefix of at least `--cache-prefix-min-tokens` tokens.

```
llama-server -m <model> ... --cache-prefix-dir /nvme/llama-prefix
```

## Why the other caches do not cover this

- A slot reuses the start of its own prompt. When another conversation used the slot in between, that prompt is gone from the slot.
- The host prompt cache (`--cache-ram`, upstream) stores whole prompts. It does not use a cached prompt that shares less than a quarter of its tokens with the new prompt, because the load takes the entry out of the cache. A 15K-token system prompt in front of a 100K-token conversation is below that. It also skips prompts larger than its budget, and it is empty after a restart.
- Context checkpoints (see [Server context checkpoints](server-checkpoints.md)) belong to one slot and one prompt.

## How it works

1. **Finding a shared prefix.** When a prompt starts processing, the server compares it with the prompts that it holds: the prompts of the slots, of the host prompt cache, and the stored prefixes. The longest common prefix after which the two prompts differ is a shared prefix, if it is at least `--cache-prefix-min-tokens` tokens and it ends before the first assistant message of the new prompt. One conversation alone never makes a snapshot. The first assistant message bound keeps out the places where a conversation differs from its own earlier requests (a reply that tokenizes differently, an edited later message); the host prompt cache handles those.
2. **Taking the snapshot.** The slot stops its prompt batch at the end of the shared prefix and stores the full state of the sequence: the target context, the draft context and the speculative state. The first repeat of a prefix pays for this once (one drain of the prompt pipeline and a copy to host memory).
3. **Restoring.** A later prompt that starts with a stored prefix, and that the slot cannot serve further from its own memory, gets the state back and processes only the rest. The server restores only when the estimated time (disk read, counted twice, plus the copy to the device) is less than the estimated time to process the tokens it saves. The estimates come from measured speeds; before any prompt is timed, a gain of at least `--cache-prefix-min-tokens` counts as worth it.

A snapshot is a full state, so it restores models with sliding-window or recurrent layers without a checkpoint.

## Options

| option | default | effect |
| --- | --- | --- |
| `--cache-prefix-mib N` | min(4096, RAM / 16) | RAM for the snapshots, apart from `--cache-ram`. `0` turns the cache off, `-1` removes the limit. A snapshot larger than half the budget is not kept. |
| `--cache-prefix-min-tokens N` | `1024` | the shortest shared prefix that gets a snapshot |
| `--cache-prefix-dir DIR` | off | also write the snapshots to files in DIR |
| `--cache-prefix-disk-mib N` | `16384` | the most disk space that the files of one model use |

Each option also has an environment variable (`LLAMA_ARG_CACHE_PREFIX_MIB`, `..._MIN_TOKENS`, `..._DIR`, `..._DISK_MIB`).

RAM eviction drops the snapshots that never had a hit first, then the ones that were used the longest ago. A snapshot that is dropped from RAM stays on disk.

## Files

- Name: `dpfx-<model key>-<n_tokens>-<token hash>.bin`. The model key covers the model and draft files (path, size, time), the context size per sequence, the KV types, flash attention and the SWA setting. A change of any of them gives other files.
- Format version 1: a 72-byte header, the tokens of the prefix, then the data of the target context, the draft context and the speculative state, in the byte order of the machine. A checksum covers the tokens and the data.
- The server writes the files from a thread of its own, with at most two in the queue. When the queue is full, the snapshot stays in RAM only.
- At start, the server reads the headers and tokens of the files with its model key. A file that does not pass the checks is removed. When a read fails its checksum, or the state does not restore, the file is removed and the prompt is processed as usual.
- Servers of different models can share a directory: a server leaves the files of other model keys alone, and its disk budget counts only its own files. The files of a model that no server uses any more stay until you remove them.
- The time of a file is its last use, so a later server keeps the recently used ones when it needs room.

## Limits

- Chat requests only (`/v1/chat/completions`, `/v1/messages`, `/v1/responses`) with a jinja template, as the bound needs the message list. Not for `/completion`, prompts with images or audio, LoRA adapters, `--cache-reuse` or aLoRA.
- The snapshot is the full state at that point, so its size grows with the prefix: for example 39 MiB for 3K tokens of Ling-3.0-tiny, 260 MiB for 3K tokens of Gemma 4 26B with its MTP draft, 1.5 GiB for 3K tokens of llama-2-7b (f16 KV, no GQA).
- With `--cache-ram 0`, the server learns a shared prefix only while the first conversation is still in a slot.
- A harness that changes text inside the shared part for each conversation (a date, a directory listing) moves the shared prefix to before that text.

## Validation (2026-11-08, R9700)

- Restoring a snapshot gives the same logprobs, bit for bit, as reusing the same prefix in the slot: llama-2-7b (plain KV), Ling-3.0-tiny (hybrid), GPT-OSS-20B (SWA), Granite-4.0-H-Small (hybrid recurrent), Gemma 4 26B with an MTP draft (SWA, same draft acceptance with a fixed draft depth). The same holds for a restore from a file after a restart.
- Cost on the R9700: restore about 22 GB/s (1.5 GiB in 68 ms), snapshot 1.5-2 GB/s (it includes the drain of the prompt pipeline).
- `tests/test-server-prefix-cache.cpp` covers the selection rules, the RAM eviction, the files across restarts, damaged files and the disk budget.
