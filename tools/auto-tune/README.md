# Launch-setting tuner

`tune.py` finds faster `llama-server` settings for one model on the devices you already chose. You give it your usual
command and a set of prompts. It tries other batch sizes, thread counts and speculative decoding settings in one running
server, keeps what is faster, and checks the result against your command with fresh launches. It writes a report and the
tuned command. It does not change your placement (`--device`, `-ts`, `-ngl`, `-ot`), context size or KV cache type.

```sh
python3 tools/auto-tune/tune.py --out /tmp/tune-mymodel -- \
    build/bin/llama-server -m model.gguf -ngl 999 -fa on -c 16384 --device ROCm0,ROCm1
```

When it finishes, `report.md` in the output directory has the tuned command and a comparison with your command, by
prompt domain. `command.txt` has the command alone. A run takes one to three hours for a large model.

## What it tunes

The tuner works in two phases. Each phase changes one setting at a time and keeps a change only when it is at least 2 %
better. The server re-creates its context for every change, which takes seconds, so the weights load only a few times.

Phase 1 tunes `-ub`, `-b`, `-ubd` (with a draft model or MTP), `-t` and `-tb`. For models with experts on the CPU
(`-cmoe`, `--n-cpu-moe`, or `-ot` rules that put experts on the CPU) it also tunes the expert cache margin
(`LLAMA_MOE_CACHE_AUTO_MARGIN_MIB`). The score combines decode speed (80 %) and prompt speed at 256 and 2048 tokens
(20 %). In this phase, decode runs without speculative decoding. The acceptance rate of a draft varies from run to run,
and those changes would hide the effect of the settings; plain decode still shows the cost of threads and cache size.

Phase 2 tunes speculative decoding, judged by decode speed alone:

- The draft method is decided once. If your command has a draft model, the tuner uses it. Otherwise, if the model has
  MTP layers, it uses MTP. Otherwise it uses no draft model.
- With a draft method, it tunes `--spec-draft-n-max`, `--spec-draft-p-min` and adaptive drafting on or off.
- Then it tries each n-gram method, `ngram-simple`, `ngram-map-k` and `ngram-mod`, alongside the draft method. It tunes
  the three parameters of each one. An n-gram draft is used when the text repeats; the draft model drafts otherwise.
- The three fastest settings run again on the whole prompt set, and the fastest one wins.

The tuner tries one n-gram method at a time, not several together.

## Prompts

The default prompt set, `prompts/default.yaml`, has 48 prompts in 8 domains: math, reasoning, knowledge, code writing,
code editing, summarizing, writing and chat. Speculative decoding depends a lot on the text: n-gram methods help most
when the output copies from the prompt, as in code editing. If you mostly run one kind of work, use your own prompts.

A prompt file is either YAML or plain text. In YAML, each item has a `domain`, an `id` and a `prompt`. An `answer` field
is allowed and ignored. In plain text, separate the prompts with lines that contain only `---`; they all count as one
domain.

```yaml
- domain: math
  id: math-7times10
  prompt: |
    What is 7x10?
    End your answer with your final answer in bold, on a separate line.
  answer: "70"
```

Every prompt generates up to 512 tokens (`--max-tokens`). Sampling is greedy unless you pass `--sampling`, for example
`--sampling '{"temperature": 0.7, "top_p": 0.95}'`. The report gives the speed of each domain. The overall decode speed is
the geometric mean of the domains, so each domain counts the same however many prompts it has.

## Options

| option | default | meaning |
| --- | --- | --- |
| `--out DIR` | required | where the report, the tuned command, every measurement (`results.jsonl`) and the server logs go |
| `--prompts FILE` | `prompts/default.yaml` | the prompt set |
| `--max-tokens N` | 512 | output tokens per prompt |
| `--prefill L,...` | `256,2048` | prompt lengths for the prompt-speed part of the phase 1 score |
| `--decode-weight W` | 0.8 | weight of decode speed in the phase 1 score |
| `--sampling JSON` | `{"temperature": 0}` | sampling fields of every request |
| `--phase2-prompts N` | 2 | prompts per domain for each phase 2 setting |
| `--max-threads N` | physical cores less 2 per GPU | the most threads to try |
| `--margin F` | 0.02 | how much better a change must be to be kept |
| `--max-hours H` | 4 | time for the search; the final check runs after it |
| `--port N` | 18190 | port of the server the tuner starts |
| `--skip-phase1`, `--skip-phase2`, `--no-confirm` | | leave out a phase or the final check |

Nothing else should use the GPUs or the CPU while the tuner runs, or the timings mean little. The tuner starts the server
itself with `--tune-endpoints` and its own `--port`; leave those out of your command.

## Limits

- Placement is yours: the tuner does not move layers or experts between devices.
- Settings that change the output (KV cache type, quantization) are not tuned.
- The prompt-processing threshold of staged CPU experts (`GGML_SCHED_STAGE_MIN_TOKENS`) is not tuned yet.
- The tuner measures one request at a time. Several parallel requests can favor other settings.
