# Auto-tune the launch settings

The auto-tuner finds faster `llama-server` settings for your model on the hardware you already chose. You give it the command you normally use. It tries other batch sizes, thread counts and speculative decoding settings in one running server. It keeps the faster settings, checks them against your command with fresh launches, and writes the tuned command to a file.

The tuner does not move layers or experts. It keeps your `--device`, `-ts`, `-ngl`, `-ot`, context size and KV cache type.

## Quick start

Run the tuner with your usual server command after `--`:

```sh
python3 tools/auto-tune/tune.py --out /tmp/tune-mymodel -- \
    build/bin/llama-server -m model.gguf -ngl 999 -fa on -c 16384 --device ROCm0,ROCm1
```

Leave out `--tune-endpoints` and `--port` from your command. The tuner adds `--tune-endpoints` and uses its own port.

Do not run anything else on the GPUs or the CPU while the tuner runs. Other load changes the timings, and the results become unreliable. A run takes one to three hours for a large model. The default time limit for the search is four hours, and the final check runs after the search.

## What you get

The output directory holds three files:

- `report.md` has the tuned command and a comparison with your command. The comparison shows the speed of each prompt domain.
- `command.txt` has only the tuned command.
- `results.jsonl` has every measurement, and the server logs are in the same directory.

## How the search works

The tuner works in two phases. Each phase changes one setting at a time. It keeps a change only when the result is at least 2 % faster. The server rebuilds its context for each change, which takes seconds. The model weights load only a few times.

Phase 1 tunes the batch sizes (`-b`, `-ub`, and `-ubd` when a draft model is set), the thread counts (`-t`, `-tb`) and, for models with experts in host memory, the expert cache margin (`LLAMA_MOE_CACHE_AUTO_MARGIN_MIB`). It scores 80 % decode speed and 20 % prompt speed at 256 and 2048 tokens. Phase 1 decodes without speculative decoding. Speculative decoding adds noise from run to run, and that noise would hide the effect of the other settings.

Phase 2 tunes speculative decoding, and it judges the result by decode speed alone. The draft method is chosen once. If your command names a draft model, the tuner uses it. Otherwise it uses MTP when the model has MTP layers, and it uses no draft when neither applies. The tuner then tunes the draft depth and adaptive drafting. It also tries one n-gram method at a time (`ngram-simple`, `ngram-map-k` or `ngram-mod`) alongside the draft. The three fastest settings run again on the whole prompt set, and the fastest one wins.

## Prompts

The default set has 48 prompts in eight domains: math, reasoning, knowledge, code writing, code editing, summarizing, writing and chat. Use your own prompts if you mostly run one kind of work. Speculative decoding depends strongly on the text. N-gram drafts help most when the output repeats the prompt, as in code editing.

Pass your own file with `--prompts FILE`. A YAML file has one item per prompt, with a `domain`, an `id` and a `prompt`. A plain text file separates prompts with lines that contain only `---`, and all of them count as one domain.

The overall speed is the geometric mean of the domain speeds. Each domain therefore counts the same, whatever the number of prompts in it.

## Settings you may want to change

Most runs need no options. These are the ones that matter most:

- `--max-hours H` sets the time limit for the search. The default is 4.
- `--max-tokens N` sets the output length per prompt. The default is 512.
- `--sampling JSON` changes the sampling. Sampling is greedy by default. Use it when you run with a temperature, so that the tuner measures your real setting.
- `--max-threads N` sets the highest thread count to try.
- `--margin F` sets how much faster a change must be to be kept. The default is 0.02.
- `--skip-phase1`, `--skip-phase2` and `--no-confirm` leave out a phase or the final check.

## Example

On Qwen 3.8 Flash Next with three R9700 cards (measured 2026-10-30), the tuner found no setting faster than the existing command. The final check was 0.3 % faster, which is within the noise. The run took about 70 minutes. A tuned command can match a command that someone has already tuned by hand.

## Limits

- The tuner does not tune placement, context size, KV cache type or quantization. Settings that change the output are not tuned.
- The prompt-processing threshold for staged CPU experts (`GGML_SCHED_STAGE_MIN_TOKENS`) is not tuned.
- The tuner measures one request at a time. If you serve several parallel requests, other settings may be faster for you.

## Related pages

- [Speculative decoding](speculative-decoding.md) explains the draft settings that phase 2 tunes.
- [Tune endpoints](tune-endpoints.md) describes the routes the tuner uses to change a running server.
