# Speculative decoding

Speculative decoding lets the model accept several tokens in one step. A small draft model, or a small head in the model itself, proposes the next tokens. The main model then checks all of them in one pass. Accepted tokens cost much less than normal decode steps.

delphi.cpp ships three draft methods for its anchor models. Each method has its own setup. With MTP, the draft depth adapts by itself. DFlash2 and DSpark verify a fixed depth, 3 by default. DSpark also has an opt-in adaptive verification prefix.

- MTP uses the multi-token prediction layers that some models carry, such as Qwen 3.8 Flash Next.
- DFlash2 uses a separate draft model for GLM-5.3 Flash.
- DSpark uses a separate draft model for DeepSeek V4.1 Flash.

## Qwen: MTP

Qwen 3.8 Flash Next carries its own MTP layers. Pass the method name:

```sh
llama-server -m qwen.gguf -ngl 999 --spec-type draft-mtp --device ROCm3,ROCm2,ROCm0 -ts 16,16,17 -fa on -c 65536
```

On three R9700 cards (measured 2026-10-26), this gave a real-text decode speed of 76.0 tokens per second with a warm file. The result depends on the quantized file. The acceptance rate changes with the quantization, so compare files with the same method.

## GLM-5.3 Flash: DFlash2

DFlash2 needs a draft GGUF file, which you pass with `--spec-draft-model`. Put the draft on a GPU that has room for it. The draft shares the output tensor of the target model, so keep it on the GPU of the last layers. This is the command that runs GLM-5.3 Flash on three CMP 170HX cards, with the whole model in VRAM:

```sh
llama-server -m glm.gguf -ngl 999 --numa tensors --device CUDA0,CUDA1,CUDA2 -ts 17,14,16 -t 48 -fa on -c 300000 -b 10240 -ub 1024 -bs \
  --spec-type draft-dflash --spec-draft-model GLM-DFlash2.gguf --spec-draft-device CUDA2 --spec-draft-ngl all -ubd 512 --parallel 1
```

With thinking off, it decoded short chats at 65 to 71 tokens per second, and 48 to 52 after a 250K-token prompt (measured 2026-11-01). Prompts ran at 1030 to 1223 tokens per second from 16K to 250K tokens. CUDA2 had 2 GiB left after the 250K prompt.

Use `--parallel 1` with DFlash2. A crash with several server slots was fixed on 2026-11-01, but DFlash2 with more than one slot has not been tested on long prompts since.

## DeepSeek V4.1 Flash: DSpark

DSpark also takes a separate draft file, which needs a GPU with room for it. DSpark drafts a block that is longer than the number of tokens it checks. This is the command that runs DeepSeek V4.1 Flash on two R9700 cards, with the routed experts in system RAM:

```sh
llama-server -m deepseek-00001-of-00002.gguf --numa tensors -ngl 999 --device ROCm0,ROCm1 -ts 48,52 -cmoe -ot engram_embd=CPU \
  -lm mlock -lzm on -t 48 -fa on -c 65536 -b 4096 -ub 4096 \
  --spec-type draft-dspark --spec-draft-model DSPARK.gguf --spec-draft-device ROCm1 --spec-draft-ngl all \
  --chat-template-file models/templates/deepseek-ai-DeepSeek-V4.1.jinja \
  --moe-stage-min-tokens 1024 --graph-alt-slots 7 --moe-cache-refresh-max 512 -bs --parallel 1
```

It decoded at 39 to 40 tokens per second (measured 2026-11-01). `--graph-alt-slots 7` keeps a graph for every draft width, so a change of width does not rebuild one. `-ts 48,52` keeps layers 20 to 39 on one card, because they read the KV of layer 20. `--chat-template-file` gives the model the V4.1 prompt format for tool calls and thinking. The template inside older GGUF files is the V4 one.

`-c 300000` also works. It leaves less memory for the expert cache, so short replies decode at about 38 tokens per second instead of 40.

For adaptive DSpark verification, add `--spec-draft-adaptive-mode block --spec-draft-n-max 5 --spec-draft-n-min-adaptive 1 --parallel 1`. The full trained block is still generated; the server selects its verification prefix from confidence and live costs. This configuration gained 4.0% across a four-task test at temperature 1. See [Adaptive speculative draft depth](speculative-draft-depth.md#dspark-adaptive-verification-prefix-block-mode) for the measurements and limits.

DSpark draft files without the `dflash.dsv41_semantics` key and without the `output_hc_fn` weights, such as Unsloth's DeepSeek-V4.1-Flash-1017 draft, load as V4.1 drafts. The log warns about the missing key.

## Choosing the draft method

If you pass `--spec-draft-model` and leave out `--spec-type`, delphi.cpp reads the draft file and picks the method. A draft file with the `dflash` architecture gets DSpark when it has a Markov head, and DFlash otherwise. A DFlash file with a selector runs the DFlash2 path.

For MTP layers inside the main model, pass `--spec-type draft-mtp`. The MTP layers are not a separate file.

Use a comma-separated list in `--spec-type` only to combine methods, for example a draft with an n-gram method.

## The draft depth adapts by itself

With `draft-mtp`, delphi.cpp chooses how many tokens to draft on each step. The draft keeps proposing tokens while they are likely to be accepted. It stops when one more token would probably cost more time than it saves. The default maximum is 7 draft tokens. You don't need to set anything to get this behavior.

On GLM-5.3 Flash with three CMP 170HX cards (greedy, eight prompts, 2048 output tokens, measured in October 2026 on an INT4 build of the model), the adaptive depth gave 73.4 tokens per second. A fixed depth of 3 gave 71.6.

Two options change the behavior:

- `--spec-draft-n-max N` sets the maximum depth. Use a smaller value for short outputs.
- `--no-spec-draft-adaptive` turns the adaptive depth off. The depth is then fixed at `--spec-draft-n-max`, which defaults to 3. Use this when you need repeatable output. The adaptive depth changes the batch shape from round to round, so its output is not bit-for-bit repeatable, even with greedy sampling.

On Qwen 3.8 Flash Next, short outputs lost about 2 % with the adaptive depth. On long outputs it gained about 4 %. If your traffic is mostly short replies, try `--no-spec-draft-adaptive`.

## Changing the settings of a running server

`POST /speculative` changes the speculative settings without a reload. The body holds the same options as the command line:

```sh
curl -X POST http://localhost:8080/speculative -H "Content-Type: application/json" \
  -d '{"args": ["--spec-draft-n-max", "5", "--no-spec-draft-adaptive"]}'
```

The new depth cannot be larger than the value the server started with. `GET /speculative` shows the active settings. For more detail, see [Adaptive speculative draft depth](speculative-draft-depth.md).

## Sampling with a draft

By default the draft picks its top token. Pass `--spec-draft-sampling probabilistic` to sample from the draft distribution instead. The output distribution of the target model stays exact. This setting applies to `draft-simple` and `draft-mtp`. At a temperature above 0, it accepts more draft tokens for a fixed depth. It gave no gain with the adaptive depth. See [Adaptive speculative draft depth](speculative-draft-depth.md) for the measurements.

## Limits

- N-gram drafts help most when the output copies text from the prompt, as in code editing.
- `--spec-synth-rates` and `--spec-synth-len` make the acceptance synthetic. They are for speed tests only. Never judge the quality of output that came from them.
- DFlash2 with `--parallel` above 1 has not been tested on long prompts since the 2026-11-01 fix, as described above.
- Each draft depth uses its own compute graph. The server keeps the graphs of the other widths (`--graph-alt-slots`), so a change of depth does not rebuild a graph. Before that, on ROCm, a change of depth cost 150 to 190 ms (2026-10-13).

## Related pages

- [Adaptive speculative draft depth](speculative-draft-depth.md) gives the full depth rule and the measurements.
- [Auto-tune](auto-tune.md) tunes the draft depth and the n-gram settings for your prompts.
- [Expert cache](expert-cache.md) and [Performance and placement](performance-and-placement.md) cover the other speed settings for MoE models.
