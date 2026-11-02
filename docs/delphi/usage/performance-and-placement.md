# Performance and placement

This page covers where delphi.cpp puts weights and experts, what it does on its own, and the settings worth changing.

Most of the placement work happens by itself. You choose the devices with `--device` and `-ts`, and you choose which experts stay in host memory with `-cmoe`, `--n-cpu-moe` or `-ot`. delphi.cpp then decides when a host-resident weight runs on a GPU and when it runs on the CPU.

## What happens automatically

The scheduler runs a host-resident weight on a GPU when a wide batch makes the copy worth it. This is the offload cost model. It estimates the time of a copy and a GPU run against a CPU run, and it chooses the cheaper one for each operation. Prompt processing benefits most. Decode batches are small, so their weights stay on the CPU.

You can turn the cost model off with `GGML_OFFLOAD_COST_MODEL=0`. Setting `GGML_OP_OFFLOAD_MIN_BATCH` to a number also replaces the cost model, with the older rule that offloads a batch of at least that many tokens (the default for that rule is 32). `GGML_OFFLOAD_CALIBRATION` names a file of stored link rates. Use it to get the same placement on every run.

The expert cache also runs on its own. See [Expert cache](expert-cache.md).

## Weight streaming for long prompts

Weight streaming copies the expert weights to a GPU during a prompt, ahead of the layer that uses them, and runs the experts there. It helps when the experts sit in host memory and the prompt is long. It is on by default, and it does nothing for a model whose experts are all in VRAM. On DeepSeek V4.1 Flash with R9700 cards, it raised 4K-token prompts from 444 to 637 tokens per second, and decode did not change (2026-11-01).

Prompt batches below a threshold keep their experts on the CPU, because streaming copies every expert of a layer whatever the batch size. The server measures the threshold at startup. That adds a few seconds to the startup, and about 25 s at `-ub 4096` in one test. Set it yourself with `--moe-stage-min-tokens N` to skip the measurement. DeepSeek V4.1 Flash uses 1024.

Each streaming GPU uses two slots by default, and each slot holds one layer's expert tensor. On a GPU with an expert cache, the slots are borrowed from the cache while a prompt runs. With several streaming GPUs, each streams the experts of its own share of the layers. These options change the behavior:

- `--no-moe-stage` turns streaming off.
- `--moe-stage-min-tokens N` sets the smallest prompt batch that streams.
- `--moe-stage-slots N` sets the slots per GPU. More slots run further ahead and use more VRAM.
- `--moe-stage-devices DEV,...` names the GPUs that stream. The default is the expert-cache GPUs.
- `--no-moe-stage-spread` turns that split of the layers over the streaming GPUs off.

Streaming needs a fast link. On a PCIe 4.0 x2 link, the cost model keeps the experts on the CPU.

Use a large `-ub` if you have the VRAM for it. Each prompt batch copies the experts of every layer once, so a larger batch spreads that copy over more tokens. With a fast GPU, prompt speed is then set by the batch size, the link speed and the size of the expert weights. DeepSeek V4.1 Flash uses `-b 4096 -ub 4096`.

The full list of streaming settings is in [Runtime environment variables](environment-variables.md).

## CPU experts on a dual-socket host

When the experts stay on the CPU, `--numa tensors` splits each expert across the NUMA nodes of the host. Each node computes its own share. Use it when the experts run on the CPU of a host with more than one NUMA node.

`--numa tensors` also turns on the CPU expert prefetch. While a GPU runs the attention of a layer, the CPU reads the experts that the next layer will most likely need into its caches. On DeepSeek V4.1 Flash, the next layer's router predicts those experts (the pre-gate). The prefetch made decode 6.5 % faster on DeepSeek V4.1 Flash (2026-11-01).

- `--cpu-expert-prefetch N` sets how many experts per layer the CPU reads ahead. The default is 6, and `0` turns the prefetch off.
- `--no-moe-pregate` turns the prediction off.
- `--no-moe-shared-overlap` stops the shared expert from running on the GPU while the CPU runs the routed experts (DeepSeek V4.1).

Leave CPU cores free for the GPU drivers. On a 64-core host with GPUs, use `-t 48` or `-t 56`, not `-t 64`.

## Finding the bottleneck

Two tools show where time goes. Use them before you change placement.

The [bottleneck profiler](bottleneck-profiler.md) runs a fixed workload and writes a report. The report lists the share of time in each part of the pipeline and the best gain that a change could give.

The [execution trace](exec-trace-record.md) records each host and device event. Use it when the profiler shows waits but not their cause. The reference for the report's rows is [Bottleneck report metrics](../reference/bottleneck-report-metrics.md).

## Limits

- Weight streaming needs a fast host link. On a PCIe 4.0 x2 link, the cost model kept the experts on the CPU.
- Two-GPU expert splits and layer splits do not always overlap their work. Measure your layout before you rely on a second card for prompts.
- The measurements reflect the hardware and the model files at the time of the measurement. A new quantized file can change the best split.

## Related pages

- [Expert cache](expert-cache.md) covers the cache for experts that run on the GPU.
- [Speculative decoding](speculative-decoding.md) has the full commands for GLM-5.3 Flash and DeepSeek V4.1 Flash.
- [Auto-tune](auto-tune.md) searches the batch and thread settings for one layout.
- [Runtime environment variables](environment-variables.md) lists every placement and scheduler variable.
