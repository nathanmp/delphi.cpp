# delphi.cpp: a fork of llama.cpp

delphi.cpp runs large mixture-of-experts (MoE) models on one machine that has much more host RAM than VRAM. The routed experts stay in host memory, and the CPU and the GPUs share the work. Dense weights, attention and the most-used experts go to the GPUs. The main targets are DeepSeek V4.1 Flash, GLM-5.3 Flash and Qwen 3.8 Flash Next, on AVX2 dual-socket CPUs, RDNA4 GPUs (for example the R9700) and SM_80 GPUs (for example the CMP 170HX).

Refer to:

* [docs/delphi/usage](docs/delphi/usage) for how to use each feature, with full command lines.
* [docs/delphi/usage/environment-variables.md](docs/delphi/usage/environment-variables.md) for the environment variables and their defaults.
* [docs/delphi/changelog.md](docs/delphi/changelog.md) for the full list of changes, with measurements.

## Notable changes

Most of these features are on by default. Each has an option or environment variable that turns it off, so you can compare.

### Experts in host RAM

* **NUMA tensor split (`--numa tensors`).** On a host with more than one socket, each routed expert is split by output rows across the NUMA nodes. Each node computes its rows from its own memory, so no socket reads the weights over the slow socket link. Workers claim small pieces of work and can take work from the other node when they finish early. AVX2 kernels decode each weight block once for up to four tokens.
* **CPU expert prefetch and router pre-gate.** While the GPU runs a layer's attention, the CPU reads the experts that the next layer will most likely use into its caches. DeepSeek V4.1 Flash predicts these experts with the next layer's router (the pre-gate). Decode was 6.5 % faster.
* **GPU hot-expert cache.** Spare VRAM on each `--device` GPU holds the experts that recent tokens used most. A routed expert that is in the cache runs on the GPU, and the others run on the CPU from host RAM. The cache refreshes in the background during decode, and it works with any MoE model that has its experts in host RAM. See [Expert cache](docs/delphi/usage/expert-cache.md).
* **Weight streaming for long prompts.** For a wide prompt batch, the experts of each layer are copied to a GPU ahead of the layer that uses them, and the GPU runs them. The copies use slots that the expert cache lends during the prompt. The server measures, at startup, the batch size where streaming becomes faster than the CPU. DeepSeek V4.1 Flash 4K-token prompts: 444 -> 637 tokens per second.

### Model support

Models from mainline llama.cpp are supported. Work on this fork focuses primarily on:

* **DeepSeek V4.1 Flash** (`deepseek41`): engrams, hyper-connections, compressed and sparse attention, the DSpark draft head, vision and the 1017 chat template (use `--chat-template-file models/templates/deepseek-ai-DeepSeek-V4.1.jinja` with older GGUF files).
  * The causal encoder-decoder prefill (from DeepInfer) runs the decoder layers only on the last rows of a prompt: 1.5-1.9x faster prefill.
  * Layer-bank prefill keeps one layer's experts on the GPU for several ubatches while the next layer's experts upload: 16K and 32K prompts take about half the time.
  * Long-context work keeps decode at 31-36 tokens per second up to 250K tokens of context, and compute buffers no longer grow with the context length.
* **GLM-5.3 Flash** (`glm5next`): KDA linear attention, mHC residual streams, the pooled sparse indexer, vision, MTP and the DFlash2 draft model.
  * The KV window (`LLAMA_KV_RESIDENT_CELLS=auto`) keeps part of the attention KV cache in host memory. On 3 CMP 170HX GPUs this gives a 1M-token context, or three 512K-token sessions.
* **Qwen 3.8 Flash Next** with MTP, tensor split and a pooled-key cache for its sparse attention. **Muse Spark Open** (text, vision and audio) and **MiMo** are also supported.

### Quantization

* **ik_llama.cpp quant types** (IQ2_K..IQ6_K and IQ2_KS..IQ5_KS) on the CPU, CUDA and HIP, including MoE kernels.
* **Q4_G128**: symmetric INT4 with one BF16 scale per 128 values, with fused INT4 MoE kernels on CUDA and RDNA4.
* **Automatic quantizer** (`tools/auto-quantize`): measures how much each tensor group loses at each quant type, then chooses the types that give the lowest loss for your memory budget on each device. See [Auto-quantize](docs/delphi/usage/auto-quantize.md).

### Speculative decoding

* Draft heads and draft models: MTP (GLM, Qwen), DFlash2 (GLM), DSpark (DeepSeek) and EAGLE3.
* The draft depth adapts per round by default. For DSpark, `--spec-draft-adaptive-mode block` chooses how much of each drafted block to verify.
* `--spec-draft-sampling probabilistic` keeps the exact output distribution at temperatures above 0 and accepts more draft tokens there than greedy drafting.
* `POST /speculative` changes the speculative settings of a running server. See [Speculative decoding](docs/delphi/usage/speculative-decoding.md).

### Serving several sessions

* **Per-sequence KV streams and one-sequence prompt ubatches**: with `--parallel N`, a long prompt gets its own ubatches, so long-prompt optimizations still apply when several sessions share the server.
* **Staggered verify and verify streams**: with pipeline parallelism over several GPUs, the verify steps of different sessions overlap on different GPUs, and a session keeps generating while another session's prompt runs in the background. GLM-5.3 Flash with 3 sessions: +20-25 % aggregate decode. The longest decode pause during a 12.8K-token prompt fell from about 4.5 s to about 1.3 s.
* **Prefix snapshots** (`--cache-prefix-dir`): when two conversations share a system prompt, the server saves the full state at the point where they differ, in RAM and on disk. A new conversation then skips that part of the prompt, also after a restart. DeepSeek V4.1 Flash with a 15K-token system prompt: first token in 4.0 s instead of 20.3 s.
* **Prompt checkpoints** at the end of each chat message, with spill to disk, so that editing an earlier message reprocesses less.

### Memory

* **Dynamic vision** (default `--mmproj-memory dynamic`): the vision encoder stays in host RAM and uses the language model's compute memory only while it encodes an image. Use `--mmproj-memory resident` to keep it on the GPU.
* Compute buffers for long contexts are split into chunks (`LLAMA_CHUNK_MIB`), so a larger context leaves more VRAM for the expert cache.
* **VRAM sharing (experimental)**: several servers can lend each other idle GPU memory for their expert caches. See [VRAM sharing](docs/delphi/usage/vram-sharing.md).

### Tools

* **Auto-tune** (`tools/auto-tune`): searches the batch, thread and speculative settings against a bank of prompts, then checks the best result with fresh launches. See [Auto-tune](docs/delphi/usage/auto-tune.md).
* **Bottleneck profiler** (`tools/bottlenecks/profile.py`) and **execution traces** (`GGML_EXEC_TRACE`): show where the time goes, per device, per operation and per pipeline stage, and the largest gain that a change could give. See [Performance and placement](docs/delphi/usage/performance-and-placement.md).

### Kernels

* Many CUDA, HIP and CPU kernel changes for the target hardware: packed MoE kernels, sparse FlashAttention on RDNA4, WMMA lightning indexer, fused hyper-connection and router operations, and graph-capture and scheduler fixes for multi-GPU pipelines.

## Tuning tips

Start with the defaults and the commands in [docs/delphi/usage](docs/delphi/usage). Most placement decisions are automatic. Change one setting at a time, and measure each change.

### Settings to always give

* **Set `-c` and `--parallel`.** Without `-c`, the server uses the model's training context (1M tokens for DeepSeek V4.1 Flash and GLM-5.3 Flash), and the compute buffers for that may not fit. Without `--parallel`, the server can choose 4 slots.
* **Read the startup log after each change.** It shows the expert-cache slots of each tier and whether each GPU streams. Search for `failed to initialize speculative decoding context`: when the draft cannot start, the server continues without speculative decoding, and only the log tells you.

### Prompt speed with experts in host RAM

* **Use a large `-ub`.** Weight streaming copies every expert of a layer once per ubatch, so the copy time per token falls as the ubatch grows. With fast GPUs, long-prompt speed is set by the ubatch size, the size of the expert weights and the PCIe speed. A larger `-ub` uses more VRAM for compute buffers and leaves less for the expert cache.
* **Make `-b` several times larger than `-ub`.** DeepSeek V4.1 Flash keeps a layer's experts on the GPU for all ubatches of one call. With `-b 4096 -ub 4096` there is only one ubatch per call, and prompts were about 40 % slower than with `-b 16384 -ub 4096`.
* **The PCIe link must be fast.** On a PCIe 4.0 x2 link, the cost model keeps the experts on the CPU, and streaming does nothing.
* **Short prompts.** Prompts below `--moe-stage-min-tokens` run their experts on the CPU. Set `LLAMA_MOE_CACHE_PROMPT_ROWS` to the same value minus one so that those prompts also use the experts in the GPU cache. On DeepSeek V4.1 Flash, `LLAMA_MOE_CACHE_PROMPT_ROWS=2047` with `--moe-stage-min-tokens 2048` reduced the time to first token by 23-35 % for 200-1500 new tokens.

### Expert cache

* **`--moe-cache-margin`** sets the VRAM that each GPU keeps free for buffers that grow after load. Increase it if a GPU runs out of memory with several sessions or large images. DeepSeek V4.1 Flash with `--parallel 3` needs 4096.
* **`--moe-cache-async N`** sets how many experts each background refresh uploads. A larger value refills the cache faster after a prompt, but each expert needs scratch memory on every cache GPU (18 MiB for DeepSeek V4.1 Flash), which the cache then cannot use. In one DeepSeek V4.1 Flash test, 64 to 128 gave about the same decode speed.
* **`--moe-cache-refresh-max N`** caps the refresh that runs before the first token after a prompt. The background refresh uploads the rest. A smaller value gives a faster first token, and a larger value gives a fuller cache for the first tokens.

### CPU threads and several servers

* **Leave cores for the GPU drivers.** On a 64-core host with GPUs, use `-t 48` or `-t 56`, not `-t 64`.
* **Give each server its own cores.** `--numa tensors` pins its workers to the lowest CPUs of each NUMA node that the process can use. If another server runs on the same CPUs, its busy threads can halve decode speed. Run all-GPU models without `--numa tensors`, and use `taskset` to put each server on CPUs that the others do not use.
* **Stop memory compaction on large-RAM hosts.** When the kernel compacts memory during decode, it moves the pages of the host experts, and decode can stop for seconds. Set `vm.compaction_proactiveness=0` and `vm.watermark_boost_factor=0`, and keep them in `/etc/sysctl.d/` so that they stay after a reboot.
* **Raise the memlock limit** if you use `-lm mlock`. A low limit makes mlock fail.

### Long contexts on GLM-5.3 Flash

* `LLAMA_KV_RESIDENT_CELLS=auto` moves each GPU's missing KV memory to host RAM. Increase `LLAMA_KV_WINDOW_MARGIN_MIB` (default 1024) if a GPU runs out of memory when several deep sessions decode at the same time. Three 512K-token sessions on 3 CMP 170HX needed 2048.
* `LLAMA_GLM5_POOL_KEYS_F16=1` (opt-in) stores the pooled indexer keys as F16. It made the first token at 250K about 20 % faster and decode rounds 8 % faster, for a KLD of 0.009 at 16K.

### Measuring a change

* **Compare A/B/A.** Run the old setting again after the new one. Heat, other processes and GPU clock modes move results by more than many changes. On R9700 cards, compare only runs with the same `power_dpm_force_performance_level`.
* **Do not judge a change from one prompt.** Measure the time per verify step at each verify width, or use the [bottleneck profiler](docs/delphi/usage/bottleneck-profiler.md), not one request's tokens per second.
* **Judge numerical changes with KLD at 16K tokens or less.** On long raw-text corpora, an instruct model can enter a region where its predictions are poor, and there any small change in arithmetic gives a large KLD (about 0.26 at 64K on GLM-5.3 Flash). That KLD does not show a defect.
* **[Auto-tune](docs/delphi/usage/auto-tune.md)** searches batch, thread and speculative settings automatically. A run takes about 70 minutes.

## Credits

* [llama.cpp](https://github.com/ggml-org/llama.cpp) for the underlying project.
* [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp/) for IQn_K/KS tensors.
* [vcruz305](https://github.com/vcruz305/llama.cpp) for support for DeepSeek V4.1 Flash.
* [DeepInfer](https://github.com/deepseek-ai/DeepInfer) for the Causal Encoder-Decoder architecture.
* [buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp) for AVX2 improvements and design improvements for the expert cache.
* [Strata](https://github.com/Niko1221/Strata) for design improvements for the expert cache.
* [llama-cpp-rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts) for the ring buffer used to speed up long-context prefill, adaptive speculative decoding (WIP), and some other RDNA4 improvements.
* [ubergarm](https://gist.github.com/ubergarm/edfeb3ff9c6ec8b49e88cdf627b0711a/) for default calibration data for the automatic quantizer.
* Any AI company committed to making local models.

## License

MIT license. Go nuts with this.

PRs are welcome. Support is on a best-effort basis. Untested on Windows.