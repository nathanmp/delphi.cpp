# Expert cache for MoE models

The expert cache speeds up a Mixture-of-Experts model when its routed experts sit in system RAM. It copies the experts that the model uses most often into GPU memory. Those experts then run on the GPU, and the rest keep running on the CPU.

You don't need to set anything. The cache turns on by itself when the routed experts are in host memory and a GPU has free memory left. Unlike llama.cpp, which runs every host-resident expert on the CPU, delphi.cpp moves the hot experts to the GPU and keeps the cold ones on the CPU.

## When it helps

The cache helps when all of these hold:

- The routed experts are in host memory, for example with `-cmoe`, `--n-cpu-moe N` or `-ot exps=CPU`.
- The rest of the model fits on one or more GPUs with some memory left over.
- The model's routing is uneven, so a small set of experts receives most of the tokens. Check this with a short run before you rely on the cache.

By default the cache serves decode and small batches. It refreshes after each prompt, so a prompt that is very different from the previous one gets less benefit.

## Quick start

Start the server as you normally do with experts in host memory. Add nothing for the cache:

```sh
llama-server -m model.gguf -ngl 999 -cmoe --numa tensors --device ROCm1 -fa on -c 8192
```

Check the startup log. For each GPU that gets a cache tier, the log shows a line that starts with `expert cache auto:`. It shows the free memory, the reserve kept free and the size of the tier. A GPU with too little free memory logs `no tier` instead. If the model does not meet the requirements under Limits, the log shows `expert cache off:` with the reason, and the model runs without the cache.

The tiers are allocated after every model has loaded, so the draft model and the MTP head are in place first.

## How auto mode sizes the cache

Auto mode gives each GPU that holds cached layers one tier. The tier size is the GPU's free memory less a reserve of 1024 MiB. The reserve covers buffers that grow later, such as graph reallocations. The cache also keeps room for its own refresh scratch.

By default, auto mode does not give a tier to a GPU that holds no cached layers.

Weight streaming for long prompts (on by default, see [Performance and placement](performance-and-placement.md)) borrows its slots from the tier on the same GPU while a prompt runs, and the cache takes them back at its next refresh. A GPU whose tier is smaller than one streaming slot does not stream, and the log says so.

## Settings you may want to change

Most runs need no settings. These are the ones that some users need. Each command-line option sets the environment variable in parentheses, so older scripts that set the variable still work:

- `--moe-cache-margin MIB` (`LLAMA_MOE_CACHE_AUTO_MARGIN_MIB`) raises or lowers the reserve per GPU. The default is 1024. Raise it when a buffer that grows later fails to allocate. On DeepSeek V4.1 Flash with vision, a 1000-token image failed to allocate a 388 MB compute buffer with the default reserve. A value of 6144 fixed it (2026-10-23). A per-device value is also accepted, for example `1024,ROCm1:6144`.
- `--moe-cache MIB` (`LLAMA_MOE_CACHE_MIB`) replaces auto mode with a fixed size: one tier of that many MiB on the GPU of the layers. `--moe-cache 0` turns the cache off.
- `--moe-cache-refresh-max N` (`LLAMA_MOE_CACHE_REFRESH_MAX`) limits the experts uploaded in the refresh after a prompt. The decode-time refreshes upload the rest. The default uploads all of them. DeepSeek V4.1 Flash uses 512, so that the first tokens of a reply do not wait for a full refresh.
- `--moe-cache-roles DEV:ROLE,...` (`LLAMA_MOE_CACHE_ROLES`) sets the role of each GPU's tier: `rotate` follows the routing (the default), `static` keeps the hottest experts, `pinned` is static and never lends slots to streaming, and `off` gives the GPU no tier.
- `LLAMA_MOE_CACHE_TIERS` adds tiers on named GPUs, for example `ROCm0:16384`.
- `LLAMA_MOE_CACHE_PROMPT_ROWS` lets prompt batches up to that many rows use the cache too. By default only decode and small batches use it.
- `LLAMA_MOE_CACHE_SHARED=1` leaves the cache without a fixed tier. It takes memory from the VRAM sharing daemon instead. See [VRAM sharing](vram-sharing.md).

`llama-server --help` lists every option. The full list of variables, with their defaults, is in [Runtime environment variables](environment-variables.md).

## Example

DeepSeek V4.1 Flash runs with its dense layers on two R9700 cards and its IQ4_KS routed experts in system RAM. At `-c 65536`, auto mode gives ROCm0 1219 expert slots and ROCm1, which also holds the DSpark draft, 712 slots (about 34 GiB in total). That cache serves 54 to 60 % of the expert selections, and the model decodes at 39 to 40 tokens per second with DSpark (measured 2026-10-31 and 2026-11-01). A longer context leaves less memory for the cache. At `-c 300000` the cards get 1235 and 513 slots, and short replies decode at about 38 tokens per second.

The cache changes the output only by the order of floating-point sums. On an earlier DeepSeek file, the KL divergence from the CPU path was 0.015 (2026-10-13).

## Limits

- The cache works only for a feed-forward block of the SiLU type (or the clipped SiLU of DeepSeek V4), with separate gate, up and down experts. The experts must have no biases and no scales, and all experts of one projection must have the same shape. In auto mode, a model that does not meet this runs without the cache and logs `expert cache off:`. With an explicit `--moe-cache` size, it fails to start with an error.

- A cache tier on a GPU that does not hold the cached layers costs one copy per layer, so auto mode keeps tiers on the layer GPUs.
- With a draft model, the draft's memory comes out of the same free memory. Check the startup log after you add a draft.
- The cache uses the routing of recent tokens. A workload that changes its experts quickly gets fewer hits.

## Related pages

- [Performance and placement](performance-and-placement.md) covers the other settings that decide where experts run.
- [VRAM sharing](vram-sharing.md) covers sharing one GPU between servers.
- [Auto-quantize](auto-quantize.md) plans the expert types that the cache serves.
