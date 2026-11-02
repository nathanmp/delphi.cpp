# Share GPUs between servers (experimental)

VRAM sharing lets two or more `llama-server` processes use the same GPUs. Each server keeps its dense weights in its own memory. While a server serves a request, it borrows the free memory of the shared GPUs for its expert cache. When the request ends, the memory goes back to the pool.

Use it when two models need the same GPUs at different times, and neither model fits alone with its expert cache.

## How it works

A small daemon, `tools/vramd/delphi-vramd.py`, tracks the memory of each GPU. Each server asks the daemon for memory when it needs it. A request on a server's home GPU can revoke the cache memory of other servers. Servers that are in the middle of a prompt keep their memory. A prompt also waits while another server runs a prompt, because the two prompts would share the host's transfer bandwidth.

The daemon frees a server's memory when the server closes its connection, so a crashed server does not hold memory.

Each borrowed block becomes one tier of the expert cache. A tier is a separate buffer on one GPU. The server keeps at most two tiers per GPU. See [Expert cache](expert-cache.md) for how tiers work.

## Quick start

Start the daemon with the memory limit of each GPU that you want to share:

```sh
python3 tools/vramd/delphi-vramd.py --socket /tmp/delphi-vramd.sock --gpu ROCm0=30.5G --gpu ROCm1=30.5G --grace 120
```

The `--gpu` value is the memory that all servers may use together on that GPU. Leave room for the driver, and for the desktop if the GPU drives a display.

Start each server with the socket path and the sharing switch:

```sh
LLAMA_VRAMD=/tmp/delphi-vramd.sock LLAMA_MOE_CACHE_SHARED=1 \
  llama-server -m model.gguf -ngl 999 -cmoe --numa tensors --device ROCm0,ROCm1 -ts 1,0
```

Weight streaming is on by default, so the servers stream long prompts without extra settings. A server that cannot reach the daemon keeps its static memory and runs as it would without sharing.

## Settings

- `--socket` sets the Unix socket path. The default is `/tmp/delphi-vramd.sock`.
- `--gpu NAME=SIZE` sets the memory limit of one GPU. Give one for each shared GPU.
- `--grace SECONDS` sets how long an idle server keeps its cache on other servers' GPUs. The default is 120.
- `--overhead SIZE` sets the memory per server and GPU that the leases do not cover. The default is 1G.
- `--log FILE` writes the daemon log to a file. Without it, the log goes to stderr.

The server reads `LLAMA_VRAMD` for the socket path, and it reads `LLAMA_MOE_CACHE_SHARED=1` to start without a fixed cache tier.

## Limits

- The feature is experimental. The daemon's policy numbers are placeholders and have not been tuned on a real deployment.
- A tier on another GPU adds work for every layer. A tier on the server's home GPU is the cheaper choice.
- Each server reserves the overhead on each GPU. The default is 1 GiB.

## Related pages

- [Expert cache](expert-cache.md) explains the cache tiers that the leases create.
- [Runtime environment variables](environment-variables.md) lists the variables that govern the expert cache and the scheduler.
