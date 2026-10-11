# delphi-vramd

Device memory leases for llama-servers that share GPUs: each server keeps its own dense weights, and borrows the free memory
of the shared GPUs for its expert cache while it serves a request. Experimental; the policy numbers are placeholders.

```
python3 tools/vramd/delphi-vramd.py --socket /tmp/delphi-vramd.sock --gpu ROCm0=30.5G --gpu ROCm1=30.5G [--grace 120] [--log FILE]

LLAMA_VRAMD=/tmp/delphi-vramd.sock LLAMA_MOE_CACHE_SHARED=1 GGML_SCHED_STAGE=1 GGML_SCHED_STAGE_SPREAD=1 \
    llama-server ... -cmoe --device ROCm0,ROCm1 -ts 1,0
```

`--gpu` is the memory that all registered servers may use on that GPU together (leave room for the driver and, on a display
GPU, the desktop). A server that cannot connect keeps its static memory.

## Server side (tools/server/server-vram.cpp)

| variable | default | meaning |
| --- | --- | --- |
| `LLAMA_VRAMD` | unset | daemon socket |
| `LLAMA_MOE_CACHE_SHARED` | 0 | create the expert cache without a static tier (`LLAMA_MOE_CACHE_MIB` adds one) |
| `LLAMA_VRAMD_CHUNK_MIB` | 4096 | lease (and expert cache tier) size; each tier adds a partial expert pass per layer |
| `LLAMA_VRAMD_MIN_MIB` | 1024 | smallest cache lease worth a tier (at least one staging ring where staging needs it) |
| `LLAMA_VRAMD_MAX_TIERS_PER_GPU` | 2 | leases per GPU; the daemon makes the chunks larger to stay within it |
| `LLAMA_VRAMD_WANT_MIB` | unlimited | cache memory to ask for per GPU |
| `LLAMA_SHARE_IDLE_COMPUTE_S` | 30 with a daemon | free the compute buffers after this idle time |
| `LLAMA_SHARE_IDLE_KV_S` | 120 with a daemon | save idle slots to the host prompt cache (`--cache-ram`) and free the KV cache |
| `LLAMA_VRAMD_IDLE_SHRINK_S` | 600 | give back the cache leases on GPUs other than the home GPU |

The idle timers also work without a daemon (set them explicitly).

Per request: before the first prompt batch the server asks for cache memory on its home GPU (the layer device) and waits for
it; then it asks for the other GPUs and takes what is free. Each lease becomes an expert cache tier, filled at the next small
batch; staging rings borrow the largest tier on their GPU, and staging is off on GPUs without a tier large enough for a ring.
A revoked lease is dropped at the next batch boundary.

## Protocol

Unix socket, one JSON object per line.

Server to daemon: `hello {model, pid, home}`, `gpu {gpu, resident, scratch, kv}` per GPU, `ready`; `phase {state: idle|prefill|decode}`;
`request {req, gpu, purpose: cache|scratch|kv, bytes, min, chunk, max_leases, wait, prefill}`; `release {lease}`; `released {lease}`
(answers a revoke); `cancel {req}`; `status`.

Daemon to server: `welcome`; `lease {req, lease, gpu, purpose, bytes}` (scratch and kv leases at registration have req -1);
`wait {req}`; `done {req, granted}`; `revoke {lease, deadline_ms}`; `status {...}`.

## Policy

- A server's request on its home GPU, and any scratch or KV request, may revoke other servers' cache leases there, unless the
  holder is in a prompt.
- A request elsewhere gets free memory, or the cache of servers idle for longer than `--grace`.
- A prompt waits while another server is in a prompt (they would split the host's staging bandwidth).
- A closed connection frees all of its leases (crash cleanup).
