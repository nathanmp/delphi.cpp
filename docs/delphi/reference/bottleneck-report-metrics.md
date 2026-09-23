# Bottleneck report metrics

`tools/bottlenecks/profile.py` writes a normalized JSON result and a Markdown summary (see `../usage/bottleneck-profiler.md`). This page explains every row that can appear in the summary's `measured decode span`, `measured prefill span`, and `measured detail` tables, and what a large value of each row implies about where to optimize. It assumes you can read code but not that you know llama.cpp internals.

## What the report measures

The profiler runs a real inference workload and records an execution trace. It reports two workloads.

- **Decode** is token-by-token generation. The analysis sums every target and draft generation execution that falls inside a client request window, then divides by the emitted output tokens. Speculative decoding therefore counts target and draft work together. The unit is `ms/output token`.
- **Prefill** is bulk prompt processing. The analysis selects the complete execution sequence for the exact-length prefill request and divides by the client input tokens. The unit is `ms/client token`.

The trace runs once per arm. The decode and prefill tables come from the same trace file, so their rows are not independent measurements.

For each production arm, the report can also contain a matching `-op-diagnostic` arm. That arm disables GPU graphs and GPU/CPU fusion and times single operations. Its numbers rank candidate work. They are not production timings. See the operation diagnostic section at the end.

## How to read the trace header lines

The arm header contains the workload lines and the trace coverage.

- `Decode workload: ... TG X t/s; TG wall Y ms/output token` is the end-to-end speed-bench result. `Y` is the denominator for every decode gain ceiling.
- `Short-prompt setup` is speed-bench's own prompt phase. The prompt is much shorter than one ubatch, so setup work dominates. Do not use it for long-prompt decisions.
- `Bulk prefill (prefill-4096): ... wall Z ms/input token` is the end-to-end prefill result. `Z` is the denominator for every prefill gain ceiling.
- `Decode trace (submit, integrity not-run): ... traced llama_decode host wall ... (C% of TG wall)` is the summed host wall of the traced `llama_decode` calls. `C` below 100% is normal. Sampling, server control, and asynchronous device completion happen outside those host calls.
- `Prefill trace: N selected executions, M trace-token executions; instrumented window ... ms/client token` is the traced prefill wall that covers all selected logical batches and ubatches.

The decode and prefill sections each have a `measured ... span` table and a `measured detail` table. Treat the tables as one hierarchy: the span table gives the category totals, and the detail table breaks those totals into the specific backend, wait reason, or copy path.

## The optimistic gain ceiling

For a measured interval `t` and the end-to-end wall `W`, the report computes `share = t / W` and prints `1 / (1 - share) - 1` as a percentage. This is the throughput gain if the interval became zero and all other work stayed the same. An interval that is half of the wall gives a 100% ceiling, because the same work then fits in half the time.

The ceiling is optimistic and diagnostic:

- It ignores that some of the interval may not be removable.
- It assumes the rest of the workload does not slow down or gain new dependencies.
- Rows overlap and are not additive. Their sum, and the sum of their ceilings, can exceed 100%.
- If `share >= 1`, the report prints `not bounded`, because no serial interval can exceed the wall it is part of.

Use the ceiling to rank work, not to promise a speedup. Confirm any change with a new end-to-end measurement.

## Evidence labels

Every row states how it was measured. The summary uses two one-letter markers, defined once near the top of the report.

| marker | full label | source | meaning |
| --- | --- | --- | --- |
| `H` | host-clock span | Host clock around host scheduler code | Measures host work, enqueue, or blocked time, not device execution. Other work can overlap it, so it is an upper bound. Rows are non-additive; do not add them. Prefill rows sum several trace contexts. |
| `D` | device-event interval | Backend timing events on the GPU stream | A device interval, summed or grouped per backend. Rows are non-additive; do not add them. |

A `DEVICE_EVENT` interval is placed on the host clock by the backend timing events. It includes every gap between kernels inside the span, such as host launch latency. It is therefore an upper bound of kernel busy time, not the sum of kernel durations. In `submit` mode one span covers a whole eager submission or a whole graph launch, so the interval includes the gaps in that launch.

## Measured span rows

The span rows are category totals. The detail rows below them break each total into its parts.

### CPU compute

**What it is.** The sum of `HOST_SAFE` CPU-compute spans. The scheduler runs a CPU split synchronously on the host, so a span covers every graph node that the scheduler assigned to the CPU backend, not one node. In a hybrid MoE build this is usually the routed experts (`MUL_MAT_ID`), the MoE GLU, and the embedding `GET_ROWS` operations.

**What a high value means.** The host CPU is the place where a large part of the token time is spent. Look at the matching `CPU split on <backend>` rows for the per-backend split, and the operation diagnostic CPU tables to name the dominant operations. Optimize CPU kernel selection, thread count and affinity, NUMA binding, quantization type and repacking, the number of active experts, and the offload policy that decides how many experts run on the CPU. A value near the wall means the model is CPU-bound and the GPU is underused.

### copy handling

**What it is.** The sum of `HOST_SAFE` copy-handling spans. The scheduler records one outer span per tensor that it copies between backends at a split boundary. The span starts before any wait and ends after the copy finishes, so it includes the source wait, the destination wait, and the copy body. In the default production detail only this outer span is recorded.

**What a high value means.** Cross-backend data movement is a large part of host time. Use the `measured detail` table to find the dominating copy path and direction. In hybrid MoE inference the largest rows are commonly host-resident expert weights (`moe_expert_async`), synchronous fallbacks (`fallback_sync`), and input staging (`user_input_sync`). Reduce the number or size of tensors that cross a backend boundary, keep more work on one backend, or make the copies asynchronous. A copy-handling row can overlap a CPU-compute row, so compare them, do not add them.

### host submit

**What it is.** The sum of `HOST_SAFE` backend-submit spans. This is the host time to enqueue non-CPU graph compute to a backend: a CUDA or HIP graph launch, or a group of eagerly launched kernels. It measures the enqueue call, not device execution.

**What a high value means.** The host is slow to hand work to the GPU. Common causes are many small submissions, a graph that keeps falling back to eager launches, a graph that is updated or recaptured every call, or allocation work inside the launch path. Look at `GPU submission on <backend>` for the per-GPU split and at the GPU graph lifecycle in the JSON (`capture`, `update`, `replay`, `fallback`). A large `fallback` count means graph replay is not used and every call is launched eagerly. A stable graph that replays is usually much cheaper to submit.

### device submit sum

**What it is.** The sum of `DEVICE_EVENT` device intervals for the coarse spans: one interval per eager backend submission and one per whole-graph launch. It is device time, not host time.

**What a high value means.** The GPU stream is busy for a large part of the wall time. If this row dominates and CPU compute is small, the workload is GPU-bound. Look at `device interval on <backend>` to find the busiest GPU, then at the operation diagnostic tables for that workload to name candidate kernels. Because a device interval includes gaps, it is an upper bound on kernel work. It can also be larger than the traced host wall, because device work continues after the host enqueues it.

### unaccounted host time

**What it is.** The execution wall minus the union of every host-clock span. It is time inside the traced `llama_decode` call that no measured span covers, such as graph construction, scheduler reserve, offload-cost calibration, or allocator work.

**What a high value means.** A large value is the strongest signal that the report is missing work. The span rows cannot explain the token time, so look outside them: graph build and reserve, repeated calibration or planning, allocation, and anything that runs before the first scheduler split. A value near zero means the traced spans account for the wall. This row is complementary to the span rows, not additive with them.

### Pipeline parallelism

**What it is.** A line above the workload tables states whether the scheduler enabled pipeline parallelism and whether it kept it. It is read from the run log, not the trace.

**What it means.** `enabled and kept` means the scheduler can overlap work across devices. `enabled, then disabled after a compute buffer allocation failure` means a larger compute buffer could not be reserved, so the scheduler fell back to a serial schedule without any command-line change. `not enabled` means one of the enable conditions did not hold, commonly a single device or a tensor override. Compare this line between arms before attributing a throughput difference to a kernel.

## Resource utilization

After each workload's detail table, a resource table shows how busy the CPU and each device were. It is derived from the same host spans and device events.

| row | meaning |
| --- | --- |
| `CPU compute` | Union of the host CPU-compute spans divided by the wall. |
| `device <name>` | Union of that device's coarse intervals divided by the wall. |
| `all devices` | The per-device busy times summed and divided by the wall. With N devices the maximum is N x 100%, so a value near 100% on a multi-device host means only one device's worth of work is running. |
| `devices busy at the same time` | The device-device overlap from the host-clock device placement, shown only on a multi-device host. Near zero means the device intervals do not overlap there. Cross-device placement is less certain than per-device utilization, so treat this row as a hint. |
| `CPU and device both busy` | Time the CPU and at least one device are both busy. A value near zero means the host and device work are serialized. |
| `neither CPU nor device` | Wall time with no CPU compute and no device work, such as copy waits, host submission, or allocator time. It is the end-to-end wall minus the combined busy union, so it is consistent with the utilization denominator. |

The CPU and device rows are upper bounds: a host span can include blocked time and a device interval includes gaps. Use the table to separate a serial hybrid schedule from an overlapped one before optimizing either side.

## Recurring work

A single line lists per-token counts for work that should not scale with tokens: offload decisions and offload measurements, cross-backend copies, backend splits, and graph capture, update, replay, and fallback. A count that grows with the token count, or an `offload_measurements` count far above the number of distinct tensor shapes, points to repeated setup rather than useful compute. This is the cheapest place to catch a calibration or reserve loop.

## Measured detail rows

All rows in this table are non-additive. Use them to explain the span row above them.

### CPU split on CPU

**What it is.** The same events as the `CPU compute` span, grouped by backend. The backend is normally `CPU`. If several CPU-like backends run CPU splits, each one gets a row.

**What a high value means.** Same as `CPU compute`: the host CPU does a large share of the token work. Confirm that all CPU compute is on the expected device, then look at CPU kernels, threads, quantization, and offload policy.

### GPU submission on GPU

**What it is.** The same events as the `host submit` span, grouped by GPU. It is the host enqueue time per GPU.

**What a high value means.** One GPU is expensive to launch work on. Look at that GPU's graph lifecycle (`capture`, `update`, `replay`, `fallback`), the number of kernel launches it receives, and whether the graph is split or updated often. A high fallback count is a strong signal to fix graph reuse before tuning kernels.

### device interval on GPU

**What it is.** The same device events as the `device submit sum` span, grouped by GPU. It is the device interval per GPU.

**What a high value means.** This GPU is the busiest device for the workload. Use the operation diagnostic tables to find candidate kernels on it. Compare the device intervals across GPUs: a strong imbalance means the work split is uneven or one GPU is slower.

### wait previous_split on GPU

**What it is.** The scheduler must finish the previous split before it can start a split on a different backend that has no inputs of its own, because the allocator can reuse buffer regions across splits. The host waits on an event or on a backend synchronize. The backend in the row name is the previous split's backend.

**What a high value means.** The model alternates between backends, and each switch drains the previous backend. Reduce backend switches per graph, keep consecutive nodes on the same backend, or give each split more work so the switch cost is amortized.

### wait moe_ids on GPU

**What it is.** To stage only the used expert weights, the scheduler copies the `MUL_MAT_ID` ids tensor back to the host and synchronizes to read it. The wait is this readback synchronization. The backend in the row name is the backend that holds the ids tensor.

**What a high value means.** Expert-ID readback is on the critical path. Look at expert caching and routing stability, and at how often the ids tensor changes. When nearly all experts are active, the scheduler takes a full-weight copy instead and avoids the readback; a routing distribution that activates most experts can therefore be cheaper than a scattered one.

### wait eval_callback on GPU

**What it is.** The scheduler runs a split node by node or group by group under an evaluation callback, then synchronizes after each group. This path is used by tooling and tests, not by normal generation.

**What a high value means.** A callback-driven evaluation path is active. Confirm that the measured arm does not enable it before you treat the value as production cost.

### wait scheduler_synchronize on GPU

**What it is.** An explicit whole-scheduler synchronize. The host drains every backend. It appears when the application or the scheduler must read results back or must reallocate.

**What a high value means.** The workload drains the GPU often. Look at output readback frequency, allocation changes, and any per-token synchronization in the application. Batching readbacks or keeping the graph allocated reduces this.

### wait graph_reallocation on GPU

**What it is.** The graph size or the node-to-backend assignment changed, so the scheduler synchronizes every backend and then reserves and reallocates the graph.

**What a high value means.** The graph topology is not stable across calls. Varying batch size, varying context length, or offload decisions that change the node count can cause this. A fixed ubatch size and a stable placement let the graph and its allocations be reused.

### wait copy_dst_reuse on GPU

**What it is.** Before the scheduler overwrites a non-input tensor in the destination, it waits for the destination backend to finish using the previous contents. It records this wait when no reusable event is available.

**What a high value means.** Buffer-reuse hazards stall the host. Look at event availability and graph and allocator reuse. A stable graph with reusable events normally removes this wait.

### copy handling user_input_sync CPU -> GPU

**What it is.** A tensor marked as a graph input is copied immediately and synchronously to the destination, so the application can reuse the source buffer. This is the normal path for input tensors. The direction is usually `CPU -> GPU`: tokens, positions, embeddings, and other per-step inputs. The outer span includes the destination wait and the copy body.

**What a high value means.** Input staging blocks the host, or the destination backend must be drained before the copy. Look at the number and size of graph inputs, at host and pinned memory, and at split boundaries that leave an input on the wrong backend. Keeping the input resident on the compute backend, or reducing the number of inputs that cross a boundary, removes the copy.

### copy handling moe_expert_async CPU -> GPU

**What it is.** Host-resident MoE expert weights are staged to the GPU. The scheduler copies only the experts used by the current batch, in groups of consecutive expert IDs.

**What a high value means.** Expert weight staging is a large part of host time. Look at the expert offload and cache policy, the number of active experts per token, the routing locality, and the weight quantization. More weight on the GPU, better routing locality, or smaller weights reduce both the bytes and the number of copy calls.

### copy handling backend_async source -> destination

**What it is.** The default asynchronous inter-backend copy. The scheduler enqueues the copy when the backend pair supports `cpy_tensor_async`. The direction can be `CPU <-> GPU` or `GPU <-> GPU`.

**What a high value means.** Even an asynchronous path costs host time to set up and can serialize when the stream is busy. Look at the number of split boundaries and the tensors that cross them. Fewer or larger boundary tensors are cheaper than many small ones.

### copy handling fallback_sync source -> destination

**What it is.** The asynchronous copy path was not available or failed, so the scheduler synchronizes the source and the destination and then copies synchronously. The direction varies; both `CPU -> GPU` and `GPU -> CPU` occur.

**What a high value means.** This is a strong signal. Cross-backend copies cannot use the fast path, and the host is blocked. Look at whether the two backends support `cpy_tensor_async` for this tensor type and layout, whether the tensors are host-accessible, and whether the model placement forces copies that could be avoided. Removing a fallback copy is often cheaper than making an existing asynchronous copy faster.

### copy body <path> source -> destination

**What it is.** Full detail only (`GGML_EXEC_TRACE_DETAIL=full`). This is the time inside the copy call after the waits. For a synchronous copy it is the host transfer time; for an asynchronous copy it is the enqueue time.

**What a high value means.** The transfer payload or the copy implementation is the cost. Compare this row with the matching source-wait and destination-wait rows to decide whether to reduce bytes or to reduce stalls.

### copy source wait <path> source -> destination

**What it is.** Full detail only. The host waits until the source backend has produced the tensor before the copy can start.

**What a high value means.** The source device is busy, or the dependency chain stalls the copy. Look at the work that must finish before the copy, and at the placement that created the dependency.

### copy destination wait <path> source -> destination

**What it is.** Full detail only. The host waits until the destination backend can safely receive the tensor, using an event or a backend synchronize.

**What a high value means.** The destination device is busy. Look at the queue depth on the destination and at the placement that routes the tensor there.

## Operation diagnostic tables

Each `-op-diagnostic` arm repeats the workload with GPU graphs and GPU/CPU fusion disabled, so single operations can be timed. The tables group the time by GGML operation, by recurring node role, and, when several backends contribute, by backend and operation. Separate CPU tables group CPU node timing.

A recurring node role is the operation plus its trace name, or, for generic names, its source tensors and shape, with layer numbers normalized. Inspect the JSON for groups that the summary omits.

The `conditional max throughput gain` column uses the matching production arm's wall time as the denominator. It is a screening bound, not a production ceiling. Disabling graphs and fusion changes execution cost. Use these tables to name candidate operations, then confirm the effect on the production arm.

## Quick map from the largest row to the first thing to inspect

| largest row | likely cause | first things to inspect |
| --- | --- | --- |
| `CPU compute` / `CPU split on CPU` | CPU-bound routed experts or offloaded ops | CPU kernel and thread settings, NUMA, quantization and repacking, active experts, offload policy |
| `copy handling` with `fallback_sync` | Cross-backend copies cannot run asynchronously | async-copy support for the backend pair, tensor layout, and placement |
| `copy handling` with `moe_expert_async` | Expert weight staging from host RAM | expert cache and offload policy, active experts per token, routing locality, weight size |
| `copy handling` with `user_input_sync` | Input staging blocks the host | number and size of graph inputs, pinned memory, split boundaries |
| `host submit` / `GPU submission on GPU` | Many or falling-back GPU launches | graph lifecycle (`replay` vs `fallback`), graph updates, number of submissions |
| `device submit sum` / `device interval on GPU` | GPU kernel-bound workload | operation diagnostic tables, kernel and quantization choices, work split across GPUs |
| `unaccounted host time` | Work outside every measured span | graph build and reserve, repeated calibration or planning, allocation, offload cost model |
| `CPU and device both busy` near zero | Host and device are serialized | cross-backend handoffs, async copies, layer pipelining, split balance |
| `all devices` utilization far below N x 100% | Devices are idle or work is on one device | work split across devices, pipeline parallelism, placement |
| `offload_measurements` high or growing | Repeated offload calibration | calibration cache keying, tensor shape variety, thread-count changes |
| `wait previous_split` | Backend alternation serializes splits | number of backend switches per graph, work per split |
| `wait moe_ids` | Expert-ID readback synchronization | expert caching, routing stability, fraction of active experts |
| `wait graph_reallocation` | Unstable graph topology | fixed ubatch and batch shape, stable offload decisions |
| `wait scheduler_synchronize` | Frequent full GPU drains | output readback frequency, per-token synchronization, allocation changes |
| `wait copy_dst_reuse` | Buffer-reuse hazard | event availability, graph and allocator reuse |
| `copy body` | Payload or copy implementation | transfer size, copy type, layout |
| `copy source wait` / `copy destination wait` | Dependency or device-busy stall | producer and consumer placement, destination queue depth |

## From a report to a change

1. Find the largest share in the span table, using the end-to-end wall as the denominator.
2. Open the detail table to find the specific backend, wait reason, or copy path.
3. Open the operation diagnostic tables to name candidate operations.
4. Pick a cause that a code or configuration change can address.
5. Measure the change end to end. The report locates work; it does not prove a speedup.
