# Recording an execution trace

Set `GGML_EXEC_TRACE` to a file path and run any llama.cpp binary. The trace is written when the process exits normally.

```text
GGML_EXEC_TRACE=/tmp/run.trace ./build/bin/llama-cli ...
```

For routine tests use the `--numa tensors` mode so CPU and memory placement stays representative. Example decode run:

```text
GGML_EXEC_TRACE=/tmp/run.trace ./build/bin/llama-bench \
    --numa tensors -ngl 999 -ncmoe 999 --device ROCm3 \
    -m model.gguf -p 512 -n 128
```

An invalid output path reports an error and disables tracing; inference continues.

## Environment variables

| variable | default | meaning |
| --- | --- | --- |
| `GGML_EXEC_TRACE` | unset (disabled) | raw trace output path |
| `GGML_EXEC_TRACE_MAX_EVENTS` | 1048576 (max 134217728) | preallocated event capacity; the trace sets its `truncated` flag if exhausted |
| `GGML_EXEC_TRACE_DETAIL` | `production` | set to `full` to also record nested copy source waits, destination waits, copy bodies, and MoE-ID waits |
| `GGML_EXEC_TRACE_GPU_TIMING` | `off` | device-event GPU timing: `submit` (`1`) for one span per eager submission and graph launch, `op` to add one span per eagerly launched node or fused node group |
| `GGML_EXEC_TRACE_GPU_TIMING_EVENTS` | 4096 (min 2, max 1048576, rounded down to even) | per-backend timing-event pool size; each span uses two events |
| `GGML_EXEC_TRACE_INTEGRITY` | unset (disabled) | structural witness output path used by the integrity harness |
| `GGML_CUDA_OP_TRACE` | unset | CUDA operation tracing; treated as contamination by the integrity harness |
| `GGML_HIP_OP_TRACE` | unset | HIP operation tracing; treated as contamination by the integrity harness |
| `GGML_HIP_DISABLE_GRAPHS` | unset | disables HIP graphs; treated as contamination by the integrity harness |

`GGML_EXEC_TRACE_GPU_TIMING` has no effect without `GGML_EXEC_TRACE`, and the integrity-only witness never records timing events. Related offload knob: `GGML_OFFLOAD_COST_MODEL` selects the transfer-aware offload cost model whose decisions the trace records as `OFFLOAD_DECISION`, `OFFLOAD_BYTES`, and `OFFLOAD_CALIBRATION` events.

## GPU timing rules

Each HIP/CUDA backend context creates a fixed pool of timing-enabled events when the backend is created. The timer obeys these rules:

- It records events only on the context main stream, which already carries the work.
- It never records an event during stream capture. A graph launch is timed as a whole from outside the graph, so the graph topology and the capture, update, and replay behavior do not change.
- It does not allocate after backend creation, does not call an event or stream synchronization, and adds no device synchronization.
- Spans between two completion points form one epoch. Elapsed times are read after the existing stream synchronization in the backend `synchronize` call. When the backend is freed, a non-blocking event query decides whether an open epoch is complete.
- A span that cannot be resolved without a wait is recorded as `UNRESOLVED`. When the pool is full, new spans are recorded as `NOT_RECORDED` until the next completion point. A rejected runtime call is `FAILED`. The timer never waits for events to become free.

Operation spans cover only the main stream outside concurrent regions. Nodes on forked streams are covered by the enclosing submission span. HIP/CUDA graphs contain no timing events, so graph replays have no per-node device timing.

The run metadata records `gpu_timing`, `gpu_timing_event_capacity`, and the started and released span counts. A started span that was never released belongs to a backend that was not freed before the trace was written.

## Placement on the host clock

The decoder places device intervals on the host clock without a clock-rate model. All spans of an epoch share one device anchor, the start event of the first span. The host reads its clock before it records each start and end event, so the device reaches the event after that sample. The device completes each end event before completion is known. These constraints bound the anchor. The host span around the synchronize that gave the completion time narrows the bound:

- If that synchronize blocked for 50 us or more, the device was busy when it started and finished at most 50 us before it returned. The decoder uses the upper bound as the placement (`anchor=completion`). This is the normal case for a layer split: the host queues every device, then waits for each in turn.
- If it returned at once, the device finished before it started. The decoder lowers the upper bound to that point and uses the lower bound as the placement (`anchor=enqueue`), as it does when no synchronize gave the completion time.

The width of the remaining bound is the placement uncertainty. An epoch whose bounds cannot all hold is reported as `inconsistent`.

## Clock and metadata

The host clock is `CLOCK_MONOTONIC_RAW` on Linux. At startup and finalization, a `CLOCK_SYNC` event brackets one `CLOCK_MONOTONIC` sample between two trace-clock samples. Offline tools use the bracket midpoints to align the clocks and account for rate drift; half of each bracket width bounds its sampling uncertainty. Run metadata also records the build commit, host CPU model, CPU affinity, detail level, and GPU timing mode.

The decoder prints trace self-overhead metadata: attempted and dropped events, interned strings, allocated event-buffer bytes, growth count, and serialization duration. Serialization happens after measured execution and does not include a backend synchronization. The fixed preallocated buffer has zero growth operations.

## Integrity witness

`GGML_EXEC_TRACE_INTEGRITY` writes a fixed-size structural witness used by the integrity harness. It does not allocate the raw event buffer or read the host clock at each trace site. It is an internal validation mechanism rather than a user trace format. You normally do not set it by hand; `exec-trace-integrity.py` sets it for each run. See `exec-trace-integrity.md`.
