# Execution-trace profiler

The execution-trace profiler records the host-visible structure of llama.cpp inference: which graphs ran, how the scheduler split them across backends, which implementation each backend chose for each node, and how long the host spent submitting, waiting, and copying. It does not change scheduler or backend synchronization, and it is disabled by default.

This page explains what the profiler is and which tool to reach for. The Recording, inspection, validation, and visualization pages under `docs/delphi/usage/` give commands, flags, and environment variables.

| task | tool |
| --- | --- |
| Record a raw trace | Set `GGML_EXEC_TRACE`; see `../usage/exec-trace-record.md` |
| Inspect a trace on the command line | `tools/exec-trace/exec-trace-dump.py`; see `../usage/exec-trace-dump.md` |
| Decide whether a trace is production-representative | `tools/exec-trace/exec-trace-integrity.py`; see `../usage/exec-trace-integrity.md` |
| Align vendor profiler data with a trace | `tools/exec-trace/exec-trace-gpu-activity.py` plus a provider adapter (`exec-trace-rocm.py` for ROCprofiler-SDK); see `../usage/exec-trace-gpu-activity.md` |
| View a unified host plus GPU timeline | `tools/exec-trace/exec-trace-perfetto.py`; see `../usage/exec-trace-perfetto.md` |

## What gets recorded

The tracer lives in `ggml/src/ggml-exec-trace.cpp` with hooks in the scheduler (`ggml/src/ggml-backend.cpp`), the CPU and CUDA backends, and `src/llama-context.cpp` (execution and ubatch boundaries). Any llama.cpp binary therefore supports tracing through environment variables; no flags are needed.

A raw trace records:

- execution and ubatch begin/end records with call kind (decode, prefill, encode), sizes, thread counts, and status;
- backend, buffer-type, device-name, and device-description strings;
- graph instances and scheduler topology reuse;
- GGML node names, operations, output types and dimensions, source relationships, source dimensions, and view relationships;
- effective matmul `M`, `N`, and `K`, plus the selected-expert dimension for `MUL_MAT_ID`;
- scheduler split ranges, backends, input dependencies, and source splits;
- deterministic hashes of ordered operations/shapes, node-to-backend assignments, and split backend/device sequences, plus the cross-backend edge count;
- host time spent submitting non-CPU work, synchronously computing CPU splits, waiting in existing scheduler synchronization calls, and handling cross-backend transfers;
- backend implementation and variant choices for traced CPU and HIP/CUDA operations (dispatch records);
- transfer-aware offload decisions, byte counts, and calibration data, when offload runs;
- HIP/CUDA graph capture, update, replay, and fallback behavior with fallback reasons;
- optional device-event GPU timing spans (see below).

The default `production` detail level records complete structural and dispatch data, scheduler/backend spans, and one outer span for each scheduler copy. `GGML_EXEC_TRACE_DETAIL=full` also records nested copy source waits, destination waits, copy bodies, and MoE-ID waits. Full detail is a higher-overhead diagnostic mode; the integrity harness reports it as non-production-representative.

## Timing provenance

Every timed interval has one of three provenance classes. Reports and exports label each interval with its class.

| provenance | source | use |
| --- | --- | --- |
| `HOST_SAFE` | host clock around host calls: execution, ubatch, scheduler submission, wait, and copy spans | production-representative when the integrity gate passes; measures host work, submission, or blocked time only |
| `DEVICE_EVENT` | timing-enabled HIP/CUDA events recorded by the backend on its own stream | device intervals from a production build; the integrity harness measures their perturbation |
| `VENDOR_EXTERNAL` | offline provider data, such as ROCprofiler-SDK | diagnostic only; never production-representative |

Host spans measure host-side work, submission, or blocked time only. They are not GPU execution or DMA time; `GPU_TIMING` spans carry device time. A device interval includes every gap between kernels on the stream inside the span, such as host launch latency. It is therefore an upper bound of kernel busy time, not the sum of kernel durations.

## Device-event GPU timing

`GGML_EXEC_TRACE_GPU_TIMING` (values `submit`/`1` or `op`) records GPU intervals with backend timing events. The timer records events only on the context main stream, never during stream capture, never allocates after backend creation, and never waits. A graph launch is timed as a whole from outside the graph. Spans that cannot be resolved without a wait are recorded as `UNRESOLVED`; spans dropped because the event pool was full are `NOT_RECORDED`; rejected runtime calls are `FAILED`. See `../usage/exec-trace-record.md` for the full rules.

## Integrity gate

The paired-run harness (`exec-trace-integrity.py`) determines whether a trace is production-representative. It runs a structural witness, clean baselines, and traced runs of the same `llama-bench` command, then compares throughput medians and structural fingerprints. A structural mismatch rejects the trace even when throughput is unchanged. A GPU graph replay-to-launch fallback is therefore a failure, not harmless timing noise. See `../usage/exec-trace-integrity.md`.

## Raw format

The raw file uses schema `ggml-exec-trace-v4`: a 96-byte header, fixed-size 80-byte events, then an interned string table. All integers use little-endian byte order. Events carry execution, ubatch, graph, node, split, backend, device, and thread IDs plus four fixed 64-bit arguments; readers must use the event kind and ignore kinds they do not know. GPU timing kinds are additive, so older decoders keep working. The full event table is in `../usage/exec-trace-dump.md`.

## Limitations

- Host spans do not show per-worker CPU scheduling; the trace exposes one synchronous `CPU_COMPUTE` backend span.
- Vendor profiler data (ROCprofiler-SDK, CUPTI) is diagnostic only; its collection perturbs the run by tens of percent.
- Contamination detection (debug builds, sanitizers, graph-disable and operation-trace settings, linked profiler libraries, profiler launchers) is not exhaustive. Driver-level tracing, system profilers attached after launch, thermal or power-state changes, and unrelated background activity can remain undetected. Review raw samples and dispersion.
- Trace output is finalized at normal process exit. `ggml_exec_trace_flush()` exists for controlled shutdown and tests, but callers must only use it when no trace writer is active. It does not synchronize backend work.

## Tests

`tests/test-exec-trace.cpp` covers disabled, enabled, detailed, truncation, integrity, and GPU-timing modes. Each offline script has a Python test (`tests/test-exec-trace-dump.py`, `test-exec-trace-integrity.py`, `test-exec-trace-gpu-activity.py`, `test-exec-trace-rocm.py`, `test-exec-trace-perfetto.py`). All run under ctest.

## Provenance

Port of the legacy `feat/execution-trace-v2` and `feat/execution-trace-gpu-timing` work, adapted to current upstream. Introduced in `24137eafb`. Transfer-aware offload events (`OFFLOAD_DECISION`, `OFFLOAD_BYTES`, `OFFLOAD_CALIBRATION`) were added with the offload cost model in `e04b0384a`. This document supersedes the former single-file `docs/delphi/execution-trace.md`, which predated the offload events and the dispatch views.
