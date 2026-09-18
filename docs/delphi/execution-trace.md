# Execution trace

The execution tracer records the host-visible structure of llama.cpp inference without changing scheduler or backend synchronization. It is disabled by default.

Set `GGML_EXEC_TRACE` to write a raw trace when the process exits:

```text
GGML_EXEC_TRACE=/tmp/run.trace ./build/bin/llama-cli ...
```

`GGML_EXEC_TRACE_MAX_EVENTS` changes the preallocated event capacity from its default of 1,048,576 records. The trace sets its `truncated` flag if this capacity is exhausted. An invalid output path reports an error and disables tracing; inference continues.

The default `production` detail level records complete structural and dispatch data, scheduler/backend spans, and one outer span for each scheduler copy. Set `GGML_EXEC_TRACE_DETAIL=full` to also record nested copy source waits, destination waits, copy bodies, and MoE-ID waits. Full detail is a higher-overhead diagnostic mode; the integrity harness reports it as non-production-representative.

Decode a trace with:

```text
python3 tools/exec-trace/exec-trace-dump.py /tmp/run.trace
```

Add `--scheduler-summary` to aggregate host scheduler span counts and durations. Add `--timeline` to print chronological begin/end records. Add `--gpu-timing` to summarize device-event GPU timing, and `--gpu-timing-json PATH` to write that summary as JSON. These views are derived offline from the raw events.

## Timing provenance

Every timed interval has one of three provenance classes. Reports and exports label each interval with its class.

| provenance | source | use |
| --- | --- | --- |
| `HOST_SAFE` | host clock around host calls: execution, ubatch, scheduler submission, wait, and copy spans | production-representative when the integrity gate passes; measures host work, submission, or blocked time only |
| `DEVICE_EVENT` | timing-enabled HIP/CUDA events recorded by the backend on its own stream | device intervals from a production build; the integrity harness measures their perturbation |
| `VENDOR_EXTERNAL` | offline provider data, such as ROCprofiler-SDK | diagnostic only; never production-representative |

## Device-event GPU timing

Set `GGML_EXEC_TRACE_GPU_TIMING` together with `GGML_EXEC_TRACE` to record GPU intervals with backend timing events. It has no effect without the raw trace, and the integrity-only witness never records timing events.

| value | spans |
| --- | --- |
| `off` or `0` | none (default) |
| `submit` or `1` | one span for each eager HIP/CUDA backend submission and one span for each HIP/CUDA graph launch |
| `op` | `submit` spans, plus one span for each eagerly launched node or fused node group on the main stream |

Each HIP/CUDA backend context creates a fixed pool of timing-enabled events when the backend is created. `GGML_EXEC_TRACE_GPU_TIMING_EVENTS` sets the pool size (default 4096, rounded down to an even number; each span uses two events). The timer obeys these rules:

- It records events only on the context main stream, which already carries the work.
- It never records an event during stream capture. A graph launch is timed as a whole from outside the graph, so the graph topology and the capture, update, and replay behavior do not change.
- It does not allocate after backend creation, does not call an event or stream synchronization, and adds no device synchronization.
- Spans between two completion points form one epoch. Elapsed times are read after the existing stream synchronization in the backend `synchronize` call. When the backend is freed, a non-blocking event query decides whether an open epoch is complete.
- A span that cannot be resolved without a wait is recorded as `UNRESOLVED`. When the pool is full, new spans are recorded as `NOT_RECORDED` until the next completion point. A rejected runtime call is `FAILED`. The timer never waits for events to become free.

Operation spans cover only the main stream outside concurrent regions. Nodes on forked streams are covered by the enclosing submission span. HIP/CUDA graphs contain no timing events, so graph replays have no per-node device timing.

A device interval includes every gap between kernels on the stream inside the span, such as host launch latency. It is therefore an upper bound of kernel busy time, not the sum of kernel durations.

The decoder places device intervals on the host clock without a clock-rate model. All spans of an epoch share one device anchor, the start event of the first span. The host reads its clock before it records each start and end event, so the device reaches the event after that sample. The device completes each end event before completion is known. These constraints bound the anchor. The decoder uses the lower bound as the placement and reports the width of the bound as the placement uncertainty. An epoch whose bounds cannot all hold is reported as `inconsistent`.

The run metadata records `gpu_timing`, `gpu_timing_event_capacity`, and the started and released span counts. A started span that was never released belongs to a backend that was not freed before the trace was written.

## Integrity gate

Use the paired-run harness to determine whether a trace is production-representative:

```text
python3 tools/exec-trace/exec-trace-integrity.py \
    --binary build/bin/llama-bench \
    --report /tmp/integrity.json \
    --launcher "numactl --cpunodebind=0 --membind=0" \
    -- -m model.gguf -p 512 -n 0
```

Arguments after `--` are passed to `llama-bench`. The default is five repetitions per process. Use `--repetitions` to change it. The harness forces JSON benchmark output and uses the same binary and arguments for five processes (seven with `--gpu-timing`):

1. A trace-disabled structural witness with `GGML_EXEC_TRACE_INTEGRITY` enabled. This run also warms file-backed routed-expert pages before the timed pair.
2. A clean timed baseline with tracing and the integrity observer disabled.
3. A timed trace run with the structural observer enabled.
4. A second timed trace run.
5. A second clean timed baseline.

With `--gpu-timing submit` or `--gpu-timing op`, two timing runs (`GGML_EXEC_TRACE` plus `GGML_EXEC_TRACE_GPU_TIMING`) run between the two trace runs, so the timed order is clean, trace, timing, timing, trace, clean. `--gpu-timing-events` sets the event pool size for these runs. The report then has a `gpu_timing` section that compares the timing runs with the clean runs and with the safe trace, checks every structural fingerprint against the witness, and reports how many spans resolved, were not recorded, stayed unresolved, or failed, and whether each epoch has a consistent host placement. The classification is `low-perturbation` (at most 2% slowdown, complete coverage), `low-perturbation, incomplete coverage`, `perturbative`, or `invalid`. The `GGML_EXEC_TRACE_GPU_TIMING` runtime flag is expected in the timing runs and is not treated as contamination there. The harness exits with status 2 when the safe trace or the requested timing mode is not trustworthy.

The separate structural witness prevents observer work from being hidden in the measured trace overhead. Reversing the second timed pair balances process-order effects such as page-cache, thermal, and GPU-clock drift. The machine-readable JSON report contains the exact commands, relevant environment, process wall times, every `llama-bench` wall-time and throughput sample, median/min/max values, and sample dispersion. The human-readable output shows the main gate result.

The gate compares ordered graph shape/operation data, total node count, scheduler split structure, node-to-backend assignments, scheduler copy paths, backend dispatch choices, GPU graph lifecycle behavior, and scheduler evaluation-callback use. A structural mismatch rejects the trace even when throughput is unchanged. A GPU graph replay-to-launch fallback is therefore a failure, not harmless timing noise.

Throughput classification uses the median sample on each side:

- At most 1% slowdown: preferred and production-safe.
- More than 1% and at most 2%: acceptable for Milestone 1.
- More than 2%: investigate; the report is not production-representative.

The report also rejects truncated traces, dropped events, debug or sanitizer builds detected by the runtime or adjacent CMake cache, graph-disable and operation-trace environment settings, directly linked profiler libraries, and an external profiler named in `--launcher`. It ignores profiler support libraries pulled in transitively by a normal GPU runtime. Detection is not exhaustive. Driver-level tracing, system profilers attached after launch, thermal or power-state changes, and unrelated background activity can remain undetected. Review the raw samples and dispersion; the harness does not remove outliers.

`GGML_EXEC_TRACE_INTEGRITY` writes the fixed-size structural witness used by the harness. It does not allocate the raw event buffer or read the host clock at each trace site. It is an internal validation mechanism rather than a user trace format.

## Raw schema version 4

All integers use little-endian byte order. The file contains a 96-byte header, fixed-size 80-byte events, then the interned string table. Header fields give the schema version, header and event sizes, trace flags, monotonic start timestamp, event count, string count, string-table byte count, event capacity, attempted and dropped event counts, buffer growth count, and post-execution serialization duration. The fixed preallocated buffer has zero growth operations. Each string entry has a 32-bit ID and byte length followed by UTF-8 data. ID zero means that no value is available.

Every event can identify its execution, ubatch, graph, node, scheduler split, backend, device, and trace-local thread. Four fixed 64-bit arguments carry event-specific data. Readers must use the event kind and ignore kinds that they do not know.

IDs are one-based. Zero means that the relationship is not available. String arguments are IDs in the trailing intern table.

| event | flags | args 0..3 |
| --- | --- | --- |
| `RUN_METADATA` | - | key string, value string, -, - |
| `EXECUTION_BEGIN` | execution kind | token count, logical batch size, ubatch size, batch threads in high 32 bits and decode threads in low 32 bits |
| `EXECUTION_END` | - | status, -, -, - |
| `UBATCH_BEGIN` | - | token count, sequence count, unique sequence count, output count |
| `UBATCH_END` | - | status, -, -, - |
| `BACKEND` | device type | backend name, buffer type, device name, device description |
| `GRAPH` | bit 0 means reused topology | topology graph ID, node count, leaf count, split count |
| `NODE` | numeric GGML op | graph node index, tensor name, op name, output type name |
| `NODE_DIMS` | - | output dimensions 0..3 |
| `NODE_SOURCE` | - | source slot, source node ID, numeric source type, source name |
| `NODE_SOURCE_DIMS` | source slot | source dimensions 0..3 |
| `NODE_VIEW` | - | source node ID, view offset, source name, - |
| `NODE_MATMUL` | numeric GGML op | effective M, N, K, selected-expert dimension or zero |
| `SCHED_SPLIT` | - | first node, exclusive last node, node count, input count |
| `SCHED_SPLIT_INPUT` | source backend ID | input index, source node ID, source split ID, source name |
| `FINGERPRINT` | - | op/shape hash, assignment hash, split-sequence hash, cross-backend edge count |
| `BACKEND_SUBMIT` | - | host end timestamp, status, first node, exclusive last node |
| `CPU_COMPUTE` | - | host end timestamp, status, first node, exclusive last node |
| `BACKEND_WAIT` | wait reason | host end timestamp, wait API, -, - |
| `COPY_HANDLING` | copy path | host end timestamp, destination tensor ID, byte count, source device/backend IDs |
| `COPY_WAIT_SRC` | copy path | host end timestamp, wait API, destination device/backend IDs, byte count |
| `COPY_WAIT_DST` | copy path | host end timestamp, wait API, source device/backend IDs, byte count |
| `COPY_BODY` | copy path | host end timestamp, destination tensor ID, byte count, source device/backend IDs |
| `DISPATCH` | numeric GGML op | implementation ID, implementation name, variant name/ID, implementation-specific configuration |
| `GPU_GRAPH` | capture/update/replay/fallback state | GPU graph ID, node count, lifecycle reason, - |
| `CLOCK_SYNC` | secondary clock kind | secondary-clock timestamp, trace-clock timestamp after the secondary sample, -, - |
| `GPU_TIMING` | scope (bits 0-7), status (bits 8-15), provenance (bits 16-23) | host time before the end-event record, device start offset from the epoch anchor, device duration, epoch ID in high 32 bits and scope detail in low 32 bits |
| `GPU_TIMING_EPOCH` | provenance (bits 16-23) | host time when completion was known, epoch ID, span count, completion source |

`GPU_TIMING` and `GPU_TIMING_EPOCH` are additive schema-v4 kinds (27 and 28). For `GPU_TIMING`, `timestamp_ns` is the host time before the start-event record. Scopes are `EAGER_SUBMIT` (detail: node count), `GRAPH_LAUNCH` (detail: DELPHI GPU graph ID), and `EAGER_OP` (node ID set; detail: nodes in the fused group). Statuses are `RESOLVED`, `UNRESOLVED`, `NOT_RECORDED`, and `FAILED`; offsets and durations are zero unless the span is resolved. For `GPU_TIMING_EPOCH`, `timestamp_ns` is the host time before the anchor record, and the completion source is `SYNCHRONIZE`, `QUERY`, or `NONE`. A span keeps the execution, ubatch, graph, split, backend, and device IDs that were current when its start event was recorded, although the span is written when its epoch resolves.

Timeline events use `timestamp_ns` as the host start timestamp and `args[0]` as the host end timestamp. A packed backend reference stores the device ID in the high 32 bits and the backend ID in the low 32 bits. Copy paths distinguish immediate user-input copies, selective asynchronous MoE expert copies, backend asynchronous copies, and fallback blocking copies. Wait metadata distinguishes backend synchronization from event synchronization and records the scheduler reason where available. Nested copy wait and body records are present only when `GGML_EXEC_TRACE_DETAIL=full`.

The current events record:

- execution and ubatch begin/end records with call kind, sizes, thread counts, and status;
- backend, buffer-type, device-name, and device-description strings;
- graph instances and topology reuse;
- GGML node names, operations, output types and dimensions, source relationships and source dimensions, and view relationships;
- effective matmul `M`, `N`, and `K`, plus the selected-expert dimension for `MUL_MAT_ID`;
- scheduler split ranges, backends, input dependencies, and source splits;
- deterministic hashes of ordered operations/shapes, node-to-backend assignments, and split backend/device sequences, plus the cross-backend edge count.
- host time spent submitting non-CPU work, synchronously computing CPU splits, waiting in existing scheduler synchronization calls, and handling cross-backend transfers;
- source/destination identity, transferred bytes, and an outer copy-handling span; full detail adds separate source-wait, destination-wait, and blocking-copy spans for scheduler paths that already expose those phases;
- backend implementation and variant choices for traced CPU and HIP/CUDA operations;
- HIP/CUDA graph capture, update, replay, and fallback behavior with fallback reasons.

The decoder prints trace self-overhead metadata: attempted and dropped events, interned strings, allocated event-buffer bytes, growth count, and serialization duration. Serialization happens after measured execution and does not include a backend synchronization.

The host clock is `CLOCK_MONOTONIC_RAW` on Linux. At startup and finalization, a `CLOCK_SYNC` event brackets one `CLOCK_MONOTONIC` sample between two trace-clock samples. Offline tools use the bracket midpoints to align the clocks and account for rate drift; half of each bracket width bounds its sampling uncertainty. Execution, ubatch, scheduler, and copy spans measure host-side work, submission, or blocked time only. They are not GPU execution or DMA time; `GPU_TIMING` spans carry device time. Dispatch records in one backend submission share a correlation timestamp because they describe implementation choices rather than per-node timing. Summary categories can overlap because full-detail copy handling contains nested wait and copy-body spans. The tracer does not add backend/device synchronization, evaluation callbacks, graph subdivisions, or tensor reads.

For reused scheduler graphs, the trace emits a new graph instance that refers to the first graph ID containing that topology. Node and split metadata are stored once for that scheduler topology; every graph instance still has its own fingerprint record.

Trace output is finalized at normal process exit. `ggml_exec_trace_flush()` exists for controlled shutdown and tests, but callers must only use it when no trace writer is active. It does not synchronize backend work.

## GPU activity import

Offline GPU activity uses the provider-neutral schema `delphi-gpu-activity-v1`, defined in `tools/exec-trace/exec-trace-gpu-activity.py`. A provider adapter writes one document with these parts:

- `provider`: name, provenance, `production_representative`, timestamp clock, and `graph_node_identity` (`none`, or `provider_reported` when activities carry provider graph IDs that DELPHI does not map to GGML nodes);
- `devices`: provider-scoped `device_ref`, type, GPU index, and names;
- `api_calls`: host API records with thread, stream, correlation IDs, provider timestamps, and a `graph_launch` flag;
- `activities`: `kernel`, `copy`, or `memset` records with device references, queue and stream, provider timestamps, correlation IDs, and copy bytes and direction.

Validation rejects `VENDOR_EXTERNAL` data that claims to be production-representative and graph IDs from a provider that declares none. `exec-trace-gpu-activity.py validate DOC` checks a document, and `exec-trace-gpu-activity.py merge TRACE DOC -o OUT` aligns it with a raw trace. The merge is the same for every provider: clock alignment, device mapping, conservative correlation, and device intervals. A provider whose clock is not `CLOCK_MONOTONIC` needs `--clock-offset-ns` until the trace records an anchor for that clock.

### ROCprofiler-SDK

`exec-trace-rocm.py` is the ROCprofiler-SDK adapter. ROCprofiler timing is `VENDOR_EXTERNAL`: tracing perturbs the observed run, so use it only for qualitative checks. Use device-event timing and the integrity harness for production-representative measurements. Collect asynchronous activity with `rocprofv3`, then merge it offline. The following configuration was validated with ROCprofiler-SDK 1.3.2 from ROCm 7.14.0; check `rocprofv3 --version` and the matching [AMD documentation](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/docs-7.14.0/how-to/using-rocprofv3.html) before reusing it with another version.

The smallest activity-only configuration is:

```text
rocprofv3 --kernel-trace --memory-copy-trace --group-by-queue \
    --output-format json --output-directory /tmp/trace --output-file rocm \
    -- <application> <arguments>
```

Add `--hip-runtime-trace` for DELPHI correlation. HIP API records provide host timestamps and correlation IDs that group kernels or copies under the API operation that submitted them. This does not require `GGML_HIP_OP_TRACE`, ROCTx, graph disabling, or new application synchronization. A complete collection has this form:

```text
GGML_EXEC_TRACE=/tmp/trace/delphi.trace \
rocprofv3 --hip-runtime-trace --kernel-trace --memory-copy-trace --group-by-queue \
    --output-format json --output-directory /tmp/trace --output-file rocm \
    -- numactl --cpunodebind=0 --membind=0 <application> <arguments>
```

Merge the files without rerunning inference:

```text
python3 tools/exec-trace/exec-trace-rocm.py \
    /tmp/trace/delphi.trace /tmp/trace/rocm_results.json \
    -o /tmp/trace/merged.json --activity-output /tmp/trace/activity.json
```

`--activity-output` also writes the `delphi-gpu-activity-v1` provider document. The merged output uses schema `delphi-gpu-activity-merged-v1` and keeps normalized records and relationships in separate tables. It copies the provider block. `gpu_activities` contains kernel and copy records with the provider `device_ref`, stream/queue, name, provider (`external_*`) and aligned timestamps, correlation IDs, copy bytes/direction, graph IDs when the provider supplies them, and `provenance`. It references `gpu_api_calls`, `host_submissions`, `host_copies`, `dispatches`, and `gpu_graph_events` by integer ID. `device_intervals` derives kernel-busy, DMA, combined-active, and idle intervals within the first-to-last DELPHI execution window. `host_wait_overlaps` reports how much of each host wait overlaps kernel, DMA, combined GPU activity, or no observed GPU activity.

ROCprofiler-SDK can place invalid UTF-8 bytes in traced API string arguments. The importer replaces invalid sequences because it does not consume those argument strings, keeps JSON structure validation strict, and reports the replacement count in `diagnostics.rocprofiler_utf8_replacements`.

ROCprofiler-SDK timestamps on the validated host use `CLOCK_MONOTONIC`. DELPHI schema-v4 traces bracket that clock at startup and finalization. The importer uses both anchors to convert timestamps to the trace's `CLOCK_MONOTONIC_RAW` domain while accounting for clock-rate drift, and it reports the larger bracket sampling uncertainty. Early schema-v4 traces with one anchor use a constant offset. The importer applies a separate 10 us boundary tolerance when it tests whether a HIP API call falls in a host span; this tolerance is not reported as clock uncertainty. Older traces need an explicitly measured `--clock-offset-ns`.

The merge maps a single active GPU automatically. With multiple active GPUs it pairs ordered DELPHI device IDs with ordered provider `gpu_index` values. Use repeatable `--device-map DELPHI_DEVICE_ID=ROCPROFILER_GPU_INDEX` arguments when visibility or device ordering makes that inference unsafe.

Cross-trace relationships are conservative:

| relationship | meaning |
| --- | --- |
| `ordered_graph_launch` | provider correlation links activity to a graph-launch API, and that API is the only launch paired with a DELPHI replay in the host submission; the cross-trace pairing remains heuristic |
| `api_within_copy` | a correlated copy API falls within one DELPHI copy-handling span |
| `ordered_within_dispatch` | a correlated API falls within one backend submission, but no GGML node boundary is known |
| `time_window_candidate` | only aligned time, device, and submission order provide a candidate |
| `unattributed` | no safe candidate was found |

ROCprofiler-SDK 1.3.2 does not emit HIP graph execution or graph-node IDs in direct JSON. DELPHI can therefore correlate graph launches and repeated activity groups, but it cannot claim an exact HIP-graph-node-to-GGML-node mapping on this installation. The adapter preserves `graph_exec_id` and `graph_node_id` when a newer JSON producer supplies them and then sets `graph_node_identity` to `provider_reported`, without upgrading the GGML relationship to exact. AMD HIP can also report an in-graph copy as a `__amd_rocclr_copyBuffer` kernel instead of a copy record.

A future CUDA provider (CUPTI activity records or an Nsight Systems export) needs only an adapter that writes `delphi-gpu-activity-v1`: map CUDA devices to `device_ref` and `gpu_index`, set `graph_launch` for `cudaGraphLaunch`, keep CUPTI correlation IDs in `correlation_ids.internal`, and declare its timestamp clock. CUPTI graph node IDs, if collected, are `provider_reported` identity; they do not map to GGML nodes by themselves.

## Unified timeline and Perfetto export

`exec-trace-perfetto.py` combines the raw DELPHI trace, its device-event timing, and an optional merged GPU activity document. It does not rerun inference or add runtime instrumentation:

```text
python3 tools/exec-trace/exec-trace-perfetto.py \
    /tmp/trace/delphi.trace \
    --gpu-activity /tmp/trace/merged.json \
    --execution 1 \
    -o /tmp/trace/trace.perfetto.json \
    --unified-output /tmp/trace/unified.json \
    --summary /tmp/trace/summary.json \
    --tree /tmp/trace/tree.md
```

`--execution` selects one execution ID. Omit it to cover the interval from the first execution begin through the last execution end. `--gpu-activity` is optional; `--rocm-merged` is an older name for it. The exporter accepts `delphi-gpu-activity-merged-v1` and upgrades legacy `delphi-rocm-activity-v1` files. It rejects a truncated raw trace and a merged document whose schema or execution intervals do not match the raw trace.

Open `trace.perfetto.json` with the **Open trace file** action at [ui.perfetto.dev](https://ui.perfetto.dev/). The output uses the Chrome Trace Event JSON format accepted by Perfetto. Timestamps in Trace Event records are microseconds relative to the selected execution window so large absolute nanosecond values do not lose precision. Each record also keeps its original aligned nanosecond timestamps in its arguments.

The timeline groups records into these logical processes and tracks:

- host threads, including scheduler submissions, waits, and copy handling;
- CPU backend execution spans;
- scheduler graph, split, GGML node, and cross-backend dependency metadata;
- correlated HIP API calls on the host threads observed by ROCprofiler;
- dispatch and HIP graph lifecycle events for each GPU device;
- `DEVICE_EVENT` submission and operation tracks for each GPU device, placed on the host clock, with status, epoch, and placement uncertainty;
- separate kernel and DMA tracks for each observed queue/stream, labeled with the provider provenance;
- an activity/idle union track for each GPU device.

Node and dispatch events include the available GGML operation, tensor name, output type and shape, backend, implementation and variant. Matmul records also include `M`, `N`, `K`, selected-expert count when applicable, and source-0 weight type and shape. GPU activity keeps its correlation relationship and confidence. When a host submission contains exactly one dispatch, the activity also carries that dispatch as a candidate; otherwise it stays explicitly scoped to the host submission.

Logical flow arrows connect traced producer/copy/destination tensor IDs and connect correlated host API, copy, or submission records to GPU activity. A host copy-handling span is still distinct from an observed GPU DMA record. The former measures host code; the latter measures activity reported by ROCprofiler.

`unified.json` uses schema `delphi-unified-timeline-v1`. It is the canonical offline timeline/DAG representation and contains tracks, overlapping events, flows, the selected window, and the summary. The Perfetto file and optional tree are derived views. `tree.md` groups execution -> graph -> split -> GGML node -> dispatch and then summarizes correlated GPU activity. It is for navigation only and does not represent overlap or a call tree.

The summary distinguishes span sums from interval unions. `timing_provenance` names the class of each section. `gpu_device_event` reports the union of `DEVICE_EVENT` submission and graph-launch intervals, its intersection with CPU compute, status counts, and the largest placement uncertainty. `gpu` reports provider activity with its provider name and provenance. It reports execution wall time, CPU-compute span sum and union, backend-wait sum, copy-wait sum, copy-handling sum, GPU kernel/DMA/combined-active unions, observed idle time, and the intersection of CPU-compute and GPU-active unions. Overall GPU unions merge activity across devices, so simultaneous work on two GPUs counts once; the per-device table retains each device separately. Observed idle means the complement of reported kernel/DMA activity within the selected execution window, not proof that the physical GPU had no other work.

The raw trace currently exposes a synchronous `CPU_COMPUTE` backend span but not individual CPU worker intervals. Perfetto therefore shows a logical CPU backend execution track, not per-worker scheduling. ROCprofiler timing remains diagnostic rather than production-absolute. The final balanced integrity gate measured -1.51% median decode throughput and -0.01% prefill throughput with DELPHI. A separate balanced decode matrix measured -0.55% for DELPHI, -60.25% for ROCprofiler, and -52.10% for both relative to clean execution. The external-profiler samples had high dispersion, including very slow first samples. Use the integrity harness, without ROCprofiler, to decide whether the in-process trace is production-representative.

## One-token reference procedure

The following produces a small trace that contains one generated token while preserving the required single-socket placement and normal asynchronous execution:

```text
mkdir -p /tmp/delphi-one-token
GGML_EXEC_TRACE=/tmp/delphi-one-token/delphi.trace \
HIP_VISIBLE_DEVICES=1 \
rocprofv3 --hip-runtime-trace --kernel-trace --memory-copy-trace --group-by-queue \
    --output-format json --output-directory /tmp/delphi-one-token --output-file activity \
    -- numactl --cpunodebind=0 --membind=0 build-hip/bin/llama-cli \
    -m /path/to/model.gguf -p "Trace one token." -n 1 -ngl 999 -cmoe \
    -no-cnv --single-turn </dev/null

python3 tools/exec-trace/exec-trace-rocm.py \
    /tmp/delphi-one-token/delphi.trace \
    /tmp/delphi-one-token/activity_results.json \
    -o /tmp/delphi-one-token/merged.json

python3 tools/exec-trace/exec-trace-dump.py /tmp/delphi-one-token/delphi.trace

python3 tools/exec-trace/exec-trace-perfetto.py \
    /tmp/delphi-one-token/delphi.trace \
    --gpu-activity /tmp/delphi-one-token/merged.json \
    --execution 3 \
    -o /tmp/delphi-one-token/trace.perfetto.json \
    --unified-output /tmp/delphi-one-token/unified.json \
    --summary /tmp/delphi-one-token/summary.json \
    --tree /tmp/delphi-one-token/tree.md
```

The validated command above used execution 3 for the real prompt evaluation after two warm-up executions. Use the decoder output to select the corresponding execution if this changes. A useful reference artifact shows scheduler splits and node/dispatch metadata, CPU and GPU backend work, cross-backend handling, HIP graph lifecycle, actual kernel/DMA activity, observed idle gaps, and the correlation confidence on every host-to-GPU relationship.

The final Milestone 1 reference export selected one prompt execution with 934 GGML nodes, 34 scheduler splits, 129 dispatch decisions, 17 HIP graph lifecycle records, 380 GPU kernels, and 32 DMA records. The unified view contained 3,052 events and 520 flow relationships; the compact Perfetto JSON was 1.6 MiB. The raw trace dropped no events. The artifact is not checked in because the ROCprofiler input and normalized intermediate total approximately 94 MiB; use the procedure above to reproduce it.
