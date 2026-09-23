# exec-trace-dump: inspect a raw trace

`tools/exec-trace/exec-trace-dump.py` prints a summary of a raw execution trace. All views are derived offline from the recorded events.

## Synopsis

```text
python3 tools/exec-trace/exec-trace-dump.py /tmp/run.trace
python3 tools/exec-trace/exec-trace-dump.py /tmp/run.trace --scheduler-summary
python3 tools/exec-trace/exec-trace-dump.py /tmp/run.trace --timeline
python3 tools/exec-trace/exec-trace-dump.py /tmp/run.trace --dispatches
python3 tools/exec-trace/exec-trace-dump.py /tmp/run.trace --offload-decisions
python3 tools/exec-trace/exec-trace-dump.py /tmp/run.trace --gpu-timing --gpu-timing-json /tmp/gpu-timing.json
```

## Options

| flag | output |
| --- | --- |
| (none) | header, run metadata, execution/ubatch records, backend/graph/node counts, fingerprint records |
| `--dispatches` | every backend dispatch decision (implementation and variant per node) |
| `--offload-decisions` | every transfer-aware offload decision |
| `--scheduler-summary` | host scheduler span counts and durations, aggregated by category |
| `--timeline` | chronological begin/end records for host spans |
| `--gpu-timing` | summary of `DEVICE_EVENT` GPU timing spans and epochs |
| `--gpu-timing-json PATH` | write the GPU timing summary as JSON |

Summary categories can overlap because full-detail copy handling contains nested wait and copy-body spans. Dispatch records in one backend submission share a correlation timestamp because they describe implementation choices rather than per-node timing.

## Event table (schema v4)

Every event can identify its execution, ubatch, graph, node, scheduler split, backend, device, and trace-local thread. IDs are one-based; zero means the relationship is not available. String arguments are IDs in the trailing intern table; ID zero means no value is available.

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
| `OFFLOAD_DECISION` | selected side (CPU or candidate) | CPU compute, CPU transfer, candidate compute, and candidate transfer nanoseconds |
| `OFFLOAD_BYTES` | - | candidate weight H2D bytes, CPU input D2H bytes, CPU output H2D bytes, operation count |
| `OFFLOAD_CALIBRATION` | NUMA node + 1, or zero if unknown | CPU operations/s, candidate operations/s, H2D bytes/s, D2H bytes/s |

For `GPU_TIMING`, `timestamp_ns` is the host time before the start-event record. Scopes are `EAGER_SUBMIT` (detail: node count), `GRAPH_LAUNCH` (detail: DELPHI GPU graph ID), and `EAGER_OP` (node ID set; detail: nodes in the fused group). Statuses are `RESOLVED`, `UNRESOLVED`, `NOT_RECORDED`, and `FAILED`; offsets and durations are zero unless the span is resolved. For `GPU_TIMING_EPOCH`, `timestamp_ns` is the host time before the anchor record, and the completion source is `SYNCHRONIZE`, `QUERY`, or `NONE`. A span keeps the execution, ubatch, graph, split, backend, and device IDs that were current when its start event was recorded, although the span is written when its epoch resolves.

Timeline events use `timestamp_ns` as the host start timestamp and `args[0]` as the host end timestamp. A packed backend reference stores the device ID in the high 32 bits and the backend ID in the low 32 bits. Copy paths distinguish immediate user-input copies, selective asynchronous MoE expert copies, backend asynchronous copies, and fallback blocking copies. Wait metadata distinguishes backend synchronization from event synchronization and records the scheduler reason where available. Nested copy wait and body records are present only when `GGML_EXEC_TRACE_DETAIL=full`.

For reused scheduler graphs, the trace emits a new graph instance that refers to the first graph ID containing that topology. Node and split metadata are stored once for that scheduler topology; every graph instance still has its own fingerprint record.

The tracer does not add backend/device synchronization, evaluation callbacks, graph subdivisions, or tensor reads.
