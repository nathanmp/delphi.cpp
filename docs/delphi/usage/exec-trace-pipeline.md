# exec-trace-pipeline: prefill pipeline timeline

`tools/exec-trace/exec-trace-pipeline.py` shows how the ubatches of large executions flow through the devices. It is meant for pipeline-parallel (layer split) prefill. For every execution with at least `--min-tokens` tokens it lists:
- each ubatch on each device, with its first device start, last device end and busy time, in ms from the start of the execution;
- the host waits by reason and device;
- how long 0, 1, 2, ... devices were busy at the same time, with the work of any execution (consecutive prompt batches overlap in a pipeline).

With more than one execution shown, a last line gives the same overlap over the whole window.

## Synopsis

```text
python3 tools/exec-trace/exec-trace-pipeline.py TRACE [--min-tokens N] [--last N]
```

| flag | meaning |
| --- | --- |
| `--min-tokens N` | smallest execution to show (default 512) |
| `--last N` | show only the last N matching executions, e.g. the batches of the last prompt |

Record with `GGML_EXEC_TRACE=<file>` and `GGML_EXEC_TRACE_GPU_TIMING=submit` (or `op`).

## Reading it

In a well-fed pipeline each stage starts the next ubatch as soon as it finishes the previous one, and the first stage is never idle until the last ubatch. Things to look for:
- A gap on the first stage between two executions, while the later stages finish, is a drain. Look for the host wait that caused it, such as `graph_reallocation`.
- A stage whose busy time per ubatch is clearly higher than the others limits the throughput: the bound is roughly the number of ubatches times its time per ubatch, plus the fill and drain.
- Busy time that grows ubatch by ubatch on one device, while the others stay flat, can be thermal throttling.
