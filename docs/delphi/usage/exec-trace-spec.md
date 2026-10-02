# exec-trace-spec: speculative decoding breakdown

`tools/exec-trace/exec-trace-spec.py` summarizes the speculative decoding in an execution trace. It groups the executions tagged `generation_target` (verify passes) and `generation_draft` (draft steps, and the draft context's catch-up decodes of the verified tokens) by token count. For each group it reports:
- the number of executions;
- the host time in `llama_decode` per execution;
- the busy time of each device, and of any device;
- the CPU compute time.

`llama_decode` can return before the device work ends, so host time in the call is not the cost of an execution. The tool therefore measures one speculative cycle (the draft executions of a verify pass plus the pass) from the start of one verify pass to the next. It also gives the device busy time per cycle, the share of the cycle when every device is idle (host-side work such as sampling and draft control, or waits), and the CPU compute per cycle.

Device spans are placed at the earliest start the host clock allows. With a layer split the host submits every stage of a verify pass before the first one ends, so a later stage can start up to the whole pass later than placed. When the median placement uncertainty is more than 5% of the cycle, the tool does not report the any-device time and the idle share. It gives the sum of the device busy times instead, which is the critical path when the devices run one after another.

## Synopsis

```text
python3 tools/exec-trace/exec-trace-spec.py TRACE [--top N] [--node-regex REGEX] [--accepted-per-verify X] [--json OUT]
```

| flag | meaning |
| --- | --- |
| `--top N` | operations listed per group when the trace has operation timing (default 8) |
| `--node-regex REGEX` | count the operations whose node name matches as one group, e.g. `'attn\|kq'` for attention |
| `--accepted-per-verify X` | accepted draft tokens per verify pass, to print the time per output token and t/s |
| `--json OUT` | also write the summary as JSON |

## Recording

Device times need GPU timing. Record with `GGML_EXEC_TRACE=<file>` and `GGML_EXEC_TRACE_GPU_TIMING=submit` for production timing, or `=op` with graphs and fusion disabled (`GGML_CUDA_DISABLE_GRAPHS=1`, `GGML_HIP_DISABLE_GRAPHS=1`, `GGML_CUDA_DISABLE_FUSION=1`, `GGML_CPU_DISABLE_FUSION=1`) to rank operations. Operation-mode times are inflated by the missing graphs and fusion: use them to compare operations, not to measure throughput. CPU operations are listed when the trace has CPU node timing.

Acceptance is not in the trace. The server reports `draft_n` and `draft_n_accepted` in each response's `timings`. Divide the accepted tokens by the number of verify passes the tool reports for `--accepted-per-verify`. A synthetic acceptance arm (`--spec-synth-rates 1,1,1,0`) accepts a fixed number per pass.

## Example

GLM-5.3 Flash, MTP with two draft tokens, production timing (`submit`), whole trace:

```text
generation_draft      1 tokens x   781: host in llama_decode 1.01 ms (median 1.17), CPU compute 0.13 ms, device busy CUDA2 1.48 (any 1.48) ms
generation_draft      3 tokens x   390: host in llama_decode 0.44 ms (median 0.40), CPU compute 0.02 ms, device busy CUDA2 1.35 (any 1.35) ms
generation_target     3 tokens x   390: host in llama_decode 3.02 ms (median 1.69), CPU compute 0.20 ms, device busy CUDA0 13.44, CUDA1 14.22, CUDA2 12.91 (any 40.54) ms
cycle: 391 verify passes, 2.99 draft executions per verify; 61.80 ms verify to verify (median 49.26)
  per cycle: device busy CUDA0 13.45, CUDA1 14.22, CUDA2 17.21 (any 44.85 ms, all devices idle 27% of the cycle); CPU compute 0.47 ms; host time in llama_decode: draft 2.46 ms, verify 3.18 ms
with 1.65 accepted draft tokens per verify: 23.32 ms per output token (42.9 t/s)
```

Without request windows, the mean cycle includes the gaps between requests; the median (49.26 ms, or 18.6 ms per output token) matches the measured decode rate. The bottleneck profiler passes the client's request windows, so its cycle excludes those gaps.
