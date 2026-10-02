# exec-trace-spec: speculative decoding breakdown

`tools/exec-trace/exec-trace-spec.py` summarizes the speculative decoding in an execution trace. It groups the executions tagged `generation_target` (verify passes) and `generation_draft` (draft steps, and the draft context's catch-up decodes of the verified tokens) by token count. For each group it reports:
- the number of executions;
- the host wall time per execution;
- the busy time of each device, and of any device;
- the CPU compute time.

It then gives the wall time of one speculative cycle: the draft executions per verify pass plus one verify pass.

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

```text
generation_draft      1 tokens x   528: host wall 1.78 ms (median 1.22), CPU compute 0.20 ms, device busy CUDA2 2.31 (any 2.31) ms
generation_draft      5 tokens x   130: host wall 1.51 ms (median 1.43), CPU compute 0.04 ms, device busy CUDA2 2.63 (any 2.63) ms
generation_target     5 tokens x   130: host wall 104.43 ms (median 103.18), CPU compute 0.15 ms, device busy CUDA0 37.15, CUDA1 37.92, CUDA2 31.15 (any 106.18) ms
      10.178 ms  CUDA1 / MUL_MAT_ID
       9.365 ms  CUDA0 / MUL_MAT
cycle: 132 verify passes, 5.00 draft executions per verify; 113.11 ms per cycle (draft 8.66, verify 104.45).
```

Here a width-5 verify costs 104 ms and the four 1-token draft steps plus the catch-up decode cost 9 ms per cycle. With three accepted tokens per pass, the time per output token is 113.11 / 4 = 28.3 ms.
