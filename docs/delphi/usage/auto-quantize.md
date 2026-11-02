# Auto-quantize a model to a memory budget

The auto-quantizer chooses the quantization type of each tensor in a GGUF file. It works from a memory budget that you set for each GPU and for system RAM. It spends the budget where it lowers the predicted quality loss most. Use it when a stock quantization does not fit your memory well, or when you want a smaller model with a known quality cost.

Unlike a fixed recipe, the tool measures how each tensor group affects the output on a calibration text. It then searches for the mix with the lowest predicted KL divergence from the source model that fits your budget.

The tool lives in `tools/auto-quantize`. Its full reference is in `tools/auto-quantize/README.md`.

## What you need

You need four things before you start.

1. An importance matrix (imatrix) for the model. Calibration does not work without one. Without activation statistics, the tool ranks layers wrongly. On OLMoE-1B-7B without an imatrix, the fit was poor (chi-squared per degree of freedom of 8.6).
2. A source GGUF. A quantized source needs `allow_requantize: true` in the configuration.
3. The build binaries `llama-quant-damage`, `llama-quantize`, `llama-bench` and `test-backend-ops`, plus the backend your GPUs use. Use a separate build directory.
4. A Python environment with the `gguf-py` package from this repository.

## Quick start

The `quant_cli.py` wrapper runs every stage in one resumable command:

```sh
python tools/auto-quantize/quant_cli.py run --source /models/hf/Model --work /tmp/autoq --scratch /fast/scratch \
  --imatrix model.imatrix.gguf --calibration-text corpus.txt --devices ROCm0,ROCm1 --ctx 8192 \
  --weights loss=.4,decode=.3,short=.2,long=.1 --max-size 40GiB --output /models/out.gguf --auto
```

The `--devices` list gives the GPUs in layer order. The `--max-size` option caps the total tensor bytes. The `--auto` option encodes the best plan without asking. Without `--auto`, the tool shows a table of plans and asks you to pick one.

Run the command again after an interruption. The tool skips the stages that finished, and it runs the stages again when an input changed.

## The steps

The wrapper runs these stages in order:

1. It converts a safetensors directory to a Q8_0 GGUF, or uses a GGUF source as it is.
2. It measures the speed of each quantization type and shape, and it samples the error of each tensor.
3. It calibrates the quality model. It computes teacher logits from the source, then measures a set of tensor-group upgrades on the floor recipe.
4. It searches for plans within your budgets and weights.
5. It encodes the chosen plan and checks the result by loading it in `llama-server`.

The calibration stage is the slow one. On Qwen3.6-35B-A3B it took 24.5 minutes for 48 passes over 16K scored tokens.

If you want to run the stages one at a time, use `setup`, `plan`, `show` and `encode`. Run `quant_cli.py -h` for their options.

## Choices you make

Most of the choices come from the goal of the model.

Set the weights to say what matters to you. The weights `loss`, `decode`, `short` and `long` are relative. A loss weight of 0 means that quality does not count. A decode weight of 0 means that decode speed does not count.

Set the budget with `--max-size` for a total size, or with `--budget` and `--devices` for each device. If you need a size, `--minimize size --max-kld 0.05` finds the smallest plan under a KL ceiling. If you need the lowest loss, use `--minimize kld`.

Choose a plan with `--pick`. You can name a row number, a recipe id, or one of `best`, `min-kld`, `min-size`, `fastest-decode`, `fastest-short` and `fastest-long`.

For a source that ships its experts in MXFP4 or NVFP4, you can limit the pool to types of 4.25 bits per weight or less. The types from IQ2_K to MXFP4 are one such pool. Types above MXFP4 do not make the experts smaller than the source. The tool's default pool is wider, from IQ2_XXS to Q8_0, so set `primary_qtypes` to narrow it.

For a model that runs on another runtime, set `--runtime llama` or `--runtime ik_llama`. The tool then allows only the types that runtime loads. The default `delphi` runtime allows every type.

## Measured results

On Qwen3.6-35B-A3B with the experts at Q2_K as the floor, the plans were compared with a simple sum of local errors. At 2.2 GB and 4.2 GB above the floor, the planned mixes had 3.4 % and 13.1 % lower measured KL than the simple picks. At 9.2 GB above the floor, the two were equal within the noise. Both planned mixes were 18 to 20 % better than a uniform Q4_K.

For GLM-5.3 Flash at a 300K context, the planning run on 2026-10-31 predicted a KL of 0.0403 for its best mix. A uniform IQ4_KS expert set was predicted at 0.0447, and a uniform IQ4_XS set at 0.0478. The mix was encoded on 2026-11-01. Its measured KL was 0.0445, against 0.0461 for the file in use, which has IQ4_XS and Q4_K experts. The difference is within the noise (-0.0016 +- 0.0023 per chunk). The new mix was 5 to 12 % slower on CMP 170HX cards, where IQ4_K and IQ5_KS run slowly, so the old file stayed in use. Check the speed of a new mix on your GPUs before you switch.

For a DeepSeek V4.1 Flash calibration pass, one 2048-token KL run took 10.7 s on an R9700 and 38 s on a CMP 170HX. The calibration runs its KL passes on the first GPU in `calibration.devices`, so pick an R9700 for that run. The R9700 also leaves more CPU threads to the quantizer.

## Speed

The calibration time is set by `llama-quantize`, which is compute bound. The `quantize_threads` setting defaults to the number of allowed CPUs less 8. On a 64-thread EPYC 7532 host, 56 threads quantized 1.5 to 1.65 times as fast as 32 threads, and the output bytes were identical. The 8 spare CPUs serve the GPU drivers of the measurement run.

The projected time for DeepSeek V4.1 Flash is about 6.8 hours with the current settings, against 11 to 15 hours before the change. This is a projection from the measured stages, not a full timed run.

## Known limits

- An imatrix is required for calibration.
- The memory reserves (KV cache, compute buffers, draft and projector) are measured at the `--ctx`, `-ub` and `--parallel` of the run. Give the context that you will serve. A plan made for a small context may not fit a larger one.
- Expert caches need their own calibration. A quality table does not support them.
- The MTP blocks take the type of the pool with the most bytes, unless you pin them. The KL measurement does not run them.
- Predictions are model estimates on the calibration text. They are not bounds on other text or on task accuracy. The validation step reports how close the predictions are on held-out text.
- The default pool does not include `Q4_G128`, a type that only this build loads.
- Lazy tables such as engram tables stay at their source type unless you pin them. They do not count against the device budgets.

## Related pages

- [Expert cache](expert-cache.md) explains the cache that the planner's speed model assumes.
- [Performance and placement](performance-and-placement.md) describes how the offload and staging settings change speed.
