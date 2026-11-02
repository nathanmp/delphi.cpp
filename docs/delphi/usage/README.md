# delphi.cpp usage guides

These guides cover the features that delphi.cpp adds to llama.cpp. Each guide starts with the goal, then shows a command, then explains the settings and the limits. Most features work without any setup, and each guide says which settings you may want to change.

- [Auto-tune the launch settings](auto-tune.md): searches batch, thread and speculative settings for your prompts, and checks the result with a fresh launch.
- [Auto-quantize a model to a memory budget](auto-quantize.md): chooses a quantization type for each tensor group under your memory budget, from a calibrated quality model.
- [Expert cache for MoE models](expert-cache.md): copies the most-used host-resident experts to the GPU. It is on by default.
- [Speculative decoding](speculative-decoding.md): MTP for Qwen, DFlash2 for GLM-5.3 Flash and DSpark for DeepSeek V4.1 Flash. The draft depth adapts by default.
- [Share GPUs between servers (experimental)](vram-sharing.md): lets several servers borrow each other's idle GPU memory for their expert caches.
- [Vision (image input)](vision.md): runs the vision encoder on a GPU, with dynamic memory by default.
- [Performance and placement](performance-and-placement.md): the automatic placement features, weight streaming, CPU expert settings and the profiling tools.

The main delphi.cpp settings are also command-line options, and `llama-server --help` lists them. Each of these options has an environment variable, so scripts that set the variable keep working; a value on the command line takes precedence.

Reference pages for the settings:

- [Runtime environment variables](environment-variables.md) lists every environment variable with its default.
- [Draft depth of MTP speculative decoding](speculative-draft-depth.md) gives the full depth rule.
- [Vision memory](vision-memory.md) describes the vision memory modes in detail.
- [Tune endpoints](tune-endpoints.md) describes the server routes that the auto-tuner uses.
- [Server context checkpoints](server-checkpoints.md) describes how the server reuses prompt state for models with sliding-window or recurrent layers.
- [Bottleneck profiler recipes](bottleneck-profiler.md) and [Recording an execution trace](exec-trace-record.md) describe the profiling tools.
