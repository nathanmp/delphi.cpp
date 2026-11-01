# DeepSeek V4.1 Flash 1017 encoding fixtures

The five input/output pairs are copied unchanged from the model release's `encoding/tests` directory:

https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/main/encoding/tests

`test-jinja` compares rendered prompts with these outputs. The test converts historical tool arguments from JSON strings to objects before rendering, as the chat integration does. It does not load images or a model.

The fixtures are covered by the accompanying DeepSeek MIT license.
