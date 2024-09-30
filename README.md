# Building the associated model

This repository may contain a built [OpenAI Whisper]() model for use with ONNX. This model was built by roughly following [this tutorial](https://github.com/microsoft/onnxruntime-inference-examples/blob/main/js/ort-whisper/README.md):

To summarize, build by:
1. Cloning the [Microsoft Olive](https://github.com/microsoft/Olive/tree/main/examples/whisper) repository and [installing Olive](https://github.com/microsoft/Olive/blob/main/examples/README.md).
2. Setting up a Python virtual environment.
3. Installing dependencies:
   ```
   cd examples/whisper
   python -m pip install -r requirements.txt
   python -m pip install librosa
   ```
4. Building the model:
   ```
   python prepare_whisper_configs.py --model_name openai/whisper-tiny --multilingual --enable_timestamps --no_audio_decoder
   python -m olive run --config whisper_config_cpu_int8.json --setup
   python -m olive run --config whisper_config_cpu_int8.json
   ```
5. Uploading the model from `examples/whisper/models/whisper_cpu_int8/model.onnx`.

To use the model in multilingual mode, it was also necessary to determien the `forced_decoder_ids` corresponding to different languages ([see the example README](https://github.com/microsoft/Olive/tree/main/examples/whisper)). This was done with the following Python script:
```python
# See https://github.com/microsoft/Olive/tree/main/examples/whisper
# for additional options and explanations

from transformers import AutoConfig, AutoProcessor

model = "openai/whisper-tiny"
whisper_config = AutoConfig.from_pretrained(model)
processor = AutoProcessor.from_pretrained(model)

languages = [ "english", "spanish", "german", "french", "italian", "dutch", "korean", "thai", "russian", "portuguese", "polish", "indonesian", "hindi" ]

for lang in languages:
	task_ids = processor.get_decoder_prompt_ids(language=lang, task="transcribe", no_timestamps=False)

	forced_decoder_ids = [ whisper_config.decoder_start_token_id ]
	for [index, id] in task_ids:
		forced_decoder_ids.append(id)

	# Note: This should have 3-4 entries (depending on no_timestamps)
	print('\"{}\" -> intArrayOf({})'.format(lang, ','.join(map(str, forced_decoder_ids))))
```
