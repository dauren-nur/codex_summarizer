# Turkic Audio Summarizer

This repository contains a Gradio application that transcribes Turkic-language audio,
translates the transcription into English, and generates an abstractive summary of the
translation using Hugging Face models.

## Features

- **Automatic Speech Recognition:** Powered by [`openai/whisper-small`](https://huggingface.co/openai/whisper-small).
- **Neural Machine Translation:** Backed by [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M) to convert Turkic text into English.
- **Summarization:** Uses the MT5-based [`csebuetnlp/mT5_m2m_crossSum`](https://huggingface.co/csebuetnlp/mT5_m2m_crossSum) model to condense the translated text.
- **Turkic language support:** Choose among Turkish, Azerbaijani, Kazakh, Uzbek, Tatar, Kyrgyz, Turkmen, and Uyghur inputs.
- **Gradio UI:** Simple web interface for uploading audio or recording via a microphone.

## Local Development

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   If you prefer to install packages manually:

   ```bash
   pip install gradio transformers accelerate sentencepiece librosa
   ```

2. Launch the Gradio app:

   ```bash
   python app.py
   ```

   The app will start a local server and print the URL in the console. Open it in your browser to interact with the interface.

## Google Colab

1. Upload the repository (or copy the files) into your Colab environment.
2. Run the following commands in a Colab cell:

   ```python
   !pip install -q gradio transformers accelerate sentencepiece librosa
   !python colab_app.py
   ```

   The script preloads all models and starts a public Gradio share link suitable for notebook environments.

## Notes

- The models used are large; for best performance run on a GPU-enabled environment.
- Whisper will automatically detect the spoken language, but select the closest Turkic option to guide translation quality.
- Translation is always performed into English to match the summarization model's expectations.
