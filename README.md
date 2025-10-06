# Turkic Audio Summarizer

This repository contains a Gradio application that transcribes Turkic-language audio,
translates the transcription into a target language, and generates an abstractive
summary of the translation using Hugging Face models.

## Features

- **Automatic Speech Recognition:** Powered by [`openai/whisper-small`](https://huggingface.co/openai/whisper-small).
- **Neural Machine Translation:** Backed by the [`facebook/m2m100_418M`](https://huggingface.co/facebook/m2m100_418M) multilingual model.
- **Summarization:** Uses [`facebook/bart-large-cnn`](https://huggingface.co/facebook/bart-large-cnn) to condense translated text. The
  app always summarizes an English rendition of the transcript to match the
  summarizer's training data while still returning the user-selected
  translation.
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
- The summarizer expects English input. Selecting a different translation target may require swapping the summarization model.
