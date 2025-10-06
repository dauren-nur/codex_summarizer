"""Gradio app for speech transcription, translation, and summarization."""
from functools import lru_cache
from typing import Tuple

import gradio as gr
import numpy as np
import torch
from transformers import pipeline

ASR_MODEL_NAME = "openai/whisper-small"
TRANSLATION_MODEL_NAME = "facebook/m2m100_418M"
SUMMARIZATION_MODEL_NAME = "facebook/bart-large-cnn"

# The M2M100 tokenizer expects ISO language codes. All entries below are
# validated against the list of supported languages in
# https://huggingface.co/facebook/m2m100_418M

TURKIC_LANGUAGE_CODES = {
    "Turkish": "tr",
    "Azerbaijani": "az",
    "Kazakh": "kk",
    "Uzbek": "uz",
    "Tatar": "tt",
    "Kyrgyz": "ky",
    "Turkmen": "tk",
    "Uyghur": "ug",
}

TARGET_LANG_CODES = {
    "English": "en",
    "Turkish": "tr",
    "German": "de",
    "French": "fr",
}


def get_device() -> int:
    """Return the preferred Torch device index for pipelines."""
    if torch.cuda.is_available():
        return 0
    return -1


DEVICE = get_device()


@lru_cache()
def load_asr_pipeline():
    generate_kwargs = {"task": "transcribe"}
    if isinstance(DEVICE, int) and DEVICE >= 0:
        return pipeline(
            "automatic-speech-recognition",
            model=ASR_MODEL_NAME,
            device=DEVICE,
            generate_kwargs=generate_kwargs,
        )
    return pipeline(
        "automatic-speech-recognition",
        model=ASR_MODEL_NAME,
        generate_kwargs=generate_kwargs,
    )


@lru_cache()
def load_translation_pipeline():
    if isinstance(DEVICE, int) and DEVICE >= 0:
        return pipeline(
            "translation",
            model=TRANSLATION_MODEL_NAME,
            tokenizer=TRANSLATION_MODEL_NAME,
            device=DEVICE,
        )
    return pipeline(
        "translation",
        model=TRANSLATION_MODEL_NAME,
        tokenizer=TRANSLATION_MODEL_NAME,
    )


@lru_cache()
def load_summarization_pipeline():
    if isinstance(DEVICE, int) and DEVICE >= 0:
        return pipeline(
            "summarization",
            model=SUMMARIZATION_MODEL_NAME,
            device=DEVICE,
        )
    return pipeline("summarization", model=SUMMARIZATION_MODEL_NAME)


def translate_text(
    translator,
    text: str,
    src_lang_code: str,
    tgt_lang_code: str,
) -> str:
    """Translate text with M2M100 while respecting source and target language codes."""
    translator.tokenizer.src_lang = src_lang_code
    forced_bos_token_id = translator.tokenizer.get_lang_id(tgt_lang_code)

    translation_output = translator(
        text,
        forced_bos_token_id=forced_bos_token_id,
        clean_up_tokenization_spaces=True,
        max_length=512,
    )
    return translation_output[0]["translation_text"].strip()


def process_audio(
    audio_file: Tuple[int, np.ndarray],
    source_language_label: str,
    target_language_label: str,
    summary_max_length: int,
) -> Tuple[str, str, str]:
    """Transcribe, translate, and summarize an input audio file."""
    if audio_file is None:
        return "", "", ""

    sample_rate, audio = audio_file
    audio = audio.astype(np.float32)

    asr = load_asr_pipeline()
    transcription_result = asr({"sampling_rate": sample_rate, "array": audio})
    transcription_text = transcription_result["text"].strip()

    translator = load_translation_pipeline()
    src_lang = TURKIC_LANGUAGE_CODES[source_language_label]
    tgt_lang = TARGET_LANG_CODES[target_language_label]

    translated_text = translate_text(translator, transcription_text, src_lang, tgt_lang)

    # Summaries are produced in English for maximum faithfulness because the
    # BART CNN model is trained exclusively on English data. If the user asks
    # for a different translation target we perform an additional English
    # translation solely for summarization.
    english_translation = (
        translated_text
        if tgt_lang == TARGET_LANG_CODES["English"]
        else translate_text(translator, transcription_text, src_lang, TARGET_LANG_CODES["English"])
    )

    summarizer = load_summarization_pipeline()
    summary_output = summarizer(
        english_translation,
        max_length=summary_max_length,
        min_length=max(10, summary_max_length // 3),
        do_sample=False,
    )
    summary_text = summary_output[0]["summary_text"].strip()

    return transcription_text, translated_text, summary_text


def build_interface() -> gr.Blocks:
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # Turkic Audio Summarizer

            Upload an audio clip in a Turkic language. The app will transcribe the speech, translate
            it to your chosen target language, and provide a concise summary of the translation.
            """
        )

        with gr.Row():
            audio_input = gr.Audio(sources=["upload", "microphone"], type="numpy", label="Audio Input")

        with gr.Row():
            source_language = gr.Dropdown(
                label="Source Language",
                choices=list(TURKIC_LANGUAGE_CODES.keys()),
                value="Turkish",
            )
            target_language = gr.Dropdown(
                label="Translation Target",
                choices=list(TARGET_LANG_CODES.keys()),
                value="English",
            )
            summary_length = gr.Slider(
                label="Summary Max Length",
                minimum=30,
                maximum=200,
                value=120,
                step=10,
            )

        submit_button = gr.Button("Transcribe, Translate & Summarize")

        transcription_output = gr.Textbox(label="Transcription", lines=6)
        translation_output = gr.Textbox(label="Translation", lines=6)
        summary_output = gr.Textbox(label="Summary", lines=6)

        submit_button.click(
            fn=process_audio,
            inputs=[audio_input, source_language, target_language, summary_length],
            outputs=[transcription_output, translation_output, summary_output],
        )

    return demo


def main():
    demo = build_interface()
    demo.queue(concurrency_count=2).launch()


if __name__ == "__main__":
    main()
