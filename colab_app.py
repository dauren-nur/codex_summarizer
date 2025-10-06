"""Google Colab entrypoint for the Turkic audio summarizer app.

Usage inside Colab:

```
!pip install -q gradio transformers accelerate sentencepiece librosa
!python colab_app.py
```

The script preloads the required Hugging Face pipelines and launches a
public Gradio share suitable for notebooks.
"""
from app import (
    build_interface,
    load_asr_pipeline,
    load_summarization_pipeline,
    load_translation_pipeline,
)


def preload_models() -> None:
    """Warm up all pipelines so the first request is fast."""
    load_asr_pipeline()
    load_translation_pipeline()
    load_summarization_pipeline()


if __name__ == "__main__":
    preload_models()
    demo = build_interface()
    demo.queue(concurrency_count=2).launch(share=True)
