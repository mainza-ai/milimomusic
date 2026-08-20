---
title: Training Studio
type: entity
created: 2026-08-19
updated: 2026-08-19
sources: [sources/training-studio-guide.md, sources/readme.md]
tags: [training, lora, finetune, studio, heartmula]
aliases: [LoRA Training Studio]
---

# Training Studio

The **Training Studio** (Beta) fine-tunes the [HeartMuLa](heartmula.md) model on custom
audio datasets directly within the app, enabling **custom styles** (e.g. 'Afrobeat',
'MyVoice'). It has a **Glassmorphism** UI (semi-transparent panels, cyan/purple gradients).

## How it works (quick start)
1. Open Training Studio (🎓 button in the Style Manager).
2. Create a **Dataset** (name + target styles).
3. Upload **≥5 audio files** (MP3/WAV/FLAC); matching `.txt` files add lyrics/captions.
4. Configure training: **LoRA** (fast, ~100MB) or **Full Fine-Tune** (best quality, ~6GB).
5. Monitor progress/loss in the **Jobs** tab; **Activate** the checkpoint in the **Models** tab.

## Tabs
- **Dataset**: create/edit/delete datasets; selected cards glow with a cyan ring.
- **Training**: params — Method, Epochs (default 3), Learning Rate (default 0.0001),
  LoRA Rank (8–32, default 8). See [LoRA fine-tuning](../concepts/lora-finetuning.md).
- **Jobs**: real-time metrics; status badges Queued/Running/Completed/Failed; Loss metric
  (lower is better; if flat/rising, lower the learning rate).
- **Models**: activate/delete checkpoints. Once active, all generation uses the custom style.

## API endpoints
- `GET/POST /training/datasets`
- `POST /training/datasets/{id}/audio` (multipart with file + caption)
- `POST /training/jobs` `{dataset_id, method, epochs, learning_rate}`
- `GET /training/checkpoints`, `POST /training/checkpoints/{id}/activate`

## Storage layout
```
backend/data/
├── datasets/{id}/   manifest.json, audio/*.mp3, processed/*.pt
├── jobs/{id}/       manifest.json, logs.txt
└── checkpoints/{id}/ meta.json, adapter_model.safetensors
```

## Notes
- Training is **local** and runs through the backend (`training/*`: `data_prep.py`,
  `lora_trainer.py`, `run_training.py`).
- **Global monitoring**: a floating status widget tracks training anywhere in the app
  (`useTrainingMonitor.ts` on the frontend).
- In v2, this UI is extended into a **Voice Training Studio** for vocal-identity cloning
  via RVC v2 (§3.6, see [roadmap](../roadmap.md)).

## Related pages
- [HeartMuLa](heartmula.md) | [LoRA fine-tuning](../concepts/lora-finetuning.md)
- [Backend & API](backend-api.md) | [Roadmap (v2)](../roadmap.md)
- [Training Studio Guide source](../sources/training-studio-guide.md)
