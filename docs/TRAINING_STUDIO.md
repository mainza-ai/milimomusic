# Training Studio Guide

The Training Studio allows you to fine-tune HeartMuLa music models with your own audio samples, creating custom styles that generate music reflecting your unique sound.

> [!NOTE]
> The studio has been updated with a new **Glassmorphism** interface. It uses semi-transparent panels and vibrant cyan/purple gradients. If you see old white/solid styles, please refresh your browser.

## Quick Start

1. **Open Training Studio**: Click the "Training Studio" button (🎓 icon) in the Style Manager.
2. **Create a Dataset**: Name your collection and add target styles (e.g., "Afrobeat").
3. **Upload Audio**: Drag & drop audio files. Minimum **5 files** required.
4. **Configure Training**: 
   - **LoRA** (Lightning bolt): Fast, lightweight.
   - **Full Fine-Tune** (Fire): Best quality, heavy resource usage.
5. **Start Training**: Monitor progress bars and Loss metrics in the **Jobs** tab.
6. **Activate Model**: Go to the **Models** tab and click "Activate" on your new checkpoint.

---

## Tabs Overview

### 📦 Dataset Tab

**Create a Dataset**
- Enter a descriptive name (e.g., "Afrobeat Collection")
- Specify target styles as comma-separated values (e.g., "Afrobeat, Highlife")
- Click "Create Dataset"

**Upload Audio Files**
- Minimum **5 audio files** required
- Supported formats: MP3, WAV, FLAC
- **Lyrics/Captions**: Upload matching `.txt` files with the same filename to automatically add lyrics/captions.
- Drag & drop files or click the upload zone.

**Edit/Delete Datasets**
- Hover over any dataset card to reveal edit (✏️) and delete (🗑️) buttons.
- **Glass UI**: Selected datasets will glow with a cyan ring.

---

### ⚙️ Training Tab

Select a dataset first, then configure training parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| **Method** | **LoRA** (fast, ~100MB) or **Full** (best quality, ~6GB) | LoRA |
| **Epochs** | Number of training passes. | 3 |
| **Learning Rate** | Step size. Lower = stable, Higher = fast. | 0.0001 |
| **LoRA Rank** | Complexity of adapter (8-32). Higher = more expressive. | 8 |

---

### 📊 Jobs Tab

Monitor your training jobs with real-time metrics:

- **Status Badges**: `Queued` (Grey), `Running` (Blue), `Completed` (Green), `Failed` (Red).
- **Progress Bar**: Shows exact percentage based on training steps.
- **Metrics**: 
  - **Epoch**: Current pass through the data.
  - **Loss**: The error rate (e.g., `Loss: 0.4512`). **Lower is better**. Watch this value decrease to confirm learning.

> [!TIP]
> If Loss stays constant or increases, try lowering the **Learning Rate**.

---

### 📦 Models Tab

Manage your trained checkpoints:

1. **Activate**: Click the "Activate" button. This loads your custom weights into the generation engine.
2. **Usage**: Once active, all generation requests will use your custom style.
3. **Management**: Delete old checkpoints to save disk space.

---

## API Endpoints

For programmatic access:

```bash
# List datasets
GET /training/datasets

# Create dataset
POST /training/datasets
{"name": "My Dataset", "styles": ["Pop", "Electronic"]}

# Upload audio
POST /training/datasets/{id}/audio
(multipart/form-data with file and caption)

# Start training job
POST /training/jobs
{"dataset_id": "...", "method": "lora", "epochs": 3, "learning_rate": 0.0001}

# List checkpoints
GET /training/checkpoints

# Activate checkpoint
POST /training/checkpoints/{id}/activate
```

---

## File Locations

Training data is stored in:
```
backend/data/
├── datasets/{id}/
│   ├── manifest.json
│   ├── audio/*.mp3
│   └── processed/*.pt
├── jobs/{id}/
│   ├── manifest.json
│   └── logs.txt
└── checkpoints/{id}/
    ├── meta.json
    └── adapter_model.safetensors
```
