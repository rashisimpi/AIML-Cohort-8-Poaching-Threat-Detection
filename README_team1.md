# 🌿 Team 1 — Data Collection & Preprocessing
### Cohort 8 | IIMSTC | SDG Goal 15 : Life on Land
### Project: Poaching Threat Detection using Acoustic Sensors

---

## 👥 Team Members

| Member | Role |
|--------|------|
| **Monish** | Data Collection & Setup |
| **Kanish** | Audio Preprocessing |
| **Nithin** | Labelling & Metadata |

---

## 📌 Overview

Team 1 is responsible for the first stage of the pipeline — collecting, cleaning, standardizing, and labelling the FSC22 audio dataset so it is ready for feature extraction by Team 2.

Raw audio files from FSC22 cannot be directly fed into a machine learning model. Preprocessing cleans, standardizes, and organizes the audio so that every file is in the same format and properly labelled as either a **threat** (illegal activity) or **wildlife** (normal forest background).

---

## 📦 Dataset

| Detail | Value |
|--------|-------|
| **Name** | FSC22 — Forest Sound Classification 2022 |
| **Source** | [Kaggle — irmiot22/fsc22-dataset](https://www.kaggle.com/datasets/irmiot22/fsc22-dataset) |
| **Total Files** | 2025 audio clips |
| **Classes** | 27 forest-specific sound classes |
| **Files per Class** | 75 (perfectly balanced) |
| **Format** | WAV, 44100 Hz, 5 seconds per clip |
| **License** | CC BY-NC-SA 4.0 |

---

## 👤 Member Responsibilities

### Monish — Data Collection & Setup
Monish was responsible for acquiring and understanding the dataset. The FSC22 dataset was downloaded from Kaggle using the `kagglehub` library. Once downloaded, the dataset structure was explored and documented — it contains 2025 audio files across 27 classes with exactly 75 files per class, all stored in a single flat folder with no subfolders. The metadata CSV file was located and inspected to identify the correct column names, specifically `Dataset File Name` and `Class Name`. A corruption check was also run at this stage to confirm that all 2025 files could be successfully loaded before any heavy processing began. Finally, Monish set up the shared project folder structure so that all team members and downstream teams could work from the same organized layout.

**File:** `monish_data_collection.ipynb`

---

### Kanish — Audio Preprocessing
Kanish handled the actual transformation of raw audio into clean, standardized clips. The core preprocessing function loads each audio file as a single mono channel at the FSC22 native sample rate of 44100 Hz. Silence is then trimmed from the beginning and end of each clip using a 30 dB threshold, which removes dead space recorded before or after the actual sound event. After trimming, each clip is fixed to exactly 5 seconds — if a clip is shorter than 5 seconds after trimming it is padded with zeros at the end, and if it is longer it is cut at the 5 second mark. All processed files are saved as `.wav` files into two organized subfolders — `threat/` and `wildlife/`.

**File:** `kanish_audio_preprocessing.ipynb`

---

### Nithin — Labelling & Metadata
Nithin was responsible for the labelling logic and the final metadata output. Since FSC22 has 27 detailed sound classes, the first task was to research and decide which classes represent illegal human activity in a forest and which represent normal background sounds. The 11 classes mapped to threat are Fire, Helicopter, VehicleEngine, Axe, Chainsaw, Generator, Handsaw, Firework, Gunshot, WoodChop and TreeFalling, while the remaining 16 classes are treated as wildlife. A `get_binary_label()` function was implemented using exact FSC22 class names. After all files were processed, a `metadata.csv` file was generated containing the filename, original class, binary label, numeric encoded label, sample rate, duration, and full file path for every one of the 2025 files.

**File:** `nithin_labelling_metadata.ipynb`

---

## 🏷️ Binary Label Mapping

| Label | Class Names | Count |
|-------|-------------|-------|
| **Threat (1)** | Fire, Helicopter, VehicleEngine, Axe, Chainsaw, Generator, Handsaw, Firework, Gunshot, WoodChop, TreeFalling | 825 files |
| **Wildlife (0)** | Rain, Thunderstorm, WaterDrops, Wind, Silence, Whistling, Speaking, Footsteps, Clapping, Insect, Frog, BirdChirping, WingFlaping, Lion, WolfHowl, Squirrel | 1200 files |

---

## ⚙️ Preprocessing Pipeline

```
Raw FSC22 Audio (2025 files, flat folder)
        ↓
Read Metadata CSV → get class names (Monish)
        ↓
Corruption Check → validate all 2025 files (Monish)
        ↓
Binary Label Mapping → 27 classes → threat / wildlife (Nithin)
        ↓
Load Audio → mono, 44100 Hz (Kanish)
        ↓
Trim Silence → remove dead space, top_db=30 (Kanish)
        ↓
Pad / Cut → fix to exactly 5 seconds (Kanish)
        ↓
Save .wav → processed/threat/ and processed/wildlife/ (Kanish)
        ↓
Save metadata.csv → handed to Team 2 (Nithin)
```

---

## 📊 Results

```
Total files processed  : 2025
Corrupted files        : 0
Wildlife (label = 0)   : 1200
Threat   (label = 1)   : 825
```

---

## 📄 Output Files

### `metadata.csv` — Handed to Team 2 (Feature Extraction)

| Column | Example | Description |
|--------|---------|-------------|
| `filename` | `Chainsaw_1_10101.wav` | Processed file name |
| `original_class` | `Chainsaw` | Original FSC22 class label |
| `label` | `threat` | Binary label |
| `label_encoded` | `1` | Numeric (1=threat, 0=wildlife) |
| `sample_rate` | `44100` | Audio sample rate |
| `duration_sec` | `5.0` | Clip duration |
| `processed_path` | `data/processed/threat/...` | Path to processed file |

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install kagglehub librosa soundfile numpy pandas tqdm
```

### 2. Download the dataset
```python
import kagglehub
path = kagglehub.dataset_download("irmiot22/fsc22-dataset")
```

### 3. Run notebooks in order
```
1. monish_data_collection.ipynb       ← first
2. kanish_audio_preprocessing.ipynb   ← second
3. nithin_labelling_metadata.ipynb    ← third
```

### 4. Update paths in each notebook
```python
AUDIO_DIR    = r"your\path\to\Audio Wise V1.0"
METADATA_CSV = r"your\path\to\Metadata V1.0 FSC22.csv"
```

---

## 🛠️ Dependencies

| Library | Purpose |
|---------|---------|
| `librosa` | Audio loading, resampling, silence trimming |
| `soundfile` | Saving processed .wav files |
| `numpy` | Array operations, padding |
| `pandas` | Metadata CSV handling |
| `tqdm` | Progress bars |
| `kagglehub` | Dataset download |

---

## 🔗 Handoff to Team 2

Team 2 (Feature Extraction) reads `metadata.csv` and uses the `processed_path` column to access each clean audio file:

```python
import pandas as pd
df = pd.read_csv("metadata.csv")
# processed_path column has the full path to each clean .wav file
```

---

## 📚 References

- FSC22 Dataset: https://www.kaggle.com/datasets/irmiot22/fsc22-dataset
- Librosa Documentation: https://librosa.org/doc/latest/index.html
- SDG 15 — Life on Land: https://sdgs.un.org/goals/goal15

---

*Cohort 8 | IIMSTC | 2nd Mini Project | Team 1 — Data Collection & Preprocessing*
