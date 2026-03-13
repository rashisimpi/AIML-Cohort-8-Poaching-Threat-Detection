Audio Feature Extraction for Sound Classification

Project Overview

This project focuses on extracting meaningful features from audio signals so that the data can be used for machine learning models. Raw audio signals are complex and unstructured, so feature extraction is used to convert them into structured numerical data.

The extracted features help represent important characteristics of sound such as frequency distribution, energy levels, and spectral properties. These features can later be used for training machine learning models for sound classification tasks.

⸻

Dataset

The dataset used in this project contains multiple audio files along with metadata describing each audio sample.

Files used in the project:
	•	metadata.xls – Contains information about each audio file such as filename and class label.
	•	audio_features.xls – Contains the extracted numerical features from each audio sample.
	•	Audio files (.wav) – Raw audio recordings used for processing.

⸻

Methodology

1. Audio Preprocessing

Before extracting features, the audio files are processed to ensure consistency and quality.

The preprocessing steps include:
	•	Loading audio files using the librosa library
	•	Resampling audio signals to a fixed sampling rate
	•	Limiting audio duration
	•	Removing silent portions of the audio
	•	Saving cleaned audio for further processing

⸻

2. Feature Extraction (My Contribution)

My contribution to the project was extracting features from the processed audio signals.

Feature extraction converts raw audio signals into numerical values that capture the important properties of sound.

The following features were extracted:

MFCC (Mel Frequency Cepstral Coefficients)
MFCC features capture the spectral characteristics of audio signals and are widely used in audio and speech recognition.

Spectral Features
	•	Spectral Centroid
	•	Spectral Bandwidth
	•	Spectral Roll-off

These features describe how the frequencies are distributed in the audio signal.

Energy-Based Features
	•	RMS Energy
	•	Zero Crossing Rate

These features measure the intensity and variation of the audio signal.

For each feature, statistical values such as mean and standard deviation were calculated.

The final extracted features were stored in audio_features.xls, which contains 560 features for each audio sample.

⸻

Technologies Used
	•	Python
	•	librosa
	•	numpy
	•	pandas
	•	soundfile
	•	tqdm

⸻

Output

The final output of the project is a structured dataset containing extracted audio features 
| File Name           | Description                           |
|--------------------|---------------------------------------|
| metadata.xls        | Metadata describing audio files       |
| audio_features.xls  | Extracted audio feature dataset       |

Conclusion

This project successfully converts raw audio signals into structured numerical features through feature extraction. The generated dataset can be used for training machine learning models for audio classification and sound recognition tasks.
