# SERES
## Speech Emotion Recognition with Empathetic Video Response Synthesis

An integrated AI framework that recognizes emotions from speech, generates emotionally aware responses, and provides synchronized facial expressions through animated video responses for real-time empathetic human-computer interaction.

## Overview

SERES (Speech Emotion Recognition with Empathetic video response Synthesis) addresses the critical need for emotionally intelligent AI systems that can provide meaningful emotional support. The system combines advanced deep learning models to create an end-to-end solution that achieves 74.56% emotion recognition accuracy while generating contextually appropriate emotional responses with synchronized facial expressions in real-time.

## Architecture

```
Voice Input → Wav2Vec Classifier → Complex Emotion Mapping → Response Generation
                                                                ↓
NDI Live Stream ← Unreal Engine ← NeuroSync ← CoquiTTS Voice Synthesis
                    (MetaHuman)    (Facial Params)  (Text Response)
```

## Key Features

- **Advanced Emotion Recognition**: Uses Wav2Vec 2.0 for self-supervised speech emotion recognition
- **Complex Emotion Mapping**: Maps 8 basic emotions to 26 complex emotional states including depression, anxiety, and melancholy
- **Empathetic Response Generation**: Generates contextually appropriate responses using Meta Llama API
- **Voice Cloning**: Creates natural-sounding emotional speech using CoquiTTS
- **Synchronized Facial Animation**: Real-time facial expression generation using NeuroSync API
- **Live Video Streaming**: Streams animated MetaHuman responses via NDI to tkinter application

## File Structure

```
SERES/
├── meta.py              # Main program - processes audio input and orchestrates pipeline
├── test_wav.py          # Model training and evaluation script
├── uivideo.py           # Tkinter application with NDI plugin integration
├── uibutton.py          # UI button components and controls
├── data_path.csv        # Dataset file with emotion labels and audio file paths
└── README.md           # This file
```

## Technology Stack

### Core Technologies

- **[Wav2Vec 2.0](https://arxiv.org/abs/2006.11477)**: Self-supervised speech representation learning for emotion classification
- **[Meta Llama API](https://ai.meta.com/llama/)**: Large language model for generating empathetic text responses
- **[CoquiTTS](https://github.com/coqui-ai/TTS)**: Open-source text-to-speech framework for voice cloning
- **[NeuroSync API]([https://neurosync.ai/](https://github.com/AnimaVR/NeuroSync_Local_API))**: Transformer-based facial animation generation from audio
- **[Unreal Engine](https://www.unrealengine.com/)**: Real-time 3D animation with MetaHuman models
- **[NDI SDK]([https://www.ndi.tv/](https://ndi.video/for-developers/ndi-sdk/))**: Network Device Interface for live video streaming

### Development Environment

- **Python 3.9+**: Core programming language
- **PyTorch**: Deep learning framework for model training
- **Tkinter**: GUI framework for desktop application
- **NumPy/SciPy**: Scientific computing libraries
- **Librosa**: Audio processing and feature extraction

## Installation

### Prerequisites

1. Python 3.9 or higher
2. NVIDIA GPU with CUDA support (recommended)
3. Unreal Engine 5.0+
4. NDI Runtime

### Dependencies

```bash
pip install torch torchvision torchaudio
pip install transformers
pip install TTS
pip install librosa
pip install numpy scipy
pip install tkinter
pip install requests  # for API calls
```

### Additional Setup

1. **Unreal Engine Setup**:
   - Install Unreal Engine 5.0+
   - Install NDI Plugin for Unreal Engine
   - Import MetaHuman models

2. **API Configuration**:
   - Obtain NeuroSync API key
   - Configure Meta Llama API access
   - Set up environment variables

## Usage

### Training the Model

Run the model training script with your dataset:

```bash
python test_wav.py
```

This script handles:
- Loading RAVDESS, CREMA-D, TESS, and SAVEE datasets
- Training the Wav2Vec classifier
- Evaluating performance across epochs
- Generating confusion matrices and accuracy metrics

### Running the Main Application

Start the complete SERES system:

```bash
python meta.py
```

This launches:
- Audio input processing
- Real-time emotion recognition
- Response generation pipeline
- Unreal Engine animation
- NDI streaming to UI

### Using the UI Application

Launch the tkinter interface:

```bash
python uivideo.py
```

Features:
- Live video stream from Unreal Engine
- Audio input controls
- Real-time emotion display
- Response playback controls

## Performance Results

### Model Performance

- **Overall Accuracy**: 74.56% across multi-corpus validation
- **Training Epochs**: Optimal performance achieved at epoch 4.9
- **Cross-Dataset Validation**: Tested on 12,162 audio files from 4 datasets

### Accuracy by Emotion

Based on confusion matrix analysis:
- **Angry**: 335 correct classifications
- **Sad**: 283 correct classifications  
- **Happy**: High precision in positive emotion detection
- **Complex Emotions**: 78% appropriateness rating in user studies

### Comparative Performance

| Method | Dataset | Accuracy | Limitations |
|--------|---------|----------|-------------|
| CNN-Transformer | RAVDESS | 87.0% | Single dataset only |
| Multimodal Fusion | IEMOCAP | 83.5% | Integration issues |
| **SERES (Ours)** | **Multi-corpus** | **74.56%** | **Cross-validated** |

### User Validation Results

Study with 30 participants (15-minute interactions):
- **Emotional Appropriateness**: 78% rated appropriate/highly appropriate
- **Naturalness**: 72% rated natural/very natural  
- **Helpfulness**: 81% found helpful/very helpful for emotional support

## Complex Emotion Mapping

The system maps 8 basic emotions to 26 complex emotional states:

### Categories
- **Positive**: excited, content, amused, grateful, proud
- **Negative**: frustrated, anxious, melancholic, annoyed, worried
- **Depression-related**: depressed, grief, hopeless, apathetic, heartbroken
- **Mixed States**: nostalgic, conflicted

### Algorithm
```python
def map_complex_emotion(probability_vector):
    # Calculate emotional metrics
    flatness = 1.0 - std(probability_vector)
    negative_load = sum([p_sad, p_fear, p_disgust, p_angry])
    
    # Apply hierarchical rules
    if p_angry > 0.2 and p_sad > 0.15:
        return "frustrated"
    elif p_fear > 0.2 and p_neutral > 0.15:
        return "anxious"
    elif p_sad > 0.25 and flatness > 0.7:
        return "depressed"
    # ... additional mappings
```

## Datasets

The system uses a curated dataset compiled from four comprehensive emotion recognition datasets. Instead of using complete datasets, specific audio samples corresponding to the target 8 emotions (neutral, calm, happy, sad, angry, fear, disgust, surprise) were extracted and compiled into a custom dataset.

### Source Datasets:
1. **RAVDESS**: Ryerson Audio-Visual Database of Emotional Speech and Song
2. **CREMA-D**: Crowdsourced Emotional Multimodal Actors Dataset  
3. **TESS**: Toronto Emotional Speech Set
4. **SAVEE**: Surrey Audio-Visual Expressed Emotion Database

### Custom Dataset Structure:
- **data_path.csv**: Contains emotion labels and corresponding file paths for all selected audio samples
- **Target Emotions**: 8 basic emotions (neutral, calm, happy, sad, angry, fear, disgust, surprise)
- **Selection Criteria**: Only audio files matching the 8 target emotions were extracted from the source datasets
- **Cross-corpus Validation**: Ensures model generalization across different recording conditions and speaker demographics

The `data_path.csv` file structure:
```csv
emotion,file_path
happy,/path/to/audio/happy_sample_001.wav
sad,/path/to/audio/sad_sample_001.wav
angry,/path/to/audio/angry_sample_001.wav
...
```

## API Integration

### NeuroSync API
```python
# Generate facial parameters from audio
response = neurosync_api.generate_facial_params(
    audio_data=audio_input,
    emotion_context=detected_emotion
)
facial_params = response['blendshapes']
```

### CoquiTTS Integration
```python
# Voice cloning with emotion
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
wav = tts.tts(
    text=response_text,
    speaker_wav="reference_voice.wav",
    language="en",
    emotion=detected_emotion
)
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@article{abhimanue2024seres,
  title={Speech Emotion Recognition with Empathetic Video Response Synthesis},
  author={Abhimanue, S and John, Jyothish K},
  journal={IEEE ACCTHPA Conference},
  year={2025},
  institution={Federal Institute of Science and Technology, Kerala, India}
}
```

## Acknowledgments

- [Coqui.ai](https://coqui.ai/) for the open-source TTS framework
- [Meta AI](https://ai.meta.com/) for Wav2Vec and Llama models
- [Epic Games](https://www.epicgames.com/) for Unreal Engine and MetaHuman
- [NeuroSync](https://neurosync.ai/](https://github.com/AnimaVR/NeuroSync_Local_API)) for facial animation API
- Dataset contributors: RAVDESS, CREMA-D, TESS, SAVEE

## Contact

**Abhimanue S**  
Department of Computer Science and Engineering  
Federal Institute of Science and Technology  
Email: abhimanue901@gmail.com

---

**Note**: This system is designed for research and educational purposes in affective computing and emotional AI. For clinical or therapeutic applications, please consult with qualified mental health professionals.
