import sys
import json
import torch
import torchaudio
import soundfile
from   safetensors.torch import load_file

import NeuralNetwork02
from   Preprocessor01 import SAMPLERATE, HOP_LENGTH, NUM_MELBANDS, FFT_SIZE, NOISEFLOORDB

MODEL_PATH   = "files_modeloutputs/drum_classifier.safetensors"
CLASSES_PATH = "files_modeloutputs/drum_classifier_classes.json"

def load_model(make_plots=False):
    """
    Loads the saved model weights and class names from disk and reconstructs the model.
    safetensors stores only the weight tensors, python class names dictionary is loaded separately from json.
    """
    with open(CLASSES_PATH, "r") as f:
        metadata = json.load(f)
    classes     = metadata["classes"]
    clip_length = metadata["clip_length"]
    model = NeuralNetwork02.DrumClassifierCNN(num_classes=len(classes), make_plots=make_plots)
    model.load_state_dict(load_file(MODEL_PATH)) # load weights from safetensors
    model.eval()                                 # disables dropout so predictions are deterministic
    return model, classes, clip_length

def use_model(file_path, model, classes, clip_length):
    """
    Runs a single audio file through the model and returns sorted class probabilities / model prediction.
    The model only works correctly if the input is prepared the same way it was during training.
    """
    # Preprocessing the input sample for sending it through the learned network
    data, sr = soundfile.read(file_path, always_2d=True) # always_2d for consistent (samples, channels) shape
    waveform = torch.tensor(data.T, dtype=torch.float32) # transpose to (channels, samples)
    if waveform.shape[0] > 1: # mix stereo down to mono
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != SAMPLERATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLERATE)
    # pad or trim to the same clip length used during training
    target_len = int(SAMPLERATE * clip_length)
    if waveform.shape[1] < target_len:
        waveform = torch.nn.functional.pad(waveform, (0, target_len - waveform.shape[1]))
    else:
        waveform = waveform[:, :target_len]
    # identical mel spectrogram pipeline to Preprocessor01.py
    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLERATE, n_fft=FFT_SIZE, hop_length=HOP_LENGTH, n_mels=NUM_MELBANDS)
    power_to_db   = torchaudio.transforms.AmplitudeToDB()
    mel = power_to_db(mel_transform(waveform))
    mel = (mel + NOISEFLOORDB) / NOISEFLOORDB
    mel = mel.unsqueeze(0) # add batch dimension: (1, 1, NUM_MELBANDS, time), model expects a batch even for a single file
    with torch.no_grad():
        logits = model(mel)                                             # raw scores: (1, num_classes)
        probs  = torch.nn.functional.softmax(logits, dim=1).squeeze()   # probabilities summing to 1.0: (num_classes,)
    results = sorted(zip(classes, probs.tolist()), key=lambda x: -x[1]) # sort highest probability first
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python Inference04.py path/to/sample.wav [--enable-plots]")
        print("Drag and drop the audio file (mp3/wav/flac) into the console, it usually autogenerates the filepath")
        sys.exit(1)
    file_path  = sys.argv[1]
    make_plots = "--enable-plots" in sys.argv # plots off by default, opt-in with --enable-plots
    model, classes, clip_length = load_model(make_plots)
    results = use_model(file_path, model, classes, clip_length)
    print(f"\nResults for: {file_path}")
    print("-" * 30)
    for class_name, prob in results:
        bar = "█" * int(prob * 30)
        print(f"  {class_name:<12} {prob:>6.1%}  {bar}")
