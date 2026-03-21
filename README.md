# Drum Sample Classifier and Automatic Drum Type Sorter

This code implements a Pytorch-based CNN that classifies drum samples of whatever types you desire from audio files. 
It does so by converting them to mel spectrograms and running them through a 4-block convolutional neural network.
It is pretrained on 12 classes; Bass, Blip, Bongo, Clap, Cymbal, HiHat, Kick, Rim, SFX, Snare, Tom, and Udu; from my own drum sample collection. 
But you can use any combination of sound types you want, see retrain section below.

I made it to learn PyTorch, and a drum classifier was an interesting idea with readily available dataset for me as a hobby musician.

# Quickstart

Double-click quickstart/quickstart.bat. 
It runs the included example.mp3 sample through the trained model and prints the results. 
It will ask whether you want to see the mel spectrogram and CNN feature map plots or not.

# Automatic File Sorting

If the quickstart works well, you can copy paste your unsorted drum files into files_inferencesorting/unsorted_files.
Run InferenceMultiple05.py which automatically sorts these files into the sorted_files/<classname> folder.
Files are copied, so your original files should be save. However, files of same name in sorted_folder will be overwritten.

# Install dependencies

pip install -r requirements.txt
The CUDA versions of torch/torchaudio are pinned in requirements.txt. I used CUDA version 13.0.

# Run inference on your own file

python Inference04.py path/to/your/sample.wav
python Inference04.py path/to/your/sample.wav --enable-plots

Drag and drop a wav/mp3/flac file into the console, it usually autogenerates the path (such as in VS Code on Windows).

# Retrain on your own data

Retraining requires you to have drum samples sorted in folders. 
They can have any name themselves, and only need to be .mp3 .flac or .wav, but they need to be in the correct subfolders you need to create:

In the given directory folder files_drumtrainingdata/ you need to create subfolder per class, each containing wav/mp3/flac samples.
For example, if you have kicks and snares, make two new folders called Kick and Snare. The code will autodetect these folders.
Simply run python ModelTrainer03.py which will replace the old .safetensors.
The best model is saved automatically to files_modeloutputs/ whenever validation accuracy improves.

## File overview

files_drumtrainingdata/ -   place your sample folders in here
files_modeloutputs/     -   Saved model weights and class list 
quickstart/             -   Example sample + launcher bat 
Preprocessor01.py       -   Audio loading, resampling, mel spectrogram conversion, dataset class 
NeuralNetwork02.py      -   CNN architecture 
ModelTrainer03.py       -   Training loop 
Inference04.py          -   Run the model on a single file 
