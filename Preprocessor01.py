# This Preprocessor handles the sound-to-spectrogram image conversion.

import os         # Loading and Saving Files
import numpy      # quick math operations
import torchaudio # provides spectrogram tools
import soundfile  # Handles soundfile writing as alternative to torchcodec
import torch      # Provides torch.nn neural networks

# GLOBAL VALUES
SAMPLERATE   = 22050 # Half the Standard CD quality audio rate which all files are resampled to (only the audible part)
NUM_MELBANDS = 64    # Height of the spectrogram image (number of frequency bands)
FFT_SIZE     = 1024  # FFT window size. The larger, the better frequency but wworse time resolution.
HOP_LENGTH   = 512   # How many samples to advance between FFT windows. Smaller = more time steps = wider image.
NOISEFLOORDB = 80    # 0 dB is full volume, noisefloor this value.
USE_AUGMENT  = False # Cache spectrograms for all epochs if false, disturb them slightly each time if true (false is sufficient & 10x faster)

# Data Set Builder Class
class BuildDrumDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset Builder Class. Loads the data set, prepares the audio files, makes mel spectrograms out of them, primes torch tensor logic for the CNN.
    """
    # __init__, __len__ and __getitem__ are python-specific functions that must be named this way
    def __init__(self, folderpath, augment=False, verbose=False):
        # Basic File Handling ------------------------------------------------------------------------------------------
        self.augment                          = augment and USE_AUGMENT # whether to apply random augmentations (True for training only) to avoid overfitting
        self._mel_cache = {} if not USE_AUGMENT else None
        self.categoryclasses = sorted([f for f in os.listdir(folderpath) if os.path.isdir(os.path.join(folderpath, f))]) # Training Classes = folder names inside the sample_directory
        self.samplefilepathandcategoryindices = self.getfilespaths(folderpath) # holds (filepath, drumcategory_index) for every audio file, e.g. [("kick_02.wav", 0), ("snare_01.wav", 1)]

        durations = []
        for filepath, _ in self.samplefilepathandcategoryindices:
            info = soundfile.info(filepath)
            durations.append(info.frames / info.samplerate)
        durations = numpy.array(durations)

        self.clip_length = durations.mean() + durations.std() # The CNN expects all inputs to be the same size. Calculate pragmatic clip length as mean + 1 std of all sample durations
        if verbose:
            print(f"Clip length set to {self.clip_length}s (mean + 1 standard deviation of {len(durations)} files)")
        # Count how many samples each class has. The training logic uses this to compensate for imbalanced classes.
        self.class_counts = [0] * len(self.categoryclasses) # initialize a vector of zeroes of lenght "classes", i.e. "number of categories"
        for filepath, categorylabel in self.samplefilepathandcategoryindices:
            self.class_counts[categorylabel] += 1 # label equals position in the counts vector here
        # --------------------------------------------------------------------------------------------------------------
        # Initialize MelSpectrogram in __init__ to avoid opening it thousands of times in __getitem__ later
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLERATE, n_fft=FFT_SIZE, hop_length=HOP_LENGTH, n_mels=NUM_MELBANDS)
        self.power_to_db   = torchaudio.transforms.AmplitudeToDB() # regularizes the volume
        if verbose: 
            print(f"Found {len(self.samplefilepathandcategoryindices)} samples across {len(self.categoryclasses)} classes:")
            for catname, count in zip(self.categoryclasses, self.class_counts):
                print(f"  {catname:<15} {count} samples") #format specification :, left align <, field with 15 characters

    def __len__(self):
        return len(self.samplefilepathandcategoryindices) # important getter for the size of the dataset

    def __getitem__(self, idx):
        """
        Called by DataLoader in ModelTrainer03.py for each sample. Returns (2d mel_spectrogram image tensor, class_label).
        """
        filepath, categorylabel = self.samplefilepathandcategoryindices[idx]
        # Cache only reachable when USE_AUGMENT is False
        if self._mel_cache is not None and idx in self._mel_cache:
            return self._mel_cache[idx], categorylabel
        samplearraydata, filesamplerate = soundfile.read(filepath, always_2d=True) # the waveform data. 2d for (#samples, #channels) format so .T transpose is always valid
        waveform = torch.tensor(samplearraydata.T, dtype=torch.float32) # Torch uses it transposed. waveform = [[leftchanneldatavector], [rightchanneldata]], shape = (2, samplerate)
        # PREPROCESS, AUDIO TO MEL --------------------------------------------------------------------
        waveform = self.preprocess(waveform, filesamplerate)
        if self.augment: # Apply random augmentations during training (not inference!) to help generalisation.
            waveform = self.perturb_samples(waveform) 
        # Convert waveform to mel spectrogram to decibel scale. The Resulting tensor structure is (1, N_MELS, time_steps)
        mel = self.power_to_db(self.mel_transform(waveform))
        # Normalize to roughly the range [0, 1] which the CNN prefers.
        mel = (mel + NOISEFLOORDB) / NOISEFLOORDB
        # mel now is of dimension (1 channel, x time steps, y mel bands)
        if self._mel_cache is not None: # Cache if global augment is false
            self._mel_cache[idx] = mel
        return mel, categorylabel
    
    # Helper functions ahead
    def getfilespaths(self, folderpath):
        samples = []
        for drumcategory_index, drumcategoryname in enumerate(self.categoryclasses): # enumerate returns index, name pairs in that order
            category_directory = os.path.join(folderpath, drumcategoryname)
            if not os.path.isdir(category_directory): # skip any stray files at the top level
                continue  
            for filename in os.listdir(category_directory): # recognize the standard audio file formats
                if filename.lower().endswith(('.wav', '.mp3', '.flac')):
                    filepath = os.path.join(category_directory, filename)
                    samples.append((filepath, drumcategory_index))
        return samples
    
    def preprocess(self, waveform, filesamplerate):
        if waveform.shape[0] > 1: # Mix down to mono by averaging the two channels.
            waveform = waveform.mean(dim=0, keepdim=True) # waveform is a torch object, mean a function from torch. keepdim leaves it [[data]], i.e. (1, samplerate), not [data] 
        if filesamplerate != SAMPLERATE:
            waveform = torchaudio.functional.resample(waveform, filesamplerate, SAMPLERATE)
        # Short samples get zero-padded at the end, long ones get trimmed. 
        target_len = int(SAMPLERATE * self.clip_length)
        if waveform.shape[1] < target_len:
            pad = target_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        else: # trim
            waveform = waveform[:, :target_len]
        return waveform

    def perturb_samples(self, waveform):
        """
        Random transforms applied to some audio samples during training to artificially expand the dataset and make the model more robust.
        waveform is a torch tensor.
        """
        # torchtensor to numpy: w = waveform.numpy()
        # numpy to torchtensor: return torch.tensor(w, dtype=torch.float32)
        if numpy.random.rand() < 0.5: # Random volume change.Makes the model learn to recognize sounds at any recording level.
            waveform *= torch.FloatTensor(1).uniform_(0.5, 1.5)
        if torch.rand(1) < 0.5: # Add a tiny amount of white noise.
            waveform += torch.randn_like(waveform) * 0.005
        if torch.rand(1) < 0.4: # Randomly shift the audio slightly in time (by up to 50ms).
            shift = int(torch.randint(0, int(SAMPLERATE * 0.05), (1,)))
            waveform = torch.roll(waveform, shift, dims=1)
        return waveform