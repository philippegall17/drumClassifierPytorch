import os
import time

import torch
from   safetensors.torch import save_file
import json

import Preprocessor01
import NeuralNetwork02

torch.backends.cudnn.benchmark = True # let CUDA check multiple convolution algorithms on 1st batch, use fastest one for this problem afterwards

DATADIRECTORY = "files_drumtrainingdata" # a folder relative to where this file is placed
SAVE_PATH     = "files_modeloutputs/drum_classifier.safetensors"
CLASSES_PATH  = "files_modeloutputs/drum_classifier_classes.json" # safetensors only stores tensors, so classes are saved separately

BATCH_SIZE    = 100   # how many samples to process at once before updating weights (average gradient)                     
EPOCHS        = 70    # how many full passes over the training data.
LEARNING_RATE = 1e-3  # how big each parameter update step is initially.
VAL_SPLIT     = 0.2   # 20% of data held back for validation
MAKE_PLOTS    = False # Plot mel spectrogram + all CNN feature maps for the first training sample. Adds ~30s before epoch 1.

def make_weighted_sampler(dataset, indices):
    """
    Builds a WeightedRandomSampler so that every class appears equally often
    in training batches, regardless of how many samples it actually has.
    """
    class_weights  = [1.0 / c for c in dataset.class_counts]
    sample_weights = [class_weights[dataset.samplefilepathandcategoryindices[i][1]] for i in indices]
    sample_weights = torch.tensor(sample_weights, dtype=torch.float)
    return torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

def train():
    os.makedirs("files_modeloutputs", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}\n")

    # Load the full dataset once to get class names and amount of samples. augment=False/True/False for load/train/val separately.
    full_dataset = Preprocessor01.BuildDrumDataset(DATADIRECTORY, augment=False, verbose=True)
    num_classes  = len(full_dataset.categoryclasses) # categoryclasses is a class member defined in BuildDrumDataset

    # Split samples and their indices into train and validation sets.
    amount_validation_samples = int(len(full_dataset) * VAL_SPLIT)
    amount_training_samples   = len(full_dataset) - amount_validation_samples
    train_indices, val_indices = torch.utils.data.random_split(range(len(full_dataset)), [amount_training_samples, amount_validation_samples])

    # Get the dataset from Preprocessing and build train / validate subsets
    train_dataset = Preprocessor01.BuildDrumDataset(DATADIRECTORY, augment=True)  # random noise/gain/shift for training
    val_dataset   = Preprocessor01.BuildDrumDataset(DATADIRECTORY, augment=False) # clean reference
    train_set = torch.utils.data.Subset(train_dataset, list(train_indices))       # actually only use these subsets (Dataset, Indices)
    val_set   = torch.utils.data.Subset(val_dataset,   list(val_indices))
    
    # From the Pytorch Documentation
    # Dataset stores the samples and their corresponding labels, and DataLoader wraps an iterable around the Dataset to enable easy access to the samples. 
    workers = 0 if not Preprocessor01.USE_AUGMENT else 4
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE,
                                               sampler=make_weighted_sampler(train_dataset, list(train_indices)), # equal weight folders depending on amount of samples in them
                                               num_workers=workers, pin_memory=True) # num_workers loads batches in parallel on CPU so GPU never waits for data
    valid_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE,
                                               num_workers=workers, pin_memory=True)

    # Rare classes get a higher loss weight so mistakes on them cost more during training.
    # Formula: total_samples / (num_classes * class_count) so a class with x samples gets n times higher weight than one with n*x samples.
    # Rule: sampler balances how often classes appear, weights balance how much mistakes cost per class.
    class_weights = torch.tensor([sum(full_dataset.class_counts) / (num_classes * amt) for amt in full_dataset.class_counts], dtype=torch.float).to(device)

    # The 4 standard PyTorch building blocks
    model     = NeuralNetwork02.DrumClassifierCNN(num_classes=num_classes, make_plots=MAKE_PLOTS).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)                   # multiclass loss function using the class_weights
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)            # adapts the learning rate per parameter automatically
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3) # validation loss not improving for 3 epochs -> reduce learning rate

    current_best_validation_accuracy = 0.0
    training_start_time = time.time()
    for epoch in range(EPOCHS): # epoch = full pass over training data
        epoch_start_time = time.time()
        # Training
        model.train() # switch the DrumClassifierCNN model into training mode (or eval mode later), meaning random dropout of 40% of neurons, using BatchNorm
        train_loss, train_correct = 0.0, 0
        for batch_spectrograms, batch_labels in train_loader:
            # Refresh
            batch_spectrograms, batch_labels = batch_spectrograms.to(device), batch_labels.to(device) # batch_spectrograms: (BATCH_SIZE, channel dimension (=1), number of melbands, time)
            optimizer.zero_grad()                                                                     # zero out gradients from the previous batch which accumulate by default in PyTorch
            # Forward Pass
            outputs = model(batch_spectrograms)        # outputs: (batch_size, num_classes), one score per class per sample
            # Example how outputs and predictions looks like in the logits matrix:
            #          Kick   Snare  Hihat  Cowbell ...
            # sample1 [ 2.1,  -0.3,   0.8,   0.1  ] -> predicted_class(sample1) = 0 (Kick  Column Index is the largest entry)
            # sample2 [ 0.1,   1.9,   0.2,   0.5  ] -> predicted_class(sample2) = 1 (Snare Column Index)
            # sample3 [-0.2,   1.7,   0.8,   0.3  ] -> predicted_class(sample3) = 1 (also Snare Index)
            loss    = criterion(outputs, batch_labels) # loss: scalar, average cross entropy loss over the batchsize amount of samples in this batch
            # Backpropagation
            loss.backward()  # computes dL/dw for every weight via automatic differentiation
            optimizer.step() # nudges every weight in the direction that reduces loss
            # Accumulate results
            train_loss          += loss.item() * len(batch_labels)     # loss.item() converts tensor(lossvalue) into lossvalue, multiply by 100 to get batch total
            predicted_classes    = outputs.argmax(1)                   # predicted_classes: (batchsize,), index of highest score per sample
            correct_predictions  = (predicted_classes == batch_labels) # correct_predictions: (batchsize,) vector, True/False per sample
            train_correct       += correct_predictions.sum().item()    # scalar, running total of correct predictions across all batches this epoch
            
        avg_train_loss   = train_loss    / amount_training_samples  # divide total loss sum by number of samples to get average loss per sample
        trainingaccuracy = train_correct / amount_training_samples  # fraction of correctly classified samples, e.g. 0.87 = 87% accuracy

        # Validation to check if we are actually improving or not
        model.eval() # switch to evaluation mode, no gradient calculations needed, disables dropout so predictions are deterministic
        val_loss, val_correct = 0.0, 0
        with torch.no_grad(): # skip gradient computation for validation — not needed and saves memory
            for batch_spectrograms, batch_labels in valid_loader:
                batch_spectrograms, batch_labels = batch_spectrograms.to(device), batch_labels.to(device)
                outputs      = model(batch_spectrograms)
                loss         = criterion(outputs, batch_labels)
                val_loss    += loss.item() * len(batch_labels)
                val_correct += (outputs.argmax(1) == batch_labels).sum().item()
        avg_val_loss = val_loss    / amount_validation_samples
        validationaccuracy = val_correct / amount_validation_samples
        scheduler.step(avg_val_loss) # feed validation loss to the scheduler so it can decide to lower the LR

        epoch_duration = time.time() - epoch_start_time
        print(
            f"Epoch {epoch+1:02d}/{EPOCHS} | "
            f"Train loss: {avg_train_loss:.4f} accuracy: {trainingaccuracy:.2%} | "
            f"Val loss: {avg_val_loss:.4f} accuracy: {validationaccuracy:.2%} | "
            f"Time: {epoch_duration:.0f}s"
        )

        # Model saving
        if validationaccuracy > current_best_validation_accuracy: # Save the model only when validation accuracy improves.
            current_best_validation_accuracy = validationaccuracy
            save_file(model.state_dict(), SAVE_PATH) # safetensors saving
            with open(CLASSES_PATH, "w") as filepath:
                json.dump({
                    "classes":     full_dataset.categoryclasses,  # classes saved separately since safetensors cannot store arbitrary python objects
                    "clip_length": full_dataset.clip_length       # saved so inference doesn't need it passed in manually
                }, filepath)
            print(f"    Saved new best model (validation accuracy = {validationaccuracy:.2%})")

    total_duration = time.time() - training_start_time
    print(f"\nDone! Best validation accuracy: {current_best_validation_accuracy:.2%}")
    print(f"Total training time: {total_duration//60:.0f}m {total_duration%60:.0f}s")
    print(f"Model saved to: {SAVE_PATH}")
    print(f"Classes saved to: {CLASSES_PATH}")

if __name__ == "__main__": # such that no other python file that loads ModelTrainer03 runs train always
    train()
