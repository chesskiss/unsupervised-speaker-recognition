import os
import librosa
import numpy as np



def load_audio(root_dir):
    audioNlabels = []
    for i, speaker_dir in enumerate(os.listdir(root_dir)):
        speaker_path = os.path.join(root_dir, speaker_dir)
        if os.path.isdir(speaker_path):
            for audio_file in os.listdir(speaker_path):
                file_path = os.path.join(speaker_path, audio_file)
                if file_path.endswith('.wav'):
                    audioNlabels.append([extract_mfcc(file_path), i])
    return audioNlabels



def load_audio_from_test_dir(speaker_dir):
    """
    Load audio files from a directory for a new person and extract features.
    """
    features = []
    for audio_file in os.listdir(speaker_dir):
        file_path = os.path.join(speaker_dir, audio_file)
        if file_path.endswith('.wav'):
            mfcc_features = extract_mfcc(file_path)
            features.append(mfcc_features)
    return np.array(features)



def load_audio_from_dirs(base_dir):
    """
    Load audio files from directories and convert speaker labels to integers.
    """
    features = []
    labels = []
    label_mapping = {}  # To keep track of the mapping from speaker folder to label
    label_counter = 1
    
    for speaker_dir in os.listdir(base_dir):
        speaker_path = os.path.join(base_dir, speaker_dir)
        if os.path.isdir(speaker_path):
            if speaker_dir not in label_mapping:
                label_mapping[speaker_dir] = label_counter  # Map speaker folder to an integer
                label_counter += 1
            for audio_file in os.listdir(speaker_path):
                file_path = os.path.join(speaker_path, audio_file)
                if file_path.endswith('.wav'):
                    # Extract MFCC features
                    mfcc_features = extract_mfcc(file_path)
                    features.append(mfcc_features)
                    labels.append(label_mapping[speaker_dir])  # Store the integer label

    return np.array(features), np.array(labels)



def extract_mfcc(file_path, sr=22050, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)