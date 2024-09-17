import os
import joblib
import librosa
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

import matplotlib.pyplot as plt

from sklearn.neighbors import RadiusNeighborsClassifier
import os
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler



def load_audio_from_test_dir(speaker_dir):
    """
    Load audio files from a directory for a new person and extract features.
    """
    features = []
    for audio_file in os.listdir(speaker_dir):
        file_path = os.path.join(speaker_dir, audio_file)
        if file_path.endswith('.wav'):
            # Extract MFCC features
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



def train_radius_neighbors_classifier(features, labels, radius=1.0):
    """
    Train RadiusNeighborsClassifier with integer outlier labels.
    """
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Train Radius Neighbors Classifier with integer outlier_label=-1
    radius_classifier = RadiusNeighborsClassifier(radius=radius, outlier_label=-1)
    radius_classifier.fit(features_scaled, labels)
    print(np.unique(labels))

    return radius_classifier, scaler



def classify_new_person(radius_classifier, scaler, new_features):
    # Standardize new features using the same scaler
    new_features_scaled = scaler.transform(new_features)

    # Predict cluster labels for new features using the RadiusNeighborsClassifier
    cluster_labels = radius_classifier.predict(new_features_scaled)
    
    counts = Counter(cluster_labels)

    majority_vote = counts.most_common(1)[0][0]

    return majority_vote if majority_vote != -1 else 'Unknown'



# Example usage
if __name__ == "__main__":

    classifier = 'radius_classifier.pkl'
    if os.path.exists(classifier):
        radius_classifier = joblib.load('radius_classifier.pkl')
        scaler = joblib.load('scaler.pkl')  
    else:
        base_dir = "audio"  
        train_features, labels = load_audio_from_dirs(base_dir)
        radius_classifier, scaler = train_radius_neighbors_classifier(train_features, labels, radius=1.0)  # Adjust the radius
        joblib.dump(radius_classifier, classifier)
        joblib.dump(scaler, 'scaler.pkl')

    test_dir = "1-test" 
    test_features = load_audio_from_test_dir(test_dir)
    new_person_cluster_labels = classify_new_person(radius_classifier, scaler, test_features)
    print("New person classified as: ", new_person_cluster_labels)