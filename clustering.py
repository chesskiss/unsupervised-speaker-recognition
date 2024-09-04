import os
import librosa
import hdbscan
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram
# from pyAudioProcessing.extract_features import get_features #TODO remove if not used


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



'''
What we did:
1. Extracted features
2. Visualized data (files in first commit)
3. Used clustering and more recently radius_neighbors_classifier to classify new drivers (including unkown)
4. Created an evaluation and prediction function using the trained classifier
TODO :
1. Clean code, merge w/ "first" commit (to print graphs w/ variance)
2. Save created figure of the dataset after PCA and clustering (first commit) + understand what it means
3. Run on collab on all data
4. Add 2 more clustering algorithms
5. Generalize visualization of features (datasets) for all people and not just 1 - bonus

'''


def extract_mfcc(file_path, sr=22050, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)  # Taking the mean of MFCC across time for each file



def plot_clusters(features, labels, cluster_labels):
    # Perform dimensionality reduction for visualization (optional)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)
    
    plt.figure(figsize=(8, 6))
    unique_clusters = np.unique(cluster_labels)
    
    for cluster in unique_clusters:
        if cluster == -1:
            color = 'black'  # Noise points
        else:
            color = plt.cm.Spectral(float(cluster) / len(unique_clusters))
        plt.scatter(reduced_features[cluster_labels == cluster, 0],
                    reduced_features[cluster_labels == cluster, 1],
                    label=f'Cluster {cluster}', c=[color])
    
    plt.legend()
    plt.title('Clustering Results (PCA Reduced)')
    plt.show()


def load_audio_from_dir(speaker_dir):
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


from sklearn.neighbors import RadiusNeighborsClassifier
import os
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler

def extract_mfcc(file_path, sr=22050, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)



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
    
    counts = np.bincount(np.array(cluster_labels)) # Count occurrences of each value

    # Determine the majority vote
    majority_vote = np.argmax(counts)

    return majority_vote



def evaluate_classifier(radius_classifier, scaler, features, labels):
    """
    Evaluate the classifier on a test dataset of known speakers.
    """
    # Standardize the features using the same scaler
    features_scaled = scaler.transform(features)

    # Predict the labels
    predicted_labels = radius_classifier.predict(features_scaled)

    # Calculate evaluation metrics
    accuracy = accuracy_score(labels, predicted_labels)
    precision = precision_score(labels, predicted_labels, average='weighted', zero_division=1)
    recall = recall_score(labels, predicted_labels, average='weighted', zero_division=1)
    f1 = f1_score(labels, predicted_labels, average='weighted')

    # Confusion matrix
    conf_matrix = confusion_matrix(labels, predicted_labels)

    # Print the results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()



# Example usage
if __name__ == "__main__":
    # Step 1: Train on known speakers
    base_dir = "audio"  # Replace with your base directory containing speaker dirs
    features, labels = load_audio_from_dirs(base_dir)
    radius_classifier, scaler = train_radius_neighbors_classifier(features, labels, radius=1.0)  # Adjust the radius

    # Step 2: Load and classify new audio
    new_person_dir = "0"  # Directory containing the new person's audio files
    new_person_features = load_audio_from_dir(new_person_dir)

    # Evaluate on the same data (you can split into train/test sets for better evaluation)
    evaluate_classifier(radius_classifier, scaler, features, labels)

    new_person_cluster_labels = classify_new_person(radius_classifier, scaler, new_person_features)
    print("New person classified as: ", new_person_cluster_labels)