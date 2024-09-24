import os
import joblib
import librosa
import kmedoids
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform




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



def compute_dissimilarity_matrix(features, metric='cosine'):
    """
    Compute dissimilarity matrix using a specified distance metric (e.g., cosine, euclidean).
    """
    return squareform(pdist(features, metric))



def train_dynmsc(features, min_k=2, max_k=5, max_iter=100, metric='cosine'):
    """
    Train using DynMSC (Dynamic Medoid Silhouette Clustering) from kmedoids with automatic cluster number selection.
    """
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Compute the dissimilarity matrix
    dist_matrix = compute_dissimilarity_matrix(features_scaled, metric=metric)
    
    # Perform DynMSC clustering
    result = kmedoids.dynmsc(dist_matrix, medoids=max_k, minimum_k=min_k, max_iter=max_iter, init='random')
    
    return result, scaler



def classify_new_person_dynmsc(result, scaler, new_features, all_features, metric='cosine'):
    """
    Classify a new person based on the closest medoid from the DynMSC result.
    """
    # Standardize new features using the same scaler
    new_features_scaled = scaler.transform(new_features)
    
    # Extract the feature vectors of the medoids using their indices from the result
    medoid_feature_vectors = all_features[result.medoids]
    
    # Compute dissimilarity of new features to the medoid feature vectors
    combined_features = np.vstack([medoid_feature_vectors, new_features_scaled])
    dist_matrix = compute_dissimilarity_matrix(combined_features, metric=metric)
    
    # Get the dissimilarity matrix for the new features to medoids (rows 1 onward, columns only for medoids)
    cluster_labels = np.argmin(dist_matrix[len(medoid_feature_vectors):, :len(medoid_feature_vectors)], axis=1)
    
    counts = Counter(cluster_labels)
    majority_vote = counts.most_common(1)[0][0]
    
    return majority_vote if majority_vote != -1 else 'Unknown'



def evaluate_clustering_accuracy(result, scaler, features, labels, metric='cosine'):
    """
    Evaluate clustering accuracy by comparing predicted labels with true labels.
    """
    # Standardize the features using the same scaler
    features_scaled = scaler.transform(features)
    
    # Extract the feature vectors of the medoids using their indices from the result
    medoid_feature_vectors = features[result.medoids]
    
    # Compute dissimilarity of features to medoids
    combined_features = np.vstack([medoid_feature_vectors, features_scaled])
    dist_matrix = compute_dissimilarity_matrix(combined_features, metric=metric)
    
    # Get the dissimilarity matrix for the features to medoids
    predicted_labels = np.argmin(dist_matrix[len(medoid_feature_vectors):, :len(medoid_feature_vectors)], axis=1)
    
    # Calculate and return the clustering accuracy
    accuracy = accuracy_score(labels, predicted_labels)
    print(f"Clustering Accuracy: {accuracy}")
    
    return accuracy



if __name__ == "__main__":
    base_dir = "audio"
    train_features, labels = load_audio_from_dirs(base_dir) #TODO - why do we need it in classify_new_person_dynmsc ? Seems redungant. Put in else only

    classifier = 'dynmsc_classifier.pkl'
    if os.path.exists(classifier):
        result = joblib.load('dynmsc_classifier.pkl')
        scaler = joblib.load('dynmsc_scaler.pkl')
    else:
        result, scaler = train_dynmsc(train_features, min_k=2, max_k=5, max_iter=100, metric='cosine')
        joblib.dump(result, classifier)
        joblib.dump(scaler, 'dynmsc_scaler.pkl')

    test_dir = "1-test"
    test_features = load_audio_from_test_dir(test_dir)
    new_person_cluster_labels = classify_new_person_dynmsc(result, scaler, test_features, train_features, metric='cosine')
    accuracy = evaluate_clustering_accuracy(result, scaler, train_features, labels, metric='cosine') #TODO Accuracy is 0 - study code, debug.
    print("New person classified as: ", new_person_cluster_labels)



'Based on https://python-kmedoids.readthedocs.io/en/latest/#dynmsc'

'''    TODO - new : Look above
    . Clean code (save model, etc.) 
    . Run on collab on all data 
    . Try other pre-trained models besides EfficientNet V

TODO - old: 
- fix confusion matrix (right now only plots for known speakers)
- Generalize visualization of features (datasets) for all people and not just 1 - bonus

'''

'''
What we did:
1. Extracted features
2. Visualized data (files in first commit)
3. Used clustering and more recently radius_neighbors_classifier to classify new drivers (including unkown)
4. Created an evaluation and prediction function using the trained classifier
5. Clean code, merge w/ "first" commit (to print graphs w/ variance) 
6. Save created figure of the dataset after PCA and clustering (first commit) + understand what it means 
7. Used Deep Embedded Clustering with EfficientNet pre-trained model as an auto-encoder and RadiusNeighborsClassifier
based on 'Related paper: https://ieeexplore.ieee.org/document/9538747'
8. Performed initial visualization

'''