import os
import numpy as np
import librosa
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.metrics import accuracy_score
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def extract_features(file_path, processor, model, max_length=16000):
    # Load and preprocess the audio
    audio, sr = librosa.load(file_path, sr=16000)
    
    # Ensure consistent length
    if len(audio) > max_length:
        audio = audio[:max_length]
    else:
        audio = np.pad(audio, (0, max_length - len(audio)))
    
    # Process audio with Wav2Vec2
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract the hidden states (you can experiment with different layers)
    features = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return features

def load_audio(root_dir, processor, model):
    audioNlabels = []
    for i, speaker_dir in enumerate(os.listdir(root_dir)):
        speaker_path = os.path.join(root_dir, speaker_dir)
        if os.path.isdir(speaker_path):
            for audio_file in os.listdir(speaker_path):
                file_path = os.path.join(speaker_path, audio_file)
                if file_path.endswith('.wav'):
                    features = extract_features(file_path, processor, model)
                    audioNlabels.append([features, i])
    return audioNlabels

def classify_with_radius_neighbors(featuresNlabels, radius=1.0):
    features = np.array([f[0] for f in featuresNlabels])
    labels = np.array([f[1] for f in featuresNlabels])

    clf = RadiusNeighborsClassifier(radius=radius)
    clf.fit(features, labels)

    predicted_labels = clf.predict(features)
    
    # Print cluster information
    unique_labels = np.unique(predicted_labels)
    for label in unique_labels:
        cluster_points = features[predicted_labels == label]
        print(f"Cluster {label} has {cluster_points.shape[0]} points.")

    # PCA for visualization
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=predicted_labels, cmap='viridis', alpha=0.5)
    plt.colorbar(scatter, label='Cluster Label')
    plt.title('Cluster Visualization')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.savefig('wav2vec2_clustering_results.png')
    plt.show()
    
    # Evaluate accuracy of clustering
    accuracy = accuracy_score(labels, predicted_labels)
    print(f"Clustering Accuracy: {accuracy}")
    
    return predicted_labels, accuracy

if __name__=='__main__':
    # Load Wav2Vec2 model and processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

    featuresNlabels = load_audio(root_dir='audio', processor=processor, model=model)
    predicted_labels, accuracy = classify_with_radius_neighbors(featuresNlabels)
