import os
import joblib
import librosa
import hdbscan
import numpy as np
import seaborn as sns
from scipy.io import wavfile
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.signal import spectrogram
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# from pyAudioProcessing.extract_features import get_features #TODO remove if not used


def extract_features(file_path):
    # Load the audio file
    y, sr = librosa.load(file_path, sr=None)
    
    # Extract MFCC features (Mel Frequency Cepstral Coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Extract Chroma features (pitch class energy normalized)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    
    # Extract Spectral Contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    
    # Extract Zero-Crossing Rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    
    # You would need a different library for GFCC, as it's not included in librosa.
    # However, I'll leave it as a placeholder for when you integrate GFCC extraction.
    gfcc = None  # Placeholder for GFCC extraction method

    # Aggregate features into a dictionary
    features = {
        "mfcc": mfcc,
        "chroma": chroma,
        "spectral_contrast": spectral_contrast,
        "zero_crossing_rate": zero_crossing_rate,
        "gfcc": gfcc  # Replace this with actual GFCC extraction
    }

    return features

def extract_mfcc(file_path, sr=22050, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)  # Taking the mean of MFCC across time for each file

def cluster_audio_features(features):
    # Standardize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
    cluster_labels = clusterer.fit_predict(features_scaled)
    
    return cluster_labels

def load_audio_from_dirs(base_dir):
    features = []
    labels = []
    # Iterate over each directory (speaker)
    for speaker_dir in os.listdir(base_dir):
        speaker_path = os.path.join(base_dir, speaker_dir)
        if os.path.isdir(speaker_path):
            for audio_file in os.listdir(speaker_path):
                file_path = os.path.join(speaker_path, audio_file)
                if file_path.endswith('.wav'):
                    # Extract MFCC features
                    mfcc_features = extract_mfcc(file_path)
                    features.append(mfcc_features)
                    labels.append(speaker_dir)  # Store the speaker label for future reference
    return np.array(features), np.array(labels)

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
    plt.savefig('clustered_data.png')
    plt.show()

def plot_features(features, sr, hop_length=512):
    # Calculate the time axis based on the number of frames and hop length
    num_frames = features['mfcc'].shape[1]  # All features have the same number of frames
    duration = (num_frames * hop_length) / sr
    time = np.linspace(0, duration, num=num_frames)

    # Plot MFCC
    plt.figure(figsize=(12, 8))

    plt.subplot(4, 1, 1)
    librosa.display.specshow(features['mfcc'], x_axis='time', sr=sr, hop_length=hop_length)
    plt.colorbar()
    plt.title('MFCC')

    # Plot Chroma
    plt.subplot(4, 1, 2)
    librosa.display.specshow(features['chroma'], x_axis='time', sr=sr, hop_length=hop_length, cmap='coolwarm')
    plt.colorbar()
    plt.title('Chroma')

    # Plot Spectral Contrast
    plt.subplot(4, 1, 3)
    librosa.display.specshow(features['spectral_contrast'], x_axis='time', sr=sr, hop_length=hop_length)
    plt.colorbar()
    plt.title('Spectral Contrast')

    # Plot Zero-Crossing Rate
    plt.subplot(4, 1, 4)
    plt.plot(time, features['zero_crossing_rate'][0], label='Zero Crossing Rate')
    plt.xlabel('Time (s)')
    plt.title('Zero-Crossing Rate')
    plt.tight_layout()

    plt.savefig('features.png')
    plt.show()



# TODO remove ?
def analyze_audio(file_path):
    # Read the WAV file
    sample_rate, audio_data = wavfile.read(file_path)
    
    # Ensure audio_data is mono
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Calculate duration
    duration = len(audio_data) / sample_rate
    time = np.linspace(0., duration, len(audio_data))
    
    # Calculate spectrogram
    frequencies, times, Sxx = spectrogram(audio_data, fs=sample_rate)
    
    # Calculate additional measures
    rms = np.sqrt(np.mean(audio_data**2))
    
    magnitudes = np.abs(np.fft.rfft(audio_data))
    freqs = np.fft.rfftfreq(len(audio_data), 1/sample_rate)
    spectral_centroid = np.sum(magnitudes * freqs) / np.sum(magnitudes)
    spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid)**2) * magnitudes) / np.sum(magnitudes))
    
    return {
        'time': time,
        'audio_data': audio_data,
        'frequencies': frequencies,
        'times': times,
        'Sxx': Sxx,
        'rms': rms,
        'spectral_centroid': spectral_centroid,
        'spectral_bandwidth': spectral_bandwidth
    }
def visualize_multiple_audio(file_paths):
    n_files = len(file_paths)
    fig, axes = plt.subplots(n_files, 2, figsize=(15, 5*n_files))
    
    if n_files == 1:
        axes = axes.reshape(1, -1)
    
    for i, file_path in enumerate(file_paths):
        audio_data = analyze_audio(file_path)
        
        # Plot waveform
        axes[i, 0].plot(audio_data['time'], audio_data['audio_data'])
        axes[i, 0].set_title(f'Waveform - {os.path.basename(file_path)}')
        axes[i, 0].set_xlabel('Time (seconds)')
        axes[i, 0].set_ylabel('Amplitude')
        
        # Plot spectrogram
        spec = axes[i, 1].pcolormesh(audio_data['times'], audio_data['frequencies'], 
                                     10 * np.log10(audio_data['Sxx']), shading='gouraud')
        axes[i, 1].set_title(f'Spectrogram - {os.path.basename(file_path)}')
        axes[i, 1].set_ylabel('Frequency (Hz)')
        axes[i, 1].set_xlabel('Time (seconds)')
        plt.colorbar(spec, ax=axes[i, 1], label='Intensity (dB)')
        
        # Print additional measures
        print(f"File: {os.path.basename(file_path)}")
        print(f"RMS Energy: {audio_data['rms']:.2f}")
        print(f"Spectral Centroid: {audio_data['spectral_centroid']:.2f} Hz")
        print(f"Spectral Bandwidth: {audio_data['spectral_bandwidth']:.2f} Hz")
        print()
    
    plt.tight_layout()
    plt.show()



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
    
    plt.savefig('confusion_matrix.png')
    plt.show()



if __name__ == "__main__":
    'Plot features '
    # file_path = "audio/1/_98-121658-0000.wav"  # Replace with your .wav file path
    # y, sr = librosa.load(file_path, sr=None)
    # features = extract_features(file_path)
    # plot_features(features, sr)

    'Visualize multiple audio files'
    # audio_dir = './audio'
    # files = [
    #     os.path.join(audio_dir, speaker_dir, file)
    #     for speaker_dir in os.listdir(audio_dir)
    #     for file in os.listdir(os.path.join(audio_dir, speaker_dir))
    # ]
    # visualize_multiple_audio(files)
    
    'Visualize clusters'
    base_dir = "audio"  # Replace with your base directory containing speaker dirs
    features, labels = load_audio_from_dirs(base_dir)
    cluster_labels = cluster_audio_features(features)
    plot_clusters(features, labels, cluster_labels)

    'Evaluate classifer if exists'
    classifier = 'radius_classifier.pkl'
    if os.path.exists(classifier):
        radius_classifier = joblib.load('radius_classifier.pkl')
        scaler = joblib.load('scaler.pkl')  
        labels = [int(label) for label in labels]
        evaluate_classifier(radius_classifier, scaler, features, labels)




