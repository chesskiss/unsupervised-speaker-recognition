import librosa.display
import matplotlib.pyplot as plt
import numpy as np

import librosa
import numpy as np

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


# Example usage
if __name__ == "__main__":
    file_path = "audio/1/_98-121658-0000.wav"  # Replace with your .wav file path
    y, sr = librosa.load(file_path, sr=None)
    features = extract_features(file_path)
    plot_features(features, sr)
