import os
import librosa
import numpy as np
from sklearn.cluster import KMeans
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Input, Flatten

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



def resnet_autoencoder(input_shape): # ResNet-based autoencoder model
    resnet = ResNet50(include_top=False, input_shape=input_shape) # Encoder: Using a pretrained ResNet50
    resnet.trainable = False  # Freeze ResNet layers
    
    # Add a custom fully connected layer to compress the data
    x = Flatten()(resnet.output)
    encoded = Dense(256, activation='relu')(x)  # Bottleneck layer (latent space)

    decoded = Dense(np.prod(input_shape), activation='sigmoid')(encoded) # Decoder: Fully connected layers to reconstruct the input
    
    # Define autoencoder model
    autoencoder = Model(resnet.input, decoded)
    autoencoder.compile(optimizer=Adam(), loss='mse')
    
    encoder = Model(resnet.input, encoded) #Encoder model to extract the latent space (for clustering)
    
    return autoencoder, encoder



def cluster_mfcc_data(mfcc_features, input_shape):
    # Standardize the MFCC features
    scaler = StandardScaler()
    mfcc_scaled = scaler.fit_transform(mfcc_features)

    # Reshape MFCC features to fit the ResNet model input
    mfcc_reshaped = np.expand_dims(mfcc_scaled, axis=-1)

    # Train the ResNet-based autoencoder
    autoencoder, encoder = resnet_autoencoder(input_shape)
    autoencoder.fit(mfcc_reshaped, mfcc_reshaped, epochs=20, batch_size=32, verbose=1)

    # Extract the latent space representation of the MFCC data
    latent_features = encoder.predict(mfcc_reshaped)

    kmeans = KMeans(n_clusters=5, random_state=42)  
    clusters = kmeans.fit_predict(latent_features)

    return clusters, latent_features


'Related paper: https://ieeexplore.ieee.org/document/9538747'
if __name__ == '__main__':    
    base_dir = "audio"  
    train_features, labels = load_audio_from_dirs(base_dir)
    
    # Define the input shape for ResNet (based on your MFCC data shape)
    input_shape = (train_features.shape[1], 1)
    print(train_features.shape)
    clusters, latent_features = cluster_mfcc_data(train_features, input_shape)

    print(clusters)

'''    TODO - new :
    1. Reshape the inputshape - give gpt the entire code to fit to the previous version it gave
    2. Take gpt previous version and use 3 audio files, and check shape. Then figure out how it should be
'''

'''
What we did:
1. Extracted features
2. Visualized data (files in first commit)
3. Used clustering and more recently radius_neighbors_classifier to classify new drivers (including unkown)
4. Created an evaluation and prediction function using the trained classifier
5. Clean code, merge w/ "first" commit (to print graphs w/ variance) 
6. Save created figure of the dataset after PCA and clustering (first commit) + understand what it means 

TODO - old: 
. Run on collab on all data - Ilan
. Add 1 more clustering algorithms - Deep Embedded Clustering
https://ieeexplore.ieee.org/document/9538747
https://ieeexplore.ieee.org/document/9999360
https://www.nature.com/articles/s41598-024-51699-z
5. fix confusion matrix (right now only plots for known speakers)
6. Generalize visualization of features (datasets) for all people and not just 1 - bonus

'''
    # classifier = 'radius_classifier.pkl'
    # if os.path.exists(classifier):
    #     radius_classifier = joblib.load('radius_classifier.pkl')
    #     scaler = joblib.load('scaler.pkl')  
    # else:
        # ....
        # radius_classifier, scaler = train_radius_neighbors_classifier(train_features, labels, radius=1.0)  # Adjust the radius
        # joblib.dump(radius_classifier, classifier)
        # joblib.dump(scaler, 'scaler.pkl')

    # test_dir = "1-test" 
    # test_features = load_audio_from_test_dir(test_dir)
    # new_person_cluster_labels = classify_new_person(radius_classifier, scaler, test_features)
    # print("New person classified as: ", new_person_cluster_labels)