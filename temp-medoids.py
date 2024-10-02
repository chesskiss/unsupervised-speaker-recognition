import os
import joblib
import kmedoids
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from graph import graph, data

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from scipy.linalg import eig
from scipy.sparse.csgraph import connected_components
import seaborn as sns

def train_dynmsc(features, min_k=2, max_k=5, max_iter=100, metric='cosine'):
    """
    Train using DynMSC (Dynamic Medoid Silhouette Clustering) from kmedoids with automatic cluster number selection.
    """
    
    # Compute dissimilarity matrix using a specified distance metric (e.g., cosine, euclidean).
    dist_matrix = squareform(pdist(features, metric))

    # Perform DynMSC clustering
    result = kmedoids.dynmsc(dist_matrix, medoids=max_k, minimum_k=min_k, max_iter=max_iter, init='random')
    
    return result



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
    dist_matrix = squareform(pdist(combined_features, metric))
    
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
    dist_matrix = squareform(pdist(combined_features, metric))
    
    # Get the dissimilarity matrix for the features to medoids
    predicted_labels = np.argmin(dist_matrix[len(medoid_feature_vectors):, :len(medoid_feature_vectors)], axis=1)
    print(predicted_labels)
    # Calculate and return the clustering accuracy
    accuracy = accuracy_score(labels, predicted_labels)
    print(f"Clustering Accuracy: {accuracy}")
    
    return accuracy




def dynmsc(X, n_clusters, n_neighbors=10):
    # Compute pairwise distances
    distances = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    
    # Compute similarity matrix
    sigma = np.mean(np.sort(distances, axis=1)[:, 1:n_neighbors+1])
    S = np.exp(-distances**2 / (2 * sigma**2))
    
    # Compute graph Laplacian
    D = np.diag(np.sum(S, axis=1))
    L = D - S
    
    # Compute eigenvectors of L
    eigenvalues, eigenvectors = eig(L)
    idx = np.argsort(eigenvalues.real)
    eigenvectors = eigenvectors[:, idx[:n_clusters]]
    
    # Normalize rows of eigenvectors
    U = eigenvectors / np.linalg.norm(eigenvectors, axis=1)[:, None]
    
    # Perform k-means clustering on U
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    labels = kmeans.fit_predict(U)
    
    return labels


if __name__ == "__main__":
    data_points, labels = data()
    # graph(data_, labels)

    predicted_labels = dynmsc(data_points, n_clusters=100)

    # Calculate accuracy and F1 score
    accuracy = accuracy_score(labels, predicted_labels)
    f1 = f1_score(labels, predicted_labels, average='weighted')

    print(f"DynMsc Accuracy: {accuracy:.4f}")
    print(f"DynMsc F1 Score: {f1:.4f}")

    # Create confusion matrix
    cm = confusion_matrix(labels, predicted_labels)

    # Plot confusion matrix
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=False, cmap='Blues')
    plt.title('DynMsc Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('confusionMatrix_DynMsc.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Visualize clustering results
    plt.figure(figsize=(16, 14))
    scatter = plt.scatter(data_points[:, 0], data_points[:, 1], c=predicted_labels, cmap='tab20', alpha=0.6, s=30)

    # Custom colorbar to show all 100 labels
    cbar = plt.colorbar(scatter, label='Cluster Label', ticks=range(0, 101, 10))
    cbar.set_ticklabels(range(0, 101, 10))

    plt.title('DynMsc Cluster Visualization', fontsize=16)
    plt.xlabel('Encoded Component 1', fontsize=12)
    plt.ylabel('Encoded Component 2', fontsize=12)

    # Adjust the plot to show all labels
    plt.clim(-0.5, 99.5)

    plt.tight_layout()
    plt.savefig('DynMsc_cluster_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("DynMsc clustering results visualization saved as DynMsc_cluster_visualization.png")
    print("DynMsc confusion matrix saved as confusionMatrix_DynMsc.png")