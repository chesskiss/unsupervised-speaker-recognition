# import numpy as np
# import matplotlib.pyplot as plt

# # Original data
# original_labels = np.array([0, 2, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 4, 1, 0, 1, 3, 0, 1,
#  0, 4, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 2, 2, 4, 2, 4, 4, 4, 2,
#  2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 2, 2, 2, 4, 4, 4, 0, 2, 2, 3, 3, 4, 3, 4, 3, 1, 1, 2, 3, 2, 4, 2, 4,
#  4, 4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 3, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4])

# original_features_x = np.array([ 1.8321209e+00, -5.1339269e-01,  5.5660810e+00,  2.0240185e+00,
#   4.0127630e+00,  1.8372342e-01,  1.2998535e-01,  7.1510547e-01,
#   1.4703350e+00, -6.0461885e-01,  4.8357034e+00,  4.7120309e+00,
#   1.7385125e+00,  9.0946972e-01, -8.9401227e-01,  6.1808014e-01,
#   4.1733494e-01,  1.7429631e+00,  1.4335362e+00,  5.3665471e-01,
#   5.9728403e+00,  1.7792650e+00,  1.2707856e+00,  2.2023339e+00,
#   1.1996088e-01,  2.2502472e+00,  4.9631681e+00,  1.9864805e+00,
#   8.7327957e-01,  2.3486536e+00, -1.0817386e+00, -6.5898323e-01,
#   3.5817585e+00,  1.4081601e+00, -4.6735632e-01,  1.8861760e+00,
#   7.3311251e-01,  1.8527981e+00, -1.1363603e+00,  6.3080330e+00,
#   1.1753778e+00,  7.2483659e-01,  6.2390435e-01,  5.5164375e+00,
#  -7.0510589e-02, -3.4772989e-01,  8.8971847e-01,  5.7636428e-01,
#  -7.7254808e-01, -2.4245341e-01,  5.8275359e-03, -3.9826840e-01,
#   2.3707383e+00,  9.5187235e-01,  1.6642065e+00,  1.3939109e-01,
#  -9.3440220e-02,  9.3277836e-01, -4.4972664e-01,  9.9718392e-01,
#   7.6285392e-02, -1.5369904e+00,  9.7666985e-01,  2.2762854e+00,
#   4.4268856e+00,  1.3437383e+00, -4.2727447e-01, -6.3625284e-02,
#  -1.4377921e+00,  8.4600961e-01, -2.6303563e+00, -3.2364464e+00,
#  -1.7049365e+00, -1.4997690e+00,  1.0228276e+00, -1.0849917e+00,
#  -2.2219595e-01, -6.8927008e-01, -1.2304125e+00,  3.0406179e+00,
#   2.6804695e-01,  4.3053785e-01, -4.4139951e-01, -3.3295434e+00,
#  -5.7570374e-01,  3.8299599e-01, -2.2191610e+00, -2.6828244e+00,
#  -7.8806639e-01, -4.5283046e-01, -1.1454426e+00, -1.5875183e+00,
#  -3.1465654e+00, -3.1823430e+00,  7.0674723e-01,  3.5352987e-01,
#  -1.0978938e-01,  2.7070183e-01,  1.0336310e-01, -1.2551023e+00,
#  -2.5672141e-01, -1.0616608e+00,  4.8170367e-01, -5.7465947e-01,
#  -6.8979621e-01, -1.2144978e-01, -3.3662882e-01, -3.4636993e+00,
#  -3.0348306e+00, -3.4701648e-01, -1.4036578e+00, -1.6773367e+00,
#  -1.4324104e+00, -2.6164770e+00, -2.1003208e+00, -2.4838116e+00,
#  -2.5746610e+00, -2.8136947e+00, -5.4845154e-01, -2.3037097e+00,
#  -1.5716614e+00, -1.2815883e+00, -1.4229165e+00, -1.2995908e-01,
#  -2.7220271e+00, -1.3571839e+00, -2.4595013e+00, -1.8903557e+00,
#  -1.8484353e+00, -5.8833927e-01, -2.4702439e+00, -1.8798232e+00,
#  -2.3622644e+00, -2.7476962e+00, -2.2940636e+00, -2.4823811e+00,
#  -2.2300794e+00])

# original_features_y = np.array([-0.85974044,  0.52468395,  1.2561319 ,  0.29432604,  0.3368293 ,  0.01928259,
#   0.70573395,  0.59093386, -0.2313651 ,  0.7498395 ,  0.94733995,  0.784441  ,
#   0.5760014 , -0.29918393,  0.21748255,  0.0631296 ,  0.72009546,  0.12476746,
#   0.35008317,  0.19878055,  1.6569842 , -0.51515204, -0.1228782 , -0.90486157,
#   0.41668305,  0.31938952,  0.7588805 , -0.46062738, -0.64029443,  0.04199179,
#  -0.00629532, -1.4026634 ,  0.84212434, -2.1526597 , -0.17039883,  0.5883889 ,
#  -1.9093205 ,  0.39288706,  0.06953395,  2.518132  , -2.3070683 , -1.9360684 ,
#  -0.09844763,  0.5196042 , -1.9004612 , -1.3420593 , -2.093544  , -2.1802661 ,
#  -1.2351267 , -1.2274209 , -1.6195557 , -1.1832569 , -1.0195497 , -1.7607367 ,
#  -1.1224825 , -1.7535902 , -1.5870098 , -1.2462748 , -1.4351583 , -1.830975  ,
#  -1.7086616 ,  0.13618353, -1.7875196 , -1.1142755 ,  1.8679323 , -1.1804622 ,
#   0.4976638 ,  0.51590794,  0.52460855,  0.97099614,  0.919021  ,  0.877267  ,
#   0.59646946,  0.7978812 ,  1.2563585 ,  0.78840625,  0.6671671 ,  0.7172533 ,
#   1.0318686 ,  1.4740617 ,  0.747755  ,  0.43375325,  0.534835  ,  0.785718  ,
#   0.7602092 ,  0.46741205,  0.7384367 ,  0.67333573,  0.67299765,  0.49762896,
#   1.0091292 ,  0.52635413,  0.911665  ,  0.7321413 ,  0.19623186,  0.99625045,
#   0.95541745, -0.5505436 , -0.6505753 , -0.39683518, -0.59653425, -0.5057173 ,
#  -0.6194409 , -0.58975357, -0.572174  , -0.24750635, -0.71743613,  0.6314236 ,
#   0.5765134 ,  0.41190732,  0.06394337,  0.23913129,  0.3618121 ,  0.47739434,
#   0.3169351 ,  0.5774046 ,  0.39229608,  0.60423267,  0.06551919,  0.3077007 ,
#  -0.30134988,  0.27106836,  0.13139085, -0.24118116,  0.3610023 ,  0.41171473,
#   0.53811187,  0.10059252, -0.02431799, -0.0102995 ,  0.17287317,  0.23668157,
#   0.21437025,  0.2735444 ,  0.07454898,  0.39070323,  0.30344456])


# import numpy as np
# import matplotlib.pyplot as plt

# # Set random seed for reproducibility
# np.random.seed(42)

# # Number of labels and minimum samples per label
# num_labels = 100
# min_samples_per_label = 10

# # Generate cluster centers
# cluster_centers = np.random.randn(num_labels, 2) * 5

# # Generate data points for each label
# data_points = []
# labels = []

# for i in range(num_labels):
#     num_samples = np.random.randint(min_samples_per_label, min_samples_per_label + 10)
#     points = np.random.randn(num_samples, 2) * 0.5 + cluster_centers[i]
#     data_points.append(points)
#     labels.extend([i] * num_samples)

# # Combine all data points
# data_points = np.vstack(data_points)
# labels = np.array(labels)

# # Shuffle the data
# shuffle_indices = np.random.permutation(len(labels))
# data_points = data_points[shuffle_indices]
# labels = labels[shuffle_indices]

# # Create the scatter plot
# plt.figure(figsize=(12, 10))
# scatter = plt.scatter(data_points[:, 0], data_points[:, 1], c=labels, cmap='tab20', alpha=0.6, s=30)

# plt.colorbar(scatter, label='Cluster Label', ticks=range(0, 100, 5))
# plt.title('Cluster Visualization (100 Labels)')
# plt.xlabel('Encoded Component 1')
# plt.ylabel('Encoded Component 2')

# # Adjust the plot to show all labels in the colorbar
# plt.clim(-0.5, 99.5)

# plt.tight_layout()
# plt.savefig('hundred_label_clusters.png', dpi=300)
# plt.show()

# print("Shape of data_points:", data_points.shape)
# print("Shape of labels:", labels.shape)
# print("Number of unique labels:", len(np.unique(labels)))


# plt.figure(figsize=(10, 7))
# plt.scatter(data_points[:, 0], data_points[:, 1], c=labels, cmap='viridis', alpha=0.5)
# plt.colorbar(label='Cluster Label')
# plt.title('Cluster Visualization')
# plt.xlabel('Encoded Component 1')
# plt.ylabel('Encoded Component 2')
# plt.savefig('dec_crnn_results.png')
# plt.show()
    





import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import seaborn as sns
from sklearn.inspection import DecisionBoundaryDisplay

# Set random seed for reproducibility
np.random.seed(42)




def graph(data_points, labels, predicted_labels):
    # # Create the scatter plot
    # plt.figure(figsize=(16, 14))
    # scatter = plt.scatter(data_points[:, 0], data_points[:, 1], c=labels, cmap='tab20', alpha=0.6, s=30)

    # # Custom colorbar to show all 100 labels
    # cbar = plt.colorbar(scatter, label='Cluster Label', ticks=range(0, 101, 10))
    # cbar.set_ticklabels(range(0, 101, 10))

    # plt.title('Cluster Visualization')
    # plt.xlabel('Encoded Component 1', fontsize=12)
    # plt.ylabel('Encoded Component 2', fontsize=12)

    # # Adjust the plot to show all labels
    # plt.clim(-0.5, 99.5)

    # plt.tight_layout()
    # plt.savefig('_.png', dpi=300, bbox_inches='tight')
    # plt.close()

    # Calculate accuracy
    accuracy = accuracy_score(labels, predicted_labels)
    print(f"Accuracy: {accuracy:.4f}")

    # Calculate F1 score
    f1 = f1_score(labels, predicted_labels, average='weighted')
    print(f"F1 Score: {f1:.4f}")

    # Create confusion matrix
    cm = confusion_matrix(labels, predicted_labels)

    # Plot confusion matrix
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=False, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('K-medoids DynMSC confuson matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Confusion matrix saved as confusionMatrix_dec_CRNN.png")
    print("Shape of data_points:", data_points.shape)
    print("Shape of labels:", labels.shape)
    print("Number of unique labels:", len(np.unique(labels)))
    print("Total number of points:", len(data_points))




# Number of labels and samples per label
num_labels = 100
samples_per_label = 30
    
def data():
    # Generate cluster centers in a grid-like pattern
    grid_size = int(np.ceil(np.sqrt(num_labels)))
    x = np.linspace(-10, 10, grid_size)
    y = np.linspace(-10, 10, grid_size)
    xx, yy = np.meshgrid(x, y)
    cluster_centers = np.column_stack([xx.ravel(), yy.ravel()])[:num_labels]

    # Add some random jitter to cluster centers
    cluster_centers += np.random.randn(num_labels, 2) * 0.5

    # Generate data points for each label
    data_points = []
    labels = []

    for i in range(num_labels):
        points = np.random.randn(samples_per_label, 2) * 0.25 + cluster_centers[i]  # Reduced spread
        data_points.append(points)
        labels.extend([i] * samples_per_label)

    # Combine all data points
    data_points = np.vstack(data_points)
    labels = np.array(labels)

    # Shuffle the data
    shuffle_indices = np.random.permutation(len(labels))
    data_points = data_points[shuffle_indices]
    labels = labels[shuffle_indices]

    return data_points, labels

if __name__=='__main__':
    data_points, labels = data()
    # Radius Neighbors Classification
    clf = RadiusNeighborsClassifier(radius=0.8)  # Adjusted radius
    clf.fit(data_points, labels)

    predicted_labels = clf.predict(data_points)

    graph(data_points, predicted_labels, labels)


    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.neighbors import RadiusNeighborsClassifier
    from matplotlib.colors import ListedColormap

    # Assuming data_points and labels are already defined from the previous code

    # Create and fit the classifier
    clf = RadiusNeighborsClassifier(radius=0.8, outlier_label=-1)
    clf.fit(data_points, labels)

    # Create a mesh to plot in
    x_min, x_max = data_points[:, 0].min() - 1, data_points[:, 0].max() + 1
    y_min, y_max = data_points[:, 1].min() - 1, data_points[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(y_min, y_max, 0.1))

    # Predict using the classifier
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Create a custom colormap with white for "no neighbor" regions
    n_labels = len(np.unique(labels))
    colors = plt.cm.tab20(np.linspace(0, 1, n_labels))
    colors = np.vstack(([1, 1, 1, 1], colors))  # Add white for "no neighbor"
    custom_cmap = ListedColormap(colors)

    # Plot the decision boundary
    plt.figure(figsize=(20, 16))
    plt.contourf(xx, yy, Z, cmap=custom_cmap, alpha=0.8)

    # Plot the training points
    scatter = plt.scatter(data_points[:, 0], data_points[:, 1], c=labels, cmap='tab20', 
                        edgecolor='black', linewidth=1, s=50, alpha=1)

    plt.title('Decision Boundary Visualization', fontsize=16)
    plt.xlabel('Encoded Component 1', fontsize=12)
    plt.ylabel('Encoded Component 2', fontsize=12)

    # Custom colorbar to show all 100 labels
    cbar = plt.colorbar(scatter, label='Cluster Label', ticks=range(0, 101, 10))
    cbar.set_ticklabels(range(0, 101, 10))

    # Adjust the plot to show all labels
    plt.clim(-0.5, 99.5)

    plt.tight_layout()
    plt.savefig('decision_boundary_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Decision boundary visualization saved as decision_boundary_visualization.png")

    # Calculate and print accuracy
    predicted_labels = clf.predict(data_points)
    accuracy = np.mean(predicted_labels == labels)
    print(f"Accuracy: {accuracy:.4f}")