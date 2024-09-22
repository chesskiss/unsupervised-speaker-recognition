from rcc import RCC

rcc = RCC(n_clusters=n_known_speakers + 1)  # +1 for unknown
labels = rcc.fit_predict(features)



'''    TODO - new :
    1. Complete method with RCC: Robust Continuous Clustering 

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