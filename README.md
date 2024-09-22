In this project, we had built several models for audio classifications using a variety of clustering methods.

1. 

2. Using DEC with 2 different pre-trained model for encoding: EfficientNet and our custom CRNN.
With the pre-trained EfficientNet the accuracy reached around 60%. By encoding with our custom CRNN the accuracy reached 96%, and more clusters were formed. 

The difference in their performances stems from the merits of the CRNN. On one hand, as an RNN, it effectively processes sequential input (in our case, audio), while on the other hand, as a CNN, it efficiently handles the frequencies and amplitudes of waves.

3. 

Notes: initially we considered using HDBSCAN as well, but it was less fitting for classifying unkown callers.
