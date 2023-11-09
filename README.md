# Emotion Classification
## Introduction

This project aims to classify human audio signals into positive or negative emotions using Mel spectrogram and chromagram features. The data is collected from the Ryerson Audio-Visual Database of Emotional Speech and Song, containing 1440 audio files vocalized by 24 professional actors. Other features such as modality, emotional intensity, repetition times and actor’s gender are also included in the classifications.

## Setup and Requirements

You can install the required libraries using the following command:

```bash
pip install numpy pandas scikit-learn librosa resampy==0.3.1
```

## **Optimized Approach**

1. **Feature Extraction:**
    - Mel spectrogram and chromagram features are extracted to represent audio signals.
    - The dataset is split into 70% training and 30% testing subsets.
2. **Classification using SVM:**
    - Gradient descent algorithm is applied to handle non-linearity.
        
        $$
        J(W)={1\over2}\lambda||w||^2+{1\over N}\sum^N_{i=1}\max\{0,1-y_i(w^Tx_i+b)\}
        $$
        
    - SVM classifier achieves an accuracy of 72.7% for positive and negative emotion classification.
3. **Classification using PCA:**
    - PCA is employed to reduce dimensionality of the original data.
    - SVM classifier is applied on PCA-transformed data, achieving an accuracy of 69.26%.

## References

https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html

https://zenodo.org/record/1188976#.Y3qkx3bMLIU
