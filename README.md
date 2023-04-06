# Iris Recognition Algorithm

This repository contains an implementation of an iris recognition algorithm based on the paper by [Ma et al., 2003](https://ieeexplore.ieee.org/document/1251145) (see reference), but focuses on Image Preprocessing, Feature Extraction, and Iris Matching only and uses the provided dataset for testing.

<kbd>
<img src="https://user-images.githubusercontent.com/114009025/230469011-21655c5c-987d-4175-9954-ebece83336fd.png">
</kbd>
&emsp; &rarr; &emsp; 
<kbd>
<img src="https://user-images.githubusercontent.com/114009025/230469095-454d826f-d2e6-43a2-9ec7-ed83d5479a94.png">
</kbd>


## Experiment Design
- **Data Source**: 

  CASIA Iris Image Database (version 1.0) **[CASIA-IrisV1]**.
  
  108 eyes, 7 iris images per eye, which were captured in two sessions (3 in the first session, 4 in the second session). All images are stored as BMP format with 320x280 pixel size.
- **Usage**: 
  
    Images from the first session will be used for training and images from the second session will be used for testing.
- **Evaluation**

  Correct Recognition Rate (CRR) for the identification mode.
  Receiver Operating Characteristic (ROC) curve for the verification mode.

## Sub-Functions
- `IrisRecognition.py`: the main function, which will use all the following sub functions;
- `IrisLocalization.py`: detecting pupil and outer boundary of iris;
- `IrisNormalization.py`: mapping the iris from Cartesian coordinates to polar coordinates;
- `ImageEnhancement.py`: enhancing the normalized iris;
- `FeatureExtraction.py`: filtering the iris and extracting features;
- `IrisMatching.py`: using Fisher linear discriminant for dimension reduction and nearest center classifier for classification;
- `PerformanceEvaluation.py`: calculating the CRR for the identification mode (CRR for all three measures, i.e., L1, L2, and Cosine similarity, should be >=75%); calculating ROC curve for verification mode.

## Reference
- Ma et al., Personal Identification Based on Iris Texture Analysis, IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE, VOL. 25, NO. 12, DECEMBER 2003
- Note_CASIA-IrisV1.pdf
