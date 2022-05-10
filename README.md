# Thermal Comfort Classification
Pytorch based models for classification of thermal comfort states using a multi-modal data.

## Preprocessing Steps
Outlier removal based on interquartil range, filtering of lines with missing values is done for all items.

Normalization parameters for normalized features:
* __age__: min=0, max=100
* __rgb__: min=0, max=255 (for all 3 channels)
* __keypoints__: min=0 max=1920 (x), min=0 max=1080 (y), min=0 max=5000 (z)
* __heart-rate__: min=35, max=130
* __wrist-temp__: min and max from data
* __ambient-temp__: min and max from data
* __relative-humidity__: data already in [0, 1]

One-hot-encoding used on following features: gender, emotion, tiredness-level.

## Training Perfomance 
Coming Soon..
