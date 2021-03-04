# Decoding analyses of the confidence dataset via recurrent neural network and random forest models

# System Information
- Platform:      Linux-3.10.0-514.el7.x86_64-x86_64-with-centos-7.3.1611-Core
- Python:        3.6.3 |Anaconda, Inc.| (default, Nov 20 2017, 20:41:42)  [GCC 7.2.0]
- CPU:           x86_64: 16 cores
- numpy:         1.16.4 {blas=mkl_rt, lapack=mkl_rt}
- scipy:         1.3.1
- matplotlib:    3.1.3 {backend=agg}
- seaborn:       0.11.1
- sklearn:       0.23.2
- pandas:        1.0.1
- tensorflow:    2.0.0

# Random Forest model
<img src="https://www.kdnuggets.com/wp-content/uploads/rand-forest-2.jpg" width="100%" />

# RNN model
<p float="left">
  <img src="https://github.com/nmningmei/decoding_confidence_dataset/blob/main/figures/RNN%20model%20confidence%20database.jpg" width="40%" /> <img src="https://github.com/nmningmei/decoding_confidence_dataset/blob/main/figures/external-content.duckduckgo.com.jpg" width="40%" /> 
</p>


# Results
## Decoding scores (within domain)
Confidence             |  Adequacy
:-------------------------:|:-------------------------:
![cws](https://github.com/nmningmei/decoding_confidence_dataset/blob/main/figures/confidence/LOO_compare_RNN_RF/scores.jpg)  |  ![aws](https://github.com/nmningmei/decoding_confidence_dataset/blob/main/figures/adequacy/LOO_compare_RNN_RF/scores.jpg)

## Feature attribution (within domain)
Confidence             |  Adequacy
:-------------------------:|:-------------------------:
![cwf](https://github.com/nmningmei/decoding_confidence_dataset/blob/main/figures/confidence/LOO_compare_RNN_RF/features.jpg)  |  ![awf](https://github.com/nmningmei/decoding_confidence_dataset/blob/main/figures/adequacy/LOO_compare_RNN_RF/features.jpg)

## Trends of features (within domain)
Confidence             |  Adequacy
:-------------------------:|:-------------------------:
![cwf](https://github.com/nmningmei/decoding_confidence_dataset/blob/main/figures/confidence/LOO_compare_RNN_RF/slopes.jpg)  |  ![awf](https://github.com/nmningmei/decoding_confidence_dataset/blob/main/figures/adequacy/LOO_compare_RNN_RF/slopes.jpg)

## Decoding scores (cross domain)
Confidence             |  Adequacy
:-------------------------:|:-------------------------:
![ccs](https://github.com/nmningmei/decoding_confidence_dataset/blob/main/figures/confidence/CD/scores.jpg)  |  ![acs](https://github.com/nmningmei/decoding_confidence_dataset/blob/main/figures/adequacy/CD/scores.jpg)

## Feature attribution (cross domain)
Confidence             |  Adequacy
:-------------------------:|:-------------------------:
![ccf](https://github.com/nmningmei/decoding_confidence_dataset/blob/main/figures/confidence/CD/features.jpg)  |  ![acf](https://github.com/nmningmei/decoding_confidence_dataset/blob/main/figures/adequacy/CD/features.jpg)

## Trends of features (cross domain)
Confidence             |  Adequacy
:-------------------------:|:-------------------------:
![cwf](https://github.com/nmningmei/decoding_confidence_dataset/blob/main/figures/confidence/CD/slopes.jpg)  |  ![awf](https://github.com/nmningmei/decoding_confidence_dataset/blob/main/figures/adequacy/CD/slopes.jpg)

