# Decoding analyses of the confidence dataset via recurrent neural network and random forest models

# System Information
- Platform:      Linux-3.10.0-514.el7.x86_64-x86_64-with-centos-7.3.1611-Core
- Python:        3.6.3 |Anaconda, Inc.| (default, Nov 20 2017, 20:41:42)  [GCC 7.2.0]
- CPU:           x86_64: 16 cores
- numpy:         1.16.4 {blas=mkl_rt, lapack=mkl_rt}
- scipy:         1.3.1
- matplotlib:    3.1.3 {backend=agg}
- sklearn:       0.23.2
- pandas:        1.0.1
- tensorflow:    2.0.0

# RNN model
<p float="left">
  <img src="https://github.com/nmningmei/decoding_confidence_dataset/blob/main/figures/RNN%20model%20confidence%20database.jpg" width="40%" /> <img src="https://github.com/nmningmei/decoding_confidence_dataset/blob/main/figures/external-content.duckduckgo.com.jpg" width="40%" /> 
</p>


# Results
## Decoding scores (within domain)
Confidence             |  Adequacy
:-------------------------:|:-------------------------:
![cws](https://github.com/nmningmei/decoding_confidence_dataset/blob/main/figures/confidence/LOO_compare_RNN_RF/RNN%20vs%20RF%20LOO.jpeg)  |  ![aws](https://github.com/nmningmei/decoding_confidence_dataset/blob/main/figures/adequacy/LOO_compare_RNN_RF/RNN%20vs%20RF%20LOO.jpeg)

## Feature attribution (within domain)
Confidence             |  Adequacy
:-------------------------:|:-------------------------:
![cwf](https://github.com/nmningmei/decoding_confidence_dataset/blob/main/figures/confidence/LOO_compare_RNN_RF/RNN%20vs%20RF%20features.jpeg)  |  ![awf](https://github.com/nmningmei/decoding_confidence_dataset/blob/main/figures/adequacy/LOO_compare_RNN_RF/RNN%20vs%20RF%20features.jpeg)

## Decoding scores (cross domain)
Confidence             |  Adequacy
:-------------------------:|:-------------------------:
![cws](https://github.com/nmningmei/decoding_confidence_dataset/blob/main/figures/confidence/CD/RNN%20vs%20RF%20LOO.jpeg)  |  ![aws](https://github.com/nmningmei/decoding_confidence_dataset/blob/main/figures/adequacy/CD/RNN%20vs%20RF%20LOO.jpeg)

## Feature attribution (cross domain)
Confidence             |  Adequacy
:-------------------------:|:-------------------------:
![cwf](https://github.com/nmningmei/decoding_confidence_dataset/blob/main/figures/confidence/CD/RNN%20vs%20RF%20features.jpeg)  |  ![awf](https://github.com/nmningmei/decoding_confidence_dataset/blob/main/figures/adequacy/CD/RNN%20vs%20RF%20features.jpeg)
