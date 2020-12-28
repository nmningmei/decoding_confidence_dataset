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
RNN model                  |  LSTM cell
:-------------------------:|:-------------------------:
![rnn](https://github.com/nmningmei/decoding_confidence_dataset/blob/main/figures/RNN%20model%20confidence%20database.jpg)  |  ![lstm](https://github.com/nmningmei/decoding_confidence_dataset/blob/main/figures/external-content.duckduckgo.com.jpg)


# Decode confidence
## Decoding scores (within domain)
![confidence_w_score](https://github.com/nmningmei/decoding_confidence_dataset/blob/main/figures/confidence/LOO_compare_RNN_RF/RNN%20vs%20RF%20LOO.jpeg)

## Feature contributions (within domain)
![confidence_w_feature](https://github.com/nmningmei/decoding_confidence_dataset/blob/main/figures/confidence/LOO_compare_RNN_RF/RNN%20vs%20RF%20features.jpeg)

## Decoding scores (cross domain)
![confidence_c_score](https://github.com/nmningmei/decoding_confidence_dataset/blob/main/figures/confidence/CD/cross%20domain%20decoding%20scores.jpeg)

## Feature contributions (cross domain)
![confidence_c_feature](https://github.com/nmningmei/decoding_confidence_dataset/blob/main/figures/confidence/CD/hidden%20states%20of%20time%20steps.jpeg)

# Decode adequacy
## Decoding scores (within domain)
![adequacy_w_score](https://github.com/nmningmei/decoding_confidence_dataset/blob/main/figures/adequacy/LOO_compare_RNN_RF/RNN%20vs%20RF%20LOO.jpeg)

## Feature contributions (within domain)
![adequacy_w_feature](https://github.com/nmningmei/decoding_confidence_dataset/blob/main/figures/adequacy/LOO_compare_RNN_RF/RNN%20vs%20RF%20features.jpeg)

## Decoding scores (cross domain)
![adequacy_c_score](https://github.com/nmningmei/decoding_confidence_dataset/blob/main/figures/adequacy/CD/cross%20domain%20decoding%20scores.jpeg)

## Feature contributions (cross domain)
![adequacy_c_feature](https://github.com/nmningmei/decoding_confidence_dataset/blob/main/figures/adequacy/CD/hidden%20states%20of%20time%20steps.jpeg)
