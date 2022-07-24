# VAE-WL - Variational Autoencoder with Weighted Loss

Codebase for the paper *Missing Image Data Imputation using Variational Autoencoders with Weighted Loss*

### Paper Details
- Authors: Ricardo Cardoso Pereira, Joana Cristo Santos, José Pereira Amorim, Pedro Pereira Rodrigues, Pedro Henriques Abreu
- Abstract: Missing data is an issue often addressed with imputation strategies that replace the missing values with plausible ones. A trend in these strategies is the use of generative models, one being Variational Autoencoders. However, the default loss function of this method gives the same importance to all data, while a more suitable solution should focus on the missing values. In this work an extension of this method with a custom loss function is introduced (Variational Autoencoder with Weighted Loss). The method was compared with state-of-the-art generative models and the results showed improvements higher than 40% in several settings.
- Published in: 28th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN 2020)
- Year: 2020
- Link: https://www.esann.org/sites/default/files/proceedings/2020/ES2020-193.pdf
- Contact: rdpereira@dei.uc.pt

### Notes
- The VAE-WL package follows the scikit-learn architecture, implementing the `fit()`, `transform()` and `fit_transform()` methods.
- The data to be imputed must be a NumPy Array.
- The missing values are pre-imputed with 0.
- The Variational Autoencoder architecture can be customized through the `ConfigVAE` data class. 
- A detailed usage example for the MNIST and CIFAR-10 datasets is available in `tests/test_mnist_cifar10.py`.
