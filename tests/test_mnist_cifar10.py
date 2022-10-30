"""
Usage example of the Variational Autoencoder with Weighted Loss (VAE-WL) with the MNIST and
    CIFAR10 datasets. Several pixels are Missing Completely At Random. The simulated missing
    rate is 40%. The dataset is scaled to the range [0, 1]. The imputation is evaluated
    through the Mean Absolute Error.
"""

import tensorflow.keras.losses
from sklearn.metrics import mean_absolute_error
import numpy as np
from tensorflow.keras.datasets import mnist, cifar10
from vaewl import ConfigVAE, VAEWL
from sklearn.model_selection import train_test_split

_DATASET = "MNIST"  # MNIST or CIFAR10
_MISSING_RATE = 0.4


def _process_data(x_data):
    """
    Scales the dataset to the range [0, 1] and removes 40% of the pixels completely at random.

    Args:
        x_data: Dataset to be processed.

    Returns: Processed dataset with and without missing pixels, and the missing mask.

    """
    if len(x_data.shape) == 3:
        x_data = np.expand_dims(x_data, axis=3)
    x_data = x_data.astype('float32') / 255

    number_channels = x_data.shape[3]
    missing_mask = np.stack(
        (np.random.choice([0, 1], size=(x_data.shape[0], x_data.shape[1], x_data.shape[2]),
                          p=[1 - _MISSING_RATE, _MISSING_RATE]),) * number_channels, axis=-1)

    x_data_md = x_data * (~missing_mask.astype(bool)).astype(int) + -1.0 * missing_mask
    x_data_md[x_data_md == -1] = np.nan

    return x_data, x_data_md, missing_mask


if __name__ == '__main__':
    if _DATASET == "MNIST":
        source = mnist
    elif _DATASET == "CIFAR10":
        source = cifar10
    else:
        raise ValueError("Invalid dataset.")

    (x_train_og, _), (x_test_og, _) = source.load_data()

    x_train_test = np.concatenate((x_train_og, x_test_og), axis=0)
    x_train_val, x_test = train_test_split(x_train_test, test_size=0.2)
    x_train, x_val = train_test_split(x_train_val, test_size=0.2)

    x_train, x_train_md, _ = _process_data(x_train)
    x_val, x_val_md, _ = _process_data(x_val)
    x_test, x_test_md, missing_mask_test = _process_data(x_test)

    vae_wl_config = ConfigVAE()
    vae_wl_config.verbose = 1
    vae_wl_config.epochs = 200
    vae_wl_config.filters = [32, 64]
    vae_wl_config.kernels = 3
    vae_wl_config.neurons = [392, 196]
    vae_wl_config.dropout = [0.2, 0.2]
    vae_wl_config.latent_dimension = 32
    vae_wl_config.batch_size = 64
    vae_wl_config.activation = "relu"
    vae_wl_config.output_activation = "sigmoid"
    vae_wl_config.loss = tensorflow.keras.losses.binary_crossentropy
    vae_wl_config.input_shape = x_train.shape[1:]
    vae_wl_config.missing_values_weight = 5
    vae_wl_config.kullback_leibler_weight = 0.1

    vae_wl_model = VAEWL(vae_wl_config)
    print("[VAE-WL] Training and performing imputation...")
    vae_wl_model.fit(x_train_md, x_train, X_val=x_val_md, y_val=x_val)

    x_test_imputed = vae_wl_model.transform(x_test_md)

    missing_mask_test_flat = missing_mask_test.astype(bool).flatten()
    mae = mean_absolute_error(x_test_imputed.flatten()[missing_mask_test_flat],
                              x_test.flatten()[missing_mask_test_flat])

    print(f"[VAE-WL] MAE for the {_DATASET} dataset: {mae:.3f}")
