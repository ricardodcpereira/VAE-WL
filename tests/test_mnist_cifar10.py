import tensorflow.keras.losses
from sklearn.metrics import mean_absolute_error
import numpy as np
from tensorflow.keras.datasets import mnist, cifar10
from vaewl import ConfigVAE, VAEWL

_DATASET = "MNIST"  # MNIST or CIFAR10
_MISSING_RATE = 0.4


def _process_data(x_data):
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

    (x_train, _), (x_test, _) = source.load_data()

    x_train, x_train_md, missing_mask_train = _process_data(x_train)
    x_test, x_test_md, missing_mask_test = _process_data(x_test)

    vae_wl_config = ConfigVAE()
    vae_wl_config.verbose = 1
    vae_wl_config.epochs = 200
    vae_wl_config.filters = [32, 32]
    vae_wl_config.kernels = 3
    vae_wl_config.neurons = [392, 196]
    vae_wl_config.dropout_fc = [0.2, 0.2]
    vae_wl_config.latent_dimension = 32
    vae_wl_config.batch_size = 64
    vae_wl_config.activation = "relu"
    vae_wl_config.output_activation = "sigmoid"
    vae_wl_config.optimizer = "adam"
    vae_wl_config.loss = tensorflow.keras.losses.mean_squared_error
    vae_wl_config.input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
    vae_wl_config.missing_values_weight = 5
    vae_wl_config.kullback_leibler_weight = 0.1

    vae_wl_model = VAEWL(vae_wl_config)
    print("[VAE-WL] Training and performing imputation...")
    vae_wl_model.fit(x_train_md)

    x_test_imputed = vae_wl_model.transform(x_test_md)

    missing_mask_test_flat = missing_mask_test.astype(bool).flatten()
    mae = mean_absolute_error(x_test_imputed.flatten()[missing_mask_test_flat],
                              x_test.flatten()[missing_mask_test_flat])

    print(f"[VAE-WL] MAE for the {_DATASET} dataset: {mae:.3f}")
