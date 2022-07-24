import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from . import ConfigVAE, VariationalAutoEncoder


class VAEWL(BaseEstimator, TransformerMixin):
    _PRE_IMPUTATION_CONSTANT = 0.0

    def __init__(self, config_vae: ConfigVAE):
        self._fitted = False
        self._config_vae = config_vae
        self._vae_wl_model = VariationalAutoEncoder(config_vae)

    def fit(self, X, y=None, **fit_params):
        if not isinstance(X, np.ndarray):
            raise TypeError("'X' must be a NumPy Array.")

        X_val = None
        mis_mask_val = None
        if "X_val" in fit_params:
            X_val = fit_params["X_val"]
            if not isinstance(X_val, np.ndarray):
                raise TypeError("'X_val' must be a NumPy Array.")
            mis_mask_val = np.isnan(X_val).astype(int)

        mis_mask = np.isnan(X).astype(int)
        X_pre = np.nan_to_num(X, nan=self._PRE_IMPUTATION_CONSTANT)

        X_val_pre = None
        if X_val is not None:
            X_val_pre = np.nan_to_num(X_val, nan=self._PRE_IMPUTATION_CONSTANT)

        self._vae_wl_model.fit(X_pre, mis_mask, X_pre, X_val_pre, mis_mask_val, X_val_pre)
        self._fitted = True
        return self

    def transform(self, X, y=None):
        if not self._fitted:
            raise RuntimeError("The fit method must be called before transform.")
        if not isinstance(X, np.ndarray):
            raise TypeError("'X' must be a NumPy Array.")

        mis_mask = np.isnan(X).astype(int)
        X_pre = np.nan_to_num(X, nan=self._PRE_IMPUTATION_CONSTANT)

        X_imputed = self._vae_wl_model.encode_and_decode(X_pre, mis_mask)
        X_imputed = X_pre * (~mis_mask.astype(bool)).astype(int) + X_imputed * mis_mask

        return X_imputed
