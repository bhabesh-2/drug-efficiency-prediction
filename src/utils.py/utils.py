import numpy as np
from sklearn.metrics import mean_squared_error

def generate_synthetic_matrix(patients=60, drugs=25, rank=6):
    np.random.seed(42)

    U = np.random.randn(patients, rank)
    V = np.random.randn(drugs, rank)
    R_true = U @ V.T

    R_obs = R_true + 0.05 * np.random.randn(*R_true.shape)

    missing_mask = np.random.rand(*R_obs.shape) < 0.2
    R_train = R_obs.copy()
    R_train[missing_mask] = np.nan

    return R_true, R_train, missing_mask


def evaluate_rmse(R_true, R_pred, missing_mask):

    true_vals = R_true[missing_mask]
    pred_vals = R_pred[missing_mask]

    rmse = mean_squared_error(true_vals, pred_vals, squared=False)
    return rmse
