import numpy as np
from sklearn.decomposition import TruncatedSVD

def iterative_svd_impute(R_train, missing_mask, k=6, max_iter=10, tolerance=1e-4):

    # Initial mean imputation
    initial_mean = np.nanmean(R_train)
    R_imputed = np.where(np.isnan(R_train), initial_mean, R_train)

    prev_vals = R_imputed[missing_mask].copy()

    for i in range(max_iter):
        print(f"\nIteration {i+1}")

        svd = TruncatedSVD(n_components=k)
        Z = svd.fit_transform(R_imputed)
        R_reconstructed = svd.inverse_transform(Z)

        # Update missing only
        R_imputed[missing_mask] = R_reconstructed[missing_mask]

        # Convergence check
        current_vals = R_imputed[missing_mask]
        change = np.linalg.norm(current_vals - prev_vals) / np.linalg.norm(prev_vals)

        print(f"Change: {change:.6f}")

        if change < tolerance:
            print("Converged!")
            break

        prev_vals = current_vals.copy()

    return R_imputed
