from src.iterative_svd import iterative_svd_impute
from src.utils import generate_synthetic_matrix, evaluate_rmse

def main():
    print("\n🔬 Drug Efficacy Prediction using Iterative SVD\n")

    # Generate synthetic drug-response matrix
    R_true, R_train, missing_mask = generate_synthetic_matrix()

    # Run Iterative SVD (Soft-Impute)
    R_pred = iterative_svd_impute(R_train, missing_mask, k=6, max_iter=10)

    # Evaluate RMSE
    rmse = evaluate_rmse(R_true, R_pred, missing_mask)

    print("\n-----------------------------------------------")
    print(" Final RMSE:", rmse)
    print("-----------------------------------------------\n")

if __name__ == "__main__":
    main()

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error

# --- 1. Setup (Same synthetic data) ---
np.random.seed(42)
patients = 60
drugs = 25
rank = 6 
k = 6 # Target latent factors
MAX_ITER = 10 
TOLERANCE = 1e-4

# True matrix and observed matrix
U = np.random.randn(patients, rank)
V = np.random.randn(drugs, rank)
R_true = U.dot(V.T)
R_obs = R_true + 0.05 * np.random.randn(*R_true.shape)

# Masking setup
missing_mask = np.random.rand(*R_obs.shape) < 0.2
R_train = R_obs.copy()
R_train[missing_mask] = np.nan

# Store known values and their locations
known_mask = ~missing_mask
R_known = R_obs[known_mask]
true_vals = R_true[missing_mask]

# --- 2. Initialize and Iteratively Impute ---

# Start with a simple mean imputation (Initial guess for R_imputed)
initial_mean = np.nanmean(R_train)
R_imputed = np.where(np.isnan(R_train), initial_mean, R_train)

prev_imputed_vals = R_imputed[missing_mask].copy()

for i in range(MAX_ITER):
    # 2a. Apply Truncated SVD on the imputed matrix
    svd = TruncatedSVD(n_components=k, random_state=0)
    Z = svd.fit_transform(R_imputed)
    R_reconstructed = svd.inverse_transform(Z)

    # 2b. 'Soft' Imputation step:
    # Overwrite the missing values in R_imputed with the reconstructed values
    # Keep the known values fixed (R_obs[known_mask] is non-NaN)
    R_imputed[missing_mask] = R_reconstructed[missing_mask]
    
    # 2c. Check for convergence 
    current_imputed_vals = R_imputed[missing_mask]
    
    # Calculate the change in the imputed values
    diff = np.linalg.norm(current_imputed_vals - prev_imputed_vals) / np.linalg.norm(prev_imputed_vals)
    
    print(f"Iteration {i+1}: Convergence change = {diff:.6f}")
    
    if diff < TOLERANCE:
        print(f"Converged after {i+1} iterations.")
        break
        
    prev_imputed_vals = current_imputed_vals.copy()

# --- 3. Evaluate Predictions ---

R_pred = R_imputed # The final imputed matrix is our prediction matrix
pred_vals = R_pred[missing_mask]
rmse = mean_squared_error(true_vals, pred_vals, squared=False)

print("\n--------------------------------------------------")
print(" Iterative SVD (Soft-Impute Principle) Results")
print("--------------------------------------------------")
print(f"Final RMSE on predictions: {rmse:.4f}")
print("--------------------------------------------------")