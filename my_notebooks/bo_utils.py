import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

# Adapted from https://github.com/krasserm/bayesian-machine-learning 
# Prepared by Ritz Aguilar (2024)

def expected_improvement(X, X_train, Y_train, gp, xi=0.01):
    '''
    Adapted from https://github.com/krasserm/bayesian-machine-learning
    
    Calculates the Expected Improvement (EI) at points X based on
    existing training data (X_train, Y_train) using a Gaussian process surrogate model.
    
    Args:
        X: Locations where EI will be evaluated (m x d).
        X_train: Training data input points (n x d).
        Y_train: Corresponding observed values for training data (n x 1).
        gp: Trained GaussianProcessRegressor model.
        xi: Parameter that balances exploration and exploitation.

    Returns:
        Computed Expected Improvement values at locations X.
        
    '''
    mu, sigma = gp.predict(X, return_std=True)  # Predict mean and standard deviation
    mu_train = gp.predict(X_train)
    sigma = sigma.reshape(-1, 1)  # Ensure sigma is reshaped as a column vector
    # print("mu, sigma:", mu.shape, sigma.shape)
    
    # For noise models or use np.max(Y_train) for noiseless scenarios
    mu_train_opt = np.max(mu_train) # Best-observed value so far

    with np.errstate(divide='warn'):  # Safe handling division by zero
        imp = mu - mu_train_opt - xi  # Improvement over the best observed value
        Z = imp / sigma.flatten()  # Z-score calculation

        # Compute Expected Improvement; set EI to zero where variance is zero
        ei = np.where(sigma.flatten() > 0, 
              imp * norm.cdf(Z) + sigma.flatten() * norm.pdf(Z), 
              0)  # Compute Expected Improvement

    # print("ei.shape:", ei.shape)
    return ei.flatten()



def suggest_next_sample(utility, X_train, Y_train, gp, bounds, n_restarts=25):
    '''
    Adapted from https://github.com/krasserm/bayesian-machine-learning
    
    Suggests the next point to sample by optimizing the utility function.
    
    Args:
        acquisition: The utility function to be maximized.
        X_train: Training data input points (n x d).
        Y_train: Observed values corresponding to training points (n x 1).
        gp: A trained GaussianProcessRegressor model.
        bounds: Boundaries for the search space.
        n_restarts: Number of random starting points for optimization.

    Returns:
        The proposed next sampling location.
    '''
    dim = X_train.shape[1]
    min_val = 1  # Initialize with a large value
    min_x = None  # Placeholder for the optimal point
    
    def min_obj(X):
        # Objective to minimize: negative of the acquisition function
        return -utility(X.reshape(-1, dim), X_train, Y_train, gp).flatten()
    
    # Search for the optimal point by trying multiple random n_restart points
    for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')
        if res.fun < min_val:
            min_val = res.fun
            min_x = res.x

    return min_x.reshape(-1, 1)