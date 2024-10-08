import matplotlib.pyplot as plt
import numpy as np

# Adapted from https://github.com/krasserm/bayesian-machine-learning 
# Prepared by Ritz Aguilar (2024)

def plot_prediction(ax, gp, X, Y, X_train, Y_train, X_next=None, noise_level=0.1, show_legend=False):
    """
    Plot the Gaussian Process predictions with uncertainty intervals,
    training points, and the next proposed sampling point.
    """
    Y_pred, sigma = gp.predict(X, return_std=True)
    ax.fill_between(X.ravel(), 
                    Y_pred - 1.96 * sigma, 
                    Y_pred + 1.96 * sigma, 
                    color='#87CEFA', alpha=0.75, label='95% confidence interval')
    ax.plot(X, Y, c='#FF4500', ls=':', lw=2.5, label='ground truth')
    ax.plot(X, Y_pred, c='#1E90FF', lw=2.5, label='prediction') 
    ax.errorbar(X_train.flatten(), Y_train.flatten(), yerr=noise_level, 
                c='#FF4500', fmt='.', ms=12, capsize=5, elinewidth=2,  alpha=0.75, label='observations')  
    # ax.set_xlim([bounds[:,0], bounds[:,1]])
    ax.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)
    
    if X_next is not None:
        ax.axvline(x=X_next, ls='--', c='k', lw=2)
    if show_legend:
        ax.legend(loc="upper right")


def plot_utility(ax, X, Y, X_next, show_legend=False):
    """
    Plot the utility (acquisition) function and the next proposed sampling point.
    """
    ax.clear()  # Clear the current axes before plotting
    ax.plot(X, Y, 'r-', lw=2, label='utility function')
    ax.fill_between(X.ravel(), Y.ravel(), color='red', alpha=0.3)
    ax.axvline(x=X_next, ls='--', c='k', lw=2, label='next sampling point')
    # ax.set_xlim([bounds[:,0], bounds[:,1]])
    ax.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)
    
    if show_legend:
        ax.legend(loc="upper left")
        
        
def plot_convergence(ax, X_train, Y_train, n_init=2, show_legend=False):
    """
    Plot convergence metrics: distance between consecutive x's and the best observed Y value.
    """
    # Print shapes of incoming data
    print("X_train shape:", X_train.shape)
    print("Y_train shape:", Y_train.shape)

    # Reshape X_train and Y_train
    x = X_train[n_init:].ravel()
    y = Y_train[n_init:].ravel()
    r = range(1, len(x) + 1)

    # Distance between consecutive x's
    x_neighbor_dist = [np.abs(a - b) for a, b in zip(x, x[1:])]
    y_max_watermark = np.maximum.accumulate(y)

    # Create subplots for distance and best Y values
    ax[0].plot(r[1:], x_neighbor_dist, 'bo-')
    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel('Distance')
    ax[0].set_title('Distance between consecutive x\'s')
    ax[0].grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)

    # Plot best Y values
    ax[1].plot(r, y_max_watermark, 'ro-', label='Best Y value')
    ax[1].set_xlabel('Iteration')
    ax[1].set_ylabel('Best Y')
    ax[1].set_title('Value of best selected sample')
    ax[1].grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)

    if show_legend:
        ax[1].legend(loc="upper left")
    
    plt.tight_layout()
