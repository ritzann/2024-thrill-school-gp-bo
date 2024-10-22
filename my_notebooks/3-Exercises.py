#!/usr/bin/env python
# coding: utf-8

# ### Explanation of Changes
# 
# 1. **Initialization of 2D Inputs**:
#    - **`params_init`**: Combines the randomly initialized **pump strength (`S0_init`)** and **temperature (`T_init`)** into a single array for training.
#    - The optimization bounds are now defined for both **pump strength** (`S0`) and **temperature** (`T`).
# 
# 2. **Meshgrid for Visualization**:
#    - We create a meshgrid of **pump strength (`S0_grid`)** and **temperature (`T_grid`)** for visualization purposes. This meshgrid is used to predict the laser power (`Y_pred`) and the acquisition function (`ei_values`) across the parameter space.
# 
# 3. **Prediction and Visualization**:
#    - The GP model is used to predict laser power values across the grid. These predictions are reshaped into a 2D format for heatmap plotting.
#    - We plot the **GP's predicted laser power** as a heatmap, and overlay the actual sampled points (`params_init`) as red crosses.
#    - The **acquisition function** is also plotted as a heatmap, showing which regions of the parameter space the model finds promising.
# 
# 4. **Acquisition Function**:
#    - The next sampling point is suggested by the **`expected_improvement`** function, which returns the point expected to yield the most improvement.
# 
# 5. **Helper Functions**:
#    - **`plot_heatmap`**: This function plots the heatmap of the GP predictions or acquisition function. It highlights the suggested sample points and previously sampled points on the plot.
#    - **`plot_acquisition`**: Similar to the heatmap plot, but focuses on the acquisition function's values.
# 
# ### Suggested Exercises/Questions
# 
# 1. **Modify the Number of Parameters**:
#    - **Exercise**: Extend the optimization to include a third parameter, like **laser wavelength**. How does this affect the convergence and behavior of the Bayesian optimization? Plot the results in a 3D plot.
#      - **Goal**: To practice adding more dimensions and visualizing optimization in higher dimensions.
# 
# 2. **Explore Noise Sensitivity**:
#    - **Exercise**: Change the `noise_level` in the `laser_power` function and observe how the optimization behaves with increasing or decreasing noise. How does the GP model adapt?
#      - **Goal**: To explore how noise in measurements affects the Bayesian optimization process.
# 
# 3. **Adjust Acquisition Function**:
#    - **Question**: Why do we use **Expected Improvement** (EI) as the acquisition function? What other acquisition functions might be appropriate, and how would you implement them?
#    - **Exercise**: Implement and test the **Probability of Improvement (PI)** acquisition function. Compare its performance with EI.
#      - **Goal**: To understand the trade-offs between different acquisition strategies and experiment with alternative methods.
# 
# 4. **Plotting Convergence**:
#    - **Exercise**: Plot the convergence of the optimization by tracking the best objective value found in each iteration. How fast does the optimization converge for different noise levels?
#      - **Goal**: To understand the convergence properties of Bayesian optimization and analyze its performance over iterations.
# 
# 5. **Analyze the Influence of Initial Samples**:
#    - **Question**: How does the choice of initial samples affect the optimization results? Does random initialization work well, or would a more systematic approach (like Latin Hypercube Sampling) be better?
#    - **Exercise**: Initialize the parameters using **Latin Hypercube Sampling (LHS)** instead of random sampling, and compare the results with random initialization.
#      - **Goal**: To introduce students to more advanced initialization strategies for multi-parameter optimization.
