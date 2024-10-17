import numpy as np
import matplotlib.pyplot as plt
from SALib.sample import morris as ms
from SALib.analyze import morris as ma

import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def morris_sensitivity(func, params, N=10000, num_levels=10):
    """
    Perform Morris sensitivity analysis on the model's dominant eigenvalue using SALib.

    Parameters:
    - func: Function that computes the dominant eigenvalue given a parameter set.
    - params: Dictionary containing 'bounds' for each parameter.
    - N: Number of samples to generate for the analysis (default is 10,000).
    - num_levels: Number of levels for the Morris method (default is 10).

    Returns:
    - Si: Dictionary containing sensitivity analysis results, including 'mu', 'mu_star', and 'mu_star_conf'.
    """
    try:
        # Prepare problem statement for SALib
        param_ranges = params['bounds']
        if not param_ranges:
            logging.warning("Parameter bounds are empty.")
            return None

        problem = {
            'num_vars': len(param_ranges),
            'names': list(param_ranges.keys()),
            'bounds': np.array(list(param_ranges.values()))
        }

        # Generate samples using SALib
        param_values = ms.sample(problem, N=N, num_levels=num_levels)

        # Evaluate the system at each sample
        dom_eigenvalues = np.array([
            func({problem['names'][i]: param_values[j, i] for i in range(problem['num_vars'])})
            for j in range(param_values.shape[0])
        ])

        # Perform Morris sensitivity analysis
        Si = ma.analyze(problem, param_values, dom_eigenvalues, num_levels=num_levels, conf_level=0.95, print_to_console=False)

        param_names = Si['names']
        mu_values = Si['mu']
        mu_star = Si['mu_star']
        mu_star_conf = Si['mu_star_conf']

        os.makedirs("output/plots", exist_ok=True)

        plt.figure(figsize=(10, 6))
        plt.errorbar(param_names, mu_star, yerr=mu_star_conf, fmt='o', capsize=5, label='mu_star', color='b')
        plt.plot(param_names, mu_values, 'o', label='mu', color='r')
        plt.xlabel('Parameter')
        plt.ylabel('Sensitivity Indices')
        plt.xticks(rotation=90)
        plt.title('Morris Sensitivity Analysis of Dominant Eigenvalue')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.grid()

        output_path = "output/plots/morris_sensitivity_analysis.png"
        plt.savefig(output_path, dpi=300)
        plt.close()

        logging.info(f"Plot saved: {output_path}")

        return Si

    except Exception as e:
        logging.error(f"An error occurred during Morris sensitivity analysis: {e}")
        return None


def oat_sensitivity(jac_matrix, params):
    """
    Perform One-At-A-Time (OAT) Sensitivity Analysis using parameter bounds from params
    and plot the results.

    Parameters:
    - jac_matrix: Function to compute the Jacobian matrix.
    - params: Dictionary containing 'values' (baseline parameters) and 'bounds' for each parameter.

    Returns:
    - sensitivity: Dictionary with parameter names as keys and their sensitivity analysis results.
    """
    par = params['baseline']
    bounds = params['bounds']
    sensitivity = {}

    # Ensure output directory exists
    os.makedirs("output/plots", exist_ok=True)

    for p in par.keys():
        if p in bounds:
            lower_bound, upper_bound = bounds[p]
            baseline_value = par[p]

            param_range = np.linspace(lower_bound, upper_bound, 100)
            dom_eigenvalues = []

            # Evaluate the dominant eigenvalue for each value in the range
            for val in param_range:
                par[p] = val
                dominant_eigenvalue = np.linalg.eigvals(jac_matrix(par))[0]
                dom_eigenvalues.append(dominant_eigenvalue)

            # Reset the parameter to its baseline value after analysis
            par[p] = baseline_value
            sensitivity[p] = (param_range, dom_eigenvalues)

            # Plotting within the same function
            plt.figure()
            plt.plot(param_range, dom_eigenvalues, label=f'{p} Sensitivity')
            plt.axvline(x=baseline_value, color='r', linestyle='--', label='Baseline')
            plt.xlabel(f"{p} Value")
            plt.ylabel("Dominant Eigenvalue")
            plt.title(f"Sensitivity Analysis for {p}")
            plt.legend()
            plt.grid(True)
            output_path = f"output/plots/{p}_sensitivity.png"
            plt.savefig(output_path, dpi=300)
            plt.close()

            print(f"Plot saved: {output_path}")

    return sensitivity
