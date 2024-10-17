import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import argparse
import logging
import platform
import sys
from datetime import datetime

from jac import jacobian_A, jacobian_B, calculate_eigenvalues, jacobian_diagnostics
from SA import oat_sensitivity, morris_sensitivity


def load_params():
    params_path = Path(__file__).parent / 'params.yaml'
    with params_path.open('r') as f:
        return yaml.safe_load(f)


def log_system_info():
    """Logs system and environment details."""
    logging.info(f"System: {platform.system()} {platform.release()}")
    logging.info(f"Architecture: {platform.machine()}")
    logging.info(f"Python version: {sys.version}")
    logging.info(f"Numpy version: {np.__version__}")
    logging.info(f"Pandas version: {pd.__version__}")


def main():
    parser = argparse.ArgumentParser(description="Run Analysis.")

    parser.add_argument("--scenario",
                        choices=["baseline", "scenario_1"],
                        default="baseline",
                        help="Select the scenario to run")

    parser.add_argument("--model_type", nargs='*', choices=["A", "B"], default=["A"],
                        help="Select the model type(s) to run. Multiple choices allowed. Default is 'A'.")

    parser.add_argument('--plot', action='store_true', help='Enable plotting.')
    parser.add_argument('--oat', action='store_true', help='Enable One-At-A-Time sensitivity analysis.')
    parser.add_argument('--morris', action='store_true', help='Enable Morris sensitivity analysis.')

    args = parser.parse_args()

    log_filename = f"output/logs/{args.model_type}_{args.scenario}_{datetime.now().strftime('%Y%m%d_%H%M')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_filename)]
    )

    try:
        logging.info(f"Running analysis with scenario: {args.scenario}")
        logging.info(f"Selected model types: {args.model_type}")

        log_system_info()

        params = load_params()
        scenario_params = params[args.scenario]
        logging.info(f"Scenario parameters: {scenario_params}")

        # Select the appropriate Jacobian function
        jacobian_functions = {
            "A": jacobian_A,
            "B": jacobian_B
        }

        for model in args.model_type:
            if model in jacobian_functions:
                jacobian = jacobian_functions[model]
                logging.info("Jacobian diagnostics (Det, Trace, Condition number)", jacobian_diagnostics(jacobian(scenario_params)))
                # Calculate dominant eigenvalue
                dom = calculate_eigenvalues(jacobian(scenario_params))[0]
                logging.info(f"Dominant eigenvalue for model {model}: {dom}")
            else:
                logging.warning(f"Model type '{model}' is not recognized.")

        if args.plot:
            logging.info("Running plotting code.")
            # Run your plotting code here
            pass

        if args.oat:
            logging.info("Running One-At-A-Time (OAT) sensitivity analysis.")
            oat_sensitivity(jacobian, params)

        if args.morris:
            logging.info("Running Morris sensitivity analysis.")
            func = calculate_eigenvalues(jacobian)[0]
            morris_sensitivity(func, params)

    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
