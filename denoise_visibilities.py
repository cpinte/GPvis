""" This file is responsible for using a GP to refine the visibility data of the target. """

from scipy.optimize import minimize
import glob
import celerite
from celerite import terms
from alive_progress import alive_bar
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, Any

# Constants
MAX_RETRIES = 5
MIN_LOG_C = 1e-6
DEFAULT_LOG_C = 100.0

def validate_data(data: Dict[str, Any]) -> None:
    """Validate the input data dictionary."""
    required_keys = ['u', 'v', 'Vis', 'Wgt', 'ant1', 'ant2', 'time']
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Missing required key: {key}")

    # Check for finite values and non-negative weights
    if not np.all(np.isfinite(data['Vis'])):
        raise ValueError("Non-finite values found in Vis")
    if not np.all(np.isfinite(data['Wgt'])):
        raise ValueError("Non-finite values found in Wgt")
    if not np.all(data['Wgt'] >= 0):
        raise ValueError("Negative weights found in Wgt")

def denoise_visibilities(target: str) -> None:
    """Denoise visibility data using Gaussian Process regression.

    Args:
        target: Name of the target to process (e.g., 'HD143006')
    """
    if not isinstance(target, str) or not target:
        raise ValueError("target must be a non-empty string")

    # defining input files
    search_name = target + "_cont_avg_split_*.vis.npz"
    npz_files = glob.glob(search_name)

    if not npz_files:
        raise FileNotFoundError(f"No files found matching pattern: {search_name}")

    # iterating over all npz files
    for npz_file in npz_files:
        if not os.path.exists(npz_file):
            print(f"Warning: File {npz_file} does not exist, skipping")
            continue

        try:
            # load in data from zipped files
            data = dict(np.load(npz_file))
            validate_data(data)

            u = data['u']
            v = data['v']
            Vis = data['Vis']
            Wgt = data['Wgt']
            ant1 = data['ant1']
            ant2 = data['ant2']
            time = data['time']

            # finding unique elements for ant1 and ant2
            list_ant1 = np.unique(ant1)
            list_ant2 = np.unique(ant2)

            # Count actual pairs to process
            actual_pairs = sum(1 for a1 in list_ant1 for a2 in list_ant2
                             if len(np.where((ant1 == a1) * (ant2 == a2))[0]) > 0)

            # creating load bar
            with alive_bar(actual_pairs, title=f"Processing {npz_file}") as bar:
                for a1 in list_ant1:
                    for a2 in list_ant2:
                        indices = np.where((ant1 == a1) * (ant2 == a2))[0]
                        if len(indices) == 0:
                            continue

                        bar()

                        # sorting time in ascending order
                        x = time[indices]
                        order = np.argsort(x)
                        x_sorted = x[order]
                        Vis_sorted = Vis[indices][order]
                        yerr = 1./np.sqrt(Wgt[indices][order])

                        real_vis = None
                        imag_vis = None

                        for i, y in enumerate([Vis_sorted.real, Vis_sorted.imag]):
                            log_c = DEFAULT_LOG_C
                            error_encountered = True
                            err_counter = 0

                            while error_encountered and err_counter < MAX_RETRIES:
                                if err_counter > 0:
                                    print(f"Retry {err_counter} for baseline {a1}:{a2}")
                                err_counter += 1
                                error_encountered = False

                                try:
                                    # running Gaussian Process
                                    white_noise = terms.JitterTerm(log_sigma=np.log(np.std(y)))
                                    real = terms.RealTerm(log_a=np.log(np.var(y)), log_c=-np.log(log_c))
                                    kernel = real + white_noise

                                    gp = celerite.GP(kernel, fit_mean=False)
                                    gp.compute(x_sorted, yerr)

                                    # fit for the maximum likelihood parameters
                                    initial_params = gp.get_parameter_vector()
                                    bounds = gp.get_parameter_bounds()

                                    soln = minimize(neg_log_like, initial_params,
                                                  jac=grad_neg_log_like,
                                                  method="L-BFGS-B",
                                                  bounds=bounds,
                                                  args=(y, gp))
                                    gp.set_parameter_vector(soln.x)

                                    # make the maximum likelihood prediction
                                    t = x
                                    mu, var = gp.predict(y, t, return_var=True)

                                    # check for NaN in output
                                    if np.any(np.isnan(mu)):
                                        raise ValueError("NaN values in prediction")

                                    if i == 0:
                                        real_vis = mu
                                    else:
                                        imag_vis = mu

                                except (ValueError, RuntimeError) as e:
                                    print(f"Error in GP fitting: {str(e)}")
                                    error_encountered = True
                                    log_c = max(log_c / 2.0, MIN_LOG_C)
                                    continue

                            if DEFAULT_LOG_C != log_c:
                                print(f"Kernel altered (log_c={log_c}) to improve output")

                        if real_vis is not None and imag_vis is not None:
                            Vis[indices] = real_vis + 1j * imag_vis
                        else:
                            print(f"Warning: Failed to process baseline {a1}:{a2}")

            print(f"{npz_file} computation finished")

            # save updated visibilities in npz file
            new_name = npz_file.replace('.vis.npz', '_updated.vis.npz')
            np.savez(new_name, **data)

        except Exception as e:
            print(f"Error processing {npz_file}: {str(e)}")
            continue

def neg_log_like(params, y, gp):
    """Negative log likelihood function for GP optimization."""
    gp.set_parameter_vector(params)
    return -gp.log_likelihood(y)

def grad_neg_log_like(params, y, gp):
    """Gradient of negative log likelihood function."""
    gp.set_parameter_vector(params)
    return -gp.grad_log_likelihood(y)[1]
