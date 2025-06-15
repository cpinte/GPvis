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
from pathlib import Path

# Constants
MAX_RETRIES = 5
MIN_LOG_C = 1e-6
DEFAULT_LOG_C = 100.0

def validate_data(data: Dict[str, Any]) -> None:
    """Validate the input data dictionary."""
    required_keys = ['u', 'v', 'Vis', 'Wgt', 'ant1', 'ant2', 'time', 'spwid']
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

def validate_denoising(original: np.ndarray, denoised: np.ndarray,
                      weights: np.ndarray) -> Dict[str, float]:
    """Validate the quality of denoising.

    Args:
        original: Original complex visibilities
        denoised: Denoised complex visibilities
        weights: Visibility weights

    Returns:
        Dictionary containing validation metrics
    """
    # Calculate weighted residuals
    residuals = original - denoised
    weighted_residuals = residuals * np.sqrt(weights)

    # Calculate metrics
    metrics = {
        'mean_residual': np.mean(np.abs(residuals)),
        'std_residual': np.std(np.abs(residuals)),
        'weighted_chi2': np.sum(np.abs(weighted_residuals)**2),
        'snr_improvement': np.mean(np.abs(denoised)) / np.mean(np.abs(original))
    }

    return metrics

def is_sorted(x: np.ndarray, rtol: float = 1e-5) -> bool:
    """Check if array is sorted in ascending order.

    Args:
        x: Array to check
        rtol: Relative tolerance for floating point comparison

    Returns:
        True if array is sorted, False otherwise
    """
    # For floating point numbers, we need to handle small numerical differences
    if np.issubdtype(x.dtype, np.floating):
        return np.all(np.diff(x) >= -rtol * np.abs(x[:-1]))
    return np.all(np.diff(x) >= 0)

def plot_visibilities(x: np.ndarray, y_real: np.ndarray, y_imag: np.ndarray,
                     mu_real: np.ndarray, mu_imag: np.ndarray,
                     std_real: np.ndarray, std_imag: np.ndarray,
                     baseline: str, output_dir: str) -> None:
    """Plot and save visibility data for a baseline.

    Args:
        x: Time points
        y_real: Original real visibility data
        y_imag: Original imaginary visibility data
        mu_real: GP prediction mean for real component
        mu_imag: GP prediction mean for imaginary component
        std_real: GP prediction std for real component
        std_imag: GP prediction std for imaginary component
        baseline: Baseline identifier (e.g., "1:2")
        output_dir: Directory to save plots
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot real component
    ax1.plot(x, y_real, 'o', color='blue', markersize=2, alpha=0.5, label='Original')
    ax1.plot(x, mu_real, color='red', linewidth=1, label='GP Prediction')
    ax1.fill_between(x, mu_real + std_real, mu_real - std_real,
                    color='red', alpha=0.2, edgecolor='none',
                    label='1σ Uncertainty')
    ax1.set_ylabel('Real Visibility')
    ax1.set_title(f'Baseline {baseline}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot imaginary component
    ax2.plot(x, y_imag, 'o', color='blue', markersize=2, alpha=0.5, label='Original')
    ax2.plot(x, mu_imag, color='red', linewidth=1, label='GP Prediction')
    ax2.fill_between(x, mu_imag + std_imag, mu_imag - std_imag,
                    color='red', alpha=0.2, edgecolor='none',
                    label='1σ Uncertainty')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Imaginary Visibility')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save plot
    filename = f'baseline_{baseline}.pdf'
    plt.savefig(os.path.join(output_dir, filename),
                bbox_inches='tight', dpi=300)
    plt.close()

def denoise_visibilities(target: str, plot: bool = False) -> None:
    """Denoise visibility data using Gaussian Process regression.

    Args:
        target: Name of the target to process (e.g., 'HD143006')
        plot: Whether to generate and save plots
    """
    if not isinstance(target, str) or not target:
        raise ValueError("target must be a non-empty string")

    # Create output directory for plots if needed
    if plot:
        output_dir = f"{target}_plots"
        os.makedirs(output_dir, exist_ok=True)

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
            spwid = data['spwid']

            # finding unique elements for ant1, ant2 and spwid
            list_ant1 = np.unique(ant1)
            list_ant2 = np.unique(ant2)
            list_spwid = np.unique(spwid)

            # Count actual pairs to process
            actual_pairs = sum(1 for a1 in list_ant1 for a2 in list_ant2
                             if len(np.where((ant1 == a1) * (ant2 == a2))[0]) > 0)

            # creating load bar
            with alive_bar(actual_pairs, title=f"Processing {npz_file}") as bar:
                for a1 in list_ant1:
                    for a2 in list_ant2:
                        bar()

                        for spw in list_spwid:
                            indices = np.where((ant1 == a1) * (ant2 == a2) * (spwid == spw))[0]

                            if len(indices) == 0:
                                continue

                            # sorting time in ascending order
                            x = time[indices]

                            # Check if data is already sorted
                            if not is_sorted(x): # This should never happen now
                                print(f"Data for baseline {a1}:{a2} needs sorting")
                                order = np.argsort(x)
                                x_sorted = x[order]
                                Vis_sorted = Vis[indices][order]
                                yerr = 1./np.sqrt(Wgt[indices][order])
                            else:
                                # Data is already sorted, no need to reorder
                                order = np.arange(len(x))
                                x_sorted = x
                                Vis_sorted = Vis[indices]
                                yerr = 1./np.sqrt(Wgt[indices])

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
                                        t = x_sorted  # Predict at sorted times
                                        mu, var = gp.predict(y, t, return_var=True)
                                        std = np.sqrt(var)

                                        # check for NaN in output
                                        if np.any(np.isnan(mu)):
                                            raise ValueError("NaN values in prediction")

                                        # Reorder predictions to match original data order
                                        mu_ordered = np.zeros_like(mu)
                                        mu_ordered[order] = mu
                                        std_ordered = np.zeros_like(std)
                                        std_ordered[order] = std

                                        if i == 0:
                                            real_vis = mu_ordered
                                            real_std = std_ordered
                                            real_data = y
                                        else:
                                            imag_vis = mu_ordered
                                            imag_std = std_ordered
                                            imag_data = y

                                        # Plot if requested and we have both components
                                        if plot and real_vis is not None and imag_vis is not None:
                                            plot_visibilities(x, real_data, imag_data,
                                                              real_vis, imag_vis,
                                                              real_std, imag_std,
                                                              f"{a1}:{a2}", output_dir)

                                    except(ValueError, RuntimeError) as e:
                                        print(f"Error in GP fitting: {str(e)}")
                                        error_encountered = True
                                        log_c = max(log_c / 2.0, MIN_LOG_C)
                                        continue

                                if DEFAULT_LOG_C != log_c:
                                    print(f"Kernel altered (log_c={log_c}) to improve output")

                            if real_vis is not None and imag_vis is not None:
                                denoised_vis = real_vis + 1j * imag_vis
                                original_vis = Vis[indices]

                                # Validate denoising
                                metrics = validate_denoising(original_vis, denoised_vis, Wgt[indices])
                                if metrics['snr_improvement'] < 1.0:
                                    print(f"Warning: Denoising may have degraded signal for baseline {a1}:{a2}")
                                    print(f"SNR improvement: {metrics['snr_improvement']:.2f}")

                                Vis[indices] = denoised_vis
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
