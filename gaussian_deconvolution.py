#!/usr/bin/env python3
"""
NMR Spectrum Gaussian Deconvolution with Multi-Restart and AIC/BIC Model Selection
Author: Goutam
Date: 2025-10-31
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from tqdm import tqdm
import os

# -----------------------------
# Define Gaussian and utilities
# -----------------------------
def gaussian(x, a, b, c):
    """Single Gaussian function"""
    return a * np.exp(-(x - b) ** 2 / (2 * c ** 2))

def multi_gauss(x, *params):
    """Sum of multiple Gaussian components"""
    n = len(params) // 3
    y = np.zeros_like(x)
    for i in range(n):
        a, b, c = params[3*i:3*i+3]
        y += gaussian(x, a, b, c)
    return y

def random_guess(x, y, npeaks, lower_bounds, upper_bounds):
    """Random initial guess strictly inside finite bounds."""
    p0 = []
    for i in range(npeaks):
        amp_min, center_min, width_min = lower_bounds[3*i:3*i+3]
        amp_max, center_max, width_max = upper_bounds[3*i:3*i+3]

        # Ensure finite bounds
        amp_max = min(amp_max, 10*np.max(y))
        width_max = min(width_max, (np.max(x)-np.min(x))/2)

        a = np.random.uniform(amp_min + 1e-6, amp_max - 1e-6)
        b = np.random.uniform(center_min + 1e-6, center_max - 1e-6)
        c = np.random.uniform(width_min + 1e-6, width_max - 1e-6)

        p0.extend([a, b, c])
    return np.array(p0)


# -----------------------------
# AIC/BIC computation
# -----------------------------
def compute_aic_bic(y, yfit, k):
    resid = y - yfit
    sse = np.sum(resid ** 2)
    n = len(y)
    aic = n * np.log(sse / n) + 2 * k
    bic = n * np.log(sse / n) + k * np.log(n)
    r2 = 1 - (np.sum(resid ** 2) / np.sum((y - np.mean(y)) ** 2))
    return aic, bic, r2

# -----------------------------
# Constrained fitting routine
# -----------------------------
def fit_gaussians(x, y, npeaks, n_iter=25, save_dir=None):
    """Perform multi-start Gaussian fitting with realistic bounds and save all iteration results."""
    xmin, xmax = np.min(x), np.max(x)
    xrange = xmax - xmin
    y_max = np.max(y)

    # ---- Finite and realistic bounds ----
    lower_bounds = []
    upper_bounds = []
    for _ in range(npeaks):
        lower_bounds += [0, xmin, 0.001 * xrange]
        upper_bounds += [2 * y_max, xmax, 0.075 * xrange] #Constrain peak width of gaussian components
#0.075*xrange implies the maximum peakwidth of gaussian components will be within 7.5% of spectral range
    best_loss = np.inf
    best_popt, best_yfit = None, None

    # Create folder for iteration-level results
    if save_dir is not None:
        iter_dir = os.path.join(save_dir, "iter_fits")
        os.makedirs(iter_dir, exist_ok=True)

    for i in range(1, n_iter + 1):
        p0 = random_guess(x, y, npeaks, lower_bounds, upper_bounds)
        if not np.all(np.isfinite(p0)):
            continue

        # --- Perform bounded least-squares optimization ---
        res = least_squares(lambda p: multi_gauss(x, *p) - y, p0,
                            bounds=(lower_bounds, upper_bounds),
                            max_nfev=50000, verbose=0)

        yfit = multi_gauss(x, *res.x)
        loss = np.sum((y - yfit) ** 2)
        amps = res.x[::3]
        if np.any(amps < 0.01 * y_max):
            continue

        aic, bic, r2 = compute_aic_bic(y, yfit, len(res.x))

        # --- Save plot for each iteration ---
        if save_dir is not None:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(x, y, 'k', lw=1.3, label='Experimental')
            ax.plot(x, yfit, 'r--', lw=2, label='Fit')
            
            # Plot individual Gaussians (fitted)
            for j in range(npeaks):
                a, b, c = res.x[3*j:3*j+3]
                ax.plot(x, gaussian(x, a, b, c), lw=1, alpha=0.7)
            
            # --- Mark starting centers (initial guesses) ---
            init_centers = p0[1::3]
            for xc in init_centers:
                ax.axvline(x=xc, color='gray', ls='--', lw=1.0, alpha=0.6)
                ax.annotate(f"{xc:.2f}",
                xy=(xc, (np.max(y)/2)),        # point at middle of plot
                xytext=(5, 0),             # offset 5 points above
                textcoords='offset points',
                ha='center', va='bottom',
                fontsize=7, color='gray', rotation=90)

            ax.set_xlabel("Chemical Shift (ppm)")
            ax.set_ylabel("Intensity (a.u.)")
            ax.set_title(f"Iter {i}/{n_iter} | {npeaks} Peaks | R²={r2:.4f}")
            ax.legend(fontsize=8, ncol=2)
            ax.invert_xaxis()
            plt.tight_layout()
            plt.savefig(os.path.join(iter_dir, f"iter_{i:02d}_fit.png"), dpi=300)
            plt.close()

        # --- Track best fit ---
        if loss < best_loss:
            best_loss = loss
            best_popt = res.x
            best_yfit = yfit

    if best_popt is None:
        return None, None, None, None, None

    aic, bic, r2 = compute_aic_bic(y, best_yfit, len(best_popt))
    return best_popt, best_yfit, aic, bic, r2


# -----------------------------
# Visualization
# -----------------------------
def plot_fit(x, y, popt, yfit, npeaks, save_dir):
    """Plot data, fit, components, residuals, and label peak centers & areas."""
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                            gridspec_kw={'height_ratios': [3, 1]})

    axs[0].plot(x, y, 'k', lw=1.3, label='Experimental')
    axs[0].plot(x, yfit, 'r--', lw=2, label=f'Fit ({npeaks} peaks)')

    peak_data = []  # store (index, amplitude, center, width, area)
    for i in range(npeaks):
        a, b, c = popt[3*i:3*i+3]
        peak = gaussian(x, a, b, c)
        area = a * c * np.sqrt(2 * np.pi)

        peak_data.append((i + 1, a, b, c, area))
        axs[0].plot(x, peak, lw=1.1, alpha=0.8, label=f'Peak {i+1}')

        # Place label near the peak maximum
        ymax = np.max(peak)
        xpos = b
        ypos = ymax * 1.05
        axs[0].text(xpos, ypos, f"{b:.2f} ppm\nArea={area:.2f}",
                    fontsize=7, ha='center', va='bottom', rotation=0)

    axs[0].set_ylabel("Intensity (a.u.)")
    axs[0].legend(fontsize=7, ncol=3, loc='upper right', frameon=False)
    axs[0].set_title(f"{npeaks}-Peak Gaussian Fit")

    # Residuals
    resid = y - yfit
    axs[1].plot(x, resid, 'b', lw=1)
    axs[1].axhline(0, color='gray', ls='--', lw=1)
    axs[1].set_xlabel("Chemical Shift (ppm)")
    axs[1].set_ylabel("Residual")

    for ax in axs:
        ax.invert_xaxis()

    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"fit_{npeaks}_peaks.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    # Save peak parameters as CSV
    peak_df = pd.DataFrame(peak_data,
                           columns=["Peak#", "Amplitude", "Center (ppm)", "Width", "Area"])
    csv_path = os.path.join(save_dir, f"fit_{npeaks}_peaks_params.csv")
    peak_df.to_csv(csv_path, index=False)

# -----------------------------
# Main Program
# -----------------------------
if __name__ == "__main__":
    file_path = input("Enter path of CSV file: ").strip()
    min_peaks = int(input("Enter minimum number of peaks to test: ").strip())
    max_peaks = int(input("Enter maximum number of peaks to test: ").strip())
    n_iter = int(input("Enter number of random restarts per model (e.g. 25): ").strip())

    data = pd.read_csv(file_path)
    x, y = data.iloc[:, 0].values, data.iloc[:, 1].values
    print(f"✅ Loaded {len(x)} data points from {os.path.basename(file_path)}")

    # === Main results directory ===
    save_dir = os.path.splitext(file_path)[0] + "_auto_results"
    os.makedirs(save_dir, exist_ok=True)

    results = []

    for npeaks in tqdm(range(min_peaks, max_peaks + 1),
                       desc="Fitting models", colour="green"):

        # Create subfolder for this peak count
        peak_dir = os.path.join(save_dir, f"{npeaks}_peaks")
        os.makedirs(peak_dir, exist_ok=True)

        # Pass peak-specific folder into fitting routine
        popt, yfit, aic, bic, r2 = fit_gaussians(
            x, y, npeaks, n_iter=n_iter, save_dir=peak_dir
        )

        if popt is None:
            print(f"❌ No valid fit for {npeaks} peaks (trivial or failed).")
            continue

        print(f"\n✅ Fitted {npeaks} peaks | AIC={aic:.2f} | BIC={bic:.2f} | R²={r2:.4f}")
        results.append((npeaks, aic, bic, r2))
        plot_fit(x, y, popt, yfit, npeaks, peak_dir)


     # --- Final high-quality plot with peak-center ticks, smart labels, and R² annotation ---
        plt.figure(figsize=(7, 5), dpi=600)

        # Plot experimental and fitted curves
        plt.plot(x, y, 'k', lw=1.3, label='Experimental')
        plt.plot(x, yfit, 'r--', lw=2, label=f'Fit ({npeaks} peaks)')

        ax = plt.gca()

        # Plot individual Gaussians and mark centers
        for i in range(npeaks):
            a, b, c = popt[3*i:3*i+3]
            gauss_curve = gaussian(x, a, b, c)
            plt.plot(x, gauss_curve, lw=1, alpha=0.7)

            # Peak position and intensity
            peak_y = gaussian(b, a, b, c)

            # --- Draw short vertical tick mark (like NMR peak marker) ---
            tick_len = 0.03 * np.max(y)  # relative to intensity scale
            plt.plot([b, b], [peak_y - tick_len/2, peak_y + tick_len/2],
                    color='gray', lw=0.8)

            # --- Determine vertical position for label (avoid overlap) ---
            offset = 0.1 * np.max(y)
            label_y = peak_y + offset

            # --- Draw ledger line if label offset is significant ---
            #if label_y > peak_y * 1.1:
                #plt.plot([b, b], [peak_y, label_y * 0.98],
                        #color='gray', lw=0.6, alpha=0.6)

            # --- Add rotated label at top of each peak ---
            #plt.text(b, label_y, f"{b:.2f}",
                    #rotation=90, ha='center', va='bottom',
                    #fontsize=8, color='blue', clip_on=False)

        # --- Add R² annotation inside the plot (top-left corner) ---
        plt.text(0.02, 0.95, f"R² = {r2:.4f}",
                transform=ax.transAxes,
                ha='left', va='top', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

        # --- Cosmetic adjustments ---
        ax.set_yticklabels([])       # hide y-axis numbers but keep ticks
        ax.set_ylabel("")            # remove axis label
        ax.set_xlabel("Chemical Shift (ppm)")
        ax.invert_xaxis()
        ax.tick_params(axis='y', length=4, width=0.8)  # show small y-ticks for structure
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()

        # --- Save as high-resolution PNG ---
        plt.savefig(os.path.join(peak_dir, f"final_fit_{npeaks}_peaks.png"),
                    dpi=1200, bbox_inches='tight')
        plt.close()


    # === Summary across models ===
    if results:
        df = pd.DataFrame(results, columns=["Peaks", "AIC", "BIC", "R²"])
        df.to_csv(os.path.join(save_dir, "model_comparison.csv"), index=False)
        best_aic = df.loc[df['AIC'].idxmin()]
        best_bic = df.loc[df['BIC'].idxmin()]

        plt.figure(figsize=(7,5))
        plt.plot(df["Peaks"], df["AIC"], 'o-', label="AIC")
        plt.plot(df["Peaks"], df["BIC"], 's-', label="BIC")
        plt.xlabel("Number of Peaks")
        plt.ylabel("Criterion Value")
        plt.legend()
        plt.grid(True)
        plt.title("AIC/BIC Model Comparison")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "AIC_BIC_trend.png"), dpi=300)
        plt.close()

        print("\n📊 Best Models:")
        print(f"  AIC minimum at {int(best_aic['Peaks'])} peaks (AIC={best_aic['AIC']:.2f})")
        print(f"  BIC minimum at {int(best_bic['Peaks'])} peaks (BIC={best_bic['BIC']:.2f})")
        print(f"\n💾 All results saved in: {save_dir}")
    else:
        print("\n❌ No valid fits found (possibly all contained trivial peaks).")
