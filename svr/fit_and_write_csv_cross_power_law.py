import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from scipy.optimize import curve_fit
import itertools

sns.set_context("talk", font_scale=1.25)
sns.set_style("ticks")


# Cross-Power Law model
def cross_power_law(gamma_dot, eta_inf, eta0, m, n):
    return eta_inf + (eta0 - eta_inf) / (1 + (m * gamma_dot) ** n)


def plot_viscosity_data_with_cross_power_law(
    folder_path, save_fig=False, use_filenames_as_legend=False
):

    folder_path = folder_path.replace("\\", "/")

    # Get a list of all Excel files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith(".xlsx")]

    # Define a set of markers for differentiation
    markers = itertools.cycle(
        [
            "o",
            "s",
            "D",
            "^",
            "v",
            ">",
            "<",
            "p",
            "*",
            "h",
            "H",
            "+",
            "x",
            "1",
            "2",
            "3",
            "4",
        ]
    )

    num_colors = len(files)
    palette = sns.color_palette("Spectral", num_colors)[::-1]

    sorted_file_names = sorted(files)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create or overwrite the CSV file with headers
    csv_file_path = (
        f"output/{os.path.basename(folder_path)}_cross_power_law_results.csv"
    )
    with open(csv_file_path, "w") as f:
        f.write("ink_type,eta_inf,eta_0,m,n\n")

    # Loop through files
    for file_name in sorted_file_names:
        file_path = os.path.join(folder_path, file_name)

        if use_filenames_as_legend:
            display_name = file_name.replace(".xlsx", "")
        else:
            display_name = (
                file_name.replace(".xlsx", "").replace("C_", " °C - ").replace("_", " ")
            )

        # Read the file
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)

            # Convert viscosity from mPas to Pas
            df = df.copy()
            df.loc[:, "η in Pas"] = df["η in mPas"] * 1e-3

            # Trim data where shear rate is less than 0.01 or greater than 1000
            df_filtered = df[df["ɣ̇ in 1/s"] >= 0.01]
            df_filtered = df_filtered[df_filtered["ɣ̇ in 1/s"] <= 1000]

            # Fit Cross Power Law model
            popt, _ = curve_fit(
                cross_power_law,
                df_filtered["ɣ̇ in 1/s"],
                df_filtered["η in Pas"],
                bounds=([1e-3, 0, 0, 0], [np.inf, np.inf, np.inf, 1]),
                sigma=(1.0 / df_filtered["ɣ̇ in 1/s"]) ** 0.5,
                maxfev=10000,
            )

            print(
                f"{display_name}: eta_inf={popt[0]:.4f}, eta0={popt[1]:.4f}, m={popt[2]:.4f}, n={popt[3]:.4f}"
            )

            # Write results to csv
            with open(csv_file_path, "a") as f:
                # Add header
                if os.stat(csv_file_path).st_size == 0:
                    f.write("ink_type,eta_inf,eta0,m,n\n")
                f.write(
                    f"{display_name},{popt[0]:.4f},{popt[1]:.4f},{popt[2]:.4f},{popt[3]:.4f}\n"
                )

            # Plot experimental data
            marker = next(markers)
            color = palette[sorted_file_names.index(file_name)]
            sns.scatterplot(
                x="ɣ̇ in 1/s",
                y="η in Pas",
                data=df_filtered,
                ax=ax,
                label=display_name,
                marker=marker,
                s=55,
                edgecolor="black",
                color=color,
                zorder=3,
            )

            # Plot fitted curve
            x_fit = np.logspace(
                np.log10(0.001),
                np.log10(10000),
                1000,
            )
            y_fit = cross_power_law(x_fit, *popt)
            ax.plot(x_fit, y_fit, color=color, linestyle="-", linewidth=1.15, zorder=2)
            ax.plot(
                x_fit, y_fit, color="black", linestyle="-", linewidth=2.45, zorder=1
            )

    # Setting axis labels and scales
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(
        r"Shear Rate ($\dot{{\gamma}}$) in s$^\mathbf{-1}$", fontweight="bold"
    )
    ax.set_ylabel(r"Apparent Viscosity ($\eta$) in Pa·s", fontweight="bold")
    ax.grid(True, which="both", ls="-", alpha=0.5)

    # Customize legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1],
        labels[::-1],
        edgecolor="black",
        facecolor="white",
        framealpha=1,
        fancybox=False,
        loc="best",
    )

    ax.set_xlim(0.01, 1e3)

    plt.tight_layout()

    if save_fig:
        if not os.path.exists("images"):
            os.makedirs("images")
        plt.savefig(
            f"images/{os.path.basename(csv_file_path)}.png",
            dpi=600,
            bbox_inches="tight",
        )
    if not save_fig:
        plt.show()


if __name__ == "__main__":
    folder_path = r""
    plot_viscosity_data_with_cross_power_law(
        folder_path, save_fig=False, use_filenames_as_legend=True
    )
