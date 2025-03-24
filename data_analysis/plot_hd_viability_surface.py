import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import seaborn as sns
import pickle
from matplotlib.colors import Normalize
from scipy.interpolate import griddata
import pandas as pd

# Set up plotting style
sns.set_context("talk", font_scale=1.4)
sns.set_style("ticks")


def extract_concentrations(ink_type):
    if "_HA" not in ink_type:
        alg = float(ink_type.replace("ALG", ""))
        ha = 0.0
    else:
        parts = ink_type.split("_")
        alg = float(parts[0].replace("ALG", ""))
        ha = float(parts[1].replace("HA", ""))
    return pd.Series({"ALG": alg, "HA": ha})


def standardize_ink_type(ink_type):
    return ink_type.replace("-", "_")


# Load HD model and scalers
hd_model = keras.models.load_model("paper_3_models/HD_value/mlp_model.keras")
with open("paper_3_models/HD_value/scaler_X.pkl", "rb") as f:
    hd_scaler_X = pickle.load(f)
with open("paper_3_models/HD_value/scaler_y.pkl", "rb") as f:
    hd_scaler_y = pickle.load(f)

# Load cell viability model and scalers
viability_model = keras.models.load_model(
    "paper_3_models/PC12_cell_viability/PC12_model_with_rheo.keras"
)
with open("paper_3_models/PC12_cell_viability/scaler_X_with_rheo.pkl", "rb") as f:
    viability_scaler_X = pickle.load(f)
with open("paper_3_models/PC12_cell_viability/scaler_y_with_rheo.pkl", "rb") as f:
    viability_scaler_y = pickle.load(f)

# Load rheological data
power_law_data = pd.read_csv(
    "paper_3_models/csv_data_files/ALG_HA_cross_power_law_svr_predictions.csv"
)
rheology_data = pd.read_csv(
    "paper_3_models/csv_data_files/ALG_HA_GG_at_1hz_svr_predictions.csv"
)

# Standardize ink type format and merge rheological data
power_law_data["ink_type"] = power_law_data["ink_type"].apply(standardize_ink_type)
rheology_data["ink_type"] = rheology_data["ink_type"].apply(standardize_ink_type)
power_law_data[["ALG", "HA"]] = power_law_data["ink_type"].apply(extract_concentrations)

rheo_data = pd.merge(
    power_law_data[["ink_type", "ALG", "HA", "eta_0", "m", "n"]],
    rheology_data[["ink_type", "tan_delta"]],
    on="ink_type",
    how="left",
)

# Create mesh grid
alg_range = np.linspace(2.0, 4.5, 100)
ha_range = np.linspace(0.0, 1.0, 100)
alg_grid, ha_grid = np.meshgrid(alg_range, ha_range)

# Prepare grid points and interpolate rheological parameters
points = rheo_data[["ALG", "HA"]].values
interpolate_points = np.column_stack((alg_grid.ravel(), ha_grid.ravel()))

# Initialize X_pred array for all 6 inputs
X_pred = np.zeros((len(alg_grid.ravel()), 6))
X_pred[:, 0] = alg_grid.ravel()
X_pred[:, 1] = ha_grid.ravel()

# Interpolate rheological parameters
eta_0_values = np.log10(rheo_data["eta_0"].values)
m_values = np.log10(rheo_data["m"].values)
n_values = rheo_data["n"].values
tan_delta_values = rheo_data["tan_delta"].values

# Interpolate each parameter using cubic interpolation
X_pred[:, 2] = griddata(points, eta_0_values, interpolate_points, method="cubic")
X_pred[:, 3] = griddata(points, m_values, interpolate_points, method="cubic")
X_pred[:, 4] = griddata(points, n_values, interpolate_points, method="cubic")
X_pred[:, 5] = griddata(points, tan_delta_values, interpolate_points, method="cubic")

# Fill NaN values using linear interpolation
for i in range(2, 6):
    nan_mask = np.isnan(X_pred[:, i])
    if np.any(nan_mask):
        X_pred[nan_mask, i] = griddata(
            points, X_pred[~nan_mask, i], interpolate_points[nan_mask], method="linear"
        )

# Get HD predictions
X_pred_scaled_hd = hd_scaler_X.transform(X_pred)
hd_pred_scaled = hd_model.predict(X_pred_scaled_hd, verbose=0)
hd_predictions = hd_scaler_y.inverse_transform(hd_pred_scaled).reshape(alg_grid.shape)

# Get viability predictions
X_pred_scaled_viability = viability_scaler_X.transform(X_pred)
viability_pred_scaled = viability_model.predict(X_pred_scaled_viability, verbose=0)
viability_predictions = (
    viability_scaler_y.inverse_transform(viability_pred_scaled).reshape(alg_grid.shape)
    * 100
)

# Load experimental cell viability data
cell_data = pd.read_csv("paper_3_models/csv_data_files/PC-12_individual_data.csv")
cell_data = cell_data[cell_data["flow_rate_uL_per_s"] == 2]
cell_data = cell_data[cell_data["alg_concentration"] <= 4.5]
cell_data["relative_cell_viability"] = cell_data["relative_cell_viability"].clip(0, 100)

# Create the plot
fig = plt.figure(figsize=(11, 12))
ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

# Create surface plot with viability coloring
norm = Normalize(vmin=np.min(viability_predictions), vmax=np.max(viability_predictions))
surf = ax.plot_surface(
    alg_grid,
    ha_grid,
    hd_predictions,
    facecolors=plt.cm.RdYlGn(norm(viability_predictions)),
    alpha=0.8,
    linewidth=0.125,
    edgecolor="w",
)

# Create contour lines for cell viability
fig_contour = plt.figure()
ax_contour = fig_contour.add_subplot(111)
viability_levels = [75, 80, 85, 90, 95]
contour_linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))][::-1]

CS = ax_contour.contour(
    alg_grid, ha_grid, viability_predictions, levels=viability_levels
)
plt.close(fig_contour)

# Plot viability contour lines on 3D surface
for level, linestyle in zip(viability_levels, contour_linestyles):
    paths_vertices = CS.allsegs[viability_levels.index(level)]

    for vertices in paths_vertices:
        vertices = np.array(vertices)
        x, y = vertices[:, 0], vertices[:, 1]

        points = np.column_stack((x, y))
        points_rheo = np.zeros((len(points), 6))
        points_rheo[:, :2] = points

        for i in range(2, 6):
            points_rheo[:, i] = griddata(
                interpolate_points, X_pred[:, i], points, method="linear"
            )

        points_scaled = hd_scaler_X.transform(points_rheo)
        z_scaled = hd_model.predict(points_scaled, verbose=0)
        z = hd_scaler_y.inverse_transform(z_scaled).flatten()

        line = ax.plot(x, y, z, color="k", linewidth=1.75, linestyle=linestyle)[0]

        if len(x) > 0:
            idx = int(len(x) * 0)
            ax.text(
                x[idx],
                y[idx] - 0.085,
                z[idx] + 0.01,
                f"{level}%",
                fontsize=21,
                color="black",
                horizontalalignment="center",
                verticalalignment="center",
            )

# Add HD value contours with different colors
hd_levels = [0.2, 0.25, 0.3, 0.35]
hd_colors = (
    ["#85feb6"]
    + ["#f0ffff"]
    + sns.color_palette("Spectral_r", n_colors=3)[1:2]
    + ["#dda0dd"]
)
hd_linestyles = ["-" for _ in hd_levels]

fig_hd_contour = plt.figure()
ax_hd_contour = fig_hd_contour.add_subplot(111)
CS_hd = ax_hd_contour.contour(alg_grid, ha_grid, hd_predictions, levels=hd_levels)
plt.close(fig_hd_contour)

# Plot HD contour lines on 3D surface
for level, color, linestyle in zip(hd_levels, hd_colors, hd_linestyles):
    paths_vertices = CS_hd.allsegs[hd_levels.index(level)]

    for vertices in paths_vertices:
        vertices = np.array(vertices)
        x, y = vertices[:, 0], vertices[:, 1]
        z = np.full_like(x, level)

        line = ax.plot(x, y, z, color=color, linewidth=1.75, linestyle=linestyle)[0]

        if len(x) > 0:
            idx = int(len(x) * 0.5)  # Place label in middle of line
            if level == 0.2:
                ax.text(
                    x[idx],
                    y[idx],
                    z[idx] * 1.2,
                    f"HD = {level:.2f}",
                    color=color,
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    fontsize=21,
                )
            else:
                ax.text(
                    x[idx],
                    y[idx],
                    z[idx] * 1.15,
                    f"HD = {level:.2f}",
                    color=color,
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    fontsize=21,
                )

# Define corner coordinates
corners = [
    (alg_range[0], ha_range[0]),  # bottom-left
    (alg_range[0], ha_range[-1]),  # top-left
    (alg_range[-1], ha_range[0]),  # bottom-right
    (alg_range[-1], ha_range[-1]),  # top-right
]

# Get corresponding z-values for corner coordinates
corner_z_values = []
for alg, ha in corners:
    # Find closest indices in the grid
    alg_idx = np.abs(alg_range - alg).argmin()
    ha_idx = np.abs(ha_range - ha).argmin()

    # Get the z-value directly from our precomputed predictions
    z_value = hd_predictions[ha_idx, alg_idx]
    corner_z_values.append(z_value)

# Add blue pentagrams at corners with increased size
for (alg, ha), z in zip(corners, corner_z_values):
    ax.scatter(
        alg,
        ha,
        z,
        marker="*",
        s=800,
        color="none",
        edgecolors="tab:blue",
        linewidth=3,
        alpha=0.8,
        zorder=100,
    )

# Customize the plot
ax.set_xlabel("ALG [% (w/v)]", fontweight="bold", labelpad=18)
ax.set_ylabel("HA [% (w/v)]", fontweight="bold", labelpad=18)
ax.set_zlabel("HD [-]", fontweight="bold", labelpad=15)

# Set axis limits and ticks
ax.set_xticks([2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1.0"])
ax.set_zticks([0.5, 0.4, 0.3, 0.2, 0.1])
ax.set_zticklabels(["0.5", "0.4", "0.3", "0.2", "0.1"])
ax.set_zlim(min(ax.get_zlim()), max(ax.get_zlim()) + 0.025)

ax.set_title("\nAverage HD Value Prediction Surface\nColored by Cell Viability", y=0.98)

# Adjust tick parameters
ax.tick_params(axis="x", pad=4)
ax.tick_params(axis="y", pad=4)
ax.tick_params(axis="z", pad=6)

# Set view angle
ax.invert_zaxis()
ax.view_init(20, -135)

# Add colorbar
m = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.RdYlGn)
# horizontal colorbar on top

cbar = plt.colorbar(
    m, ax=ax, aspect=25, shrink=0.7, alpha=0.8, orientation="horizontal", pad=-0.025
)
cbar.ax.invert_xaxis()
cbar.set_label("Cell Viability [%]", fontweight="bold")

cbar.set_ticks([65, 70, 75, 80, 85, 90, 95])
cbar.set_ticklabels(["65", "70", "75", "80", "85", "90", "95"])

# Add legend for blue pentagrams
from matplotlib.lines import Line2D

legend_elements = [
    Line2D(
        [0],
        [0],
        marker="*",
        color="w",
        markerfacecolor="none",
        markeredgecolor="tab:blue",
        markersize=26,
        markeredgewidth=2.5,
        label="Edge Points",
        alpha=0.8,
    )
]
ax.legend(
    handles=legend_elements,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.91),
    fancybox=False,
    edgecolor="black",
    framealpha=1,
)

# Save the figure
plt.savefig(
    "combined_HD_viability_surface.png",
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.48,
)

# crop top and bottom of the image
from PIL import Image

img = Image.open("combined_HD_viability_surface.png")
img = img.crop((25, 400, img.width - 300, img.height - 200))
img.save("combined_HD_viability_surface.png")
