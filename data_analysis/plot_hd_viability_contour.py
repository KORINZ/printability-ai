import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import seaborn as sns
import pickle
from scipy.interpolate import griddata
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# Set up plotting style
sns.set_context("talk", font_scale=1.4)
sns.set_style("ticks")
plt.rcParams["hatch.linewidth"] = 2.5


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


# Load models and data
hd_model = keras.models.load_model("paper_3_models/HD_value/mlp_model.keras")
with open("paper_3_models/HD_value/scaler_X.pkl", "rb") as f:
    hd_scaler_X = pickle.load(f)
with open("paper_3_models/HD_value/scaler_y.pkl", "rb") as f:
    hd_scaler_y = pickle.load(f)

viability_model = keras.models.load_model(
    "paper_3_models/PC12_cell_viability/PC12_model_with_rheo.keras"
)
with open("paper_3_models/PC12_cell_viability/scaler_X_with_rheo.pkl", "rb") as f:
    viability_scaler_X = pickle.load(f)
with open("paper_3_models/PC12_cell_viability/scaler_y_with_rheo.pkl", "rb") as f:
    viability_scaler_y = pickle.load(f)

# Load and process rheological data
power_law_data = pd.read_csv(
    "paper_3_models/csv_data_files/ALG_HA_cross_power_law_svr_predictions.csv"
)
rheology_data = pd.read_csv(
    "paper_3_models/csv_data_files/ALG_HA_GG_at_1hz_svr_predictions.csv"
)

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
alg_range = np.linspace(3.5, 4.5, 200)
ha_range = np.linspace(0.4, 1.0, 200)
alg_grid, ha_grid = np.meshgrid(alg_range, ha_range)

# Prepare grid points and interpolate rheological parameters
points = rheo_data[["ALG", "HA"]].values
interpolate_points = np.column_stack((alg_grid.ravel(), ha_grid.ravel()))

# Initialize X_pred array
X_pred = np.zeros((len(alg_grid.ravel()), 6))
X_pred[:, 0] = alg_grid.ravel()
X_pred[:, 1] = ha_grid.ravel()

# Interpolate rheological parameters
eta_0_values = np.log10(np.array(rheo_data["eta_0"].values))
m_values = np.log10(np.array(rheo_data["m"].values))
n_values = np.array(rheo_data["n"].values)
tan_delta_values = np.array(rheo_data["tan_delta"].values)

for i, values in enumerate([eta_0_values, m_values, n_values, tan_delta_values], 2):
    X_pred[:, i] = griddata(points, values, interpolate_points, method="cubic")
    nan_mask = np.isnan(X_pred[:, i])
    if np.any(nan_mask):
        X_pred[nan_mask, i] = griddata(
            points, values, interpolate_points[nan_mask], method="linear"
        )

# Get predictions
X_pred_scaled_hd = hd_scaler_X.transform(X_pred)
hd_pred_scaled = hd_model.predict(X_pred_scaled_hd, verbose=0)
hd_predictions = hd_scaler_y.inverse_transform(hd_pred_scaled).reshape(alg_grid.shape)

X_pred_scaled_viability = viability_scaler_X.transform(X_pred)
viability_pred_scaled = viability_model.predict(X_pred_scaled_viability, verbose=0)
viability_predictions = (
    viability_scaler_y.inverse_transform(viability_pred_scaled).reshape(alg_grid.shape)
    * 100
)

# Find the optimal point in the hatched region (HD ≤ 0.2 and viability ≥ 95%)
intersection_mask = np.logical_and(hd_predictions <= 0.2, viability_predictions >= 95)

# Find coordinates and values in the intersection region
valid_indices = np.where(intersection_mask)
if len(valid_indices[0]) > 0:
    # Get all values in the intersection
    valid_hd = hd_predictions[valid_indices]
    valid_viability = viability_predictions[valid_indices]
    valid_alg = alg_grid[valid_indices]
    valid_ha = ha_grid[valid_indices]

    # Find index of minimum HD
    min_hd_idx = np.argmin(valid_hd)

    # Get optimal point values
    optimal_hd = valid_hd[min_hd_idx]
    optimal_viability = valid_viability[min_hd_idx]
    optimal_alg = valid_alg[min_hd_idx]
    optimal_ha = valid_ha[min_hd_idx]

    # Print information about optimal point
    print("\nOptimal Formulation Point (minimum HD with viability ≥ 95%):")
    print(f"ALG concentration: {optimal_alg:.2f}% (w/v)")
    print(f"HA concentration: {optimal_ha:.2f}% (w/v)")
    print(f"HD value: {optimal_hd:.4f}")
    print(f"Cell viability: {optimal_viability:.2f}%")

    has_optimal_point = True
else:
    print("No points found in the intersection region.")
    has_optimal_point = False

# Find additional optimal points at 90% and 85% viability thresholds
# For 90% viability
intersection_mask_90 = np.logical_and(
    hd_predictions <= 0.2, viability_predictions >= 90
)
valid_indices_90 = np.where(intersection_mask_90)
if len(valid_indices_90[0]) > 0:
    valid_hd_90 = hd_predictions[valid_indices_90]
    valid_viability_90 = viability_predictions[valid_indices_90]
    valid_alg_90 = alg_grid[valid_indices_90]
    valid_ha_90 = ha_grid[valid_indices_90]

    min_hd_idx_90 = np.argmin(valid_hd_90)
    optimal_hd_90 = valid_hd_90[min_hd_idx_90]
    optimal_viability_90 = valid_viability_90[min_hd_idx_90]
    optimal_alg_90 = valid_alg_90[min_hd_idx_90]
    optimal_ha_90 = valid_ha_90[min_hd_idx_90]

    print("\nOptimal Formulation Point (minimum HD with viability ≥ 90%):")
    print(f"ALG concentration: {optimal_alg_90:.2f}% (w/v)")
    print(f"HA concentration: {optimal_ha_90:.2f}% (w/v)")
    print(f"HD value: {optimal_hd_90:.4f}")
    print(f"Cell viability: {optimal_viability_90:.2f}%")

    has_optimal_point_90 = True
else:
    print("No points found in the intersection region for 90% viability.")
    has_optimal_point_90 = False

# For 85% viability
intersection_mask_85 = np.logical_and(
    hd_predictions <= 0.2, viability_predictions >= 85
)
valid_indices_85 = np.where(intersection_mask_85)
if len(valid_indices_85[0]) > 0:
    valid_hd_85 = hd_predictions[valid_indices_85]
    valid_viability_85 = viability_predictions[valid_indices_85]
    valid_alg_85 = alg_grid[valid_indices_85]
    valid_ha_85 = ha_grid[valid_indices_85]

    min_hd_idx_85 = np.argmin(valid_hd_85)
    optimal_hd_85 = valid_hd_85[min_hd_idx_85]
    optimal_viability_85 = valid_viability_85[min_hd_idx_85]
    optimal_alg_85 = valid_alg_85[min_hd_idx_85]
    optimal_ha_85 = valid_ha_85[min_hd_idx_85]

    print("\nOptimal Formulation Point (minimum HD with viability ≥ 85%):")
    print(f"ALG concentration: {optimal_alg_85:.2f}% (w/v)")
    print(f"HA concentration: {optimal_ha_85:.2f}% (w/v)")
    print(f"HD value: {optimal_hd_85:.4f}")
    print(f"Cell viability: {optimal_viability_85:.2f}%")

    has_optimal_point_85 = True
else:
    print("No points found in the intersection region for 85% viability.")
    has_optimal_point_85 = False

fig, ax = plt.subplots(figsize=(10, 10))

cp = ax.pcolormesh(
    alg_grid,
    ha_grid,
    viability_predictions,
    cmap="RdYlGn",
    alpha=0.8,
    vmin=np.min(viability_predictions),
    vmax=np.max(viability_predictions),
    shading="auto",
)
cbar = fig.colorbar(cp, ax=ax, label="Cell Viability [%]", alpha=0.8)
cbar.ax.yaxis.label.set_fontweight("bold")

if np.any(intersection_mask):
    ax.contourf(
        alg_grid,
        ha_grid,
        intersection_mask,
        levels=[0.5, 1.5],
        colors="none",
        hatches=["//"],
        alpha=0.35,
    )

if np.any(intersection_mask_90):
    mask_90_only = np.logical_and(intersection_mask_90, ~intersection_mask)
    ax.contourf(
        alg_grid,
        ha_grid,
        mask_90_only,
        levels=[0.5, 1.5],
        colors="none",
        hatches=["\\\\"],
        alpha=0.325,
    )

if np.any(intersection_mask_85):
    mask_85_only = np.logical_and(
        intersection_mask_85, ~np.logical_or(intersection_mask, intersection_mask_90)
    )
    ax.contourf(
        alg_grid,
        ha_grid,
        mask_85_only,
        levels=[0.5, 1.5],
        colors="none",
        hatches=["||"],
        alpha=0.3,
    )

hd_levels = [0.19, 0.2, 0.21]
hd_colors = sns.color_palette("rainbow", n_colors=len(hd_levels))
hd_contour = ax.contour(
    alg_grid,
    ha_grid,
    hd_predictions,
    levels=hd_levels,
    colors=hd_colors,
    linewidths=3.5,
)
ax.clabel(
    hd_contour,
    inline=True,
    fmt="HD = %.2f",
    manual=[(4.4, 0.76), (3.8, 0.7), (3.7, 0.5)],
)

viability_levels = [75, 80, 85, 90, 95]
contour_linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))][::-1]
viability_contour = ax.contour(
    alg_grid,
    ha_grid,
    viability_predictions,
    levels=viability_levels,
    colors="k",
    linestyles=contour_linestyles,
    linewidths=3.5,
)
ax.clabel(
    viability_contour,
    inline=True,
    fmt="%1.0f%%",
    colors="k",
    manual=[(4.4, 0.5), (4.25, 0.55), (4.2, 0.625), (4.1, 0.7), (4.0, 0.75)],
)

if has_optimal_point:
    ax.plot(
        optimal_alg,
        optimal_ha,
        color="tab:red",
        marker="o",
        markersize=18,
        markeredgecolor="k",
        markeredgewidth=2,
    )

if has_optimal_point_90:
    ax.plot(
        optimal_alg_90,
        optimal_ha_90,
        color="tab:blue",
        marker="^",
        markersize=18,
        markeredgecolor="k",
        markeredgewidth=2,
    )

if has_optimal_point_85:
    ax.plot(
        optimal_alg_85,
        optimal_ha_85,
        color="tab:green",
        marker="s",
        markersize=18,
        markeredgecolor="k",
        markeredgewidth=2,
    )

legend_elements = []

# Add intersection regions to legend with clearer labels
legend_elements.append(
    mpatches.Patch(
        facecolor="none",
        edgecolor="black",
        hatch="//",
        label="HD ≤ 0.2, Viability ≥ 95%",
    )
)

# Add regions for 90% and 85% viability to legend
legend_elements.append(
    mpatches.Patch(
        facecolor="none",
        edgecolor="black",
        hatch=r"\\",
        label="HD ≤ 0.2, Viability 90-95%",
    )
)

legend_elements.append(
    mpatches.Patch(
        facecolor="none",
        edgecolor="black",
        hatch="||",
        label="HD ≤ 0.2, Viability 85-90%",
    )
)

# Add optimal points to legend if they exist
if has_optimal_point:
    legend_elements.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="tab:red",
            markeredgecolor="k",
            markersize=16,
            markeredgewidth=2,
            label=f"Min HD (95%): {optimal_hd:.3f}",
        )
    )

if has_optimal_point_90:
    legend_elements.append(
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="tab:blue",
            markeredgecolor="k",
            markersize=16,
            markeredgewidth=2,
            label=f"Min HD (90%): {optimal_hd_90:.3f}",
        )
    )

if has_optimal_point_85:
    legend_elements.append(
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="tab:green",
            markeredgecolor="k",
            markersize=16,
            markeredgewidth=2,
            label=f"Min HD (85%): {optimal_hd_85:.3f}",
        )
    )


ax.legend(
    handles=legend_elements,
    loc="upper center",
    bbox_to_anchor=(0.55, 1.3),
    ncol=2,
    columnspacing=0.5,
    fancybox=False,
    framealpha=1,
    edgecolor="k",
)


ax.set_xlabel("ALG [% (w/v)]", fontweight="bold")
ax.set_ylabel("HA [% (w/v)]", fontweight="bold")
ax.set_xlim(3.5, 4.5)
ax.set_ylim(0.4, 1.0)
ax.set_xticks(np.arange(3.5, 4.6, 0.25))
ax.set_xticklabels(["3.5", "3.75", "4.0", "4.25", "4.5"])

ax.grid(True, linestyle="-", alpha=0.5)

plt.tight_layout()
plt.savefig(
    "HD_viability_2D_contour.png", dpi=600, bbox_inches="tight", pad_inches=0.75
)
# crop top and right and bottom of the image
from PIL import Image

img = Image.open("HD_viability_2D_contour.png")
img = img.crop((0, 425, img.width - 425, img.height - 425))
img.save("HD_viability_2D_contour.png")
