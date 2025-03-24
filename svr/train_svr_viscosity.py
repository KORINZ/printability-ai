import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict
from sklearn.metrics import r2_score
import pickle
import os

# Set up plotting style
sns.set_context("talk", font_scale=1.4)
sns.set_style("ticks")

data_path = "output/ALG_HA_new_cross_power_law_results.csv"

CROSS_VALIDATION_FOLDS = 10

ink_type = os.path.basename(data_path).split("_cross_power_law_results.csv")[0]

INK_NAME = " ".join(ink_type.split("_")[0:2]).replace(" ", "-")

# Read and preprocess data
data = pd.read_csv(data_path)


def extract_concentrations(ink_type):
    """Extract ALG and HA concentrations from the ink type."""
    parts = ink_type.split("_")
    print(parts)
    if "HA" in parts[1]:
        alg, ha = float(parts[0].replace("ALG", "")), float(parts[1].replace("HA", ""))
    else:
        alg = float(parts[0].replace("ALG", ""))
        ha = 0.0
    return pd.Series({"ALG": alg, "HA": ha})


# Extract ALG and HA concentrations
data[["ALG", "HA"]] = data["ink_type"].apply(extract_concentrations)

# Prepare input features and target variables
X = data[["ALG", "HA"]].values
Y = data[["eta_0", "m", "n"]].values

# Apply log transformation to Y
Y = np.log(Y)

# Scale the features
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# Initialize scalers for each target variable
scalers_Y = [StandardScaler() for _ in range(3)]


def train_svr_cv(X: np.ndarray, Y: np.ndarray) -> tuple[SVR, np.float64, dict]:
    """Train an SVR model using grid search with cross-validation."""
    param_grid = {
        "C": np.linspace(100, 5000, 10),
        "gamma": np.linspace(0.01, 0.1, 10),
        "epsilon": np.linspace(0.01, 0.1, 10),
    }

    svr_grid = GridSearchCV(
        estimator=SVR(),
        param_grid=param_grid,
        cv=CROSS_VALIDATION_FOLDS,
        verbose=2,
        n_jobs=-1,
    )

    svr_grid.fit(X, Y)

    best_model = svr_grid.best_estimator_
    best_params = svr_grid.best_params_

    mse_scores = -cross_val_score(
        best_model, X, Y, scoring="neg_mean_squared_error", cv=CROSS_VALIDATION_FOLDS
    )
    mean_mse = np.mean(mse_scores).round(4)

    return best_model, mean_mse, best_params


# Train models for all target variables
svr_models = []
mse_scores = []
best_params_list = []
rmse_scores_original_space = []

for i, name in enumerate(["eta_0", "m", "n"]):
    y = Y[:, i].reshape(-1, 1)
    y_scaled = scalers_Y[i].fit_transform(y)

    svr_model, mse_score, best_params = train_svr_cv(X_scaled, y_scaled.ravel())
    svr_models.append(svr_model)
    mse_scores.append(mse_score)
    best_params_list.append(best_params)

    # Calculate RMSE in original space
    y_pred_scaled = cross_val_predict(
        svr_model, X_scaled, y_scaled.ravel(), cv=CROSS_VALIDATION_FOLDS
    )
    y_pred = scalers_Y[i].inverse_transform(y_pred_scaled.reshape(-1, 1))
    y_pred = np.exp(y_pred)
    y_actual = np.exp(y)
    rmse_original = np.sqrt(np.mean((y_actual - y_pred) ** 2))
    rmse_scores_original_space.append(rmse_original)

print(f"Total data points: {len(data)}")
for i, name in enumerate(["eta_0", "m", "n"]):
    print(f"Best parameters for SVR model for {name}: {best_params_list[i]}")
    print(
        f"Cross-validated Root Mean Squared Error for {name} in original space: {rmse_scores_original_space[i]:.3f}"
    )


def plot_3d_surface(model, title, scaler_X, scaler_Y, data, index, save_fig=False):
    """Plot a 3D surface plot of the predicted values for a given feature."""
    alg_range = np.linspace(data["ALG"].min(), data["ALG"].max(), 25)
    ha_range = np.linspace(data["HA"].min(), data["HA"].max(), 25)
    alg_grid, ha_grid = np.meshgrid(alg_range, ha_range)

    grid_scaled = scaler_X.transform(
        np.column_stack((alg_grid.ravel(), ha_grid.ravel()))
    )
    predictions_scaled = model.predict(grid_scaled)
    predictions_log = scaler_Y.inverse_transform(predictions_scaled.reshape(-1, 1))[
        :, 0
    ]
    predictions_original = np.exp(predictions_log).reshape(alg_grid.shape)

    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection="3d", computed_zorder=False)

    if title == "eta_0":
        cmap = "viridis"
        zlabel = r"$\eta_{0}$ [Pa·s]"
        marker = "o"
        ax.set_title(f"Zero-Shear-Rate Viscosity" + r" ($\eta_{0}$)")
    elif title == "m":
        cmap = "plasma"
        zlabel = r"$m$ [s]"
        marker = "s"
        ax.set_title(f"Time Constant" + r" ($m$)")
    elif title == "n":
        cmap = "coolwarm"
        zlabel = r"$n$ [-]"
        marker = "^"
        ax.set_title(f"Shear-Thinning Index" + r" ($n$)")

    surf = ax.plot_surface(
        alg_grid, ha_grid, predictions_original, cmap=cmap, alpha=0.8, linewidth=0.25
    )
    ax.scatter(
        data["ALG"],
        data["HA"],
        np.exp(Y[:, index]),
        c="red",
        marker=marker,
        s=100,
        edgecolor="black",
        label="Fitted Empirical Data",
        depthshade=False,
    )

    ax.set_xlabel("ALG [% (w/v)]", fontweight="bold")
    ax.set_ylabel("HA [% (w/v)]", fontweight="bold")
    ax.set_zlabel(zlabel, fontweight="bold")
    # plt.colorbar(surf)
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1),
    )

    # Set y-axis (HA) ticks
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1.0"])

    # Add padding to labels
    ax.xaxis.labelpad = 18
    ax.yaxis.labelpad = 18
    ax.zaxis.labelpad = 12

    # Add padding to tick labels
    ax.tick_params(axis="x", pad=4)
    ax.tick_params(axis="y", pad=4)
    ax.tick_params(axis="z", pad=6)

    ax.view_init(20, -135)

    if save_fig:
        if not os.path.exists(f"SVR/{ink_type}_images"):
            os.makedirs(f"SVR/{ink_type}_images")
        plt.savefig(
            f"SVR/{ink_type}_images/{title}.png",
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.48,
        )

    plt.show()


def plot_actual_vs_predicted_cv(
    model, feature_index, scaler_Y, X, Y, title, save_fig=False
):
    # Get the target variable
    y = Y[:, feature_index].reshape(-1, 1)

    # Scale the target variable
    y_scaled = scaler_Y.fit_transform(y)

    # Predict using cross-validation on scaled data
    Y_pred_scaled = cross_val_predict(
        model, X, y_scaled.ravel(), cv=CROSS_VALIDATION_FOLDS
    )

    # Inverse transform the predictions back to log space
    Y_pred_log = scaler_Y.inverse_transform(Y_pred_scaled.reshape(-1, 1))

    # Convert from log space to original space
    Y_pred = np.exp(Y_pred_log)
    Y_actual = np.exp(y)

    # Calculate metrics
    r2 = r2_score(Y_actual, Y_pred)
    mae = np.mean(np.abs(Y_actual - Y_pred))
    rmse = np.sqrt(np.mean((Y_actual - Y_pred) ** 2))

    plt.figure(figsize=(7.8, 7.8))

    if title == "eta_0":
        unit = "Pa·s"
        marker = "o"
        plt.title(INK_NAME + r" ($\eta_{0}$)")
    elif title == "m":
        unit = "s"
        marker = "s"
        plt.title(INK_NAME + r" ($m$)")
    else:
        unit = "-"
        marker = "^"
        plt.title(INK_NAME + r" ($n$)")

    # Plot
    plt.scatter(Y_actual, Y_pred, alpha=0.5, s=100, marker=marker, color="red")
    plt.plot(
        [Y_actual.min(), Y_actual.max()], [Y_actual.min(), Y_actual.max()], "k--", lw=2
    )
    plt.xlabel(f"Actual [{unit}]", fontweight="bold")
    plt.ylabel(f"Predicted [{unit}]", fontweight="bold")

    if title == "n":
        unit = ""

    # Create a box with error evaluation results
    textstr = "\n".join(
        [f"R-squared: {r2:.3f}", f"MAE: {mae:.3f} {unit}", f"RMSE: {rmse:.3f} {unit}"]
    )
    props = dict(facecolor="white", alpha=1, edgecolor="black")
    plt.text(
        0.05,
        0.95,
        textstr,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=props,
    )

    # log scale
    # if title == "eta_0" or title == "m":
    #     plt.xscale("log")
    #     plt.yscale("log")

    plt.locator_params(axis="both", nbins=6)

    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()

    if save_fig:
        plt.savefig(
            f"SVR/{ink_type}_images/{title}_actual_vs_predicted.png",
            dpi=600,
            bbox_inches="tight",
        )

    plt.show()


# Plot 3D surfaces
for i, name in enumerate(["eta_0", "m", "n"]):
    plot_3d_surface(svr_models[i], name, scaler_X, scalers_Y[i], data, i, save_fig=True)
    plot_actual_vs_predicted_cv(
        svr_models[i], i, scalers_Y[i], X_scaled, Y, name, save_fig=True
    )

# Save models and scalers
if not os.path.exists(f"SVR/{ink_type}_model"):
    os.makedirs(f"SVR/{ink_type}_model")

for i, name in enumerate(["eta_0", "m", "n"]):
    with open(f"SVR/{ink_type}_model/svr_{name}_model.pkl", "wb") as file:
        pickle.dump(svr_models[i], file)

with open(f"SVR/{ink_type}_model/scaler_X.pkl", "wb") as file:
    pickle.dump(scaler_X, file)

for i, name in enumerate(["eta_0", "m", "n"]):
    with open(f"SVR/{ink_type}_model/scaler_Y_{name}.pkl", "wb") as file:
        pickle.dump(scalers_Y[i], file)
