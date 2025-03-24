import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pickle
import tensorflow as tf
import os

sns.set_context("talk", font_scale=1.4)
sns.set_style("ticks")

np.random.seed(42)
tf.random.set_seed(42)


def extract_concentrations(ink_type):
    """Extract ALG and HA concentrations from the ink type string."""
    parts = ink_type.split("_")
    if len(parts) > 1 and "HA" in parts[1]:
        alg = float(parts[0].replace("ALG", ""))
        ha = float(parts[1].replace("HA", ""))
    else:
        alg = float(parts[0].replace("ALG", ""))
        ha = 0.0
    return pd.Series({"ALG": alg, "HA": ha})


def standardize_ink_type(ink_type):
    """Standardize ink type format to match between datasets."""
    return ink_type.replace("-", "_")


def load_and_preprocess_data():
    """Load and preprocess the data from CSV files."""
    # Read data from CSV files
    ink_data = pd.read_csv("csv_data_files/ALG_HA_index.csv")
    results_data = pd.read_csv("csv_data_files/lattice_analysis_results.csv")
    power_law_data = pd.read_csv(
        "csv_data_files/ALG_HA_cross_power_law_svr_predictions.csv"
    )
    rheology_data = pd.read_csv("csv_data_files/ALG_HA_GG_at_1hz_svr_predictions.csv")

    # Standardize ink type format
    power_law_data["ink_type"] = power_law_data["ink_type"].apply(standardize_ink_type)
    rheology_data["ink_type"] = rheology_data["ink_type"].apply(standardize_ink_type)

    # Extract image ID from results data
    results_data["image_id"] = (
        results_data["Image"].str.extract(r"IMG_(\d+)").astype(int)
    )

    # Merge the datasets
    data = pd.merge(
        results_data[["image_id", "Average_HD"]],
        ink_data[["image_id", "ink_type"]],
        on="image_id",
    )

    # Merge rheological parameters
    data = pd.merge(
        data, power_law_data[["ink_type", "eta_0", "m", "n"]], on="ink_type", how="left"
    )

    data = pd.merge(
        data,
        rheology_data[["ink_type", "tan_delta"]],
        on="ink_type",
        how="left",
    )

    # Extract ALG and HA concentrations
    data[["ALG", "HA"]] = data["ink_type"].apply(extract_concentrations)

    return data


def prepare_data(data):
    """Prepare and split the data for training."""
    # Prepare input features and target variables
    feature_columns = [
        "ALG (% w/v)",
        "HA (% w/v)",
        r"$\log(η_0)$",
        r"$\log(m)$",
        r"$n$",
        r"$\tan(\delta)$",
    ]
    original_columns = ["ALG", "HA", "eta_0", "m", "n", "tan_delta"]
    X = data[original_columns].values
    y = data["Average_HD"].values

    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Second split: separate validation set from remaining data
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42
    )

    # Apply log transformation after splitting
    for dataset in [X_train, X_val, X_test]:
        # Log transform eta_0 (index 2)
        dataset[:, 2] = np.log10(dataset[:, 2])
        # Log transform m (index 3)
        dataset[:, 3] = np.log10(dataset[:, 3])

    # Scale the features
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train_scaled,
        y_val_scaled,
        y_test_scaled,
        scaler_X,
        scaler_y,
        feature_columns,
    )


def create_model():
    """Create and compile the MLP model."""
    l2 = keras.regularizers.l2(0.0001)
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(6,)),
            keras.layers.Dense(512, activation="relu", kernel_regularizer=l2),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(512, activation="relu", kernel_regularizer=l2),
            keras.layers.Dense(1, activation="relu"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )

    return model


def plot_training_history(history, save_fig=False):
    """Plot the training and validation loss."""
    plt.figure(figsize=(10, 6))

    best_epoch = np.argmin(history.history["val_loss"])

    plt.plot(
        range(best_epoch + 1),
        history.history["loss"][: best_epoch + 1],
        label="Training Loss",
        color="tab:blue",
    )
    plt.plot(
        range(best_epoch, len(history.history["loss"])),
        history.history["loss"][best_epoch:],
        color="tab:blue",
        alpha=0.3,
    )

    plt.plot(
        range(best_epoch + 1),
        history.history["val_loss"][: best_epoch + 1],
        label="Validation Loss",
        color="tab:orange",
    )
    plt.plot(
        range(best_epoch, len(history.history["val_loss"])),
        history.history["val_loss"][best_epoch:],
        color="tab:orange",
        alpha=0.3,
    )

    plt.axvline(
        x=best_epoch,
        color="gray",
        linestyle="--",
        label=f"Early Stopping (Epoch {best_epoch})",
    )

    plt.xlabel("Epoch", fontweight="bold")
    plt.ylabel("Loss (MSE)", fontweight="bold")
    plt.yscale("log")
    plt.title("HD Value MLP Training History")
    plt.legend(
        framealpha=1,
        edgecolor="black",
        fancybox=False,
        bbox_to_anchor=(1, 1),
        loc=1,
        borderaxespad=0,
        handlelength=1.5,
    )
    plt.xlim(0, len(history.history["loss"]) - 1)

    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()

    if save_fig:
        if not os.path.exists("MLP_multi_rheo_parameters/HD_images"):
            os.makedirs("MLP_multi_rheo_parameters/HD_images")
        plt.savefig(
            "MLP_multi_rheo_parameters/HD_images/hd_mlp_training_history.png",
            dpi=600,
            bbox_inches="tight",
        )
    plt.show()


def calculate_permutation_importance(
    model, X_test_scaled, y_test_scaled, feature_columns
):
    """Calculate and plot permutation importance using box plots with grouped features."""

    class KerasRegressor:
        def __init__(self, model):
            self.model = model

        def predict(self, X):
            return self.model.predict(X, verbose=0).ravel()

    wrapped_model = KerasRegressor(model)

    # Define feature groups
    feature_groups = {
        r"$\log(η_0), \log(m)$": [2, 3],
        # indices for eta_0 and m
        "ALG (% w/v)": [0],
        "HA (% w/v)": [1],
        "$n$": [4],
        r"$\tan(\delta)$": [5],
    }

    n_repeats = 30
    importances = {group: [] for group in feature_groups}

    # Calculate baseline score
    baseline_score = r2_score(y_test_scaled, wrapped_model.predict(X_test_scaled))

    for _ in range(n_repeats):
        X_permuted = X_test_scaled.copy()

        # Permute each group
        for group, indices in feature_groups.items():
            X_temp = X_permuted.copy()
            # Permute features in the group together
            permutation = np.random.permutation(len(X_test_scaled))
            X_temp[:, indices] = X_test_scaled[permutation][:, indices]

            # Calculate importance
            permuted_score = r2_score(y_test_scaled, wrapped_model.predict(X_temp))
            importances[group].append(baseline_score - permuted_score)

    # Create DataFrame for plotting
    importance_data = []
    for group, values in importances.items():
        importance_data.extend(
            [{"Feature": group, "Importance": value} for value in values]
        )

    importance_df = pd.DataFrame(importance_data)

    # Calculate mean importance for sorting
    mean_importance = importance_df.groupby("Feature")["Importance"].mean()
    feature_order = mean_importance.sort_values(ascending=False).index

    # Create box plot
    plt.figure(figsize=(10, 6))
    plt.axvline(x=0, color="gray", linestyle="--", alpha=0.75, lw=2)

    sns.boxplot(
        data=importance_df,
        x="Importance",
        y="Feature",
        order=feature_order,
        color="lightblue",
        linecolor="black",
        width=0.7,
        flierprops=dict(marker="o", markersize=7),
    )

    plt.xlabel("Decrease in Model Performance (R²)", fontweight="bold")
    plt.ylabel("Features", fontweight="bold")
    plt.title("HD Value MLP Feature Importance")

    plt.grid(axis="x", linestyle="-", alpha=0.5)

    plt.tight_layout()

    if not os.path.exists("MLP_multi_rheo_parameters/HD_images"):
        os.makedirs("MLP_multi_rheo_parameters/HD_images")
    plt.savefig(
        "MLP_multi_rheo_parameters/HD_images/hd_mlp_feature_importance.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.show()

    # Calculate summary statistics
    importance_summary = pd.DataFrame(
        {
            "Feature": list(feature_groups.keys()),
            "Mean_Importance": [
                np.mean(importances[group]) for group in feature_groups
            ],
            "Std_Importance": [np.std(importances[group]) for group in feature_groups],
        }
    ).sort_values("Mean_Importance", ascending=True)

    return importance_summary


def plot_test_predictions(
    model,
    X_test_scaled,
    y_test_scaled,
    scaler_y,
    scaler_X,
    X_test_original,
    save_fig=False,
):
    """Plot actual vs predicted values for the test set."""
    # Get predictions in scaled space
    y_pred_scaled = model.predict(X_test_scaled, verbose=0)

    # First inverse transform the scaled predictions
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_actual = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1))

    # Inverse transform X to get original feature values
    X_test_unscaled = scaler_X.inverse_transform(X_test_scaled)

    # Inverse log transform the relevant features
    X_test_unscaled[:, 2] = 10 ** X_test_unscaled[:, 2]  # eta_0
    X_test_unscaled[:, 3] = 10 ** X_test_unscaled[:, 3]  # m

    # Calculate metrics using the fully transformed values
    r2 = r2_score(y_actual, y_pred)
    mae = np.mean(np.abs(y_actual - y_pred))
    rmse = np.sqrt(np.mean((y_actual - y_pred) ** 2))

    plt.figure(figsize=(7.8, 7.8))
    plt.scatter(y_actual, y_pred, alpha=0.75, s=100, marker="D", color="gray")
    plt.plot(
        [y_actual.min(), y_actual.max()],
        [y_actual.min(), y_actual.max()],
        lw=3.5,
        c="r",
        ls="--",
    )

    plt.xlabel("Actual HD [-]", fontweight="bold")
    plt.ylabel("Predicted HD [-]", fontweight="bold")
    plt.title("HD Value MLP Test Set Predictions")

    textstr = "\n".join(
        [f"R-squared: {r2:.3f}", f"MAE: {mae:.3f}", f"RMSE: {rmse:.3f}"]
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

    plt.locator_params(axis="both", nbins=6)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()

    if save_fig:
        if not os.path.exists("MLP_multi_rheo_parameters/HD_images"):
            os.makedirs("MLP_multi_rheo_parameters/HD_images")
        plt.savefig(
            "MLP_multi_rheo_parameters/HD_images/hd_mlp_test_predictions.png",
            dpi=600,
            bbox_inches="tight",
        )

    plt.show()
    return r2, mae, rmse


if __name__ == "__main__":
    # Load and preprocess data
    data = load_and_preprocess_data()

    # Prepare and split data
    (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train_scaled,
        y_val_scaled,
        y_test_scaled,
        scaler_X,
        scaler_y,
        feature_columns,
    ) = prepare_data(data)

    # Store original X_test values before scaling
    X_test_original = X_test_scaled.copy()

    # Create and train model
    model = create_model()

    # Define early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
    )

    # Train model
    history = model.fit(
        X_train_scaled,
        y_train_scaled,
        epochs=1000,
        batch_size=32,
        validation_data=(X_val_scaled, y_val_scaled),
        callbacks=[early_stopping],
        verbose=1,
    )

    # Generate plots
    plot_training_history(history, save_fig=True)

    # Calculate and plot feature importance
    importance_df = calculate_permutation_importance(
        model, X_test_scaled, y_test_scaled, feature_columns
    )

    # Plot test predictions with original values
    r2, mae, rmse = plot_test_predictions(
        model,
        X_test_scaled,
        y_test_scaled,
        scaler_y,
        scaler_X,
        X_test_original,
        save_fig=True,
    )

    # Save model and scalers
    if not os.path.exists("MLP_multi_rheo_parameters/HD_model"):
        os.makedirs("MLP_multi_rheo_parameters/HD_model")

    model.save("MLP_multi_rheo_parameters/HD_model/mlp_model.keras")

    with open("MLP_multi_rheo_parameters/HD_model/scaler_X.pkl", "wb") as file:
        pickle.dump(scaler_X, file)

    with open("MLP_multi_rheo_parameters/HD_model/scaler_y.pkl", "wb") as file:
        pickle.dump(scaler_y, file)
