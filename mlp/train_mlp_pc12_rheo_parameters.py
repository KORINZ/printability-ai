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

# Set up plotting style
sns.set_context("talk", font_scale=1.4)
sns.set_style("ticks")


tf.random.set_seed(42)
np.random.seed(42)


def extract_concentrations(ink_type):
    """Extract ALG and HA concentrations from the ink type string."""
    if "_HA" not in ink_type:
        # Handle case when there's only ALG (e.g., "ALG2.0")
        alg = float(ink_type.replace("ALG", ""))
        ha = 0.0
    else:
        # Handle case with both ALG and HA (e.g., "ALG2.0_HA1.0")
        parts = ink_type.split("_")
        alg = float(parts[0].replace("ALG", ""))
        ha = float(parts[1].replace("HA", ""))
    return pd.Series({"ALG": alg, "HA": ha})


def create_ink_type(row):
    """Create ink type string based on ALG and HA concentrations."""
    if row["ha_concentration"] == 0:
        return f"ALG{row['alg_concentration']}"
    return f"ALG{row['alg_concentration']}_HA{row['ha_concentration']}"


def standardize_ink_type(ink_type):
    """Standardize ink type format to match between datasets."""
    return ink_type.replace("-", "_")


def load_and_preprocess_data():
    """Load and preprocess data from CSV files."""
    # Read data
    data = pd.read_csv("test_ALG_HA/PC-12_individual_data.csv")
    power_law_data = pd.read_csv(
        "test_ALG_HA/csv_data_files/ALG_HA_cross_power_law_svr_predictions.csv"
    )
    rheology_data = pd.read_csv(
        "test_ALG_HA/csv_data_files/ALG_HA_GG_at_1hz_svr_predictions.csv"
    )

    # Filter data
    data = data[data["flow_rate_uL_per_s"] == 2]
    data = data[data["alg_concentration"] <= 4.5]

    # Clip cell viability values
    data["relative_cell_viability"] = data["relative_cell_viability"].clip(0, 100)

    # Create ink_type column in data
    data["ink_type"] = data.apply(create_ink_type, axis=1)
    data["ink_type"] = data["ink_type"].apply(standardize_ink_type)

    # Standardize ink type format in rheology data
    power_law_data["ink_type"] = power_law_data["ink_type"].apply(standardize_ink_type)
    rheology_data["ink_type"] = rheology_data["ink_type"].apply(standardize_ink_type)

    # Merge rheological parameters
    data = pd.merge(
        data,
        power_law_data[["ink_type", "eta_0", "m", "n"]],
        on="ink_type",
        how="left",
        suffixes=("_orig", ""),
    )
    data = pd.merge(
        data,
        rheology_data[["ink_type", "tan_delta"]],
        on="ink_type",
        how="left",
    )
    print(f"Dataset size before preprocessing: {len(data)}")

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
    original_columns = [
        "alg_concentration",
        "ha_concentration",
        "eta_0",
        "m",
        "n",
        "tan_delta",
    ]
    X = data[original_columns].values
    y = data["relative_cell_viability"].values / 100.0  # Scale to [0,1]

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
        dataset[:, 2] = np.log10(dataset[:, 2])  # eta_0
        dataset[:, 3] = np.log10(dataset[:, 3])  # m

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
        X_test,
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
    plt.title("PC12 Viability MLP Training History")
    plt.legend(
        framealpha=1,
        edgecolor="black",
        fancybox=False,
        bbox_to_anchor=(1, 1),
        loc=1,
        borderaxespad=0,
        handlelength=1.5,
    )
    plt.xlim(0, len(history.history["loss"]))

    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()

    if save_fig:
        if not os.path.exists("test_ALG_HA/images_MLP_rheo"):
            os.makedirs("test_ALG_HA/images_MLP_rheo")
        plt.savefig(
            "test_ALG_HA/images_MLP_rheo/PC12_training_history.png",
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
        color="lightcoral",
        linecolor="black",
        width=0.7,
        flierprops=dict(marker="o", markersize=7),
    )

    plt.xlabel("Decrease in Model Performance (R²)", fontweight="bold")
    plt.ylabel("Features", fontweight="bold")
    plt.title("PC12 Viability MLP Feature Importance")

    # Add grid
    plt.grid(axis="x", linestyle="-", alpha=0.5)

    plt.tight_layout()

    if not os.path.exists("test_ALG_HA/images_MLP_rheo"):
        os.makedirs("test_ALG_HA/images_MLP_rheo")
    plt.savefig(
        "test_ALG_HA/images_MLP_rheo/PC12_feature_importance.png",
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
    model, X_test_scaled, y_test_scaled, scaler_y, save_fig=False
):
    """Plot actual vs predicted values for the test set."""
    y_pred_scaled = model.predict(X_test_scaled, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled) * 100  # Scale back to percentage
    y_actual = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)) * 100

    r2 = r2_score(y_actual, y_pred)
    mae = np.mean(np.abs(y_actual - y_pred))
    rmse = np.sqrt(np.mean((y_actual - y_pred) ** 2))

    plt.figure(figsize=(7.8, 7.8))
    plt.scatter(y_actual, y_pred, alpha=0.75, s=100, marker="o", color="gray")
    plt.plot(
        [y_actual.min(), y_actual.max()],
        [y_actual.min(), y_actual.max()],
        lw=3.5,
        c="r",
        ls="--",
    )
    plt.xlim([y_actual.min() - 2.5, 104])
    plt.ylim([y_actual.min() - 2.5, 104])
    plt.xlabel("Actual Cell Viability [%]", fontweight="bold")
    plt.ylabel("Predicted Cell Viability [%]", fontweight="bold")
    plt.title("PC12 Viability Test Set Predictions")

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
        if not os.path.exists("test_ALG_HA/images_MLP_rheo"):
            os.makedirs("test_ALG_HA/images_MLP_rheo")
        plt.savefig(
            "test_ALG_HA/images_MLP_rheo/PC12_test_predictions.png",
            dpi=600,
            bbox_inches="tight",
        )
    plt.show()
    return r2, mae, rmse


if __name__ == "__main__":
    # Load and preprocess data
    data = load_and_preprocess_data()
    print(f"Dataset size after preprocessing: {len(data)}")

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
        X_test_original,
    ) = prepare_data(data)

    # Create and train model
    model = create_model()
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
    )

    history = model.fit(
        X_train_scaled,
        y_train_scaled,
        epochs=1000,
        batch_size=32,
        validation_data=(X_val_scaled, y_val_scaled),
        callbacks=[early_stopping],
        verbose=1,
    )

    # Plot training history
    plot_training_history(history, save_fig=True)

    # Calculate and plot feature importance
    importance_df = calculate_permutation_importance(
        model, X_test_scaled, y_test_scaled, feature_columns
    )
    print("\nFeature Importance:")
    print(importance_df)

    # Plot test predictions
    r2, mae, rmse = plot_test_predictions(
        model, X_test_scaled, y_test_scaled, scaler_y, save_fig=True
    )
    print("\nTest Set Metrics:")
    print(f"R-squared: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # Save model and scalers
    if not os.path.exists("test_ALG_HA/models_MLP_rheo"):
        os.makedirs("test_ALG_HA/models_MLP_rheo")

    model.save("test_ALG_HA/models_MLP_rheo/PC12_model_with_rheo.keras")
    with open("test_ALG_HA/models_MLP_rheo/scaler_X_with_rheo.pkl", "wb") as file:
        pickle.dump(scaler_X, file)
    with open("test_ALG_HA/models_MLP_rheo/scaler_y_with_rheo.pkl", "wb") as file:
        pickle.dump(scaler_y, file)
