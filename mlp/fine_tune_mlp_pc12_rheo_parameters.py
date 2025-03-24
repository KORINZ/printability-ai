import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
import time
from itertools import product

np.random.seed(42)
import tensorflow as tf

tf.random.set_seed(42)

# Set plotting style
sns.set_context("talk", font_scale=1.425)
sns.set_style("ticks")


def create_model_architecture(architecture, dropout_rate, l2_lambda):
    """Create model with specified architecture and parameters."""
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(6,)))  # ALG, HA, eta_0, m, n, tan_delta

    # Add hidden layers with dropout (except for the last hidden layer)
    for i, units in enumerate(architecture):
        model.add(
            keras.layers.Dense(
                units,
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(l2_lambda),
            )
        )
        # Add dropout only between hidden layers, not after the last hidden layer
        if i < len(architecture) - 1:
            model.add(keras.layers.Dropout(dropout_rate))

    model.add(keras.layers.Dense(1, activation="relu"))
    return model


def load_and_preprocess_data():
    """Load and preprocess data."""
    # Read data
    cell_data = pd.read_csv("test_ALG_HA/PC-12_individual_data.csv")

    # drop eta_0,m,n columns
    cell_data = cell_data.drop(columns=["eta_0", "m", "n"])

    power_law_data = pd.read_csv(
        "test_ALG_HA/csv_data_files/ALG_HA_cross_power_law_svr_predictions.csv"
    )
    rheology_data = pd.read_csv(
        "test_ALG_HA/csv_data_files/ALG_HA_GG_at_1hz_svr_predictions.csv"
    )

    # Filter and preprocess cell data
    cell_data = cell_data[cell_data["flow_rate_uL_per_s"] == 2]
    cell_data = cell_data[cell_data["alg_concentration"] <= 4.5]
    cell_data["relative_cell_viability"] = cell_data["relative_cell_viability"].clip(
        0, 100
    )

    # Create and standardize ink_type
    cell_data["ink_type"] = cell_data.apply(
        lambda row: (
            f"ALG{row['alg_concentration']}_HA{row['ha_concentration']}"
            if row["ha_concentration"] > 0
            else f"ALG{row['alg_concentration']}"
        ),
        axis=1,
    )
    cell_data["ink_type"] = cell_data["ink_type"].str.replace("-", "_")
    power_law_data["ink_type"] = power_law_data["ink_type"].str.replace("-", "_")
    rheology_data["ink_type"] = rheology_data["ink_type"].str.replace("-", "_")

    # Merge rheological parameters
    data = pd.merge(
        cell_data,
        power_law_data[["ink_type", "eta_0", "m", "n"]],
        on="ink_type",
        how="left",
    )
    data = pd.merge(
        data, rheology_data[["ink_type", "tan_delta"]], on="ink_type", how="left"
    )

    return data


def prepare_data(data):
    """Prepare data for training with train-test split."""
    # Prepare input features and target
    X = data[
        ["alg_concentration", "ha_concentration", "eta_0", "m", "n", "tan_delta"]
    ].values
    y = data["relative_cell_viability"].values / 100.0  # Scale to [0,1]

    # First split: separate test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Apply log transformation to rheological parameters
    X_train_val[:, 2] = np.log10(X_train_val[:, 2])  # eta_0
    X_train_val[:, 3] = np.log10(X_train_val[:, 3])  # m
    X_test[:, 2] = np.log10(X_test[:, 2])  # eta_0
    X_test[:, 3] = np.log10(X_test[:, 3])  # m

    # Scale features using only training data
    scaler_X = MinMaxScaler()
    X_train_val_scaled = scaler_X.fit_transform(X_train_val)
    X_test_scaled = scaler_X.transform(X_test)

    return X_train_val_scaled, X_test_scaled, y_train_val, y_test, scaler_X


def train_and_evaluate_model(X_train, y_train, X_val, y_val, params):
    """Train and evaluate a model with given parameters."""
    model = create_model_architecture(
        params["architecture"], params["dropout_rate"], params["l2_lambda"]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]),
        loss="mse",
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True, verbose=0
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=1000,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=0,
    )

    return model, history


def plot_cv_results(results_df):
    """Plot CV results in three subplots based on L2 regularization."""
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))

    def create_subplot(ax, l2_value, title, ylim, show_legend=True):
        data = results_df[results_df["l2_lambda"] == l2_value].copy()

        def get_sort_key(arch_str):
            nodes = [int(x) for x in arch_str.split("-")]
            return sum(
                node * (1000 ** (len(nodes) - i)) for i, node in enumerate(nodes)
            )

        data["sort_key"] = data["arch_str"].apply(get_sort_key)
        data = data.sort_values(["n_layers", "sort_key"], ascending=[True, False])
        data["Architecture"] = data["arch_str"].apply(lambda x: f"[{x}]")

        bp = sns.boxplot(
            data=data,
            x="Architecture",
            y="fold_mse",
            hue="n_layers",
            palette="Set2",
            ax=ax,
            linewidth=2,
            linecolor="black",
            showmeans=True,
            flierprops={"markersize": 10, "marker": "o", "markeredgewidth": 2},
            meanprops={
                "marker": "^",
                "markerfacecolor": "lightblue",
                "markeredgecolor": "black",
                "markeredgewidth": 2,
                "markersize": 10,
            },
            order=data["Architecture"].tolist(),
        )

        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("Mean Squared Error (MSE)", fontweight="bold")
        ax.set_ylim(ylim)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

        if show_legend:
            handles, labels = ax.get_legend_handles_labels()
            mean_marker = plt.Line2D(
                [],
                [],
                marker="^",
                color="none",
                markerfacecolor="lightblue",
                markeredgecolor="black",
                markeredgewidth=2,
                markersize=10,
                label="Mean",
            )
            outlier_marker = plt.Line2D(
                [],
                [],
                marker="o",
                color="none",
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=2,
                markersize=10,
                label="Outlier",
            )
            legend_elements = handles + [mean_marker, outlier_marker]
            legend_labels = labels + ["Mean", "Outlier"]

            ax.legend(
                legend_elements,
                legend_labels,
                title="Number of Hidden Layers",
                frameon=True,
                edgecolor="black",
                framealpha=1,
                fancybox=False,
                ncol=2,
                columnspacing=1.0,
                handletextpad=0.5,
            )
        else:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
            ax.set_ylabel("")

        ax.grid(True, axis="y", alpha=0.3)

    # Get global y limits for consistency with padding
    y_min = results_df["fold_mse"].min()
    y_max = results_df["fold_mse"].max()
    y_range = y_max - y_min
    ylim = (y_min - 0.05 * y_range, y_max + 0.05 * y_range)

    # Create subplots for each L2 value
    create_subplot(ax1, 0.001, r"$\lambda = 10^{-3}$", ylim, show_legend=True)
    create_subplot(ax2, 0.0001, r"$\lambda = 10^{-4}$", ylim, show_legend=False)
    create_subplot(ax3, 0.00001, r"$\lambda = 10^{-5}$", ylim, show_legend=False)

    plt.tight_layout()
    if not os.path.exists("test_ALG_HA/images_MLP_rheo"):
        os.makedirs("test_ALG_HA/images_MLP_rheo")
    plt.savefig(
        "test_ALG_HA/images_MLP_rheo/pc12_cv_results_visualization.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()


def main():
    # Load and preprocess data
    data = load_and_preprocess_data()
    X_train_val_scaled, X_test_scaled, y_train_val, y_test, scaler_X = prepare_data(
        data
    )

    print(f"Training-Validation set size: {len(X_train_val_scaled)}")
    print(f"Test set size: {len(X_test_scaled)}")

    # Define hyperparameter grid
    architectures = {
        "2_layers": [[512, 512], [512, 256], [256, 256]],
        "3_layers": [[512, 512, 512], [512, 256, 128], [256, 256, 256]],
        "4_layers": [
            [512, 512, 512, 512],
            [512, 256, 128, 64],
            [256, 256, 256, 256],
        ],
    }

    param_grid = {
        "architecture": sum(architectures.values(), []),
        "dropout_rate": [0.5],
        "learning_rate": [0.001],
        "l2_lambda": [0.001, 0.0001, 0.00001],
    }

    # Create all combinations of parameters
    param_combinations = [
        dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())
    ]

    # Initialize arrays to store results
    all_results = []
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    total_combinations = len(param_combinations)
    start_time = time.time()

    # Perform CV for each parameter combination
    for param_idx, params in enumerate(param_combinations, 1):
        print(f"\nTesting combination {param_idx}/{total_combinations}")
        print(f"Architecture: {params['architecture']}")
        print(f"L2 lambda: {params['l2_lambda']}")

        fold_mses = []
        for fold_idx, (train_idx, val_idx) in enumerate(
            kf.split(X_train_val_scaled), 1
        ):
            X_train_fold = X_train_val_scaled[train_idx]
            y_train_fold = y_train_val[train_idx]
            X_val_fold = X_train_val_scaled[val_idx]
            y_val_fold = y_train_val[val_idx]

            model, history = train_and_evaluate_model(
                X_train_fold, y_train_fold, X_val_fold, y_val_fold, params
            )

            val_score = model.evaluate(X_val_fold, y_val_fold, verbose=0)
            print(f"Fold {fold_idx}/10 - MSE: {val_score:.6f}")
            fold_mses.append(val_score)

            # Store individual fold results
            all_results.append(
                {
                    "architecture": str(params["architecture"]),
                    "arch_str": "-".join(map(str, params["architecture"])),
                    "n_layers": len(params["architecture"]),
                    "l2_lambda": params["l2_lambda"],
                    "fold": fold_idx,
                    "fold_mse": val_score,
                }
            )

        print(f"Mean MSE: {np.mean(fold_mses):.6f}")

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time/60:.2f} minutes")

    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)

    # Calculate summary statistics
    summary_df = (
        results_df.groupby(["architecture", "arch_str", "n_layers", "l2_lambda"])
        .agg({"fold_mse": ["mean", "std"]})
        .reset_index()
    )

    summary_df.columns = [
        "architecture",
        "arch_str",
        "n_layers",
        "l2_lambda",
        "mean_mse",
        "std_mse",
    ]

    # Save detailed results
    if not os.path.exists("test_ALG_HA/cv_results"):
        os.makedirs("test_ALG_HA/cv_results")
    results_df.to_csv(
        "test_ALG_HA/cv_results/pc12_cv_detailed_results.csv", index=False
    )
    summary_df.to_csv("test_ALG_HA/cv_results/pc12_cv_summary_results.csv", index=False)

    # Create visualization
    plot_cv_results(results_df)

    # Print best configuration
    best_config = summary_df.loc[summary_df["mean_mse"].idxmin()]
    print("\nBest Configuration:")
    print(f"Architecture: [{best_config['arch_str']}]")
    print(f"Layers: {best_config['n_layers']} Layers")
    print(f"L2 Lambda: {best_config['l2_lambda']:.4f}")
    print(f"MSE: {best_config['mean_mse']:.6e} ± {best_config['std_mse']:.6e}")


if __name__ == "__main__":
    main()
