import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.model_selection import KFold
from train_mlp_hd_rheo_parameters import (
    load_and_preprocess_data,
    prepare_data,
)
import time
import tensorflow as tf

from itertools import product

np.random.seed(42)
tf.random.set_seed(42)

# Set plotting style
sns.set_context("talk", font_scale=1.425)
sns.set_style("ticks")


def create_model_architecture(architecture, dropout_rate, l2_lambda):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(6,)))

    for i, units in enumerate(architecture):
        model.add(
            keras.layers.Dense(
                units,
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(l2_lambda),
            )
        )
        if i < len(architecture) - 1:
            model.add(keras.layers.Dropout(dropout_rate))

    model.add(keras.layers.Dense(1))
    return model


def train_and_evaluate_model(X_train, y_train, X_val, y_val, params):
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


def plot_cv_results(results_df, dropout_rate, output_filename):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))

    def create_subplot(ax, l2_value, title, ylim, show_legend=True):
        data = results_df[
            (results_df["l2_lambda"] == l2_value)
            & (results_df["dropout_rate"] == dropout_rate)
        ].copy()

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

        ax.set_title(f"{title}\nDropout Rate: {dropout_rate}")
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

    y_min = results_df["fold_mse"].min()
    y_max = results_df["fold_mse"].max()
    y_range = y_max - y_min
    ylim = (y_min - 0.05 * y_range, y_max + 0.05 * y_range)

    create_subplot(ax1, 0.001, r"$\lambda = 10^{-3}$", ylim, show_legend=True)
    create_subplot(ax2, 0.0001, r"$\lambda = 10^{-4}$", ylim, show_legend=False)
    create_subplot(ax3, 0.00001, r"$\lambda = 10^{-5}$", ylim, show_legend=False)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=600, bbox_inches="tight")
    plt.close()


def main():
    data = load_and_preprocess_data()
    (
        X_train_full,
        X_val,
        X_test,
        y_train_full,
        y_val,
        y_test,
        scaler_X,
        scaler_y,
        feature_columns,
    ) = prepare_data(data)

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
        "dropout_rate": [0.25, 0.5],  # Added 0.25 dropout rate
        "learning_rate": [0.001],
        "l2_lambda": [0.001, 0.0001, 0.00001],
    }

    param_combinations = [
        dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())
    ]

    all_results = []
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    total_combinations = len(param_combinations)
    start_time = time.time()

    for param_idx, params in enumerate(param_combinations, 1):
        print(f"\nTesting combination {param_idx}/{total_combinations}")
        print(f"Architecture: {params['architecture']}")
        print(f"Dropout Rate: {params['dropout_rate']}")
        print(f"L2 lambda: {params['l2_lambda']}")

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train_full), 1):
            X_train_fold = X_train_full[train_idx]
            y_train_fold = y_train_full[train_idx]
            X_val_fold = X_train_full[val_idx]
            y_val_fold = y_train_full[val_idx]

            model, history = train_and_evaluate_model(
                X_train_fold, y_train_fold, X_val_fold, y_val_fold, params
            )

            val_score = model.evaluate(X_val_fold, y_val_fold, verbose=0)
            print(f"Fold {fold_idx}/10 - MSE: {val_score:.6f}")

            all_results.append(
                {
                    "architecture": str(params["architecture"]),
                    "arch_str": "-".join(map(str, params["architecture"])),
                    "n_layers": len(params["architecture"]),
                    "dropout_rate": params["dropout_rate"],
                    "l2_lambda": params["l2_lambda"],
                    "fold": fold_idx,
                    "fold_mse": val_score,
                }
            )

        print(
            f"Mean MSE: {np.mean([result['fold_mse'] for result in all_results[-10:]])}"
        )

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time/60:.2f} minutes")

    results_df = pd.DataFrame(all_results)

    summary_df = (
        results_df.groupby(
            ["architecture", "arch_str", "n_layers", "dropout_rate", "l2_lambda"]
        )
        .agg({"fold_mse": ["mean", "std"]})
        .reset_index()
    )

    summary_df.columns = [
        "architecture",
        "arch_str",
        "n_layers",
        "dropout_rate",
        "l2_lambda",
        "mean_mse",
        "std_mse",
    ]

    results_df.to_csv("cv_detailed_results.csv", index=False)
    summary_df.to_csv("cv_summary_results.csv", index=False)

    # Create separate plots for each dropout rate
    plot_cv_results(results_df, 0.25, "hd_mlp_cv_results_dropout_0.25.png")
    plot_cv_results(results_df, 0.5, "hd_mlp_cv_results_dropout_0.5.png")

    # Print best configurations for each dropout rate
    for dropout_rate in [0.25, 0.5]:
        dropout_results = summary_df[summary_df["dropout_rate"] == dropout_rate]
        best_config = dropout_results.loc[dropout_results["mean_mse"].idxmin()]
        print(f"\nBest Configuration for Dropout Rate {dropout_rate}:")
        print(f"Architecture: [{best_config['arch_str']}]")
        print(f"Layers: {best_config['n_layers']} Layers")
        print(f"L2 Lambda: {best_config['l2_lambda']:.4f}")
        print(f"MSE: {best_config['mean_mse']:.6e} ± {best_config['std_mse']:.6e}")


if __name__ == "__main__":
    main()
