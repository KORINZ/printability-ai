import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from CNN.train_cnn_hd import HDPredictionModel
import tensorflow as tf
import os
import joblib
import pandas as pd

np.random.seed(42)
tf.random.set_seed(42)


def analyze_rotation_effect(model, X_test, y_test, angles=[0, 90, 180, 270]):
    differences_data = []
    y_test_unscaled = model.scaler.inverse_transform(y_test.reshape(-1, 1))

    for angle in angles:
        rotated_images = np.array(
            [
                model.rotate_image(img, angle).reshape(*model.image_size, 1)
                for img in X_test
            ]
        )

        rot_predictions = model.model.predict(rotated_images)
        rot_predictions = model.scaler.inverse_transform(rot_predictions)

        perc_diff = np.abs(rot_predictions - y_test_unscaled) / y_test_unscaled * 100

        for diff in perc_diff:
            differences_data.append(
                {"Rotation Angle": f"{angle}", "Difference (%)": diff[0]}
            )

    return pd.DataFrame(differences_data)


def plot_rotation_analysis(differences_df):
    sns.set_context("talk", font_scale=1.4)
    sns.set_style("ticks")

    plt.figure(figsize=(8, 8))

    # Create box plot
    sns.boxplot(
        data=differences_df,
        x="Rotation Angle",
        y="Difference (%)",
        hue="Rotation Angle",
        palette="Set2",
        width=0.325,
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
        showfliers=True,
    )

    # Filter out outliers for stripplot
    def remove_outliers(group):
        q1 = group["Difference (%)"].quantile(0.25)
        q3 = group["Difference (%)"].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return group[
            (group["Difference (%)"] >= lower_bound)
            & (group["Difference (%)"] <= upper_bound)
        ]

    # Apply outlier removal by group
    non_outlier_df = differences_df.groupby("Rotation Angle", group_keys=False).apply(
        remove_outliers
    )

    # Sort the data by rotation angle (0, 90, 180, 270)
    non_outlier_df["Rotation Angle"] = non_outlier_df["Rotation Angle"].astype(int)
    non_outlier_df = non_outlier_df.sort_values("Rotation Angle")

    # add data points
    sns.stripplot(
        data=non_outlier_df,
        x="Rotation Angle",
        y="Difference (%)",
        hue="Rotation Angle",
        palette="Set2",
        alpha=0.8,
        zorder=10,
        size=7,
        linewidth=1.25,
    )

    # remove legend
    plt.legend([], [], frameon=False)

    # shift stripplot points slightly to avoid overlap
    for i, artist in enumerate(plt.gca().collections):
        artist.set_offsets(artist.get_offsets() + np.array([0.335, 0]))

    # Draw horizontal line at 0% difference
    plt.axhline(0, color="black", linestyle="--", linewidth=2, alpha=0.5)

    plt.xlabel("Rotation Angle [°]", fontweight="bold")
    plt.ylabel("Absolute Prediction Error [%]", fontweight="bold")
    plt.title("Effect of Image Rotation on CNN Predictions")

    plt.tight_layout()

    if not os.path.exists("CNNs"):
        os.makedirs("CNNs")

    plt.savefig(
        "CNNs/rotation_effect_boxplot.png", dpi=600, bbox_inches="tight", pad_inches=0.1
    )
    plt.close()


def main():
    # Load the saved model and scaler
    model = HDPredictionModel()
    model.model = tf.keras.models.load_model("CNNs/models/hd_prediction_model.keras")
    model.scaler = joblib.load("CNNs/models/hd_scaler.pkl")

    # Load test data
    csv_data = pd.read_csv("csv_data_files/lattice_analysis_results.csv")
    _, _, X_test, _, _, y_test = model.prepare_data("GUI_output", csv_data)

    # Analyze rotation effects
    differences_df = analyze_rotation_effect(model, X_test, y_test)

    # Plot results
    plot_rotation_analysis(differences_df)

    stats = differences_df.groupby("Rotation Angle")["Difference (%)"].agg(
        ["median", "mean", "std"]
    )
    print(stats.round(2))


if __name__ == "__main__":
    main()
