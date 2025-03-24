import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

np.random.seed(42)
tf.random.set_seed(42)


class EmbeddingVisualizer:
    def __init__(self, model_path, scaler_path, input_shape=(224, 224, 1)):
        self.model = tf.keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.ink_mapping = pd.read_csv("csv_data_files/ALG_HA_index.csv")
        self.ink_mapping["image_id"] = self.ink_mapping["image_id"].astype(str)

        dense_layers = [
            i
            for i, layer in enumerate(self.model.layers)
            if isinstance(layer, tf.keras.layers.Dense)
        ]
        if len(dense_layers) < 2:
            raise ValueError("Model needs at least 2 Dense layers")

        penultimate_idx = dense_layers[-2]
        input_layer = tf.keras.Input(shape=input_shape)
        x = input_layer
        for layer in self.model.layers[: penultimate_idx + 1]:
            layer.trainable = False
            x = layer(x)

        self.embedding_model = Model(inputs=input_layer, outputs=x)

    def plot_embeddings(
        self,
        embeddings,
        y,
        csv_data,
        method="tsne",
        perplexity=30,
        n_components=2,
        marker="o",
    ):
        y_original = self.scaler.inverse_transform(y.reshape(-1, 1)).ravel()

        if method.lower() == "tsne":
            reducer = TSNE(
                n_components=n_components, perplexity=perplexity, random_state=42
            )
            title = "FC3 Layer Embeddings (t-SNE)"
        else:
            reducer = PCA(n_components=n_components)
            title = "FC3 Layer Embeddings (PCA)"

        embeddings_2d = reducer.fit_transform(embeddings)
        self.save_embedding_data(embeddings_2d, y, csv_data, method)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.set_context("talk", font_scale=1.3)
        sns.set_style("ticks")
        scatter = ax.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=y_original,
            cmap="inferno_r",
            s=100,
            alpha=0.85,
            edgecolors="k",
            marker=marker,
            linewidth=1,
            zorder=100,
            vmax=0.175,
            vmin=0.545,
        )
        cbar = plt.colorbar(scatter, label="HD [-]", ax=ax, pad=0.01)
        cbar.ax.yaxis.label.set_fontweight("bold")
        cbar.ax.invert_yaxis()

        if method.lower() == "tsne":
            plt.xlabel("Dimension 1", fontweight="bold")
            plt.ylabel("Dimension 2", fontweight="bold")
        else:
            plt.xlabel(
                f"Component 1 ({reducer.explained_variance_ratio_[0]*100:.2f}%)",
                fontweight="bold",
            )
            plt.ylabel(
                f"Component 2 ({reducer.explained_variance_ratio_[1]*100:.2f}%)",
                fontweight="bold",
            )

            # If PCA, plot original axes
            plt.axhline(0, color="k", linewidth=2, zorder=1)
            plt.axvline(0, color="k", linewidth=2, zorder=1)

        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_dir = "CNNs/embeddings/fc_layer"
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            f"{save_dir}/hd_cnn_{method.lower()}_embeddings.png",
            dpi=600,
            bbox_inches="tight",
        )
        plt.close()

        if method.lower() == "pca":
            self.analyze_pca_components(embeddings)

    def extract_embeddings(self, X):
        return self.embedding_model.predict(X, verbose=0)

    def extract_concentrations(self, ink_type):
        parts = ink_type.split("_")
        if len(parts) > 1 and "HA" in parts[1]:
            alg = float(parts[0].replace("ALG", ""))
            ha = float(parts[1].replace("HA", ""))
        else:
            alg = float(parts[0].replace("ALG", ""))
            ha = 0.0
        return f"{alg}-{ha}"

    def get_image_id(self, filename):
        try:
            parts = filename.split("_")
            if len(parts) >= 3:
                return parts[2]
            return "N/A"
        except:
            print(f"Error extracting ID from filename: {filename}")
            return "N/A"

    def save_embedding_data(self, embeddings_2d, y, csv_data, method="tsne"):
        image_indices = csv_data["Image"].apply(self.get_image_id)

        compositions = []
        for idx in image_indices:
            match = self.ink_mapping[self.ink_mapping["image_id"] == idx]
            if not match.empty:
                compositions.append(
                    self.extract_concentrations(match.iloc[0]["ink_type"])
                )
            else:
                print(f"Warning: No ink type found for image {idx}")
                compositions.append("N/A")

        df = pd.DataFrame(
            {
                "image_index": image_indices,
                "ink_composition": compositions,
                "dimension_1": embeddings_2d[:, 0],
                "dimension_2": embeddings_2d[:, 1],
                "hd_value": self.scaler.inverse_transform(y.reshape(-1, 1)).ravel(),
            }
        ).sort_values("hd_value", ascending=False)

        save_dir = "CNNs/embeddings/fc_layer"
        os.makedirs(save_dir, exist_ok=True)
        csv_path = f"{save_dir}/hd_cnn_{method.lower()}_embeddings.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved embedding data to {csv_path}")

    def analyze_pca_components(self, embeddings):
        pca = PCA()
        pca.fit(embeddings)

        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

        print("\nPCA Analysis of FC Layer Embeddings:")
        print(f"Embedding dimension: {embeddings.shape[1]}")
        print(f"Variance explained by 2 components: {cumulative_variance[1]:.3f}")
        print(f"Components needed for 95% variance: {n_components_95}")


def main():
    csv_data = pd.read_csv("csv_data_files/lattice_analysis_results.csv")

    from CNN.train_cnn_hd import HDPredictionModel

    hd_model = HDPredictionModel()
    X_train, X_val, X_test, y_train, y_val, y_test = hd_model.prepare_data(
        "GUI_output", csv_data
    )

    visualizer = EmbeddingVisualizer(
        "CNNs/models/hd_prediction_model.keras", "CNNs/models/hd_scaler.pkl"
    )

    test_embeddings = visualizer.extract_embeddings(X_test)

    _, test_indices = train_test_split(
        range(len(csv_data)), test_size=0.2, random_state=42
    )
    test_csv_data = csv_data.iloc[test_indices].reset_index(drop=True)

    visualizer.plot_embeddings(
        test_embeddings, y_test, test_csv_data, method="tsne", perplexity=10, marker="^"
    )
    visualizer.plot_embeddings(
        test_embeddings, y_test, test_csv_data, method="pca", marker="o"
    )


if __name__ == "__main__":
    main()
