import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Model
from scipy import ndimage
import os
import pandas as pd
import joblib
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

sns.set_context("talk", font_scale=0.8)
sns.set_style("ticks")

np.random.seed(42)
tf.random.set_seed(42)


class ModelVisualizer:
    def __init__(
        self,
        model_path="CNNs/models/hd_prediction_model.keras",
        scaler_path="CNNs/models/hd_scaler.pkl",
        image_size=(224, 224),
    ):
        # Load the original model
        self.model = load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.image_size = image_size

        # Find the last convolutional layer
        conv_layers = []
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                conv_layers.append(layer)

        if not conv_layers:
            raise ValueError("No convolutional layer found in the model")

        # Get the last convolutional layer
        self.last_conv_layer = conv_layers[-1]
        self.last_conv_index = self.model.layers.index(self.last_conv_layer)

        # Create feature extractor model
        self.feature_model = Model(
            inputs=self.model.inputs,
            outputs=self.model.layers[self.last_conv_index].output,
        )

    def compute_gradcam(self, input_tensor):
        """Compute Grad-CAM for the input image."""
        with tf.GradientTape() as tape:
            # Get the conv layer output
            conv_output = self.feature_model(input_tensor, training=False)
            tape.watch(conv_output)

            # Get model prediction through remaining layers
            x = conv_output
            for layer in self.model.layers[self.last_conv_index + 1 :]:
                x = layer(x)
            pred = x

        # Get gradients of the target output with respect to the last conv layer
        grads = tape.gradient(pred, conv_output)

        # Compute channel-wise mean of the gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight the channels and sum
        conv_output = conv_output[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output), axis=-1)

        # ReLU and normalization
        heatmap = tf.maximum(heatmap, 0) / (
            tf.reduce_max(heatmap) + tf.keras.backend.epsilon()
        )

        return heatmap.numpy()

    def visualize_attention(self, image_path, csv_path=None):
        """Compute and visualize both saliency map and Grad-CAM."""
        processed_img = self.load_and_preprocess_image(image_path)
        prediction = self.get_prediction(processed_img)

        # Get true value if CSV path is provided
        true_value = None
        if csv_path:
            image_name = os.path.basename(image_path)
            true_value = self.get_true_value(image_name, csv_path)

        # Compute saliency map
        input_tensor = tf.convert_to_tensor(
            processed_img[np.newaxis, ..., np.newaxis], dtype=tf.float32
        )
        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            predictions = self.model(input_tensor)

        grads = tape.gradient(predictions, input_tensor)
        saliency_map = np.abs(grads.numpy()[0, ..., 0])
        saliency_map = ndimage.gaussian_filter(saliency_map, sigma=1.5)
        saliency_map = (saliency_map - saliency_map.min()) / (
            saliency_map.max() - saliency_map.min() + 1e-8
        )

        # Compute Grad-CAM
        gradcam = self.compute_gradcam(input_tensor)
        gradcam = cv2.resize(gradcam, self.image_size)
        gradcam = np.maximum(gradcam, 0)
        gradcam = gradcam / (gradcam.max() + 1e-8)

        # Print predictions
        print(f"Predicted HD: {prediction:.4f}")
        if true_value:
            print(f"True HD: {true_value:.4f}")
            print(f"Absolute error: {abs(prediction - true_value):.4f}")
            print(
                f"Relative error: {abs(prediction - true_value) / true_value * 100:.2f}%"
            )

        # Visualization
        fig = plt.figure(figsize=(12, 4))
        gs = fig.add_gridspec(1, 3, wspace=0.3)

        titles = ["Input Image", "Saliency Map", "Grad-CAM"]
        images = []

        for idx in range(3):
            ax = fig.add_subplot(gs[0, idx])
            ax.set_facecolor("white")
            for spine in ax.spines.values():
                spine.set_linewidth(2)

            if idx == 0:
                ax.set_title(titles[idx], pad=10)

                im = ax.imshow(processed_img, cmap="gray", alpha=1, vmin=0, vmax=1)
                images.append(im)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = plt.colorbar(
                    im,
                    cax=cax,
                    label="Pixel Intensity",
                    ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                )
                cbar.ax.set_yticklabels(["1.0", "0.8", "0.6", "0.4", "0.2", "0.0"])

                # reverse the colorbar
                cbar.ax.yaxis.label.set_fontweight("bold")
                cbar.ax.invert_yaxis()

                # draw border around image
                ax.add_patch(
                    plt.Rectangle(
                        (-0.5, -0.5),
                        processed_img.shape[1],
                        processed_img.shape[0],
                        fill=False,
                        edgecolor="black",
                        lw=2,
                    )
                )

            elif idx == 1:
                ax.imshow(processed_img, cmap="gray")
                im = ax.imshow(saliency_map, cmap="hot", alpha=0.85, vmin=0, vmax=1)
                ax.set_title(titles[idx], pad=10)

                # Add colorbar for saliency map
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = plt.colorbar(im, cax=cax, label="Saliency Intensity")
                cbar.ax.yaxis.label.set_fontweight("bold")

                images.append(im)
            else:
                ax.imshow(processed_img, cmap="gray")
                im = ax.imshow(gradcam, cmap="magma", alpha=0.85, vmin=0, vmax=1)
                ax.set_title(titles[idx], pad=10)

                # Add colorbar for Grad-CAM
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = plt.colorbar(im, cax=cax, label="Class Activation")
                cbar.ax.yaxis.label.set_fontweight("bold")

                images.append(im)

            ax.tick_params(axis="both", which="both", bottom=False, left=False)
            ax.axis("off")

        plt.tight_layout()

        plt.savefig(
            "attention_analysis.png",
            dpi=600,
            bbox_inches="tight",
        )

        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(processed_img, cmap="gray")
        ax[0].imshow(saliency_map, cmap="hot", alpha=0.85, vmin=0, vmax=1)
        ax[0].axis("off")

        ax[1].imshow(processed_img, cmap="gray")
        ax[1].imshow(gradcam, cmap="magma", alpha=0.85, vmin=0, vmax=1)
        ax[1].axis("off")

        plt.tight_layout()
        plt.savefig(
            "attention_analysis_overlay.png",
            dpi=600,
            bbox_inches="tight",
        )

        plt.close()

    def load_and_preprocess_image(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        img = cv2.resize(img, self.image_size) / 255.0
        return img

    def get_prediction(self, input_image):
        input_tensor = input_image[np.newaxis, ..., np.newaxis]
        prediction = self.model.predict(input_tensor, verbose=0)
        hd_value = self.scaler.inverse_transform(prediction)[0][0]
        return hd_value

    def get_true_value(self, image_name, csv_path):
        try:
            df = pd.read_csv(csv_path)
            true_value = df[df["Image"] == image_name]["Average_HD"].values[0]
            return true_value
        except Exception as e:
            print(f"Error reading true value from CSV: {e}")
            return None


def main():
    model_path = "CNNs/models/hd_prediction_model.keras"
    scaler_path = "CNNs/models/hd_scaler.pkl"
    image_path = "GUI_output/resized_IMG_4648_masked.png"
    csv_path = "csv_data_files/lattice_analysis_results.csv"

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    visualizer = ModelVisualizer(model_path=model_path, scaler_path=scaler_path)
    visualizer.visualize_attention(image_path, csv_path)

    print("Visualization completed. Check generated images for analysis.")


if __name__ == "__main__":
    main()
