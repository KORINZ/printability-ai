import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import r2_score
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

sns.set_context("talk", font_scale=1.4)
sns.set_style("ticks")

np.random.seed(42)
tf.random.set_seed(42)


class HDPredictionModel:
    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size
        self.model = None
        self.scaler = MinMaxScaler()

    def rotate_image(self, image, angle):
        """Rotate image by specified angle"""
        if angle == 0:
            return image

        # Convert to numpy array if tensor
        if isinstance(image, tf.Tensor):
            image = image.numpy()

        # Get image dimensions
        height, width = image.shape[:2]
        center = (width // 2, height // 2)

        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Perform rotation
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

        return rotated_image

    def load_and_preprocess_image(self, image_path, rotation_angle=0):
        """Load and preprocess a single image with optional rotation"""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.image_size)

        # Apply rotation if specified
        if rotation_angle != 0:
            img = self.rotate_image(img, rotation_angle)

        img = img / 255.0  # Normalize pixel values
        return img.reshape((*self.image_size, 1))

    def prepare_data(self, images_dir, csv_data):
        """Prepare image data and labels with train/validation/test split"""
        X = []
        y = []

        # First collect original images without augmentation
        for _, row in csv_data.iterrows():
            image_id = (
                row["Image"].replace("resized_IMG_", "").replace("_masked.png", "")
            )
            image_path = os.path.join(images_dir, f"resized_IMG_{image_id}_masked.png")

            if os.path.exists(image_path):
                try:
                    img = self.load_and_preprocess_image(image_path, rotation_angle=0)
                    X.append(img)
                    y.append(row["Average_HD"])
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    continue
            else:
                print(f"Image not found: {image_path}")

        if not X:
            raise ValueError("No valid images found in the dataset")

        X = np.array(X)
        y = np.array(y)

        # First split test set (20%)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )

        # Second split for train and validation (40% each)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )

        # Fit scaler only on training data
        y_train_scaled = self.scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

        # Transform validation and test using the scaler fitted on training data
        y_val_scaled = self.scaler.transform(y_val.reshape(-1, 1)).ravel()
        y_test_scaled = self.scaler.transform(y_test.reshape(-1, 1)).ravel()

        # Apply augmentation (only to training data as before)
        X_train_aug = []
        y_train_aug = []
        rotation_angles = [90, 180, 270]

        for img, label in zip(X_train, y_train_scaled):
            X_train_aug.append(img)
            y_train_aug.append(label)

            for angle in rotation_angles:
                rotated_img = self.rotate_image(img, angle)
                X_train_aug.append(rotated_img.reshape((*self.image_size, 1)))
                y_train_aug.append(label)

        return (
            np.array(X_train_aug),
            X_val,
            X_test,
            np.array(y_train_aug),
            y_val_scaled,
            y_test_scaled,
        )

    def build_model(self):
        regularizer = tf.keras.regularizers.l2(1e-4)

        self.model = models.Sequential(
            [
                # First block - larger kernel
                layers.Conv2D(
                    32,
                    (3, 3),
                    activation="relu",
                    kernel_regularizer=regularizer,
                    input_shape=(*self.image_size, 1),
                    padding="same",
                ),
                layers.MaxPooling2D((2, 2)),
                # Second block
                layers.Conv2D(
                    64,
                    (3, 3),
                    activation="relu",
                    kernel_regularizer=regularizer,
                    padding="same",
                ),
                layers.MaxPooling2D((2, 2)),
                # Third block
                layers.Conv2D(
                    128,
                    (3, 3),
                    activation="relu",
                    kernel_regularizer=regularizer,
                    padding="same",
                ),
                layers.MaxPooling2D((2, 2)),
                # Fourth block
                layers.Conv2D(
                    256,
                    (3, 3),
                    activation="relu",
                    kernel_regularizer=regularizer,
                    padding="same",
                ),
                layers.MaxPooling2D((2, 2)),
                # Dense layers
                layers.Flatten(),
                layers.Dense(512, activation="relu", kernel_regularizer=regularizer),
                layers.Dropout(0.5),
                layers.Dense(256, activation="relu", kernel_regularizer=regularizer),
                layers.Dropout(0.5),
                layers.Dense(128, activation="relu", kernel_regularizer=regularizer),
                layers.Dense(1),
            ]
        )

        self.model.compile(
            optimizer="adam",
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.MeanAbsoluteError()],
        )

        self.model.summary()

    def train(self, X_train, y_train, X_val, y_val, epochs=1000, batch_size=32):
        """Train the model"""
        return self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=15, restore_best_weights=True
                )
            ],
        )

    def predict(self, image_path):
        """Predict HD value for a single image"""
        img = self.load_and_preprocess_image(image_path, rotation_angle=0)
        prediction = self.model.predict(np.array([img]))
        return self.scaler.inverse_transform(prediction)[0][0]

    def plot_training_history(self, history):
        """Plot the training and validation loss with decreased alpha after best epoch."""
        plt.figure(figsize=(10, 6))

        # Find the epoch with the best validation loss
        best_epoch = np.argmin(history.history["val_loss"])

        # Plot training loss
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

        # Plot validation loss
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

        # Add vertical line at best epoch
        plt.axvline(
            x=best_epoch,
            color="gray",
            linestyle="--",
            label=f"Early Stopping (Epoch {best_epoch})",
        )

        plt.xlim(0, len(history.history["loss"]))

        plt.xlabel("Epoch", fontweight="bold")
        plt.ylabel("Loss (MSE)", fontweight="bold")
        plt.yscale("log")
        plt.title("HD Value CNN Training History")
        plt.legend(
            framealpha=1,
            edgecolor="black",
            fancybox=False,
            bbox_to_anchor=(1, 1),
            loc=1,
            borderaxespad=0,
            handlelength=1.5,
        )

        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.tight_layout()

        if not os.path.exists("CNNs"):
            os.makedirs("CNNs")

        plt.savefig("CNNs/hd_cnn_training_history.png", dpi=600, bbox_inches="tight")
        plt.close()

    def evaluate_predictions(self, X, y):
        """Evaluate model predictions on specified dataset"""
        # Get predictions
        predictions = self.model.predict(X)

        # Inverse transform scaled values
        y_actual = self.scaler.inverse_transform(y.reshape(-1, 1))
        y_pred = self.scaler.inverse_transform(predictions)

        r2 = r2_score(y_actual, y_pred)
        mae = np.mean(np.abs(y_actual - y_pred))
        rmse = np.sqrt(np.mean((y_actual - y_pred) ** 2))

        plt.figure(figsize=(7.8, 7.8))
        plt.scatter(y_actual, y_pred, alpha=0.75, s=100, marker="^", color="gray")
        plt.plot(
            [y_actual.min(), y_actual.max()],
            [y_actual.min(), y_actual.max()],
            lw=3.5,
            c="r",
            ls="--",
        )

        plt.xlabel("Actual HD [-]", fontweight="bold")
        plt.ylabel("Predicted HD [-]", fontweight="bold")
        plt.title("HD Value CNN Test Set Predictions")

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
        plt.savefig(f"CNNs/HD_CNNs_test_predictions.png", dpi=600, bbox_inches="tight")
        plt.close()

        return r2


def main():
    # Load data
    csv_data = pd.read_csv("csv_data_files/lattice_analysis_results.csv")

    # Initialize model
    hd_model = HDPredictionModel()

    # Prepare data with rotation augmentation
    X_train, X_val, X_test, y_train, y_val, y_test = hd_model.prepare_data(
        "GUI_output", csv_data
    )

    # Build and train model
    hd_model.build_model()
    history = hd_model.train(X_train, y_train, X_val, y_val)

    # Plot training history
    hd_model.plot_training_history(history)

    # Evaluate predictions on validation and test sets
    val_r2 = hd_model.evaluate_predictions(X_val, y_val)
    test_r2 = hd_model.evaluate_predictions(X_test, y_test)

    print(f"Validation R-squared: {val_r2:.3f}")
    print(f"Test R-squared: {test_r2:.3f}")

    if not os.path.exists("CNNs/models"):
        os.makedirs("CNNs/models")

    # Save model and scaler
    model_save_path = "CNNs/models/hd_prediction_model.keras"
    hd_model.model.save(model_save_path)

    joblib.dump(hd_model.scaler, "CNNs/models/hd_scaler.pkl")


if __name__ == "__main__":
    main()
