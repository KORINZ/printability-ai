import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import seaborn as sns
from itertools import product
import time
import joblib

sns.set_context("talk", font_scale=1.4)
sns.set_style("ticks")

np.random.seed(42)
tf.random.set_seed(42)


def load_and_preprocess_image(image_path, image_size, rotation_angle=0, img=None):
    """Load and preprocess a single image with optional rotation."""
    if img is None:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, image_size)

    if rotation_angle != 0:
        height, width = img.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        img = cv2.warpAffine(img, rotation_matrix, (width, height))

    img = img / 255.0
    return img.reshape((*image_size, 1))


def prepare_data(images_dir, csv_data, image_size=(224, 224)):
    """Prepare image data and labels with initial test split before any processing."""
    # First, split into train and final test sets (80/20)
    train_data, final_test_data = train_test_split(
        csv_data, test_size=0.2, random_state=42
    )

    def process_dataset(data):
        X = []
        y = []
        for _, row in data.iterrows():
            image_id = (
                row["Image"].replace("resized_IMG_", "").replace("_masked.png", "")
            )
            image_path = os.path.join(images_dir, f"resized_IMG_{image_id}_masked.png")

            if os.path.exists(image_path):
                try:
                    img = load_and_preprocess_image(image_path, image_size)
                    X.append(img)
                    y.append(row["Average_HD"])
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    continue
            else:
                print(f"Image not found: {image_path}")

        return np.array(X), np.array(y)

    X_train_orig, y_train_orig = process_dataset(train_data)
    X_final_test, y_final_test = process_dataset(final_test_data)

    print(f"\nFinal test set size (20%): {len(y_final_test)}")
    print(f"Training set size (80%, no augmentation): {len(y_train_orig)}")

    return X_train_orig, y_train_orig, X_final_test, y_final_test


def create_model(params, image_size):
    """Create model with specified architecture and parameters."""
    model = models.Sequential()

    # First convolutional block
    model.add(
        layers.Conv2D(
            params["filters"][0],
            (3, 3),
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(params["l2_lambda"]),
            input_shape=(*image_size, 1),
            padding="same",
        )
    )
    model.add(layers.MaxPooling2D((2, 2)))

    # Additional convolutional blocks
    for filters in params["filters"][1:]:
        model.add(
            layers.Conv2D(
                filters,
                (3, 3),
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(params["l2_lambda"]),
                padding="same",
            )
        )
        model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())

    for i, units in enumerate(params["dense_units"]):
        model.add(
            layers.Dense(
                units,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(params["l2_lambda"]),
            )
        )

        # Add dropout after each dense layer except the last one
        if i < len(params["dense_units"]) - 1:
            model.add(layers.Dropout(params["dropout_rate"]))

    model.add(layers.Dense(1))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]),
        loss="mse",
        metrics=["mae"],
    )

    return model


def augment_data(X, y, image_size):
    """Apply rotational augmentation to the data."""
    X_aug = []
    y_aug = []

    for i, (x, y_val) in enumerate(zip(X, y)):
        # Append original image
        X_aug.append(x)
        y_aug.append(y_val)

        # Add rotated versions
        for angle in [90, 180, 270]:
            img_rotated = load_and_preprocess_image(
                None, image_size, angle, img=x.reshape(image_size)
            )
            X_aug.append(img_rotated)
            y_aug.append(y_val)

    return np.array(X_aug), np.array(y_aug)


def main():
    csv_data = pd.read_csv("csv_data_files/lattice_analysis_results.csv")
    image_size = (224, 224)

    X_train_orig, y_train_orig, X_final_test, y_final_test = prepare_data(
        "GUI_output", csv_data, image_size
    )

    param_grid = {
        "filters": [[16, 32, 64, 128], [32, 64, 128, 256], [64, 128, 256, 512]],
        "dense_units": [[1024, 512, 256], [512, 256, 128], [256, 128, 64]],
        "dropout_rate": [0.5],
        "learning_rate": [0.001],
        "l2_lambda": [0.001, 0.0001, 0.00001],
    }

    param_combinations = [
        dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())
    ]

    all_results = []
    fold_results = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    total_combinations = len(param_combinations)
    start_time = time.time()

    for param_idx, params in enumerate(param_combinations, 1):
        print(f"\nTesting combination {param_idx}/{total_combinations}")
        print(
            f"Architecture: Filters={params['filters']}, Dense={params['dense_units']}"
        )
        print(f"L2 lambda: {params['l2_lambda']}")

        fold_scores = []
        scaler = MinMaxScaler()

        filter_str = "-".join(str(f) for f in params["filters"])
        dense_str = "-".join(str(d) for d in params["dense_units"])
        arch_str = f"{filter_str}_{dense_str}"

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train_orig), 1):
            # Split data
            X_train_fold = X_train_orig[train_idx]
            y_train_fold = y_train_orig[train_idx]
            X_val_fold = X_train_orig[val_idx]
            y_val_fold = y_train_orig[val_idx]

            # Augment only training data
            X_train_aug, y_train_aug = augment_data(
                X_train_fold, y_train_fold, image_size
            )

            # Scale the data
            y_train_scaled = scaler.fit_transform(y_train_aug.reshape(-1, 1)).ravel()
            y_val_scaled = scaler.transform(y_val_fold.reshape(-1, 1)).ravel()

            model = create_model(params, image_size)

            # Train the model
            history = model.fit(
                X_train_aug,
                y_train_scaled,
                epochs=1000,
                batch_size=32,
                validation_data=(X_val_fold, y_val_scaled),
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss", patience=15, restore_best_weights=True
                    )
                ],
                verbose=0,
            )

            # Calculate MSE in scaled space
            val_pred = model.predict(X_val_fold, verbose=0)
            mse = np.mean((y_val_scaled - val_pred.ravel()) ** 2)

            fold_scores.append(mse)
            print(f"Fold {fold_idx}/5 - MSE: {mse:.6e}")

            # Save individual fold results
            fold_results.append(
                {
                    "architecture": f"Filters: {params['filters']}, Dense: {params['dense_units']}",
                    "arch_str": arch_str,
                    "l2_lambda": params["l2_lambda"],
                    "fold": fold_idx,
                    "fold_mse": mse,
                }
            )

        mean_mse = np.mean(fold_scores)
        std_mse = np.std(fold_scores)
        print(f"Mean MSE: {mean_mse:.6e} ± {std_mse:.6e}")

        # Store summary results
        all_results.append(
            {
                "filters": str(params["filters"]),
                "dense_units": str(params["dense_units"]),
                "dropout_rate": params["dropout_rate"],
                "learning_rate": params["learning_rate"],
                "l2_lambda": params["l2_lambda"],
                "arch_str": arch_str,
                "fold_mse": mean_mse,
                "mse_std": std_mse,
            }
        )

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time/60:.2f} minutes")

    # Save results
    results_df = pd.DataFrame(all_results)
    detailed_results_df = pd.DataFrame(fold_results)

    results_df.to_csv("CNNs/cv_results.csv", index=False)
    detailed_results_df.to_csv("CNNs/cv_detailed_results.csv", index=False)

    # Print best configuration
    best_config = results_df.loc[results_df["fold_mse"].idxmin()]
    print("\nBest Configuration:")
    print(f"Architecture: {best_config['arch_str']}")
    print(f"L2 Lambda: {best_config['l2_lambda']:.5f}")
    print(f"MSE: {best_config['fold_mse']:.6e} ± {best_config['mse_std']:.6e}")

    # Save test set
    np.save("CNNs/X_final_test.npy", X_final_test)
    np.save("CNNs/y_final_test.npy", y_final_test)

    # Train final model with best configuration
    print("\nTraining final model with best configuration...")
    best_params = {
        "filters": eval(best_config["filters"]),
        "dense_units": eval(best_config["dense_units"]),
        "dropout_rate": best_config["dropout_rate"],
        "learning_rate": best_config["learning_rate"],
        "l2_lambda": best_config["l2_lambda"],
    }

    final_model = create_model(best_params, image_size)
    scaler = MinMaxScaler()

    # Augment the full training set
    X_train_final_aug, y_train_final_aug = augment_data(
        X_train_orig, y_train_orig, image_size
    )

    # Scale the data
    y_train_scaled = scaler.fit_transform(y_train_final_aug.reshape(-1, 1)).ravel()

    # Train final model
    final_history = final_model.fit(
        X_train_final_aug,
        y_train_scaled,
        epochs=1000,
        batch_size=32,
        validation_split=0.2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=15, restore_best_weights=True
            )
        ],
        verbose=1,
    )

    # Scale and evaluate on held-out test set
    y_test_scaled = scaler.transform(y_final_test.reshape(-1, 1)).ravel()
    final_test_pred = final_model.predict(X_final_test)
    final_test_mse = np.mean((y_test_scaled - final_test_pred.ravel()) ** 2)

    print(f"\nFinal Test Set MSE (held-out 20%): {final_test_mse:.6e}")

    # Save model and results
    final_model.save("CNNs/best_model.keras")
    joblib.dump(scaler, "CNNs/scaler.pkl")

    with open("CNNs/final_results.txt", "w") as f:
        f.write("Best Configuration:\n")
        f.write(f"Architecture: {best_config['arch_str']}\n")
        f.write(f"L2 Lambda: {best_config['l2_lambda']:.5f}\n")
        f.write(f"MSE: {best_config['fold_mse']:.6e} ± {best_config['mse_std']:.6e}\n")
        f.write(f"\nFinal Test Set MSE (held-out 20%): {final_test_mse:.6e}\n")


if __name__ == "__main__":
    main()
