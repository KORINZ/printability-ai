import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image
import torch
import os
from tkinter import filedialog
import pandas as pd
import csv


class SAM2GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SAM2 Interactive GUI")

        # Initialize variables
        self.image = None
        self.image_path = None
        self.points = []
        self.labels = []
        self.current_mask = None
        self.predictor = None
        self.mask_overlay = None

        # CSV file for logging points
        self.csv_file = "csv_data_files/segmentation_points.csv"
        self.ensure_csv_exists()

        # Create output directory
        self.output_dir = "GUI_output"
        os.makedirs(self.output_dir, exist_ok=True)

        self.setup_gui()
        self.setup_keyboard_bindings()

    def ensure_csv_exists(self):
        """Create CSV file with headers if it doesn't exist"""
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["image_name", "point_type", "x", "y"])

    def setup_keyboard_bindings(self):
        self.root.bind("<BackSpace>", self.remove_last_point)
        self.root.bind("<Delete>", self.remove_last_point)

    def remove_last_point(self, event=None):
        if self.points:
            self.points.pop()
            self.labels.pop()
            if len(self.points) > 0:
                self.run_sam2_auto()  # Re-run SAM2 with remaining points
            else:
                self.current_mask = None
                self.mask_overlay = None
            self.update_plot()
            self.status_var.set("Removed last point")

    def setup_gui(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")

        # Top controls
        controls_frame = ttk.Frame(main_frame)
        controls_frame.grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky="we")

        # Import button and instructions
        import_btn = ttk.Button(
            controls_frame, text="Import Image", command=self.import_image
        )
        import_btn.grid(row=0, column=0, padx=5)

        # Instructions
        instructions = "Left click: Positive point | Right click: Negative point | Backspace/Delete: Remove last point"
        ttk.Label(controls_frame, text=instructions).grid(row=0, column=1, padx=5)

        # Buttons frame
        buttons_frame = ttk.Frame(controls_frame)
        buttons_frame.grid(row=0, column=2, padx=5)

        # Clear and Export buttons (removed Run SAM2 button since it's automatic now)
        ttk.Button(buttons_frame, text="Clear Points", command=self.clear_points).pack(
            side=tk.LEFT, padx=2
        )
        self.export_btn = ttk.Button(
            buttons_frame,
            text="Export Grayscale",
            command=self.export_grayscale,
            state="disabled",
        )
        self.export_btn.pack(side=tk.LEFT, padx=2)

        # Create matplotlib figure
        self.fig = Figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.get_tk_widget().grid(row=1, column=0, columnspan=2)

        # Remove spines
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.spines["bottom"].set_visible(False)
        self.ax.spines["left"].set_visible(False)

        # Remove ticks and labels
        self.ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False,
        )
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Connect mouse events
        self.canvas.mpl_connect("button_press_event", self.on_click)

        # Status bar
        self.status_var = tk.StringVar(value="Import an image to begin.")
        status_label = ttk.Label(main_frame, textvariable=self.status_var)
        status_label.grid(row=2, column=0, columnspan=2, pady=(5, 0))

    def import_image(self):
        self.image_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
        )

        if self.image_path:
            self.image = np.array(Image.open(self.image_path).convert("RGB"))
            self.setup_predictor()
            self.clear_points()
            self.update_plot()
            self.status_var.set(
                "Image loaded. Left click for positive points, right click for negative points."
            )
            self.export_btn.config(state="disabled")

    def setup_predictor(self):
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            device = "cuda" if torch.cuda.is_available() else "cpu"
            sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
            sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

            self.predictor = SAM2ImagePredictor(sam2_model)
            self.predictor.set_image(self.image)

        except Exception as e:
            self.status_var.set(f"Error setting up model: {str(e)}")

    def on_click(self, event):
        if event.inaxes != self.ax or self.image is None:
            return

        x, y = event.xdata, event.ydata
        # Left click (1) for positive, right click (3) for negative
        is_positive = event.button == 1

        self.points.append([x, y])
        self.labels.append(1 if is_positive else 0)

        # Automatically run SAM2 after adding a point
        self.run_sam2_auto()

        point_type = "positive" if is_positive else "negative"
        self.status_var.set(f"Added {point_type} point at ({int(x)}, {int(y)})")

    def clear_points(self):
        self.points = []
        self.labels = []
        self.current_mask = None
        self.mask_overlay = None
        if self.image is not None:
            self.update_plot()
            self.status_var.set("Points cleared.")
            self.export_btn.config(state="disabled")

    def update_plot(self):
        self.ax.clear()
        if self.image is not None:
            self.ax.imshow(self.image)

            # Show mask overlay if available
            if self.current_mask is not None:
                self.ax.imshow(self.current_mask, alpha=0.5, cmap="coolwarm")

            if self.points:
                points = np.array(self.points)
                labels = np.array(self.labels)

                # Plot positive points
                pos_mask = labels == 1
                if np.any(pos_mask):
                    self.ax.scatter(
                        points[pos_mask, 0],
                        points[pos_mask, 1],
                        color="green",
                        marker="X",
                        edgecolors="black",
                        s=300,
                        label="Positive",
                    )

                # Plot negative points
                neg_mask = labels == 0
                if np.any(neg_mask):
                    self.ax.scatter(
                        points[neg_mask, 0],
                        points[neg_mask, 1],
                        color="red",
                        marker="X",
                        edgecolors="black",
                        s=300,
                        label="Negative",
                    )

                # Add equal-sized colored patches for background and foreground
                from matplotlib.patches import Rectangle
                import matplotlib.pyplot as plt

                cmap = plt.cm.coolwarm
                legend_handles = [
                    *self.ax.get_legend_handles_labels()[
                        0
                    ],  # existing scatter plot handles
                    Rectangle(
                        (0, 0),
                        1,
                        1,
                        facecolor=cmap(0.15),
                        edgecolor="black",
                        label="Background",
                    ),  # Blue
                    Rectangle(
                        (0, 0),
                        1,
                        1,
                        facecolor=cmap(0.85),
                        edgecolor="black",
                        label="Segmentation",
                    ),  # Red
                ]
                self.ax.legend(
                    handles=legend_handles,
                    loc="upper center",
                    fancybox=False,
                    edgecolor="black",
                    framealpha=1,
                    fontsize=18,
                    bbox_to_anchor=(0.525, 1.08),
                    ncol=2,
                    columnspacing=0.5,
                )

        self.canvas.draw()

    def run_sam2_auto(self):
        """Automatically run SAM2 and update the display"""
        if not self.points or self.predictor is None:
            return

        try:
            # Convert points and labels to numpy arrays
            input_points = np.array(self.points)
            input_labels = np.array(self.labels)

            # First prediction to get the best mask
            masks, scores, logits = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True,
            )

            # Use the best mask for final prediction
            mask_input = logits[np.argmax(scores), :, :]
            masks, scores, _ = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                mask_input=mask_input[None, :, :],
                multimask_output=False,
            )

            # Store the current mask for export and display
            self.current_mask = masks[0]

            # Enable export button after successful segmentation
            self.export_btn.config(state="normal")

            # Update the plot with the new mask
            self.update_plot()

            self.status_var.set(f"SAM2 segmentation updated (Score: {scores[0]:.3f})")

        except Exception as e:
            self.status_var.set(f"Error running SAM2: {str(e)}")

    def save_points_to_csv(self):
        """Save or update point coordinates in CSV file"""
        if not self.image_path or not self.points:
            return

        image_name = os.path.basename(self.image_path)

        # Read existing CSV file
        try:
            df = pd.read_csv(self.csv_file)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=["image_name", "point_type", "x", "y"])

        # Remove existing entries for this image
        df = df[df["image_name"] != image_name]

        # Create new entries
        new_rows = []
        for point, label in zip(self.points, self.labels):
            point_type = "positive" if label == 1 else "negative"
            new_rows.append(
                {
                    "image_name": image_name,
                    "point_type": point_type,
                    "x": round(point[0], 2),
                    "y": round(point[1], 2),
                }
            )

        # Append new rows and save
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        df.to_csv(self.csv_file, index=False)

    def export_grayscale(self):
        if self.current_mask is None:
            self.status_var.set("Error: No segmentation available to export.")
            return

        if self.image is not None:
            try:
                # Generate output filename based on input image
                base_filename = os.path.splitext(os.path.basename(self.image_path))[0]
                output_filename = f"{base_filename}_masked.png"
                figure_filename = f"{base_filename}_figure.png"
                save_path = os.path.join(self.output_dir, output_filename)
                figure_path = os.path.join("./", figure_filename)

                # Create grayscale image with white background
                pil_image = Image.fromarray(self.image)
                grayscale_image = np.array(pil_image.convert("L"))
                height, width = self.image.shape[:2]
                white_background = np.ones((height, width), dtype=np.uint8) * 255
                white_background[self.current_mask == 1] = grayscale_image[
                    self.current_mask == 1
                ]

                # Save the result
                result_image = Image.fromarray(white_background)
                result_image.save(save_path)

                # Save current figure at 600 dpi
                self.fig.savefig(
                    figure_path, dpi=600, bbox_inches="tight", pad_inches=0
                )

                # Save points to CSV after successful export
                self.save_points_to_csv()

                self.status_var.set(
                    f"Segmented grayscale image saved to: {save_path}\nFigure saved to: {figure_path}"
                )
            except Exception as e:
                self.status_var.set(f"Error saving image: {str(e)}")


def main():
    root = tk.Tk()
    app = SAM2GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
