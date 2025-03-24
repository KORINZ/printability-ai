import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
import os
import numpy as np


GOLDEN_RATIO = (1 + np.sqrt(5)) / 2


class StarAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Star Analysis GUI")

        # Initialize variables
        self.x_offset = 0
        self.y_offset = 0
        self.rotation_angle = 0
        self.current_image_path = None
        self.original_image = None
        self.output_image = None
        self.show_legend = False
        self.avg_hd_text = None
        self.arrow_annotation = None  # Track the arrow annotation

        # Fixed star parameters
        self.star_outer_radius = 375
        self.star_inner_radius = 215

        # For image positioning
        self.image_center = None
        self.x_offset = 0
        self.y_offset = 0
        self.rotation_angle = 0

        # HD values
        self.outer_hd = None
        self.inner_hd = None

        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")

        # File selection
        ttk.Button(main_frame, text="Select Image", command=self.select_image).grid(
            row=0, column=0, columnspan=2, pady=5
        )

        # Controls frame
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="5")
        control_frame.grid(row=1, column=0, columnspan=2, pady=5, sticky="we")

        # X offset controls
        ttk.Label(control_frame, text="X Offset:").grid(row=0, column=0, padx=5)
        ttk.Button(
            control_frame, text="←", command=lambda: self.adjust_offset("x", -1)
        ).grid(row=0, column=1, padx=2)
        self.x_offset_label = ttk.Label(
            control_frame, text="0", width=10, anchor="center"
        )
        self.x_offset_label.grid(row=0, column=2, padx=5)
        ttk.Button(
            control_frame, text="→", command=lambda: self.adjust_offset("x", 1)
        ).grid(row=0, column=3, padx=2)

        # Y offset controls
        ttk.Label(control_frame, text="Y Offset:").grid(row=1, column=0, padx=5)
        ttk.Button(
            control_frame, text="↑", command=lambda: self.adjust_offset("y", -1)
        ).grid(row=1, column=1, padx=2)
        self.y_offset_label = ttk.Label(
            control_frame, text="0", width=10, anchor="center"
        )
        self.y_offset_label.grid(row=1, column=2, padx=5)
        ttk.Button(
            control_frame, text="↓", command=lambda: self.adjust_offset("y", 1)
        ).grid(row=1, column=3, padx=2)

        # Rotation controls
        ttk.Label(control_frame, text="Rotation:").grid(row=2, column=0, padx=5)
        ttk.Button(
            control_frame, text="↻", command=lambda: self.adjust_rotation(-0.5)
        ).grid(row=2, column=1, padx=2)
        self.rotation_label = ttk.Label(
            control_frame, text="0°", width=10, anchor="center"
        )
        self.rotation_label.grid(row=2, column=2, padx=5)
        ttk.Button(
            control_frame, text="↺", command=lambda: self.adjust_rotation(0.5)
        ).grid(row=2, column=3, padx=2)

        # Reset button
        ttk.Button(control_frame, text="Reset All", command=self.reset_all).grid(
            row=5, column=0, columnspan=4, pady=5
        )

        # Save button
        self.save_button = ttk.Button(
            main_frame, text="Save Results", command=self.save_results
        )
        self.save_button.grid(row=3, column=0, columnspan=2, pady=5)
        self.save_button.state(["disabled"])

        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.get_tk_widget().grid(row=4, column=0, columnspan=2, pady=5)

        # Remove spines, ticks, and labels
        self.ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])

        # Add button to show/hide legend
        self.legend_button = ttk.Button(
            control_frame, text="Toggle Legend", command=self.toggle_legend
        )
        self.legend_button.grid(row=0, column=5, pady=5)

    def toggle_legend(self):
        """Toggle the legend visibility on the plot."""
        if self.show_legend and self.ax.legend_ is not None:
            self.ax.legend_.remove()
            self.show_legend = False

            # Update the Avg HD box to remove the arrow and HD text if it exists
            if self.avg_hd_text is not None:
                self.avg_hd_text.remove()
                self.avg_hd_text = None

            # Remove the arrow annotation if it exists
            if self.arrow_annotation is not None:
                self.arrow_annotation.remove()
                self.arrow_annotation = None

            # Redraw the Avg HD box without the HD indicator
            if (
                self.current_image_path is not None
                and self.original_image is not None
                and self.outer_hd is not None
                and self.inner_hd is not None
            ):
                image = self.rotate_image(self.original_image, self.rotation_angle)
                self.avg_hd_text = self.ax.text(
                    image.shape[1] * 0.001,
                    image.shape[0] * 0.001,
                    f"Avg HD: {(self.outer_hd + self.inner_hd) / 2:.3f}",
                    fontsize=21,
                    color="black",
                    ha="left",
                    va="top",
                    alpha=1.0,
                    bbox=dict(facecolor="white", edgecolor="black", pad=5, alpha=1),
                )
        else:
            self.ax.legend(
                loc="upper right",
                fontsize="20",
                fancybox=False,
                framealpha=1,
                edgecolor="black",
                bbox_to_anchor=(1.04, 1.04),
                handletextpad=0.15,
                handlelength=1.25,
                labelspacing=0.15,
            )
            self.show_legend = True

            # Update the Avg HD box to include the HD indicator if image is loaded
            if (
                self.current_image_path is not None
                and self.original_image is not None
                and self.outer_hd is not None
                and self.inner_hd is not None
            ):
                if self.avg_hd_text is not None:
                    self.avg_hd_text.remove()
                    self.avg_hd_text = None

                image = self.rotate_image(self.original_image, self.rotation_angle)
                # Add Avg HD box with HD indicator using same arrow style as in plot
                self.avg_hd_text = self.ax.text(
                    image.shape[1] * 0.001,
                    image.shape[0] * 0.001,
                    f"Avg HD: {(self.outer_hd + self.inner_hd) / 2:.3f}\n→ HD Vector",
                    fontsize=21,
                    color="black",
                    ha="left",
                    va="top",
                    alpha=1.0,
                    bbox=dict(facecolor="white", edgecolor="black", pad=5, alpha=1),
                )

                # Add the orange arrow overlay
                self.arrow_annotation = self.ax.annotate(
                    "",  # No text in annotation
                    xy=(40, 68),  # Arrow end
                    xytext=(1, 68),  # Arrow start
                    arrowprops=dict(
                        lw=2.5, alpha=1, color="darkorange", headlength=7, width=0.5
                    ),
                    annotation_clip=False,
                )

        self.canvas.draw()

    def interpolate_points(self, points, num_points=360):
        """Interpolate between points to create a denser sampling."""
        # Convert to numpy array if not already
        points = np.array(points)

        # Add first point to end to close the contour if not already closed
        if not np.array_equal(points[0], points[-1]):
            points = np.vstack([points, points[0]])

        # Calculate cumulative distances
        distances = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
        distances = np.insert(distances, 0, 0)

        # Normalize distances to [0, 1]
        distances = distances / distances[-1]

        # Create evenly spaced points
        t = np.linspace(0, 1, num_points)

        # Interpolate x and y coordinates
        interpolated_points = np.zeros((num_points, 2))
        interpolated_points[:, 0] = np.interp(t, distances, points[:, 0])
        interpolated_points[:, 1] = np.interp(t, distances, points[:, 1])

        return interpolated_points

    def generate_ideal_star(self):
        """Generate ideal star shape with proper 36-108 degree angles"""
        if self.image_center is None:
            return None

        center_x = self.image_center[0] + self.x_offset
        center_y = self.image_center[1] + self.y_offset

        points = []

        # Calculate required angles
        angles = []
        for i in range(10):  # 5 points * 2 vertices each
            if i % 2 == 0:
                # Outer point
                angle = i * 36  # Each outer point is 72° apart (360°/5)
            else:
                # Inner point - using 108° internal angle
                angle = i * 36
            angles.append(angle - 90)  # -90 to start from top

        # Create outer star points
        for angle in angles:
            rad = np.radians(angle)
            radius = (
                self.star_outer_radius
                if angles.index(angle) % 2 == 0
                else self.star_outer_radius * (1 / GOLDEN_RATIO) ** 2
            )  # Golden ratio for proper proportions
            x = center_x + radius * np.cos(rad)
            y = center_y + radius * np.sin(rad)
            points.append([[int(x), int(y)]])

        # Close outer contour
        points.append(points[0])

        # Create inner star points (similar to outer star)
        inner_points = []
        for angle in angles:
            rad = np.radians(angle)
            radius = (
                self.star_inner_radius
                if angles.index(angle) % 2 == 0
                else self.star_inner_radius * (1 / GOLDEN_RATIO) ** 2
            )
            x = center_x + radius * np.cos(rad)
            y = center_y + radius * np.sin(rad)
            inner_points.append([[int(x), int(y)]])

        # Close inner contour
        inner_points.append(inner_points[0])

        # Combine outer and inner contours
        return np.array(points + inner_points[::-1], dtype=np.int32)

    def find_star_contours(self, image):
        """Find outer and inner contours of the star in the image using improved detection."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Convert to binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find all contours
        contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None, None

        # Find outer contour (largest area)
        outer_contour = max(contours, key=cv2.contourArea)

        # Find inner contour
        # Create a mask of the outer contour
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [outer_contour], -1, (255,), -1)

        # Invert the image inside the mask
        masked_binary = cv2.bitwise_and(cv2.bitwise_not(binary), mask)

        # Find contours in the masked region
        inner_contours, _ = cv2.findContours(
            masked_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not inner_contours:
            return outer_contour, None

        # Get the largest inner contour
        inner_contour = max(inner_contours, key=cv2.contourArea)

        return outer_contour, inner_contour

    def calculate_hausdorff_distance(
        self, actual_points, ideal_points, normalization=None
    ):
        """Calculate bidirectional Hausdorff distance."""
        # Calculate pairwise distances
        distances = np.sqrt(
            np.sum(
                (actual_points.reshape(-1, 1, 2) - ideal_points.reshape(1, -1, 2)) ** 2,
                axis=2,
            )
        )

        # Calculate forward and backward HD
        forward_distances = np.min(distances, axis=1)
        backward_distances = np.min(distances, axis=0)

        forward_hd = np.max(forward_distances)
        backward_hd = np.max(backward_distances)

        # Get the direction and points that give the maximum HD
        if forward_hd >= backward_hd:
            actual_idx = np.argmax(forward_distances)
            ideal_idx = np.argmin(distances[actual_idx])
            is_forward = True
            hausdorff_dist = forward_hd
        else:
            ideal_idx = np.argmax(backward_distances)
            actual_idx = np.argmin(distances[:, ideal_idx])
            is_forward = False
            hausdorff_dist = backward_hd

        # Apply normalization only if requested
        if normalization is not None:
            hausdorff_dist = hausdorff_dist / normalization

        return hausdorff_dist, actual_idx, ideal_idx, is_forward

    def update_plot(self):
        if self.current_image_path is None:
            return

        self.ax.clear()
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])

        # Reset HD indicator and arrow annotation
        self.avg_hd_text = None
        self.arrow_annotation = None

        image = self.rotate_image(self.original_image, self.rotation_angle)

        # Create and process ideal star contour
        ideal_star = self.generate_ideal_star()
        if ideal_star is None:
            return
        ideal_points = ideal_star.reshape(-1, 2)

        mid_point = len(ideal_points) // 2
        ideal_outer = self.interpolate_points(ideal_points[:mid_point])
        ideal_inner = self.interpolate_points(ideal_points[mid_point:])

        # Find actual star contours
        outer_contour, inner_contour = self.find_star_contours(image)

        if outer_contour is not None:
            # Process outer contour - store raw HD without normalization
            actual_outer = self.interpolate_points(outer_contour.reshape(-1, 2))
            self.outer_hd_raw, outer_actual_idx, outer_ideal_idx, outer_forward = (
                self.calculate_hausdorff_distance(actual_outer, ideal_outer)
            )

            # Store normalized version for display
            self.outer_hd = (self.outer_hd_raw / self.star_outer_radius).round(3)

            # Process inner contour if it exists
            self.inner_hd_raw = None
            self.inner_hd = None
            if inner_contour is not None:
                actual_inner = self.interpolate_points(inner_contour.reshape(-1, 2))
                self.inner_hd_raw, inner_actual_idx, inner_ideal_idx, inner_forward = (
                    self.calculate_hausdorff_distance(actual_inner, ideal_inner)
                )
                # Store normalized version for display
                self.inner_hd = (self.inner_hd_raw / self.star_outer_radius).round(
                    3
                )  # Use same normalization

            # Plot the image and contours
            self.ax.imshow(image, cmap="gray")

            # Plot ideal contours
            self.ax.plot(
                ideal_outer[:, 0],
                ideal_outer[:, 1],
                "g-",
                label="Ideal Outer",
                lw=3,
                alpha=1,
            )
            self.ax.plot(
                ideal_inner[:, 0],
                ideal_inner[:, 1],
                "g--",
                label="Ideal Inner",
                lw=3,
                alpha=1,
            )

            # Plot actual contours
            actual_outer_closed = np.vstack([actual_outer, actual_outer[0]])
            self.ax.plot(
                actual_outer_closed[:, 0],
                actual_outer_closed[:, 1],
                "r-",
                label="Actual Outer",
                lw=3,
                alpha=1,
            )

            # Draw HD arrow for outer contour based on direction
            if outer_forward:
                # Actual → Ideal (forward HD)
                xy = (
                    ideal_outer[outer_ideal_idx][0],
                    ideal_outer[outer_ideal_idx][1],
                )  # arrow head
                xytext = (
                    actual_outer[outer_actual_idx][0],
                    actual_outer[outer_actual_idx][1],
                )  # arrow tail
            else:
                # Ideal → Actual (backward HD)
                xy = (
                    actual_outer[outer_actual_idx][0],
                    actual_outer[outer_actual_idx][1],
                )  # arrow head
                xytext = (
                    ideal_outer[outer_ideal_idx][0],
                    ideal_outer[outer_ideal_idx][1],
                )  # arrow tail

            self.ax.annotate(
                "",
                xy=xy,
                xytext=xytext,
                arrowprops=dict(
                    lw=2.5, alpha=1, color="darkorange", headlength=7, width=0.5
                ),
            )

            # Add HD value with edgecolor="black" to improve visibility
            self.ax.text(
                xytext[0],
                xytext[1],
                f"{self.outer_hd:.3f}",
                fontsize=21,
                color="red",
                ha="right",
                va="bottom",
                alpha=1.0,
                bbox=dict(facecolor="white", edgecolor="black", pad=1.5, alpha=0.85),
                zorder=10,
            )

            if inner_contour is not None:
                actual_inner_closed = np.vstack([actual_inner, actual_inner[0]])
                self.ax.plot(
                    actual_inner_closed[:, 0],
                    actual_inner_closed[:, 1],
                    "b-",
                    label="Actual Inner",
                    lw=3,
                    alpha=1,
                )

                # Draw HD arrow for inner contour based on direction
                if inner_forward:
                    # Actual → Ideal (forward HD)
                    xy = (
                        ideal_inner[inner_ideal_idx][0],
                        ideal_inner[inner_ideal_idx][1],
                    )  # arrow head
                    xytext = (
                        actual_inner[inner_actual_idx][0],
                        actual_inner[inner_actual_idx][1],
                    )  # arrow tail
                else:
                    # Ideal → Actual (backward HD)
                    xy = (
                        actual_inner[inner_actual_idx][0],
                        actual_inner[inner_actual_idx][1],
                    )  # arrow head
                    xytext = (
                        ideal_inner[inner_ideal_idx][0],
                        ideal_inner[inner_ideal_idx][1],
                    )  # arrow tail

                self.ax.annotate(
                    "",
                    xy=xy,
                    xytext=xytext,
                    arrowprops=dict(
                        lw=2.5, alpha=1, color="darkorange", headlength=7, width=0.5
                    ),
                )

                # Add with edgecolor="black" to improve visibility
                self.ax.text(
                    xytext[0],
                    xytext[1],
                    f"{self.inner_hd:.3f}",
                    fontsize=21,
                    color="blue",
                    ha="right",
                    va="bottom",
                    alpha=1.0,
                    bbox=dict(
                        facecolor="white", edgecolor="black", pad=1.5, alpha=0.85
                    ),
                    zorder=10,
                )

            if self.show_legend:
                # First add the Avg HD label
                self.avg_hd_text = self.ax.text(
                    image.shape[1] * 0.001,
                    image.shape[0] * 0.001,
                    f"Avg HD: {(self.outer_hd + self.inner_hd) / 2:.3f}\n→ HD Vector",
                    fontsize=21,
                    color="black",
                    ha="left",
                    va="top",
                    alpha=1.0,
                    bbox=dict(facecolor="white", edgecolor="black", pad=5, alpha=1),
                )

                # Add the orange arrow overlay and store the reference to it
                self.arrow_annotation = self.ax.annotate(
                    "",  # No text in annotation
                    xy=(40, 68),  # Arrow end
                    xytext=(1, 68),  # Arrow start
                    arrowprops=dict(
                        lw=2.5, alpha=1, color="darkorange", headlength=7, width=0.5
                    ),
                    annotation_clip=False,
                )
            else:
                self.avg_hd_text = self.ax.text(
                    image.shape[1] * 0.001,
                    image.shape[0] * 0.001,
                    f"Avg HD: {(self.outer_hd + self.inner_hd) / 2:.3f}",
                    fontsize=21,
                    color="black",
                    ha="left",
                    va="top",
                    alpha=1.0,
                    bbox=dict(facecolor="white", edgecolor="black", pad=5, alpha=1),
                )

            if self.show_legend:
                self.ax.legend(
                    loc="upper right",
                    fontsize="20",
                    fancybox=False,
                    framealpha=1,
                    edgecolor="black",
                    bbox_to_anchor=(1.04, 1.04),
                    handletextpad=0.15,
                    handlelength=1.25,
                    labelspacing=0.15,
                )
        self.canvas.draw()
        self.save_button.state(["!disabled"])

    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.tif *.tiff")]
        )
        if file_path:
            self.current_image_path = file_path
            self.original_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if self.original_image is not None:
                height, width = self.original_image.shape
                self.image_center = (width // 2, height // 2)
            self.reset_all()
            self.update_plot()

    def adjust_offset(self, axis, amount):
        """Adjust the offset of the ideal star."""
        if axis == "x":
            self.x_offset += amount * 5
            self.x_offset_label.config(text=str(self.x_offset))
        else:
            self.y_offset += amount * 5
            self.y_offset_label.config(text=str(self.y_offset))
        self.update_plot()

    def adjust_rotation(self, amount):
        """Adjust the rotation angle of the ideal star."""
        self.rotation_angle += amount
        self.rotation_angle = round(self.rotation_angle, 1)
        self.rotation_label.config(text=f"{self.rotation_angle}°")
        self.update_plot()

    def rotate_image(self, image, angle):
        """Rotate the image by the given angle."""
        if angle == 0:
            return image

        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(
            image,
            rotation_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255,),
        )
        return rotated_image

    def reset_all(self):
        self.x_offset = 0
        self.y_offset = 0
        self.rotation_angle = 0
        self.x_offset_label.config(text="0")
        self.y_offset_label.config(text="0")
        self.rotation_label.config(text="0°")

        # Reset show_legend to ensure consistency
        self.show_legend = False

        # Clear any existing arrow annotation
        if self.arrow_annotation is not None:
            self.arrow_annotation.remove()
            self.arrow_annotation = None

        if self.current_image_path:
            self.update_plot()

    def save_results(self):
        if self.current_image_path is None:
            return

        # Create directories if they don't exist
        if not os.path.exists("stars/results"):
            os.makedirs("stars/results")

        # Save results to CSV
        import csv
        import datetime

        results_path = "stars/results/star_analysis_results.csv"
        now = datetime.datetime.now()

        header = [
            "Date",
            "Time",
            "Image",
            "X_Offset",
            "Y_Offset",
            "Rotation",
            "Star_Size",
            "Star_Thickness",
            "Outer_HD",
            "Inner_HD",
        ]

        row_data = [
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            os.path.basename(self.current_image_path),
            self.x_offset,
            self.y_offset,
            self.rotation_angle,
            self.star_outer_radius,
            self.star_inner_radius,
            self.outer_hd if self.outer_hd is not None else "N/A",
            self.inner_hd if self.inner_hd is not None else "N/A",
        ]

        file_exists = os.path.exists(results_path)

        with open(results_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(row_data)

        plot_path = (
            f"stars/results/star_analysis_{os.path.basename(self.current_image_path)}"
        )
        plt.savefig(plot_path, dpi=600, bbox_inches="tight")


if __name__ == "__main__":
    root = tk.Tk()
    app = StarAnalysisGUI(root)
    root.protocol("WM_DELETE_WINDOW", root.quit)
    root.mainloop()
