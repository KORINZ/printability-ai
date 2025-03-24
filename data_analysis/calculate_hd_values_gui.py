import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
import os
import numpy as np
from calculate_hd_values import (
    generate_ideal_square_contour,
    find_contour_and_hd_in_roi,
)
from typing import Final

SCALE_1mm_IN_PIXEL: Final[int] = 250 // 5  # 800 * 800 pixels


class LatticeAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Lattice Analysis GUI")

        # Initialize variables
        self.x_offset = 0
        self.y_offset = 0
        self.rotation_angle = 0
        self.current_image_path = None
        self.hd_list = None
        self.pr_list = []  # List for Pr values
        self.avg_hd = None
        self.avg_pr = None  # Average Pr value
        self.output_image = None
        self.original_image = None
        self.show_hd = True  # Toggle between HD and Pr display
        self.pore_contours = []  # Store contours for area calculation
        self.construct_area_mm2 = None  # Add new variable for construct area

        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")

        # File selection
        ttk.Button(main_frame, text="Select Image", command=self.select_image).grid(
            row=0, column=0, columnspan=2, pady=5
        )

        # Offset and rotation controls
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

        # Toggle HD/Pr button
        self.toggle_button = ttk.Button(
            control_frame, text="Show Pr Values", command=self.toggle_display
        )
        self.toggle_button.grid(row=3, column=0, columnspan=2, pady=5)

        # Reset button
        ttk.Button(control_frame, text="Reset All", command=self.reset_all).grid(
            row=3, column=2, columnspan=2, pady=5
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

        # remove spines
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.spines["bottom"].set_visible(False)
        self.ax.spines["left"].set_visible(False)
        # remove ticks and labels
        self.ax.tick_params(
            axis="both", which="both", bottom=False, top=False, left=False, right=False
        )
        self.ax.set_xticks([])
        self.ax.set_yticks([])

    def toggle_display(self):
        if self.current_image_path is None:
            return
        self.show_hd = not self.show_hd
        self.toggle_button.config(
            text="Show HD Values" if not self.show_hd else "Show Pr Values"
        )
        self.update_plot()

    def calculate_construct_area(self, image):
        # Convert to grayscale if image is RGB
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Create a mask for real construct pixels (not rotation artifacts)
        # Real construct pixels will be black in the original image (value < 255)
        # Rotation artifacts will have been created as black pixels (value == 0)
        construct_mask = (gray < 255) & (gray > 0)

        # Count only the valid construct pixels
        construct_pixels = np.sum(construct_mask)

        # Convert to mm²
        area_mm2 = construct_pixels / (SCALE_1mm_IN_PIXEL**2)
        return round(area_mm2, 3)

    def calculate_pr_value(self, contour):
        if contour is None:
            return "N/A"  # Return N/A for invalid contours

        # Calculate perimeter and area
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)

        if area <= 0:
            return "N/A"  # Return N/A for zero area

        # Calculate Pr value: L²/(16*A)
        pr_value = (perimeter * perimeter) / (16 * area)
        return pr_value

    def run_analysis_with_modified_find_contours(self):
        # Start with the original image and apply rotation
        if self.original_image is None and self.current_image_path is not None:
            self.original_image = cv2.imread(self.current_image_path)
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            # Create white background image
            self.original_image_with_white_bg = np.full_like(self.original_image, 255)
            # Only copy the non-white pixels from original image
            mask = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY) < 255
            self.original_image_with_white_bg[mask] = self.original_image[mask]
            self.original_image = self.original_image_with_white_bg

        # Rotate the image
        image = self.rotate_image(self.original_image, self.rotation_angle)

        # Create white background for output image
        output_image = np.full_like(image, 255)

        if image is None:
            raise ValueError("Image not found")

        # Only copy the non-white pixels from rotated image
        mask = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) < 255
        output_image[mask] = image[mask]

        contour_thickness = 4
        epsilon_factor = 0.0005
        hd_values_list = []
        pr_values_list = []
        pore_status_list = []
        pore_centers = []
        pore_contours = []

        image_size = image.shape[0]

        # Define parameters based on G-code specifications
        num_pores_per_line = 5
        line_thickness_mm = 0.4
        inner_construct_size_mm = 12.0
        total_construct_size_mm = 12.4

        # Convert to pixels
        line_thickness_px = int(line_thickness_mm * SCALE_1mm_IN_PIXEL)
        total_construct_size_px = int(total_construct_size_mm * SCALE_1mm_IN_PIXEL)
        inner_construct_size_px = int(inner_construct_size_mm * SCALE_1mm_IN_PIXEL)

        # Calculate pore width
        pore_width_mm = inner_construct_size_mm / num_pores_per_line
        pore_width_px = int(pore_width_mm * SCALE_1mm_IN_PIXEL)

        # Calculate starting points
        start_x = (image_size - total_construct_size_px) // 2 + self.x_offset
        start_y = (image_size - total_construct_size_px) // 2 + self.y_offset

        # Inner grid start
        inner_start_x = start_x + line_thickness_px // 2
        inner_start_y = start_y + line_thickness_px // 2

        # Create an overlay for semi-transparent green lines
        overlay = np.zeros_like(output_image, dtype=np.uint8)

        # Draw outer boundary on overlay
        cv2.rectangle(
            overlay,
            (start_x, start_y),
            (start_x + total_construct_size_px, start_y + line_thickness_px),
            (0, 255, 0),
            -1,
        )
        cv2.rectangle(
            overlay,
            (start_x, start_y + total_construct_size_px - line_thickness_px),
            (start_x + total_construct_size_px, start_y + total_construct_size_px),
            (0, 255, 0),
            -1,
        )
        cv2.rectangle(
            overlay,
            (start_x, start_y),
            (start_x + line_thickness_px, start_y + total_construct_size_px),
            (0, 255, 0),
            -1,
        )
        cv2.rectangle(
            overlay,
            (start_x + total_construct_size_px - line_thickness_px, start_y),
            (start_x + total_construct_size_px, start_y + total_construct_size_px),
            (0, 255, 0),
            -1,
        )

        # Draw inner grid lines on overlay
        for i in range(1, num_pores_per_line):
            # Calculate position for vertical lines
            x = inner_start_x + (i * pore_width_px) - (line_thickness_px // 2)
            cv2.rectangle(
                overlay,
                (x, inner_start_y),
                (x + line_thickness_px, inner_start_y + inner_construct_size_px),
                (0, 255, 0),
                -1,
            )

            # Calculate position for horizontal lines
            y = inner_start_y + (i * pore_width_px) - (line_thickness_px // 2)
            cv2.rectangle(
                overlay,
                (inner_start_x, y),
                (inner_start_x + inner_construct_size_px, y + line_thickness_px),
                (0, 255, 0),
                -1,
            )

        # Blend the overlay with the output image
        alpha = 0.5
        mask = cv2.cvtColor(overlay, cv2.COLOR_RGB2GRAY) > 0
        output_image[mask] = cv2.addWeighted(
            output_image[mask], 1 - alpha, overlay[mask], alpha, 0
        )

        contour_color = (255, 0, 0) if self.show_hd else (0, 0, 255)

        # Process each pore
        ideal_contour = generate_ideal_square_contour(pore_width_px)

        for i in range(num_pores_per_line):
            for j in range(num_pores_per_line):
                x_start = inner_start_x + (j * pore_width_px)
                y_start = inner_start_y + (i * pore_width_px)
                x_end = x_start + pore_width_px
                y_end = y_start + pore_width_px

                center_x = (x_start + x_end) // 2
                center_y = (y_start + y_end) // 2
                pore_centers.append((center_x, center_y))

                x_start = max(0, x_start)
                y_start = max(0, y_start)
                x_end = min(image.shape[1], x_end)
                y_end = min(image.shape[0], y_end)

                if x_end > x_start and y_end > y_start:
                    pore_roi = image[y_start:y_end, x_start:x_end].copy()
                    pore_contour, hausdorff_distance = find_contour_and_hd_in_roi(
                        pore_roi,
                        epsilon_factor,
                        pore_width_px,
                        line_thickness_px,
                    )

                if pore_contour is None or hausdorff_distance is None:
                    hausdorff_distance = 1.0
                    pr_value = "N/A"
                    pore_status_list.append("invalid")
                    pore_contours.append(None)
                else:
                    pore_status_list.append("valid")
                    pore_contour[:, :, 0] += x_start
                    pore_contour[:, :, 1] += y_start
                    pore_contours.append(pore_contour)
                    pr_value = self.calculate_pr_value(pore_contour)
                    cv2.drawContours(
                        output_image,
                        [pore_contour],
                        -1,
                        contour_color,
                        contour_thickness,
                        lineType=cv2.LINE_AA,
                    )

                hd_values_list.append(hausdorff_distance)
                pr_values_list.append(pr_value)

        # Calculate average Pr value excluding N/A values
        valid_pr_values = [pr for pr in pr_values_list if pr != "N/A"]
        average_pr = np.mean(valid_pr_values).round(3) if valid_pr_values else "N/A"
        average_hd = np.mean(hd_values_list).round(3)

        hd_values_list = [round(val, 3) for val in hd_values_list]
        pr_values_list = [
            round(val, 3) if isinstance(val, float) else val for val in pr_values_list
        ]

        # Calculate construct area
        self.construct_area_mm2 = self.calculate_construct_area(image)

        return (
            hd_values_list,
            pr_values_list,
            average_hd,
            average_pr,
            output_image,
            pore_centers,
            pore_contours,
            self.construct_area_mm2,  # Add construct area to return values
        )

    def update_plot(self):
        if self.current_image_path is None:
            return

        self.ax.clear()

        # Remove the tick labels
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        (
            self.hd_list,
            self.pr_list,
            self.avg_hd,
            self.avg_pr,
            self.output_image,
            pore_centers,
            self.pore_contours,
            self.construct_area_mm2,
        ) = self.run_analysis_with_modified_find_contours()

        # Display the image
        self.ax.imshow(self.output_image)

        # Display either HD or Pr values based on toggle
        values_to_show = self.hd_list if self.show_hd else self.pr_list
        average_value = self.avg_hd if self.show_hd else self.avg_pr
        metric_name = "HD" if self.show_hd else "Pr"

        for i, (center_x, center_y) in enumerate(pore_centers):
            value = values_to_show[i]
            if isinstance(value, str):
                text_value = value
                text_color = "white"
            else:
                text_value = f"{value:.3f}"
                text_color = "black" if value != 1.0 else "white"

            self.ax.text(
                center_x,
                center_y,
                text_value,
                color=text_color,
                fontsize=16,
                ha="center",
                va="center",
            )

        # Add legend patches
        from matplotlib.patches import Rectangle

        pore_label = (
            "Pore Contour with HD Values"
            if self.show_hd
            else "Pore Contour with Pr Values"
        )
        legend_handles = [
            Rectangle(
                (0, 0),
                1,
                1,
                facecolor=(0, 1, 0),
                edgecolor="black",
                label="Ideal Construct",
            ),  # Green
            Rectangle(
                (0, 0),
                1,
                1,
                facecolor=(1, 0, 0) if self.show_hd else (0, 0, 1),
                edgecolor="black",
                label=pore_label,
            ),  # Red for HD, Blue for Pr
        ]

        self.ax.legend(
            handles=legend_handles,
            loc="upper center",
            fancybox=False,
            edgecolor="black",
            framealpha=1,
            fontsize=16,
            bbox_to_anchor=(0.525, 1.08),
        )

        # Update result label with construct area
        if self.show_hd:
            valid_count = sum(1 for val in values_to_show if val != 1.0)
        else:
            valid_count = sum(1 for val in values_to_show if val != "N/A")
        total_count = len(values_to_show)

        # Format average value display
        avg_display = (
            f"{average_value:.3f}"
            if isinstance(average_value, (float, int))
            else average_value
        )

        # Set title including construct area
        # self.ax.set_title(
        #     f"Average {metric_name}: {avg_display}, Valid: {valid_count}/{total_count}, "
        #     f"Area: {self.construct_area_mm2} mm²",
        #     pad=35,
        # )

        self.canvas.draw()
        self.save_button.state(["!disabled"])

    def save_results_to_csv(self, data, csv_path):
        """
        Save analysis results to a CSV file, overwriting previous entries for the same image.
        """
        import csv
        import datetime
        from tempfile import NamedTemporaryFile
        import shutil

        # Create header
        header = [
            "Date",
            "Time",
            "Image",
            "X_Offset",
            "Y_Offset",
            "Rotation",
            "Average_HD",
            "Average_Pr",
            "Construct_Area_mm2",
        ]

        # Add headers for individual values
        num_values = len(data["hd_values"])
        for i in range(num_values):
            header.extend([f"HD_{i+1}", f"Pr_{i+1}"])

        # Get current date and time
        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        # Prepare new row data
        current_image = os.path.basename(data["image_path"])
        new_row_data = [
            date_str,
            time_str,
            current_image,
            data["x_offset"],
            data["y_offset"],
            data["rotation_angle"],
            data["average_hd"],
            (
                data["average_pr"]
                if isinstance(data["average_pr"], (float, int))
                else "N/A"
            ),
            data["construct_area_mm2"],
        ]

        # Add individual HD and Pr values
        for hd, pr in zip(data["hd_values"], data["pr_values"]):
            new_row_data.extend([hd, pr])

        # Create a temporary file
        tempfile = NamedTemporaryFile(mode="w", delete=False, newline="")

        if os.path.exists(csv_path):
            # If file exists, read existing data and update
            with open(csv_path, "r", newline="") as csvfile, tempfile:
                reader = csv.reader(csvfile)
                writer = csv.writer(tempfile)

                # Write header
                header_row = next(reader, None)
                if header_row:
                    writer.writerow(header_row)
                else:
                    writer.writerow(header)

                # Track if found and updated the existing entry
                entry_updated = False

                for row in reader:
                    if row[2] == current_image:  # Check image name (index 2)
                        writer.writerow(new_row_data)
                        entry_updated = True
                    else:
                        writer.writerow(row)

                if not entry_updated:
                    writer.writerow(new_row_data)
        else:
            with tempfile:
                writer = csv.writer(tempfile)
                writer.writerow(header)
                writer.writerow(new_row_data)

        # Replace the original file with the temporary file
        shutil.move(tempfile.name, csv_path)

    def save_results(self):
        if self.current_image_path is None or self.hd_list is None:
            return

        # Create the directory if it doesn't exist
        if not os.path.exists("csv_data_files"):
            os.makedirs("csv_data_files")

        # Prepare data for saving
        data = {
            "image_path": self.current_image_path,
            "x_offset": self.x_offset,
            "y_offset": self.y_offset,
            "rotation_angle": self.rotation_angle,
            "average_hd": self.avg_hd,
            "average_pr": self.avg_pr,
            "hd_values": self.hd_list,
            "pr_values": self.pr_list,
            "construct_area_mm2": self.construct_area_mm2,
        }

        # Save to CSV
        self.save_results_to_csv(
            data, csv_path="csv_data_files/lattice_analysis_results.csv"
        )

        # Save the plot image - overwrite existing one
        if not os.path.exists("hd_values_images"):
            os.makedirs("hd_values_images")

        output_image_path = f"hd_values_images/analysis_results_{os.path.basename(self.current_image_path)}"
        plt.savefig(output_image_path, dpi=600, bbox_inches="tight", pad_inches=0)

    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.tif *.tiff")]
        )
        if file_path:
            self.current_image_path = file_path
            self.original_image = None  # Reset original image
            self.reset_all()
            self.update_plot()

    def adjust_offset(self, axis, amount):
        if axis == "x":
            self.x_offset += amount
            self.x_offset_label.config(text=str(self.x_offset))
        else:
            self.y_offset += amount
            self.y_offset_label.config(text=str(self.y_offset))
        self.update_plot()

    def adjust_rotation(self, amount):
        self.rotation_angle += amount
        self.rotation_angle = round(self.rotation_angle, 1)
        self.rotation_label.config(text=f"{self.rotation_angle}°")
        self.update_plot()

    def rotate_image(self, image, angle):
        if angle == 0:
            return image

        height, width = image.shape[:2]
        center = (width // 2, height // 2)

        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Create a white background
        rotated_image = np.full_like(image, 255)

        # Rotate the image
        temp_rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

        # Only copy non-white pixels from the rotated image
        mask = cv2.cvtColor(temp_rotated, cv2.COLOR_RGB2GRAY) < 255
        rotated_image[mask] = temp_rotated[mask]

        return rotated_image

    def reset_all(self):
        self.x_offset = 0
        self.y_offset = 0
        self.rotation_angle = 0
        self.x_offset_label.config(text="0")
        self.y_offset_label.config(text="0")
        self.rotation_label.config(text="0°")
        if self.current_image_path:
            self.update_plot()


def save_results_to_csv(data, csv_path):
    """Save analysis results to a CSV file."""
    import csv
    import datetime

    # Create header and row data
    header = [
        "Date",
        "Time",
        "Image",
        "X_Offset",
        "Y_Offset",
        "Rotation",
        "Average_HD",
        "Average_Pr",
    ]

    # Add headers for individual values
    num_values = len(data["hd_values"])
    for i in range(num_values):
        header.extend([f"HD_{i+1}", f"Pr_{i+1}"])

    # Get current date and time
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    # Prepare row data
    row_data = [
        date_str,
        time_str,
        os.path.basename(data["image_path"]),
        data["x_offset"],
        data["y_offset"],
        data["rotation_angle"],
        data["average_hd"],
        data["average_pr"] if isinstance(data["average_pr"], (float, int)) else "N/A",
    ]

    # Add individual HD and Pr values
    for hd, pr in zip(data["hd_values"], data["pr_values"]):
        row_data.extend([hd, pr])

    # Write to CSV
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row_data)


if __name__ == "__main__":
    root = tk.Tk()
    app = LatticeAnalysisGUI(root)
    root.protocol("WM_DELETE_WINDOW", root.quit)
    root.mainloop()
