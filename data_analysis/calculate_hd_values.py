import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from typing import Final

SCALE_1mm_IN_PIXEL: Final[int] = 250 // 5  # 800 * 800 pixels


def save_results_to_csv(
    image_path, x_offset, y_offset, average_hd, hd_values, csv_path
):
    import pandas as pd
    import os
    import re

    # Extract numeric ID from filename using regex
    filename = os.path.basename(image_path)
    match = re.search(r"(\d+)", filename)
    image_id = match.group(1) if match else filename

    # Create data dictionary
    data = {
        "image_id": [image_id],
        "offset_x_y": [f"({x_offset},{y_offset})"],
        "average_hd": [average_hd],
    }

    # Add individual HD values
    for i, hd in enumerate(hd_values, 1):
        data[f"hd_{i}"] = [hd]

    # Create DataFrame for new data
    df_new = pd.DataFrame(data)

    try:
        if os.path.exists(csv_path):
            # Read existing CSV
            df_existing = pd.read_csv(csv_path)

            # Convert image_id to string in both dataframes to ensure consistent comparison
            df_existing["image_id"] = df_existing["image_id"].astype(str)
            df_new["image_id"] = df_new["image_id"].astype(str)

            # Remove old entry for this image_id
            mask = df_existing["image_id"].astype(str) != image_id
            df_existing = df_existing[mask]

            # Add new data
            df_final = pd.concat([df_existing, df_new], ignore_index=True)

            # Sort by image_id to maintain consistent ordering
            df_final["image_id"] = df_final["image_id"].astype(
                str
            )  # Ensure string type for sorting
            df_final = df_final.sort_values("image_id").reset_index(drop=True)

        else:
            df_final = df_new

        df_final.to_csv(csv_path, index=False)
        print(
            f"Results saved to {csv_path} (ID: {image_id}, offsets: x={x_offset}, y={y_offset}, avg_HD: {average_hd:.2f})"
        )

    except Exception as e:
        print(f"Error saving to CSV: {str(e)}")


def resample_contour(contour, num_points):
    """Resample a contour to have num_points points."""
    contour = contour.reshape(-1, 2)
    x, y = contour[:, 0], contour[:, 1]

    # Create a parameterization variable
    t = np.linspace(0, 1, len(contour))

    # Create interpolating functions for x and y
    fx = interp1d(t, x, kind="linear")
    fy = interp1d(t, y, kind="linear")

    # Create new parameterization variable for resampled points
    new_t = np.linspace(0, 1, num_points)

    # Generate resampled x and y points
    new_x = fx(new_t)
    new_y = fy(new_t)

    # Combine x and y to get new resampled contour
    new_contour = np.array([new_x, new_y]).T
    new_contour = new_contour.reshape(-1, 1, 2).astype(np.int32)

    return new_contour


def hausdorff_distance(setA, setB):
    """Calculate the Hausdorff Distance between two sets of points."""

    def one_way_hausdorff(set_from, set_to):
        """Calculate the one-way Hausdorff Distance from set_from to set_to."""
        return max(np.min(np.linalg.norm(set_to - a, axis=1)) for a in set_from)

    return max(one_way_hausdorff(setA, setB), one_way_hausdorff(setB, setA))


def generate_inner_boundary_contour(pore_width_px, line_thickness_px, num_points=100):
    """Generate the inner boundary contour of the pore area."""
    # Calculate the actual boundary coordinates considering the line thickness
    start = line_thickness_px // 2
    end = pore_width_px - line_thickness_px // 2

    # Create arrays for x and y coordinates
    points = []

    # Generate points for all four sides
    sides = [
        # Top edge: vary x, constant y=start
        [(x, start) for x in np.linspace(start, end, num_points // 4)],
        # Right edge: constant x=end, vary y
        [(end, y) for y in np.linspace(start, end, num_points // 4)],
        # Bottom edge: vary x, constant y=end
        [(x, end) for x in np.linspace(end, start, num_points // 4)],
        # Left edge: constant x=start, vary y
        [(start, y) for y in np.linspace(end, start, num_points // 4)],
    ]

    # Flatten the points list
    points = [point for side in sides for point in side]

    # Convert to numpy array with correct shape for OpenCV contours
    return np.array(points, dtype=np.int32).reshape(-1, 1, 2)


def find_contour_and_hd_in_roi(roi, epsilon_factor, pore_width_px, line_thickness_px):
    """Modified function to calculate HD between pore contour and inner boundary."""
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Generate inner boundary contour
    inner_boundary = generate_inner_boundary_contour(pore_width_px, line_thickness_px)

    # Center the inner boundary in the ROI
    roi_center_x = roi.shape[1] // 2
    roi_center_y = roi.shape[0] // 2

    M_boundary = cv2.moments(inner_boundary)
    if M_boundary["m00"] != 0:  # Check for division by zero
        cX_boundary = int(M_boundary["m10"] / M_boundary["m00"])
        cY_boundary = int(M_boundary["m01"] / M_boundary["m00"])

        dx = roi_center_x - cX_boundary
        dy = roi_center_y - cY_boundary
        centered_boundary = inner_boundary + [dx, dy]
    else:
        return None, None

    if not contours:
        return None, None

    # Filter and process contours
    valid_contours = []
    min_area = (pore_width_px * 0.1) ** 2
    max_perimeter = 500

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if perimeter > max_perimeter:
            continue

        if area > min_area:
            valid_contours.append((contour, perimeter))

    if not valid_contours:
        return None, None

    longest_contour = max(valid_contours, key=lambda x: x[1])[0]
    epsilon = epsilon_factor * cv2.arcLength(longest_contour, True)
    approx_contour = cv2.approxPolyDP(longest_contour, epsilon, True)

    try:
        # Resample contour
        resampled_approx_contour = resample_contour(
            approx_contour, centered_boundary.shape[0]
        )

        # Calculate area ratio
        area_ratio = cv2.contourArea(approx_contour) / (pore_width_px**2)

        if area_ratio < 0.01:
            return None, None

        # Calculate HD between pore contour and inner boundary
        hd = (
            hausdorff_distance(
                np.squeeze(resampled_approx_contour), np.squeeze(centered_boundary)
            )
            / pore_width_px
        ).round(3)

        if hd > 1.0:
            return None, None

        # Also draw the inner boundary for visualization
        cv2.drawContours(roi, [centered_boundary], -1, (0, 255, 0), 1)

        return resampled_approx_contour, hd

    except Exception as e:
        print(f"Error processing contour: {str(e)}")
        return None, None


def find_all_contours_and_hd(
    image_path, margin=35, should_plot=False, x_offset=0, y_offset=0
):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = mask_outer_background(image_path)

    contour_thickness = 5
    epsilon_factor = 0.0005
    hd_values_list = []
    pore_status_list = []

    output_image = image.copy()
    image_size = image.shape[0]

    # Define parameters
    construct_size_mm = 12.2
    nozzle_diameter_mm = 0.2  # Inner diameter of the nozzle
    line_center_offset_mm = nozzle_diameter_mm / 2
    num_pores_per_line = 5

    # Calculate actual line thickness in pixels
    line_thickness_mm = nozzle_diameter_mm
    line_thickness_px = int(line_thickness_mm * SCALE_1mm_IN_PIXEL)

    # Calculate offset from line center in pixels
    line_center_offset_px = int(line_center_offset_mm * SCALE_1mm_IN_PIXEL)

    # Calculate pore width considering the line thickness
    pore_width_mm = (
        construct_size_mm - (line_thickness_mm * (num_pores_per_line + 1))
    ) / num_pores_per_line
    pore_width_px = int(pore_width_mm * SCALE_1mm_IN_PIXEL)

    # Calculate the total lattice size in pixels
    total_lattice_size_px = (pore_width_px * num_pores_per_line) + (
        line_thickness_px * (num_pores_per_line - 1)
    )

    # Modified starting points to use the offset parameters
    start_x = (
        (image_size - total_lattice_size_px) // 2 + line_center_offset_px + x_offset
    )
    start_y = (
        (image_size - total_lattice_size_px) // 2 + line_center_offset_px + y_offset
    )

    # Draw outer boundary with actual line width
    # Draw each side of the boundary separately as filled rectangles
    outer_box_start_x = start_x - line_thickness_px
    outer_box_start_y = start_y - line_thickness_px
    outer_box_end_x = start_x + total_lattice_size_px
    outer_box_end_y = start_y + total_lattice_size_px

    # Top horizontal line
    cv2.rectangle(
        output_image,
        (outer_box_start_x, outer_box_start_y),
        (outer_box_end_x, outer_box_start_y + line_thickness_px),
        (0, 255, 0),
        -1,  # Filled rectangle
    )

    # Bottom horizontal line
    cv2.rectangle(
        output_image,
        (outer_box_start_x, outer_box_end_y - line_thickness_px),
        (outer_box_end_x, outer_box_end_y),
        (0, 255, 0),
        -1,
    )

    # Left vertical line
    cv2.rectangle(
        output_image,
        (outer_box_start_x, outer_box_start_y),
        (outer_box_start_x + line_thickness_px, outer_box_end_y),
        (0, 255, 0),
        -1,
    )

    # Right vertical line
    cv2.rectangle(
        output_image,
        (outer_box_end_x - line_thickness_px, outer_box_start_y),
        (outer_box_end_x, outer_box_end_y),
        (0, 255, 0),
        -1,
    )

    # Draw inner grid lines with proper offset
    for i in range(1, num_pores_per_line):
        # Vertical lines
        x = (
            start_x
            + (i * (pore_width_px + line_thickness_px))
            - line_thickness_px
            - line_center_offset_px
        )
        cv2.rectangle(
            output_image,
            (x, start_y - line_thickness_px),
            (x + line_thickness_px, start_y + total_lattice_size_px),
            (0, 255, 0),
            -1,
        )

        # Horizontal lines
        y = (
            start_y
            + (i * (pore_width_px + line_thickness_px))
            - line_thickness_px
            - line_center_offset_px
        )
        cv2.rectangle(
            output_image,
            (start_x - line_thickness_px, y),
            (start_x + total_lattice_size_px, y + line_thickness_px),
            (0, 255, 0),
            -1,
        )

    # Calculate the step size between pores
    step_size = pore_width_px + line_thickness_px

    # Generate ideal contour with offset consideration
    ideal_contour = generate_ideal_square_contour(
        pore_width_px + 2 * line_center_offset_px
    )

    if should_plot:
        fig, axs = plt.subplots(
            num_pores_per_line, num_pores_per_line, figsize=(12, 12)
        )

    for i in range(num_pores_per_line):
        for j in range(num_pores_per_line):
            # Calculate ROI coordinates with offset
            x_start = start_x + (j * step_size) - margin - line_center_offset_px
            y_start = start_y + (i * step_size) - margin - line_center_offset_px
            x_end = x_start + pore_width_px + 2 * line_center_offset_px + (2 * margin)
            y_end = y_start + pore_width_px + 2 * line_center_offset_px + (2 * margin)

            # Ensure coordinates are within image bounds
            x_start = max(0, x_start)
            y_start = max(0, y_start)
            x_end = min(image.shape[1], x_end)
            y_end = min(image.shape[0], y_end)

            if x_end > x_start and y_end > y_start:
                pore_roi = image[y_start:y_end, x_start:x_end].copy()
                pore_contour, hausdorff_distance = find_contour_and_hd_in_roi(
                    pore_roi,
                    epsilon_factor,
                    ideal_contour,
                    pore_width_px + 2 * line_center_offset_px,
                )

                # Calculate center position for ideal contour in the full image
                roi_center_x = (x_start + x_end) // 2
                roi_center_y = (y_start + y_end) // 2

                # Center the ideal contour
                M_ideal = cv2.moments(ideal_contour)
                cX_ideal = int(M_ideal["m10"] / M_ideal["m00"])
                cY_ideal = int(M_ideal["m01"] / M_ideal["m00"])

                # Translate ideal contour to ROI center in the full image
                dx = roi_center_x - cX_ideal
                dy = roi_center_y - cY_ideal
                centered_ideal_contour = ideal_contour + [dx, dy]

                # Assign HD=1.0 for invalid/covered pores
                if pore_contour is None or hausdorff_distance is None:
                    hausdorff_distance = 1.0
                    pore_status_list.append("invalid")
                else:
                    pore_status_list.append("valid")
                    # Draw detected contour (red) on both ROI and output image
                    cv2.drawContours(
                        pore_roi,
                        [pore_contour],
                        -1,
                        (255, 0, 0),
                        contour_thickness,
                    )
                    cv2.drawContours(
                        output_image,
                        [pore_contour],
                        -1,
                        (255, 0, 0),
                        contour_thickness,
                        offset=(x_start, y_start),
                    )

                hd_values_list.append(hausdorff_distance)

                if should_plot:
                    axs[i, j].imshow(pore_roi)
                    axs[i, j].set_xticks([])
                    axs[i, j].set_yticks([])
                    if hausdorff_distance == 1.0 and pore_status_list[-1] == "invalid":
                        axs[i, j].set_title("HD: 1.00 (Invalid)", fontsize=10)
                    else:
                        axs[i, j].set_title(
                            f"HD: {hausdorff_distance:.2f}", fontsize=10
                        )

    if should_plot:
        plt.tight_layout()

        plt.figure(figsize=(10, 8))
        plt.imshow(output_image)
        avg_hd = np.mean(hd_values_list).round(3)
        valid_count = sum(1 for status in pore_status_list if status == "valid")
        total_count = len(pore_status_list)
        plt.title(f"Average HD: {avg_hd}, Valid: {valid_count}/{total_count}")
        plt.tight_layout()
        plt.show()

    average_hd = np.mean(hd_values_list).round(3)
    return hd_values_list, average_hd


def calculate_roi_coordinates(
    center, pore_width, line_thickness, i, j, margin, image_shape
):
    """Calculate the coordinates of the region of interest for a pore."""

    x_offset = i * (pore_width + line_thickness)
    y_offset = j * (pore_width + line_thickness)
    x_start = max(0, center[0] + x_offset - margin)
    y_start = max(0, center[1] + y_offset - margin)
    x_end = min(image_shape[1], center[0] + x_offset + pore_width + margin)
    y_end = min(image_shape[0], center[1] + y_offset + pore_width + margin)
    return x_start, y_start, x_end, y_end


def mask_outer_background(image_path):
    """Mask out the outer background, identifying the lattice contour using position and shape."""

    # Read and convert the image
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # change to mono color (all white or all black)
    _, gray_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)

    # Use adaptive thresholding to better handle varying lighting conditions
    thresh = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on multiple criteria
    valid_contours = []
    image_center = np.array([image.shape[1] // 2, image.shape[0] // 2])

    for contour in contours:
        # Calculate contour properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        M = cv2.moments(contour)

        # Calculate contour center
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            contour_center = np.array([cx, cy])

            # Calculate distance from image center
            distance_from_center = np.linalg.norm(contour_center - image_center)

            # Calculate shape complexity (ratio of area to squared perimeter)
            # Square-like shapes have higher ratios
            shape_factor = (
                4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            )

            distance_score = 1 / (1 + distance_from_center / 100)  # Normalize distance
            shape_score = shape_factor  # Already normalized between 0 and 1
            area_score = min(
                area / (image.shape[0] * image.shape[1]), 1
            )  # Normalize area

            total_score = (distance_score + shape_score + area_score) / 3

            valid_contours.append((contour, total_score))

    if not valid_contours:
        print("Warning: No valid lattice contour found")
        return image

    # Sort by score and take the highest scoring contour
    lattice_contour = max(valid_contours, key=lambda x: x[1])[0]

    # Create mask
    mask = np.zeros_like(gray_image)
    cv2.fillPoly(mask, [lattice_contour], (255, 255, 255))

    # Perform morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Apply mask to original image
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    return masked_image


def generate_ideal_square_contour(square_size, num_points=100):
    points = []
    # Top edge
    for x in np.linspace(0, square_size, num_points):
        points.append([x, 0])
    # Right edge
    for y in np.linspace(0, square_size, num_points):
        points.append([square_size, y])
    # Bottom edge
    for x in np.linspace(square_size, 0, num_points):
        points.append([x, square_size])
    # Left edge
    for y in np.linspace(square_size, 0, num_points):
        points.append([0, y])

    return np.array(points, dtype=np.int32).reshape((-1, 1, 2))


if __name__ == "__main__":
    image_path = r""
    image_path = image_path.replace("\\", "/")

    x_offset = -2
    y_offset = -5

    hd_list, avg_hd = find_all_contours_and_hd(
        image_path, margin=0, should_plot=True, x_offset=x_offset, y_offset=y_offset
    )

    # Save results to CSV
    save_results_to_csv(
        image_path=image_path,
        x_offset=x_offset,
        y_offset=y_offset,
        average_hd=avg_hd,
        hd_values=hd_list,
        csv_path="csv_data_files/lattice_analysis_results.csv",
    )

    print(f"Hausdorff Distance values for each pore: {hd_list}")
    print(f"Average Hausdorff Distance: {avg_hd}")
