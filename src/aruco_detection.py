def extract_white_bits(image, corners, grid_size=4):
    """
    Extracts and visualizes the white bits inside a detected ArUco marker.
    Args:
        image: Original image (numpy array)
        corners: Marker corner coordinates (4x2 array)
        grid_size: Number of bits per side (default 4 for DICT_4X4)
    Returns:
        bits: 2D numpy array of 0 (black) and 1 (white) bits
        warped: Warped marker image
    """
    # Define destination points for perspective transform
    dst_pts = np.array([
        [0, 0],
        [grid_size*10-1, 0],
        [grid_size*10-1, grid_size*10-1],
        [0, grid_size*10-1]
    ], dtype=np.float32)
    src_pts = corners.astype(np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (grid_size*10, grid_size*10))
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bits = np.zeros((grid_size, grid_size), dtype=np.uint8)
    cell_size = 10
    for i in range(grid_size):
        for j in range(grid_size):
            cell = bw[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            bits[i, j] = 1 if np.mean(cell) > 127 else 0
    return bits, warped
import cv2
import argparse
import sys
import numpy as np
from hylite.analyse import band_ratio
import hylite
from hylite import io


def getAruco(image, border_detect = False):
    # Common ArUco dictionaries to try (prioritize smaller sizes for your marker)
    ARUCO_DICT = {
        "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
        # "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        # "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        # "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        # "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        # "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL
    }

    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", required=True, help="path to input image")
    # args = vars(ap.parse_args())
    

    # image = cv2.imread(path)
    if image is None:
        print("Error: Could not load image.")
        sys.exit(1)

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Default parameters
    aruco_params = cv2.aruco.DetectorParameters()

    print("[INFO] Detecting markers...")
    marker_dict = {}  # Dictionary to store marker id and corners
    for dict_name, dict_id in ARUCO_DICT.items():
        print(f"Trying {dict_name}...")
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            print(f"[INFO] Detected {len(ids)} markers using {dict_name}")
            print("Marker IDs:", ids.flatten().tolist())
            # Save marker ids and corners in the dictionary
            for marker_id, marker_corners in zip(ids.flatten(), corners):
                marker_dict[int(marker_id)] = marker_corners
            # Draw detected markers
            cv2.aruco.drawDetectedMarkers(image, corners, ids)
            # Process all borders if requested
            if border_detect:
                detect_white_border2(corners, image)  # Example for the first marker
            else:
                # Resize for display if large
                display = cv2.resize(image, (960, max(540, int(h * 960 / w))))
                cv2.imshow('Detected ArUco Markers', display)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            # # Save output
            # cv2.imwrite('detected_markers.jpg', image)
            # print("Output saved as 'detected_markers.jpg'")
            break
    else:
        print("[INFO] No markers detected. Check image quality, lighting, or try custom dictionary. Rejected candidates:", len(rejected))
    return marker_dict

def calculate_expected_border_from_geometry(marker_corners, marker_size_mm=18, total_size_mm=25):
    """
    Calculate expected outer border corners based on known marker geometry.
    
    Args:
        marker_corners: 4x2 array of inner marker corner coordinates
        marker_size_mm: Physical size of inner marker (default 18mm)
        total_size_mm: Total size including border (default 25mm)
    
    Returns:
        dict: Contains:
            - 'expected_outer': 4x2 array of expected outer corner coordinates
            - 'avg_marker_size_px': Average marker size in pixels
            - 'pixels_per_mm': Pixel-to-mm ratio
            - 'border_size_px': Expected border size in pixels
            - 'border_size_mm': Border size in mm
    """
    marker_corners = marker_corners.astype(np.float32)
    border_size_mm = (total_size_mm - marker_size_mm) / 2
    
    # Calculate marker size in pixels (average of sides)
    side1 = np.linalg.norm(marker_corners[1] - marker_corners[0])
    side2 = np.linalg.norm(marker_corners[2] - marker_corners[1])
    avg_marker_size_px = (side1 + side2) / 2
    
    # Calculate pixel-to-mm ratio
    pixels_per_mm = avg_marker_size_px / marker_size_mm
    border_size_px = border_size_mm * pixels_per_mm
    
    # Calculate expected outer border corners using known geometry
    center = np.mean(marker_corners, axis=0)
    
    # Normalize vectors from center to each corner
    directions = marker_corners - center
    distances = np.linalg.norm(directions, axis=1, keepdims=True)
    normalized_dirs = directions / distances
    
    # Expected distance from center to outer corner
    half_marker_diag = np.linalg.norm(marker_corners[0] - center)
    outer_diagonal_px = (total_size_mm / 2) * pixels_per_mm
    scale_factor = outer_diagonal_px / half_marker_diag
    
    # Calculate expected outer border corners
    expected_outer = center + normalized_dirs * (distances * scale_factor)
    
    return {
        'expected_outer': expected_outer.astype(np.float32),
        'avg_marker_size_px': avg_marker_size_px,
        'pixels_per_mm': pixels_per_mm,
        'border_size_px': border_size_px,
        'border_size_mm': border_size_mm
    }


def detect_white_border(corners, image):
    # bounding box around the inner corners
    image_copy = image.copy()  # Create copy once for all markers
    
    for corner in corners:
        x, y, w, h = cv2.boundingRect(corner[0].astype(np.float32))
        pad = 20
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(image.shape[1], x + w + pad)
        y1 = min(image.shape[0], y + h + pad)

        roi = image[y0:y1, x0:x1].copy()

        # find outer white rectangle in the ROI
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_roi = cv2.GaussianBlur(gray_roi, (5, 5), 1.0)
        edges = cv2.Canny(gray_roi, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = max(contours, key=cv2.contourArea)
        
        # Use minAreaRect to get exactly 4 corner points for a rectangular shape
        rect = cv2.minAreaRect(best)
        box = cv2.boxPoints(rect)  # Get 4 corners of the rotated rectangle
        
        # convert back to image coordinates, expect 4 points
        outer_refined = box.astype(np.float32)
        outer_refined[:, 0] += x0
        outer_refined[:, 1] += y0
        
        # Plot the outer border on the image
        pts = outer_refined.astype(np.int32)
        cv2.polylines(image_copy, [pts], True, (0, 255, 0), 3)  # Green border
        for i, pt in enumerate(pts):
            cv2.circle(image_copy, tuple(pt), 8, (0, 0, 255), -1)  # Red circles at corners
            cv2.putText(image_copy, str(i), tuple(pt + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
    # Display the result (outside loop with all borders)
    display = cv2.resize(image_copy, (960, max(540, int(image.shape[0] * 960 / image.shape[1]))))
    cv2.imshow('Outer Border Detection', display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_white_border2(corners, image):
    """
    Detect white borders using known marker geometry.
    Compares expected border (from geometry) with detected border.
    
    Args:
        corners: ArUco marker corners
        image: Input image
    """
    # Known physical dimensions (in mm)
    marker_size_mm = 18  # Inner marker size
    total_size_mm = 25   # Total size including border
    
    image_copy = image.copy()
    
    for corner in corners:
        marker_corners = corner[0].astype(np.float32)
        
        # Calculate expected outer border using geometry
        geom_result = calculate_expected_border_from_geometry(marker_corners, marker_size_mm, total_size_mm)
        expected_outer = geom_result['expected_outer']
        pixels_per_mm = geom_result['pixels_per_mm']
        border_size_px = geom_result['border_size_px']
        avg_marker_size_px = geom_result['avg_marker_size_px']
        border_size_mm = geom_result['border_size_mm']
        
        print(f"\n--- Marker Geometry Analysis ---")
        print(f"Marker size in pixels: {avg_marker_size_px:.2f}px")
        print(f"Pixels per mm: {pixels_per_mm:.2f} px/mm")
        print(f"Expected border size: {border_size_px:.2f}px ({border_size_mm}mm)")
        
        # Plot marker corners (inner)
        marker_pts = marker_corners.astype(np.int32)
        cv2.polylines(image_copy, [marker_pts], True, (255, 0, 0), 2)  # Blue for marker
        
        # Plot expected outer border
        outer_pts = expected_outer.astype(np.int32)
        cv2.polylines(image_copy, [outer_pts], True, (0, 165, 255), 2)  # Orange for expected
        
        # Try to detect actual outer border
        x, y, w, h = cv2.boundingRect(marker_corners)
        pad = 20
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(image.shape[1], x + w + pad)
        y1 = min(image.shape[0], y + h + pad)

        roi = image[y0:y1, x0:x1].copy()

        # find outer white rectangle in the ROI
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_roi = cv2.GaussianBlur(gray_roi, (5, 5), 1.0)
        edges = cv2.Canny(gray_roi, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            best = max(contours, key=cv2.contourArea)
            
            # Use minAreaRect to get exactly 4 corner points for a rectangular shape
            rect = cv2.minAreaRect(best)
            box = cv2.boxPoints(rect)  # Get 4 corners of the rotated rectangle
            
            # convert back to image coordinates, expect 4 points
            actual_outer = box.astype(np.float32)
            actual_outer[:, 0] += x0
            actual_outer[:, 1] += y0
            
            # Plot actual outer border (green)
            actual_pts = actual_outer.astype(np.int32)
            cv2.polylines(image_copy, [actual_pts], True, (0, 255, 0), 3)  # Green border
            
            for i, pt in enumerate(actual_pts):
                cv2.circle(image_copy, tuple(pt), 8, (0, 0, 255), -1)  # Red circles at corners
                cv2.putText(image_copy, str(i), tuple(pt + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Draw legend
    cv2.putText(image_copy, "Blue=Marker Inner, Orange=Expected Outer, Green=Detected Outer", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display the result
    display = cv2.resize(image_copy, (960, max(540, int(image.shape[0] * 960 / image.shape[1]))))
    cv2.imshow('Outer Border Detection with Geometry', display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()





def extract_white_bits(image, corners, grid_size=4):
        """
        Extracts and visualizes the white bits inside a detected ArUco marker.
        Args:
            image: Original image (numpy array)
            corners: Marker corner coordinates (4x2 array)
            grid_size: Number of bits per side (default 4 for DICT_4X4)
        Returns:
            bits: 2D numpy array of 0 (black) and 1 (white) bits
            warped: Warped marker image
        """
        # Define destination points for perspective transform
        dst_pts = np.array([
            [0, 0],
            [grid_size*10-1, 0],
            [grid_size*10-1, grid_size*10-1],
            [0, grid_size*10-1]
        ], dtype=np.float32)
        src_pts = corners.astype(np.float32)
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, M, (grid_size*10, grid_size*10))
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bits = np.zeros((grid_size, grid_size), dtype=np.uint8)
        cell_size = 10
        for i in range(grid_size):
            for j in range(grid_size):
                cell = bw[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
                bits[i, j] = 1 if np.mean(cell) > 127 else 0
        return bits, warped

        ids, bboxs, _ = getAruco(image, bands=[0,1,2])  # Example bands
        for i, corners in enumerate(bboxs):
            bits, warped = extract_white_bits(image, corners, grid_size=4)
            print(f"Marker {ids[i]} bits:\n{bits}")
            cv2.imshow(f"Marker {ids[i]}", warped)
            cv2.waitKey(0)
        cv2.destroyAllWindows()

def temp():
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", required=True, help="path to input image")
    # ap.add_argument("-b", "--border", action="store_true", help="also detect outer white border corners")
    # args = vars(ap.parse_args())

    image = cv2.imread('/Users/nova98/Documents/Nova/Marker-detection/test/img12.png')
    if image is None:
        print(f"Error: Could not load image at '/Users/nova98/Documents/Nova/Marker-detection/test/img12.png'")
        sys.exit(1)

    marker_dict = getAruco(image, border_detect=True)

    if marker_dict:
        print(f"\n[RESULTS] Detected {len(marker_dict)} marker(s):")
        for marker_id, corners in marker_dict.items():
            pts = corners[0]  # shape (4, 2): TL, TR, BR, BL
            print(f"  Marker ID {marker_id}:")
            labels = ["top-left", "top-right", "bottom-right", "bottom-left"]
            for label, pt in zip(labels, pts):
                print(f"    {label}: ({pt[0]:.1f}, {pt[1]:.1f})")
        # Show final image with detected markers and borders
        corners = list(marker_dict.values())
        if corners:
            detect_white_border2(corners, image)
        else:
            print("No markers detected for border display.")

if __name__ == "__main__":
    path = '/Users/nova98/Documents/Nova/Marker-detection/test/img12.png'
    image = cv2.imread(path)
    marker_dict = getAruco(image, border_detect=False)
    # temp()
