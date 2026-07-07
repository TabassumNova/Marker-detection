import cv2
import argparse
import sys
import numpy as np


def getAruco(image, aruco_dict_id, visualisation = True):
    if image is None:
        print("Error: Could not load image.")
        sys.exit(1)

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Default parameters
    aruco_params = cv2.aruco.DetectorParameters()

    print("[INFO] Detecting markers...")
    marker_dict = {}
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, ids, rejected = detector.detectMarkers(gray)

    inner_vis = image.copy()
    outer_vis = image.copy()
    if ids is not None and len(ids) > 0:
        print(f"[INFO] Detected {len(ids)} markers")
        print("Marker IDs:", ids.flatten().tolist())
        # Draw only detected marker inner corners on inner_vis.
        cv2.aruco.drawDetectedMarkers(inner_vis, corners, ids)

        # Always detect border for each marker id and store both inner/outer corners.
        for marker_id, marker_corners in zip(ids.flatten(), corners):
            marker_id = int(marker_id)
            inner_pts = marker_corners[0].astype(np.int32)
            outer_pts = detect_white_border(marker_corners, image)
            marker_dict[marker_id] = {
                "inner_corners": inner_pts.copy(),
                "outer_corners": outer_pts.copy() if outer_pts is not None else None
            }

            if outer_pts is not None:
                cv2.polylines(outer_vis, [outer_pts], True, (0, 255, 0), 2)
                for i, pt in enumerate(outer_pts):
                    cv2.circle(outer_vis, tuple(pt), 2, (0, 0, 255), -1)

            cv2.putText(
                inner_vis,
                f"{marker_id}",
                tuple(inner_pts[0] + np.array([0, -8])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            cv2.putText(
                outer_vis,
                f"{marker_id}",
                tuple(inner_pts[0] + np.array([0, -8])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

        cv2.imwrite('detected_inner_corners.jpg', inner_vis)
        cv2.imwrite('detected_outer_corners.jpg', outer_vis)

        if visualisation:
            # Final visualization in separate windows.
            inner_display = cv2.resize(inner_vis, (960, max(540, int(h * 960 / w))))
            outer_display = cv2.resize(outer_vis, (960, max(540, int(h * 960 / w))))
            cv2.imshow('Detected ArUco Inner Corners', inner_display)
            cv2.imshow('Detected ArUco Outer Corners', outer_display)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("[INFO] No markers detected. Check image quality, lighting, or try custom dictionary. Rejected candidates:", len(rejected))
    return marker_dict


def find_large_contour(contours):
    best = None
    best_area = 0.0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        rect_area = w * h
        if rect_area > best_area:
            best_area = rect_area
            best = c
    return best

def detect_white_border(corner, image, pad=20):
    """Detect border points for a single ArUco marker corner set."""
    marker_corners = corner[0].astype(np.float32)
    x, y, w, h = cv2.boundingRect(marker_corners)
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(image.shape[1], x + w + pad)
    y1 = min(image.shape[0], y + h + pad)

    roi = image[y0:y1, x0:x1].copy()
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_roi, 50, 150)

    contour_result = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contour_result[0] if len(contour_result) == 2 else contour_result[1]

    if not contours:
        return None

    best = find_large_contour(contours)
    if best is None:
        return None

    rect = cv2.minAreaRect(best)
    box = cv2.boxPoints(rect)
    outer_refined = box.astype(np.float32)
    outer_refined[:, 0] += x0
    outer_refined[:, 1] += y0

    return outer_refined.astype(np.int32)



if __name__ == "__main__":
    path = '/Users/nova98/Documents/Nova/Helios+/FX10/20260410/FX10_Test_2026-04-10_11-23-40/FX10_Test_2026-04-10_11-23-40.png'
    image = cv2.imread(path)
    marker_dict = getAruco(image, aruco_dict_id=cv2.aruco.DICT_4X4_1000)
    pass
