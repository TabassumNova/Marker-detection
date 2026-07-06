# Marker-detection
Aruco and barcode markers

Aruco creation
---------------------------

- `python3 ./src/aruco_creation.py --rows 9 --columns 12 -T charuco_board --square_size 13 --marker_size 8 -f ./aruco_pattern/DICT_6X6_1000.json`
- out.svg will be created in the working directory

Aruco inner and outer border detection
---------------------------

<img src="images/Aruco_detection.png" alt="Aruco Detection" width="600" height="400" />

# References
- https://docs.opencv.org/4.x/da/d0d/tutorial_camera_calibration_pattern.html#autotoc_md255
- https://www.geeksforgeeks.org/python/how-to-generate-barcode-in-python/
- https://www.geeksforgeeks.org/python/detect-and-read-barcodes-with-opencv-in-python/
