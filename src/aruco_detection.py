import cv2
import argparse
import sys
import numpy as np
from hylite.analyse import band_ratio
import hylite
from hylite import io

# helper function for detecting aruco codes during calibration phase
def getAruco(image, bands, scale=0, denoise=0, thresh=False, pad=5, debug=False):
    """
    Detect Aruco markers in the specified image.

    *Arguments*:
      - image = the image to detect markers in.
      - bands = Either: (1) a list of 3 bands to include when calculating the grayscale image to detect with, or;
                        (2) a tuple containing ([b0:b1], [b2:b3]) defining the LWIR band ratio to use to detect targets.
      - scale = a kernel size for long-wavelength background removal (using a gaussian filter). Set to 0 to disable.
      - denoise = a blur to apply before Aruco detection to remove speckle noise. Default is 0.
      - thresh = True if an Otsu's threshold should be applied to the image before detection to improve contrast.
      - pad = padding to apply to the image borders to allow corner detection near edges.
      - debug = True if a matplotlib figure showing the results should be rendered. Default is False.

    *Returns*:
     - IDs = a list containing the Aruco IDs of any detected markers, or [] if none are found.
     - bbox = a list containing the four coordinates of each marker corner, or [] if no makrers are found.
     - debug = a (m,n,3) RGB image array as returned by cv2.aruco.drawDetectedMarker(...)
    """

    # convert to greyscale
    if isinstance(bands[0], list) or isinstance(bands[0], tuple):  # use band ratio
        img = band_ratio(image, (bands[0][0], bands[0][1]), (bands[1][0], bands[1][1]))
        gray = img.data[..., 0]
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)


        # do percentile clip
        mn, mx = np.nanpercentile(gray, (1, 99))
        gray -= mn
        gray /= (mx - mn)

        # convert to integer
        gray = np.clip(gray * 255, 0, 255).astype(np.uint8)
        img = np.dstack([gray, gray, gray])  # create greyscale image for output

    else:  # extract bands
        img = np.clip(image.data[..., [image.get_band_index(b, thresh=100) for b in bands]], 0, 1)

        # convert to int for opencv
        img = (255 * img).astype(np.uint8)

        # greyscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # remove background by subtracting long wavelengths
    if scale > 0:
        if scale % 2 == 0:  # scale must be an odd number
            scale += 1

        blur = cv2.GaussianBlur(gray, (scale, scale), 0)
        gray = gray - blur

    # apply denoising blur
    if denoise > 0:
        gray = cv2.GaussianBlur(gray, (denoise, denoise), 0)

    # Otsu's thresholding
    if thresh:
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        bw = gray  # no threshold

    if pad > 0:
        t = bw
        bw = np.zeros((pad * 2 + t.shape[0], pad * 2 + t.shape[1]), dtype=np.uint8)
        bw[pad:-pad, pad:-pad] = t

    # detect marker
    # arDict = cv2.aruco.Dictionary(cv2.aruco.DICT_4X4_250)
    # arParam = cv2.aruco.DetectorParameters_create()
    # bboxs, ids, rejected = cv2.aruco.detectMarkers(bw.T, arDict, parameters=arParam)

    aruco_params = cv2.aruco.DetectorParameters()
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    bboxs, ids, rejected = detector.detectMarkers(gray)


    # if ids is not None and len(ids) > 0:
    #     # print(f"[INFO] Detected {len(ids)} markers using {dict_name}")
    #     # print("Marker IDs:", ids.flatten().tolist())
    #     h, w = image.shape[:2]
        
    #     # Draw detected markers
    #     cv2.aruco.drawDetectedMarkers(image, bboxs, ids)
        
    #     # Resize for display if large
    #     display = cv2.resize(image, (960, max(540, int(h * 960 / w))))
    #     cv2.imshow('Detected ArUco Markers', display)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
        

    # fix padding
    for b in bboxs:
        b -= pad

    img = np.transpose(img, (2, 1, 0)).copy()
    C = {190: (0, 0, 255), 13: (0, 255, 0), 126: (0, 255, 255), 17: (255, 0, 255), 10: (0, 0, 128), 62: (128, 0, 128)}

    # fill rejected markers
    for bbx in rejected:
        for cx, cy in bbx[0, :, :]:
            img[:,
            max(int(cy) - 2, 0):min(int(cy) + 2, img.shape[1]),
            max(int(cx) - 2, 0):min(int(cx) + 2, img.shape[2])] = np.array([255, 0, 0])[:, None, None]  # fill with red

    # fill accepted markers and return
    if ids is not None:

        for bbx, fid in zip(bboxs, ids):
            # fill box
            c = np.array(C.get(int(fid), (255, 0, 0)))
            xmn, xmx = np.percentile(bbx[0, :, 0], (0, 100)).astype(np.int)
            ymn, ymx = np.percentile(bbx[0, :, 1], (0, 100)).astype(np.int)
            img[:, ymn:ymx, xmn:xmx] = np.clip(img[:, ymn:ymx, xmn:xmx] + 0.4 * c[:, None, None], 0, 255)

            # plot corner
            cy = int(bbx[0, 0, 1])
            cx = int(bbx[0, 0, 0])
            img[:,
            max(int(cy) - 2, 0):min(int(cy) + 2, img.shape[1]),
            max(int(cx) - 2, 0):min(int(cx) + 2, img.shape[2])] = np.array([255, 255, 255])[:, None,
                                                                  None]  # fill with white

        return ids[:, 0], np.array([bb[0, :, :] for bb in bboxs]).astype(np.int), np.transpose(img, (1, 2, 0))

    else:  # return empty lists if no markers found
        return np.array([]), np.array([]), np.transpose(img, (1, 2, 0))

def getArucoOld(path):
    # Common ArUco dictionaries to try (prioritize smaller sizes for your marker)
    ARUCO_DICT = {
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL
    }

    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", required=True, help="path to input image")
    # args = vars(ap.parse_args())
    

    image = cv2.imread(path)
    if image is None:
        print("Error: Could not load image.")
        sys.exit(1)

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Default parameters
    aruco_params = cv2.aruco.DetectorParameters()

    print("[INFO] Detecting markers...")
    for dict_name, dict_id in ARUCO_DICT.items():
        print(f"Trying {dict_name}...")
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        
        corners, ids, rejected = detector.detectMarkers(gray)
        
        if ids is not None and len(ids) > 0:
            print(f"[INFO] Detected {len(ids)} markers using {dict_name}")
            print("Marker IDs:", ids.flatten().tolist())
            
            # Draw detected markers
            cv2.aruco.drawDetectedMarkers(image, corners, ids)
            
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


if __name__ == "__main__":
    path = '/Users/nova98/Documents/Nova/Spectrolysis/3D_localization/dict_4x4.png'
    getArucoOld(path)
    # image = io.load(path)
    # image = cv2.imread(path)
    # ids, bbox, debug = getAruco(image, bands=[[0, 98, 42]])
    # print("IDs:", ids)
    # print("Bounding Boxes:", bbox)
    # import matplotlib.pyplot as plt
    # plt.imshow(debug)
    # plt.title("Detected ArUco Markers")
    # plt.axis('off')
    # plt.show()