import cv2
from pyzbar.pyzbar import decode
import matplotlib.pyplot as plt
 
def detect_and_decode_barcode(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect barcodes in the grayscale image
    barcodes = decode(gray)

    # Loop over detected barcodes
    for barcode in barcodes:
        # Extract barcode data and type
        barcode_data = barcode.data.decode("utf-8")
        barcode_type = barcode.type

        # Print barcode data and type
        print("Barcode Data:", barcode_data)
        print("Barcode Type:", barcode_type)

        # Draw a rectangle around the barcode
        (x, y, w, h) = barcode.rect
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Put barcode data and type on the image
        cv2.putText(image, f"{barcode_data} ({barcode_type})",
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Convert image from BGR to RGB (Matplotlib uses RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # plt.imshow(image_rgb)
    # plt.axis('off')
    # # plt.show()
    # plt.savefig('detected_barcodes.jpg', bbox_inches='tight')
    cv2.imwrite('detected_barcodes_cv2.jpg', image_rgb)
 
# Main
if __name__ == '__main__':
 
  # Read image
  path = '/Users/nova98/Documents/Nova/Marker-detection/new_code1.png'
  # Read input image
  image = cv2.imread(path)

  detect_and_decode_barcode(image)