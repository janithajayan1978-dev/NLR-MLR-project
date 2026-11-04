
import cv2

# Test camera indexes 0 through 5
for index in range(6):
    camera = cv2.VideoCapture(index, cv2.CAP_MSMF)

    if camera.isOpened():
        ok, frame = camera.read()

        if ok and frame is not None:
            print(f"Camera found at index {index}")
            cv2.imshow(f"Camera {index}", frame)
            cv2.waitKey(3000)
            cv2.destroyAllWindows()
        else:
            print(f"Index {index}: opened, but no image")

        camera.release()
    else:
        print(f"Index {index}: unavailable")

print("Camera scan finished.")
