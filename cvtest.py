import cv2

print("cv2 path:", cv2.__file__)
for line in cv2.getBuildInformation().splitlines():
    if "GUI:" in line or "QT" in line or "GTK" in line:
        print(line)