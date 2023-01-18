import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

filename = sys.argv[1] if len(sys.argv) > 1 else "32px.png"

# Load the image and convert it to grayscale
image = cv2.imread(filename)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Use the Laplacian detector to find the edges
edges = cv2.Laplacian(gray, cv2.CV_8U, ksize=1)

# Use the Hough transform to detect lines in the image
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=10, maxLineGap=0)

# Select only lines that are perpendicular to the x or y axis
lines = [line for line in lines if np.abs(line[0][0] - line[0][2]) < 10 or np.abs(line[0][1] - line[0][3]) < 10]

# iterate the numbers that the resolution is divisible by
divisible = {}
for i in range(1, image.shape[1]):
    if image.shape[1] % i == 0:
        # Sum the length of lines in columns that are divisible by i
        sum = np.sum([np.abs(line[0][0] - line[0][2]) for line in lines if line[0][0] % i == 0 or line[0][2] % i == 0])
        # Sum the length of lines in rows that are divisible by i
        sum += np.sum([np.abs(line[0][1] - line[0][3]) for line in lines if line[0][1] % i == 0 or line[0][3] % i == 0])
        # Print i and the sum
        divisible[i] = sum

# plot divisible
plt.plot(list(divisible.keys()), list(divisible.values()))
plt.show()






# show overlaying edges on image
img1 = image.copy()
img1[edges > 0] = [0, 255, 0]
cv2.imshow("edges", cv2.resize(img1, (0, 0), fx=3, fy=3))
cv2.waitKey(0)


# show lines in image
img1 = image.copy()
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img1, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imshow("lines", cv2.resize(img1, (0, 0), fx=3, fy=3))
cv2.waitKey(0)
