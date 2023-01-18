import sys
import cv2
import numpy as np
import scipy as sp
import pandas as pd

WINDOW = 3

def smooth(s, win):
    return pd.Series(s).rolling(window=win, center=True).mean().ffill().bfill()

def peaks_of(data):
    column_edges = np.clip(smooth(data, 2 * WINDOW)-smooth(data, 10 * WINDOW), 0, np.inf)
    return sp.signal.find_peaks(column_edges, width=WINDOW)[0]

filename = sys.argv[1] if len(sys.argv) > 1 else "32px.png"

# read image
img = cv2.imread(filename)

# convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find edges using Laplacian edge detection
edges = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)

# find the percentage of pixels of contour are filled in each column and row
column_edges = {i: np.sum(edges[:, i]) / 255 for i in range(1, img.shape[1] - 1) if np.sum(edges[:, i + 1]) > 0}
row_edges = {i: np.sum(edges[i, :]) / 255 for i in range(1, img.shape[0] - 1) if np.sum(edges[i + 1, :]) > 0}

# smooth the column edges
col_peaks = peaks_of(np.array(list(column_edges.values())))
row_peaks = peaks_of(np.array(list(row_edges.values())))

# average distance between peaks
avg_dist = (np.average(np.diff(col_peaks)) + np.average(np.diff(row_peaks))) / 2

# round to the nearest number that the resolution is divisible by
divisible_by = [i for i in range(1, img.shape[1]) if img.shape[1] % i == 0]
rounded = min(divisible_by, key=lambda x: abs(x - avg_dist))

print("avg_dist", rounded)



# # show
# cv2.imshow("image", cv2.resize(img, (0, 0), fx=3, fy=3))
# cv2.waitKey(0)

# show overlaying edges on image
img1 = img.copy()
img1[edges > 0] = [0, 255, 0]
cv2.imshow("edges", cv2.resize(img1, (0, 0), fx=3, fy=3))
cv2.waitKey(0)

# plot column_edges
img2 = img.copy()
# iterate the columns of the image
for i in range(0, img.shape[1]):
    # find how many pixels of contour are filled in column i
    sum = np.sum(edges[:, i]) / 255
    if sum > 1:
        # draw a line on the image with the height of 'sum'
        cv2.line(img2, (i, 0), (i, int(sum)), (0, 255, 0), 1)

# show
cv2.imshow("column_edges", cv2.resize(img2, (0, 0), fx=3, fy=3))
cv2.waitKey(0)

img3 = img.copy()
# iterate the rows of the image
for i in range(0, img.shape[0]):
    # find how many pixels of contour are filled in row i
    sum = np.sum(edges[i, :]) / 255
    if sum > 1:
        # draw a line on the image with the height of 'sum'
        cv2.line(img3, (0, i), (int(sum), i), (0, 255, 0), 1)

# show
cv2.imshow("row_edges", cv2.resize(img3, (0, 0), fx=3, fy=3))
cv2.waitKey(0)

# plot smoothed lines
img2 = img.copy()
# iterate the columns of the image
for i in range(0, img.shape[1] - 1):
    # find how many pixels of contour are filled in column i
    sum = 200 if i in col_peaks else 0
    if sum > 1:
        # draw a line on the image with the height of 'sum'
        cv2.line(img2, (i, 0), (i, int(sum)), (0, 255, 0), 1)

for i in range(0, img.shape[0]):
    # find how many pixels of contour are filled in row i
    sum = 200 if i in row_peaks else 0
    if sum > 1:
        # draw a line on the image with the height of 'sum'
        cv2.line(img2, (0, i), (int(sum), i), (0, 255, 0), 1)

# show
cv2.imshow("row_edges", cv2.resize(img2, (0, 0), fx=3, fy=3))
cv2.waitKey(0)
