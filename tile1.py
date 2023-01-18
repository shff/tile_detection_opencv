import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("vikings.png")

# convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# show
cv2.imshow("image", cv2.resize(img, (0, 0), fx=2, fy=2))
cv2.waitKey(0)

# find edges using Laplacian edge detection
edges = cv2.Laplacian(gray, cv2.CV_8U, ksize=5)

# show
cv2.imshow("edges", cv2.resize(edges, (0, 0), fx=2, fy=2))
cv2.waitKey(0)

# find the percentage of pixels of contour are filled in each column and row
column_edges = {i: np.sum(edges[:, i]) / 255 / img.shape[0] for i in range(0, img.shape[1])}
row_edges = {i: np.sum(edges[i, :]) / 255 / img.shape[1] for i in range(0, img.shape[0])}




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
cv2.imshow("column_edges", cv2.resize(img2, (0, 0), fx=2, fy=2))
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
cv2.imshow("row_edges", cv2.resize(img3, (0, 0), fx=2, fy=2))
cv2.waitKey(0)



exit()

# N = len(column_edges) #number of elements
# t = np.linspace(0, N * 3600, N) #converting hours to seconds
# fft = np.fft.fft(list(column_edges.values()))

# fftfreq = np.fft.fftfreq(N)

# # T = t[1] - t[0]

# # f = np.linspace(0, 1 / T, N)
# # plt.ylabel("Amplitude")
# # plt.xlabel("Frequency [Hz]")
# plt.plot(fftfreq, fft)
# plt.show()


# find frequency of spikes in column_edges
# column_freq = {}
# for i in range(1, len(column_edges)):
#     # find the percentage of pixels of contour are filled in column i
#     column_freq[i] = column_edges[i] / column_edges[i - 1]

# # convolve column_edges
# window = np.ones(int(1))/float(1)
# column_edges_convolved = np.convolve(list(column_edges.values()), window, mode='same')

# # plot
# plt.plot(column_edges_convolved)
# plt.show()

# # find dominant frequency
# fft = np.fft.fft(column_edges_convolved)
# freq = np.fft.fftfreq(len(fft))
# plt.plot(freq, fft)
# plt.show()





# # filter column edges, only show values bigger than 0.6
# column_edges_filtered = {}
# for i in range(1, len(column_edges)):
#     if column_edges[i] > 0.6:
#         column_edges_filtered[i] = column_edges[i]
#     else:
#         column_edges_filtered[i] = 0

# # plot column_edges
# plt.plot(column_edges_filtered.values())
# plt.show()

# # find frequency of spikes in column_edges_filtered
# fft = np.fft.fft(list(column_edges_filtered.values()))

# # plot
# plt.plot(fft)
# plt.show()

# exit()


##
## Failed experiment
##
column_sums = {}
for i in range(1, 256):
    # find out how the sum of the column_edges that are divisible by i
    column_sum = 0
    for j in range(1, len(column_edges)):
        if j % i == 0:
            column_sum += column_edges[j]
    column_sums[i] = column_sum

row_sums = {}
for i in range(1, 256):
    # find out how the sum of the row_edges that are divisible by i
    row_sum = 0
    for j in range(1, len(row_edges)):
        if j % i == 0:
            row_sum += row_edges[j]
    row_sums[i] = row_sum

if debug:
    img2 = img.copy()
    # iterate map column_sums
    for i in column_sums:
        # draw a line on the image with the height of 'sum'
        cv2.line(img2, (i, 0), (i, int(column_sums[i] * 10)), (0, 255, 0), 1)
    show(img2)

if debug:
    img2 = img.copy()
    # iterate map column_sums
    for i in row_sums:
        # draw a line on the image with the height of 'sum'
        cv2.line(img2, (0, i), (int(row_sums[i] * 10), i), (0, 255, 0), 1)
    show(img2)

# column_sums = []
# for i in range(0, int(max(img.shape[0], img.shape[1]) / 2)):
#     # calculate the sum of column_edges in n % i
#     column_sum = 0
#     for j in [32]:
#         if i % j == 0:
#             column_sum += column_edges[j]
#     column_sums.append(column_sum)

# row_sums = []
# for i in range(0, int(max(img.shape[0], img.shape[1]) / 2)):
#     # calculate the sum of row_edges in n % i
#     row_sum = 0
#     for j in range(1, len(row_edges)):
#         if i % j == 0:
#             row_sum += row_edges[j]
#     row_sums.append(row_sum)




# # highlight the position of column_sum in the image
# if debug:
#     img2 = img.copy()
#     for i in range(0, len(column_sums)):
#         cv2.line(img2, (i, 0), (i, int(column_sums[i])), (0, 255, 0), 1)

#     show(img2)

# # highlight the position of row_sums in the image
# if debug:
#     img2 = img.copy()
#     for i in range(0, len(row_sums)):
#         cv2.line(img2, (0, i), (int(row_sums[i]), i), (0, 255, 0), 1)

#     show(img2)




# # use edge detection to find the outline of the image
# edges = cv2.Canny(gray, 100, 200)

# # show edges
# cv2.imshow("edges", cv2.resize(edges, (0, 0), fx=2, fy=2))
# cv2.waitKey(0)

# # show the image at twice the original size
# cv2.imshow("edges", cv2.resize(edges, (0, 0), fx=2, fy=2))
# cv2.waitKey(0)

# # find discontinuities in the image
# contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# # show the image at twice the original size
# cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
# cv2.imshow("contours", cv2.resize(img, (0, 0), fx=2, fy=2))
# cv2.waitKey(0)

# # low pass the countours, to find the edges of the tiles
# kernel = np.ones((5, 5), np.float32) / 25
# dst = cv2.filter2D(contours, -1, kernel)

# # show the image at twice the original size
# cv2.imshow("low pass", cv2.resize(dst, (0, 0), fx=2, fy=2))
# cv2.waitKey(0)



# ret, thresh = cv2.threshold(gray, 255, 255, 255)

# # show the image at twice the original size
# cv2.imshow("binary", cv2.resize(thresh, (0, 0), fx=2, fy=2))
# cv2.waitKey(0)

# # detect the contours in the grayscale image
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# # show the image at twice the original size
# cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
# cv2.imshow("contours", cv2.resize(img, (0, 0), fx=2, fy=2))
# cv2.waitKey(0)
