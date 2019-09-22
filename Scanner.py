import cv2
import numpy as np
import argparse
import math

def getDistance(x1,y1,x2,y2):
    return math.sqrt( pow( abs(x2)-abs(x1) ,2) + pow( abs(y2)-abs(y1) ,2) )

def arrangePoints(unarranged):
    unarranged = unarranged.reshape((4,2))
    arranged = np.zeros((4, 2), dtype=np.float32)

    add = unarranged.sum(1)
    arranged[0] = unarranged[np.argmin(add)]
    arranged[2] = unarranged[np.argmax(add)]

    diff = np.diff(unarranged, axis=1)
    arranged[1] = unarranged[np.argmin(diff)]
    arranged[3] = unarranged[np.argmax(diff)]

    return arranged


# get image from arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to the image file")
ap.add_argument("-e", "--enhance", help="black enhancement level (1-10)")
ap.add_argument("-l", "--length", help="image maximum side length")
ap.add_argument("-s", "--save", help="output image name and location")
args = vars(ap.parse_args())
f1 = float(args["enhance"])
image = cv2.imread(args["image"])

# scale image
scale_percent = 70  # percent of original size
dim = (int(image.shape[1] * scale_percent / 100), int(image.shape[0] * scale_percent / 100))
image = cv2.resize(image, dsize=dim, interpolation=cv2.INTER_AREA)

# get grayscale image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# blur image for better contour detection
gray = cv2.GaussianBlur(gray, (9, 9), 1)

# detect edges
edges = cv2.Canny(gray, 30, 50)

# get contours
dummy, contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# sort contours
contours = sorted(contours, key=cv2.contourArea, reverse=True)

height, width, channels = image.shape
paper = [[0, 0], [width, 0], [width, height], [0, height]]
# search contours for paper
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)

    if len(approx) == 4:
        paper = approx
        # arrange detected edges
        paper = arrangePoints(paper)
        break

# draw edges on original image
cv2.circle(image, (paper[0][0], paper[0][1]), 2, (0, 0, 255), 2)
cv2.circle(image, (paper[1][0], paper[1][1]), 2, (0, 255, 0), 2)
cv2.circle(image, (paper[2][0], paper[2][1]), 2, (255, 0, 0), 2)
cv2.circle(image, (paper[3][0], paper[3][1]), 2, (255, 255, 0), 2)


# detect width to height ratio and set image size accordingly
width = getDistance(paper[0][0], paper[0][1], paper[1][0], paper[1][1])
height = getDistance(paper[0][0], paper[0][1], paper[3][0], paper[3][1])

if width < height:
    ratio = width/height
    height = int(args["image"])
    width = height * ratio
if height < width:
    ratio = height/width
    width = int(args["image"])
    height = width*ratio
frame = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

# get perspective transformation matrix
M = cv2.getPerspectiveTransform(paper, frame)
# apply matrix to get perspective corrected image
warped = cv2.warpPerspective(image, M, (int(width), int(height)))

# get grayscale version of warped image
final = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
# get image threshold and enhance black
ret, finalThresh = cv2.threshold(final, 127, 255, 0)
final = cv2.addWeighted(final, 1-f1/10, finalThresh, f1/10, 0)

# display
cv2.imshow(args["image"], image)
cv2.imshow("warped", final)

# save
cv2.imwrite(args["save"], final)

cv2.waitKey(0)
cv2.destroyAllWindows()