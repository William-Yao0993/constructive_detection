import sys
import math
import cv2
import numpy as np

img_path = r'datasets\data.png'
img = cv2.imread(img_path, cv2.IMREAD_COLOR)

edges = cv2.Canny(img, 500,500)

cedges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
lines = cv2.HoughLinesP(
    edges,
    rho=1,
    theta=np.pi/180,
    threshold= 100,
    minLineLength=500,
    maxLineGap=200
     )

# lines = cv2.HoughLines(
#     edges,
#     rho = 1,
#     theta= np.pi /180,
#     threshold= 200
# )

if lines is not None: 
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv2.line(edges, (x1,y1), (x2,y2), (0,0,255), 2)
#cv2.imshow("windows",img)
cv2.imshow("detection", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()