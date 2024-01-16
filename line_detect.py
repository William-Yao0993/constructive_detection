import sys
import math
import cv2
import numpy as np

img_path = r'datasets\data.png'
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(img, 250,200)
height, width =  img.shape[:2]

mask = np.ones((height,width), dtype=np.int8)
# Mask Out Left Elevator 
cv2.rectangle(  mask,
               (int(width*0.2), int(height*0.4)), 
               (int(width*0.33), int(height*0.58)) ,0,-1)
# Mask Out Lifts 
cv2.rectangle(  mask,
               (int(width*0.42), int(height*0.43)), 
               (int(width*0.59), int(height*0.66)) ,0,-1)
# Mask Out Right Stair
cv2.rectangle(  mask,
               (int(width*0.72), int(height*0.49)), 
               (int(width*0.86), int(height*0.66)) ,0,-1)
# Mask Out Top Edges 
cv2.rectangle(  mask,
               (int(width*0.0), int(height*0.0)), 
               (int(width*1.0), int(height*0.05)) ,0,-1)
cv2.rectangle(  mask,
               (int(width*0.0), int(height*0.0)), 
               (int(width*0.34), int(height*0.15)) ,0,-1)
# Mask Out Left Edges
cv2.rectangle(  mask,
               (int(width*0.0), int(height*0.0)), 
               (int(width*0.07), int(height*1.0)) ,0,-1)
# Mask Out Right Edges
cv2.rectangle(  mask,
               (int(width*0.973), int(height*0.0)), 
               (int(width*1), int(height*1.0)) ,0,-1)
# Mask Out Bottom Edge
cv2.rectangle(  mask,
               (int(width*0.0), int(height*0.925)), 
               (int(width*1), int(height*1.0)) ,0,-1)
masked_edges = cv2.bitwise_and(edges,edges, mask = mask)
cedges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
lines = cv2.HoughLinesP(
    masked_edges,
    rho=1,
    theta=90*(np.pi/180),
    threshold= 100,
    minLineLength=50,
    maxLineGap=8
     )

# lines = cv2.HoughLines(
#     edges,
#     rho = 1,
#     theta= np.pi /180,
#     threshold= 200
# )


angles = []
if lines is not None: 
    for line in lines:
        x1,y1,x2,y2 = line[0]
        angle = np.degrees(np.arctan2(y2-y1, x2-x1)) % 180
        angles.append(angle)
        cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 2)
mostCommonAngle = np.bincount(np.round(angles).astype(int)).argmax() 
print ('most common angle ' + str(mostCommonAngle))

angleDiffList = []
angleDiffOrthogonalCandidateList = []
for angle in angles: 
    diff = np.abs(angle - mostCommonAngle)
    angleDiffList.append(angle)
    if diff > 50 and diff < 135:
        angleDiffOrthogonalCandidateList.append(diff)
mostCommonDiff = 0
if angleDiffOrthogonalCandidateList:
    #print(angleDiffOrthogonalCandidateList)  
    mostCommonDiff = np.bincount(np.round(angleDiffOrthogonalCandidateList).astype(int)).argmax()
print ('most common diff angle ', str(mostCommonDiff))

for line, diff in zip(lines, angleDiffList): 
    x1,y1,x2,y2 = line[0]
    angle = np.degrees(np.arctan2(y2-y1, x2-x1)) % 180
    if (diff < 5) or (np.abs(diff - mostCommonDiff) < 5): 
        cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 2)
#        break
cv2.namedWindow('detection',cv2.WINDOW_NORMAL)
cv2.resizeWindow('detection', 1920, 1000)   
cv2.imshow("detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()