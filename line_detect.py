import cv2
import numpy as np

img_path = r'datasets\data.png'
img = cv2.imread(img_path)


# Image Augmentation

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cv2.namedWindow('gray',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('gray', 960, 960)   
# cv2.imshow('gray', gray)
kernel_size =3
blur_gray = cv2.GaussianBlur(gray,(kernel_size,kernel_size),0)
# kernel_size =5
# blur_gray2 = cv2.GaussianBlur(gray,(kernel_size,kernel_size),0)
# cv2.namedWindow('GaussianBlur',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('GaussianBlur', 960, 960)   
# cv2.imshow('GaussianBlur', blur_gray)


 

low_threshold = 50
high_threshold = 550
edges = cv2.Canny(blur_gray, low_threshold,high_threshold)
# cv2.namedWindow('edge',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('edge', 960, 960)   
# cv2.imshow('edge', edges)



# low_threshold = 60
# high_threshold = 550
# edges2 = cv2.Canny(blur_gray2, low_threshold,high_threshold)
# cv2.namedWindow('edge2',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('edge2', 960, 960)   
# cv2.imshow('edge2', edges2)


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

cv2.namedWindow('masked_edges',cv2.WINDOW_NORMAL)
cv2.resizeWindow('masked_edges', 960, 960)   
cv2.imshow('masked_edges', masked_edges)


# masked_edges2 = cv2.bitwise_and(edges2,edges2, mask = mask)
# cv2.namedWindow('masked_edges2',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('masked_edges2', 960, 960)   
# cv2.imshow('masked_edges2', masked_edges2)
# lines = cv2.HoughLinesP(
#     masked_edges,
#     rho=1,
#     theta=90*(np.pi/180),
#     threshold= 100,
#     minLineLength=55,
#     maxLineGap=10
#      )

# # print(lines.__len__())   
# angles = []
# if lines is not None: 
#     for line in lines:
#         x1,y1,x2,y2 = line[0]
#         angle = np.degrees(np.arctan2(y2-y1, x2-x1)) % 180
#         angles.append(angle)

#         # cv2.line(img, (x1,y1), (x2,y2), (255,0,0), 2)
# mostCommonAngle = np.bincount(np.round(angles).astype(int)).argmax() 
# print ('most common angle ' + str(mostCommonAngle))

# angleDiffList = []
# angleDiffOrthogonalCandidateList = []
# for angle in angles: 
#     diff = np.abs(angle - mostCommonAngle)
#     angleDiffList.append(angle)
#     if diff > 50 and diff < 135:
#         angleDiffOrthogonalCandidateList.append(diff)
# mostCommonDiff = 0
# if angleDiffOrthogonalCandidateList:
#     #print(angleDiffOrthogonalCandidateList)  
#     mostCommonDiff = np.bincount(np.round(angleDiffOrthogonalCandidateList).astype(int)).argmax()
# print ('most common diff angle ', str(mostCommonDiff))
# selectedLines = []
# for line, diff in zip(lines, angleDiffList): 
#     x1,y1,x2,y2 = line[0]
#     angle = np.degrees(np.arctan2(y2-y1, x2-x1)) % 180
#     if (diff < 5) or (np.abs(diff - mostCommonDiff) < 5): 
#         #cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 2)
#         selectedLines.append(line)

     
# ###--------------------------------------------------------------------------------------------------------------------
# # Kmeans to seperate Horizontal/Vertical lines  
# features = np.array([[np.cos(2*angle), np.sin(2*angle)] for angle in np.radians(angles)]).astype(np.float32)
# # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
# criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# # Kmean++ for initial Centre
# flags = cv2.KMEANS_PP_CENTERS
# _,labels,centres = cv2.kmeans(features, 2, None, criteria, 10, flags)

# horizontalLines = lines[labels.ravel() == 1]
# verticalLines = lines[labels.ravel() == 0]
# for line in horizontalLines:
#     x1,y1,x2,y2 = line[0]
#     cv2.line(img, (x1,y1),(x2,y2), (0,255,0),2)
# for line in verticalLines:
#     x1,y1,x2,y2 = line[0]
#     cv2.line(img, (x1,y1),(x2,y2), (255,0,0),2)
# print(len(horizontalLines))
# print(len(verticalLines))

# intersections = set([])
# for i, l1 in enumerate(horizontalLines):
#     x1,y1,x2,y2 = l1[0]
#     #cv2.line(img, (x1,y1),(x2,y2), (0,255,0),2)
#     theta1 = np.arctan2(y2-y1 ,x2-x1) / 90
#     rho1 = x1*np.cos(theta1) + y1*np.sin(theta1) 
    
#     for l2 in verticalLines:
#         x3,y3,x4,y4 = l2[0]
#         if (x1 <=x3 <= x2):
#         #cv2.line(img, (x3,y3),(x4,y4), (255,0,0),2)
#             theta2 = np.arctan2(y4-y3, x4-x3) /90
#             rho2 = x3* np.cos(theta2) + y3*np.sin(theta2)

#             A = np.array([
#                 [np.cos(theta1), np.sin(theta1)],
#                 [np.cos(theta2), np.sin(theta2)]
#             ]) 
#             b = np.array([rho1,rho2])
#             x,y = np.linalg.solve(A,b)
#             x,y = int(np.round(x)), int(np.round(y))

                
#             intersections.add((x,y))
#     print(f"{i}: intersections {len(intersections)}")
    
# print(len(intersections))
# for point in intersections:
#     cv2.circle(img, (point[0],point[1]), 5, (0,0,255), -1)
# cv2.namedWindow('detection',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('detection', 1920, 1000)   
# cv2.imshow("detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
