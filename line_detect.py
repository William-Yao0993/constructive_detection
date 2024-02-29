import cv2
import numpy as np

img_path = r'datasets\data.png'
img = cv2.imread(img_path)
# cv2.namedWindow('img',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('img', 960, 960)   
# cv2.imshow("img",img)
rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
def quantizeImage(img,k,num_channels):
    arr = np.float32(img).reshape(-1,num_channels)
    condition = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10,1.0)
    ret,label,center = cv2.kmeans(arr,k,None,condition,10,cv2.KMEANS_PP_CENTERS)
    FrequentClusters = np.bincount(label.flatten()).argsort()[0]
    mostFrequentColor = center[FrequentClusters]
    # for x in mostFrequentColor:
    #     if (x > 255/2):
    #         x = 255
    #     else:
    #         x = 0

    center[FrequentClusters] = np.zeros(num_channels,dtype=np.int8)
    center = center.astype(np.uint8)
    quantiImg = center[label.flatten()].reshape(img.shape)
    
    return quantiImg, mostFrequentColor

# Image Augmentation
hls = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
 
lower = np.uint8([10, 200, 0])
upper = np.uint8([255, 255, 255])
white_mask = cv2.inRange(hls, lower, upper)
hls_mask = cv2.bitwise_or(img,img,mask=white_mask)
# cv2.namedWindow('hlsMask',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('hlsMask', 960, 960)   
# cv2.imshow("hlsMask",hls_mask)

grayHLS = cv2.cvtColor(hls_mask, cv2.COLOR_BGR2GRAY)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

k=3
im = img.copy()
channels = im.shape[-1]
print(channels)
quantiImg,c = quantizeImage(im,k, channels)
grayQuanti = cv2.cvtColor(quantiImg,cv2.COLOR_BGR2GRAY)
# cv2.namedWindow(f'quantiImg_{k}',cv2.WINDOW_NORMAL)
# cv2.resizeWindow(f'quantiImg_{k}', 960, 960)   
# cv2.imshow(f'quantiImg_{k}',quantiImg)
cv2.imwrite(f'quantiImg_{k}.jpg',quantiImg)
# quantiImgHLS = quantizeImage(grayHLS,k)
# cv2.namedWindow(f'quantiImg_HLS{k}',cv2.WINDOW_NORMAL)
# cv2.resizeWindow(f'quantiImg_HLS{k}', 960, 960)   
# cv2.imshow(f'quantiImg_HLS{k}',quantiImgHLS)
# cv2.namedWindow('gray',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('gray', 960, 960)   
# cv2.imshow('gray', gray)
# cv2.namedWindow('grayHLS',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('grayHLS', 960, 960)   
# cv2.imshow('grayHLS', grayHLS)
kernel_size =3
blur_gray = cv2.GaussianBlur(gray,(kernel_size,kernel_size),0)
blur_grayHLS = cv2.GaussianBlur(grayHLS,(kernel_size,kernel_size),0)
# kernel_size =5
# blur_gray2 = cv2.GaussianBlur(gray,(kernel_size,kernel_size),0)
# cv2.namedWindow('GaussianBlur',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('GaussianBlur', 960, 960)   
# cv2.imshow('GaussianBlur', blur_gray)

 
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
masked_edges = cv2.bitwise_or(gray,gray, mask = mask)
masked_edgesHLS = cv2.bitwise_or(grayHLS,grayHLS, mask = mask)
masked_edgesQuanti = cv2.bitwise_or(grayQuanti,grayQuanti, mask = mask)
#masked_edges2 = cv2.bitwise_or(blur_gray,blur_gray,mask= mask)
# cv2.namedWindow('masked_edges',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('masked_edges', 960, 960)   
# cv2.imshow('masked_edges', masked_edges)
## Canny recommended a upper:lower ratio between 2:1 and 3:1.
low_threshold = 150
high_threshold = int(3* low_threshold)
apertureSize = 5
edges = cv2.Canny(masked_edges, low_threshold,high_threshold)
edgesHLS = cv2.Canny(masked_edgesHLS, low_threshold,high_threshold,apertureSize=apertureSize)
edgesQuanti = cv2.Canny(masked_edgesQuanti, low_threshold,high_threshold)
# cv2.namedWindow('edge',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('edge', 960, 960)   
# cv2.imshow('edge', edges)
# cv2.namedWindow('edgeHLS',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('edgeHLS', 960, 960)   
# cv2.imshow('edgeHLS', edgesHLS)

# cv2.namedWindow('edgeGaussianBlur',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('edgeGaussianBlur', 960, 960)   
# cv2.imshow('edgeGaussianBlur', edges2)

cv2.imwrite('edgeHLS.jpg', edgesHLS)
cv2.imwrite('edge.jpg', edges)
cv2.imwrite('edgesQuanti.jpg', edgesQuanti)
rho=10 
theta=90*(np.pi/180)
threshold= 100
minLineLength=55
maxLineGap=10 
lines= cv2.HoughLinesP(
    edges,
    rho=rho,
    theta=theta,
    threshold= threshold,
    minLineLength=minLineLength,
    maxLineGap=maxLineGap 
     )
# lines2 = cv2.HoughLinesP(
#     edges2,
#     rho=rho,
#     theta=theta,
#     threshold= threshold,
#     minLineLength=minLineLength,
#     maxLineGap=maxLineGap 
#      )
 
linesHLS = cv2.HoughLinesP(
    edgesHLS,
    rho=rho,
    theta=theta,
    threshold= threshold,
    minLineLength=minLineLength,
    maxLineGap=maxLineGap 
     )
linesQuanti = cv2.HoughLinesP(
    edgesHLS,
    rho=rho,
    theta=theta,
    threshold= threshold,
    minLineLength=minLineLength,
    maxLineGap=maxLineGap 
     )

def line_filter(lines,img):

    angles = []
    if lines is not None: 
        for line in lines:
            x1,y1,x2,y2 = line[0]
            angle = np.degrees(np.arctan2(y2-y1, x2-x1)) % 180
            angles.append(angle)

            # cv2.line(img, (x1,y1), (x2,y2), (255,0,0), 2)
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
    selectedLines = []
    for line, diff in zip(lines, angleDiffList): 
        x1,y1,x2,y2 = line[0]
        angle = np.degrees(np.arctan2(y2-y1, x2-x1)) % 180
        if (diff < 5) or (np.abs(diff - mostCommonDiff) < 5): 
            #cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 2)
            selectedLines.append(line)

        
    # ###--------------------------------------------------------------------------------------------------------------------
    # # Kmeans to seperate Horizontal/Vertical lines  
    features = np.array([[np.cos(2*angle), np.sin(2*angle)] for angle in np.radians(angles)]).astype(np.float32)
    # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # Kmean++ for initial Centre
    flags = cv2.KMEANS_PP_CENTERS
    _,labels,centres = cv2.kmeans(features, 2, None, criteria, 10, flags)

    horizontalLines = lines[labels.ravel() == 0]
    verticalLines = lines[labels.ravel() == 1]
    for line in horizontalLines:
        x1,y1,x2,y2 = line[0]
        cv2.line(img, (x1,y1),(x2,y2), (0,255,0),2)
    for line in verticalLines:
        x1,y1,x2,y2 = line[0]
        cv2.line(img, (x1,y1),(x2,y2), (255,0,0),2)
img_cp = img.copy()
line_filter(lines,img_cp)
# cv2.namedWindow('HoughLines',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('HoughLines', 960, 960)   
# cv2.imshow("HoughLines", img_cp)
imgHLS = img.copy()
line_filter(linesHLS,imgHLS)
imgQuanti = img.copy()
line_filter(linesQuanti,imgQuanti)
cv2.imwrite("lines.jpg",img_cp)
cv2.imwrite("linesHLS.jpg",imgHLS)
cv2.imwrite("linesQuanti.jpg",imgQuanti)
# cv2.namedWindow('HoughLinesHLS',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('HoughLinesHLS', 960, 960)   
# cv2.imshow("HoughLinesHLS", imgHLS)
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
# cv2.resizeWindow('detection', 960, 960)   
# cv2.imshow("detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
