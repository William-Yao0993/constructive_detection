import cv2
import numpy as np

img_path = r'datasets\data.png'


def quantizeImage(img,k):
    channels = img.shape[-1]
    arr = np.float32(img).reshape(-1,channels)
    condition = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10,1.0)
    ret,label,center = cv2.kmeans(arr,k,None,condition,10,cv2.KMEANS_PP_CENTERS)
    frequentClusters = np.bincount(label.flatten()).argsort()[0]
    mostFrequentColor = center[frequentClusters]
    center[frequentClusters] = np.zeros(channels,dtype=np.int8)
    center = center.astype(np.uint8)
    quantiImg = center[label.flatten()].reshape(img.shape)
    
    return quantiImg, mostFrequentColor

def cvt_hls_inrange(img, lower, upper):
    '''
    Process in HLS channel and reduce color in range [lower,upper]
    '''
    hls = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
    lower = np.uint8([10, 200, 0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(hls, lower, upper)
    hls_mask = cv2.bitwise_or(img,img,mask=white_mask)
    return hls_mask

def extract_panel_area(img):
    '''
        manually extract none-panel area from image
        Return extracted image for edges and HoughLines
        #TODO
        Could have extract them more wisely by different channels info???

    '''
    height, width =  img.shape[:2]
    mask = np.ones((height,width), dtype=np.int8)
    # Mask Out Left Elevator 
    cv2.rectangle(mask,
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
    masked_edges = cv2.bitwise_or(img,img, mask = mask)
    return masked_edges 
def find_angles(lines):
    '''
    Return angles for HoughLinesP or HoughLine line segment
    '''
    if lines is None:
        return None
    elif lines.shape[-1] == 2:      
        # HoughLines -> rho,theta (ndarray N*1*2)
        return lines[...,-1].reshape(-1)
    
    # HoughLinesP -> x1,y1,x2,y2 (ndarray N*1*4)    
    x1,y1,x2,y2= lines[...,0],lines[...,1],lines[...,2],lines[...,3]
    angles = np.arctan2(np.abs(y2-y1),np.abs(x2-x1)).reshape(-1)
    return angles

def degree_filter(lines, tolerance):
    '''
        Filter out the perpendicular lines for 0 or 90 degrees with the tolerance for HoughLinesP 
    '''
    
    degrees = np.degrees(find_angles(lines))
    #print(degrees)
    #print(degrees.shape)
    
    mostCommonAngle = np.bincount(np.round(degrees).astype(np.int16)).argmax() 
    print ('most common angle ' + str(mostCommonAngle))
    angleDiffs =np.abs(degrees -mostCommonAngle)
    #print(angleDiffs)
    mask = (angleDiffs >50) & (angleDiffs <135)
    angleDiffOrthogonalCandidates = angleDiffs[mask]
    mostCommonDiff = np.bincount(np.round(angleDiffOrthogonalCandidates).astype(np.int16)).argmax()
    print ('most common diff angle ', str(mostCommonDiff))
    
    tolerance_mask = (angleDiffs < tolerance) | (np.abs(angleDiffs-mostCommonDiff) < tolerance)
    orthogonal_lines = lines[tolerance_mask]
    return orthogonal_lines


def lines_kmeans(lines,show= None):
    '''
        Use Kmeans to seperate distinguishable group in lines by angles 
    '''
    angles = find_angles(lines)
    # Kmeans to seperate Horizontal/Vertical lines  
    # scale angles in range [0,360] 
    # Find x and y coordinate in Cartesian coordinate system
    x,y= np.cos(2*angles), np.sin(2*angles)
    features=np.stack((x,y),axis=-1)
    # type, max_iter,epsilon
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # Kmean++ for initial Centre
    flags = cv2.KMEANS_PP_CENTERS
    nclusters=2
    _,labels,centres = cv2.kmeans(features, nclusters, None, criteria, 10, flags)

    group0 = lines[labels.ravel() == 0]
    group1 = lines[labels.ravel() == 1]
    if show is not None:
        for line in group0:
            x1,y1,x2,y2 = line[0]
            cv2.line(show, (x1,y1),(x2,y2), (0,255,0),2)
        for line in group1:
            x1,y1,x2,y2 = line[0]
            cv2.line(show, (x1,y1),(x2,y2), (255,0,0),2)
    return group0,group1
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


if __name__ == '__main__':
    img = cv2.imread(img_path)
    img = extract_panel_area(img)
    #---------------------------------------------------------------------------------------------------------------------------------------------
    # Parameters 
    #---------------------------------------------------------------------------------------------------------------------------------------------
    # Gaussian
    kernel_size =(3,3)  
    
    # HLS Range
    green = np.uint8([10, 200, 0]) # Color lower bound  
    white = np.uint8([255, 255, 255]) # Coloer higher bound
    
    # Quantize level 
    quanti_k=3

    ## Canny 
    # Recommended a upper:lower ratio between 2:1 and 3:1.
    canny_low_threshold = 150
    canny_high_threshold = int(3* canny_low_threshold)
    apertureSize = 5
    
    # HoughLines
    rho=10 
    theta=90*(np.pi/180)
    threshold= 100
    minLineLength=55
    maxLineGap=10 
    #-----------------------------------------------------------------------------------------------------------------------------------------------
    # Augmentation 
    #-----------------------------------------------------------------------------------------------------------------------------------------------
    # Three trials
    # hls_reduced_range = cvt_hls_inrange(img,green,white) # 1: HLS
    # grayHLS = cv2.cvtColor(hls_reduced_range, cv2.COLOR_BGR2GRAY)
    # blur_grayHLS = cv2.GaussianBlur(grayHLS,kernel_size,0)
    # edgesHLS = cv2.Canny(blur_grayHLS, canny_low_threshold,canny_high_threshold)
    # linesHLS = cv2.HoughLinesP(edgesHLS,rho=rho,theta=theta,threshold= threshold,minLineLength=minLineLength,maxLineGap=maxLineGap)

    
    
    # quantiImg,c = quantizeImage(img,quanti_k) # 2 Quantize to k groups
    # grayQuanti = cv2.cvtColor(quantiImg,cv2.COLOR_BGR2GRAY)
    # blur_grayQuanti = cv2.GaussianBlur(grayQuanti,kernel_size,0)
    # edgesQuanti = cv2.Canny(blur_grayQuanti, canny_low_threshold,canny_high_threshold)
    # linesQuanti = cv2.HoughLinesP(edgesQuanti,rho=rho,theta=theta,threshold= threshold,minLineLength=minLineLength,maxLineGap=maxLineGap)


    rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # 3 No augment
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.GaussianBlur(gray,(kernel_size),0)
    edges = cv2.Canny(blur_gray, canny_low_threshold,canny_high_threshold)
    linesP= cv2.HoughLinesP(edges,rho=rho,theta=theta,threshold= threshold,minLineLength=minLineLength,maxLineGap=maxLineGap)
    lines = cv2.HoughLines(edges,rho,theta,threshold)
    orthgonal_lines = degree_filter(lines,5)
    orthgonal_linesP = degree_filter(linesP,5)
    print(orthgonal_lines.shape)
    print(orthgonal_linesP.shape)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # TODO: 
    # use panel info as line segments length 
    # use rho,theta to draw segment within panel 
    # use line segments from HoughLinesP to verify above and connect in-between gap

    
     