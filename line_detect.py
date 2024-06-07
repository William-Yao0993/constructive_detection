import cv2
import numpy as np
##---------------------------------------------------------------------------------------------------------------------------
# Augmentation 
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

def get_roi_mask(img):
    '''
        Generate a Binary Mask 
        Non roi -> 0
        roi -> 1 
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
    return mask
def extract_panel_area(img):
    '''
        manually extract none-panel area from image
        Return extracted image for edges and HoughLines
        #TODO
        Could have extract them more wisely by different channels info???

    '''
    mask=get_roi_mask(img)
    masked_img = cv2.bitwise_or(img,img, mask = mask)
    return masked_img 
# Augmentation END
##---------------------------------------------------------------------------------------------------------------------------






##-----------------------------------------------------------------------------------------------------------------
# Canny && HoughLines Post-processing 
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

# Canny && HoughLines Post-processing END
##-----------------------------------------------------------------------------------------------------------------


##-----------------------------------------------------------------------------------------------------------------
# YOLO Post-processing
def get_panel_info(img):
    '''
    Return bboxes from YOLO prediction
    '''
    from ultralytics import YOLO
    model = YOLO(r'models\panel.pt')
    results = model(img)
    
    return results[0].boxes.numpy()
def xyxy_to_xywh(xyxys:np.ndarray) -> np.ndarray:
    '''
    top left point and bottom right point  (x0,y0,x1,y1) 
    -> anchor point and width height  (anchor_x,anchor_y,w,h)  
    '''
    x0=xyxys[...,0]
    y0=xyxys[...,1]
    x1=xyxys[...,2]
    y1=xyxys[...,3]
    
    
    w= np.abs(x1-x0)
    h= np.abs(y1-y0)
    anchor_x = x0+w/2
    anchor_y = y0+h/2
    return np.stack((anchor_x,anchor_y,w,h),axis=-1)
def xywh_to_xyxy(xyxys:np.ndarray) -> np.ndarray:
    '''
    anchor point and width height  (anchor_x,anchor_y,w,h)  
    -> top left point and bottom right point  (x0,y0,x1,y1) 
    '''
    x=xyxys[...,0]
    y=xyxys[...,1]
    w=xyxys[...,2]
    h=xyxys[...,3]
    
    x0=x-w/2
    y0=y-h/2
    x1=x0+w
    y1=y0+h
    return np.stack((x0,y0,x1,y1),axis=-1)
def get_orig_xyxy(bboxes):
    xyxyn =bboxes.xyxyn
    orig_h, orig_w = bboxes.orig_shape
    orig_xyxy = xyxyn.copy()
    orig_xyxy[...,0] *= orig_w
    orig_xyxy[...,1] *= orig_h
    orig_xyxy[...,2] *= orig_w
    orig_xyxy[...,3] *= orig_h
    return orig_xyxy
def get_orig_xywh(bboxes):
    xywhn =bboxes.xywhn
    orig_h, orig_w = bboxes.orig_shape
    orig_xywh = xywhn.copy()
    orig_xywh[...,0] *= orig_w
    orig_xywh[...,1] *= orig_h
    orig_xywh[...,2] *= orig_w
    orig_xywh[...,3] *= orig_h
    return orig_xywh

def bboxes_in_roi(bboxes,mask,threshold):
    '''
        Filter out bounding boxes if overlap ratio over threshold  
    '''
    orig_xyxy = get_orig_xyxy(bboxes).astype(np.int_)
    overlap=[]
    for x1,y1,x2,y2 in orig_xyxy:
        area = mask[y1:y2,x1:x2]
        overlap.append(np.sum(area)/area.size)
    arr =np.array(overlap)
    mask = arr > threshold
    return bboxes[mask]

def bboxes_nms(bboxes,threshold):

    '''
        Apply Non-maximum-suppression algorithm to filter out overlapping bboxes based on given IoU threshold
         
        Return:
        a list of filtered bboxes in xyxy format [N*1*4]
    '''
    if bboxes is None:
        return None
    
    # TODO
    # O(n^2) could be imporved by better search
    # Dymanic Programming or Hashtable ??
    xywh= bboxes.xywh
    bboxes_areas= xywh[...,2]*xywh[...,3] 
    indices= bboxes_areas.argsort()[::-1] # descending order
    orig_xyxys_sorted = get_orig_xyxy(bboxes)[indices]

    orig_xyxys_sorted=orig_xyxys_sorted.tolist()
    candidates =[]
    while len(orig_xyxys_sorted) != 0:
        box1 = orig_xyxys_sorted.pop()
        candidates.append(box1)
        for box2 in orig_xyxys_sorted:
            if get_iou(box1,box2) > threshold:
                orig_xyxys_sorted.remove(box2) 
    arr = np.array(candidates)
    return arr   

def box_area(xyxy):
    x1,y1,x2,y2=xyxy
    return abs(x2-x1)*abs(y2-y1)

def box_areas(xyxys):
    xywhs = xyxy_to_xywh(xyxys)
    return xywhs[...,2]*xywhs[...,3]

def get_iou(xyxy1,xyxy2):

    area1 =box_area(xyxy1)
    area2 = box_area(xyxy2)

    # IoU Anchor 
    x1_iou = max(xyxy1[0],xyxy2[0])
    y1_iou = max(xyxy1[1],xyxy2[1])
    x2_iou = min(xyxy1[2],xyxy2[2])
    y2_iou = min(xyxy1[3],xyxy2[3])

    w_iou = max(0,x2_iou-x1_iou)
    h_iou = max(0,y2_iou-y1_iou)
    if w_iou ==0 or h_iou==0:
        return 0.0
    intersection = w_iou*h_iou
    union = area1+area2 - intersection
    return intersection/union

def bbox_kmeans(xyxys:np.ndarray,nclusters:int)-> list:
    '''
        Clusters bounding boxes by its shape 
        Return A List of seperated boxes-groups
    '''
    # areas = box_areas(xyxys).astype(np.float32)
    xywhs = xyxy_to_xywh(xyxys)
    features = np.stack([xywhs[...,2],xywhs[...,3]],axis=-1)
    critertia = (cv2.TermCriteria_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _,lbs,_ = cv2.kmeans(features.astype(np.float32),nclusters,None,critertia,20,flags)
    # TODO
    # should return groups of bboxes clustered by area (Not lbs)
    grouped_xyxys=[]
    for lb in np.unique(lbs):
        group = xyxys[lbs.flatten() ==lb]
        grouped_xyxys.append(group)
    return grouped_xyxys
# YOLO Post-processing END
##-----------------------------------------------------------------------------------------------------------------

##-----------------------------------------------------------------------------------------------------------------
# Line segments & Intersections
def get_houghlines_x0y0(lines:np.ndarray) -> np.ndarray: 
    '''
        get x0, y0 by given rho, theta 
    '''
    if lines.shape[-1] != 2:
        return None
    thetas =lines[...,1]
    rhos = lines[...,0]
    a = np.cos(thetas)
    b = np.sin(thetas)
    x0 = a*rhos
    y0 = b*rhos 
    return np.stack((x0,y0),axis=-1)
def get_lines_segments_in_bboxes(lines, bbox_xyxys):
    '''
        get lines segments based to fulfill in all bboxes
        Return: np.ndarray in xyxy format
    ''' 
    lines_segments = []
    for bbox_xyxy in bbox_xyxys:
        segments = get_lines_segments_in_bbox(lines,bbox_xyxy)
        if segments is not None:
            lines_segments.extend(segments)
    return np.array(lines_segments)
def get_lines_segments_in_bbox(lines,bbox_xyxy):
    ''''
        extend lines segment to fulfill in a bbox
        Return: np.ndarray in xyxy format 
    '''
    theta= lines[...,1]
    houghlines_x0y0 = get_houghlines_x0y0(lines)
    bx0,by0,bx1,by1 = bbox_xyxy
    w,h = np.abs(bx1-bx0), np.abs(by1-by0)
    fit_in_x = (houghlines_x0y0[...,0] >= bx0) & (houghlines_x0y0[...,0] <= bx1)   
    fit_in_y = (houghlines_x0y0[...,1] >= by0) & (houghlines_x0y0[...,1] <= by1)
    
    masked_houghlines_x0y0 = houghlines_x0y0[fit_in_x | fit_in_y]
    masked_theta= theta[fit_in_x | fit_in_y]
    if len(masked_houghlines_x0y0)!=0:
        x0= np.maximum(masked_houghlines_x0y0[...,0],bx0)
        y0= np.maximum(masked_houghlines_x0y0[...,1],by0)
        x1= x0+w*(np.sin(masked_theta))
        y1= y0+h*(np.cos(masked_theta))
        #print(x0.shape,y0.shape,x1.shape,y1.shape)
        return np.stack((x0,y0,x1,y1),axis=-1)
    return None

##-----------------------------------------------------------------------------------------------------------------
# Erosion & Dilation
def erode_and_dilate(img):
    skel = np.zeros_like(img)
    erosion_kernel_H = cv2.getStructuringElement(cv2.MORPH_RECT,(1,3))
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
    erosion_map_H = img.copy()
    while(np.count_nonzero(erosion_map_H)!=0):
        curr_erosion = cv2.erode(erosion_map_H,erosion_kernel_H) # Horizontal Erosion
        opening = cv2.dilate(curr_erosion,dilation_kernel,iterations=10) # Both directional dilation
        temp = cv2.subtract(img,opening) 
        skel = cv2.bitwise_or(skel,temp)
        erosion_map_H=curr_erosion
    return skel
# Erosion & Dilation END
##-----------------------------------------------------------------------------------------------------------------


def rect_fitting(bi:np.ndarray,threshold)-> np.ndarray:
    '''
        Rectanlge fitting a binary image
        Return:
        mask map of the estimated maximized rectangle(0), rest(1) above threshold, otherwise np.ones_like(bi) 
    '''
    mask = np.ones_like(bi,dtype=np.uint8)
    ratio = np.sum(bi==0) / np.sum(mask)
    if ratio >= threshold:
        ys,xs = np.nonzero(bi==0)
        p1 = xs.min(),ys.min()
        p2 = xs.max(),ys.max()
        cv2.rectangle(mask,p1,p2,0,-1)
    return mask 




if __name__ == '__main__':

    #---------------------------------------------------------------------------------------------------------------------------------------------
    # Parameters 
    #---------------------------------------------------------------------------------------------------------------------------------------------
    # PATH    
    img_path = r'datasets\data.png'

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
    canny_high_threshold = int (3*canny_low_threshold)
    #apertureSize = 5
    
    # HoughLines
    rho=10 
    theta=90*(np.pi/180)
    threshold= 100
    minLineLength=55
    maxLineGap=10 

    # Color in BGR for cv2 drawing
    
    RED = (0,0,255)
    BLUE  = (255,0,0)
    GREEN = (0,255,0)
    YELLOW = (0,255,255)
    SKY_BLUE = (255,255,0)
    PINK = (255,0,255)
    COLORS = [RED,BLUE,GREEN,YELLOW,SKY_BLUE,PINK]
    # Bounding Box 
    bbox_nclusters = 5 # Numbers of bbox type in image
    
    # Line distances 
    LINE_DISTANCE_IN_RED_H = 19.5
    LINE_DISTANCE_IN_RED_V = 10.
    LINE_DISTANCE_IN_BLUE_H = 15.8
    LINE_DISTANCE_IN_BLUE_V = 18.9
    LINE_DISTANCE_IN_GREEN_H = 14.8
    LINE_DISTANCE_IN_GREEN_V = 10.17
    LINE_DISTANCE_IN_YELLOW_H =14.
    LINE_DISTANCE_IN_YELLOW_V =18.2
    LINE_DISTANCE_IN_SKY_BLUE_H =21.
    LINE_DISTANCE_IN_SKY_BLUE_V =12.67

    # Morphology Kernels
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,1)) 
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,3)) 
    #-----------------------------------------------------------------------------------------------------------------------------------------------
    # Experiments Start
    #-----------------------------------------------------------------------------------------------------------------------------------------------
    img = cv2.imread(img_path)
    img = extract_panel_area(img)

    # Bounding Box info 
    bboxes = get_panel_info(img_path)
    overlap = 0.8 # threshold for bboxes in non-roi 
    bboxes = bboxes_in_roi(bboxes,get_roi_mask(img), overlap)
    bboxes_orig_xyxys= bboxes_nms(bboxes,threshold=0.1) 
    grouped_xyxys = bbox_kmeans(bboxes_orig_xyxys,bbox_nclusters)

    # Three trials

    ## 1 HLS
    # hls_reduced_range = cvt_hls_inrange(img,green,white) 
    # grayHLS = cv2.cvtColor(hls_reduced_range, cv2.COLOR_BGR2GRAY)
    # blur_grayHLS = cv2.GaussianBlur(grayHLS,kernel_size,0)
    # edgesHLS = cv2.Canny(blur_grayHLS, canny_low_threshold,canny_high_threshold)
    # linesHLS = cv2.HoughLinesP(edgesHLS,rho=rho,theta=theta,threshold= threshold,minLineLength=minLineLength,maxLineGap=maxLineGap)

    ## 2 Quantize to k groups
    # quantiImg,c = quantizeImage(img,quanti_k) 
    # grayQuanti = cv2.cvtColor(quantiImg,cv2.COLOR_BGR2GRAY)
    # blur_grayQuanti = cv2.GaussianBlur(grayQuanti,kernel_size,0)
    # edgesQuanti = cv2.Canny(blur_grayQuanti, canny_low_threshold,canny_high_threshold)
    # linesQuanti = cv2.HoughLinesP(edgesQuanti,rho=rho,theta=theta,threshold= threshold,minLineLength=minLineLength,maxLineGap=maxLineGap)

    ## 3 Original Gray Image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #gray = cv2.GaussianBlur(gray,(kernel_size),0)


    edges = cv2.Canny(gray, canny_low_threshold,canny_high_threshold)
    # cv2.namedWindow('edges',cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('edges',980,980)
    # cv2.imshow('edges',edges)

    closing = cv2.morphologyEx(edges,cv2.MORPH_CLOSE,rect_kernel,iterations=5)
    rect_threshold = 0.10
    mask = np.ones_like(gray,dtype=np.uint8)
    for xyxys in grouped_xyxys:
        for x0,y0,x1,y1 in xyxys.astype(np.int_):
            w, h = np.abs(x1-x0),np.abs(y1-y0)
            region = closing[y0:y1,x0:x1]
            rect_mask = rect_fitting(region,rect_threshold) 
            mask[y0:y1,x0:x1] = rect_mask
    roi_mask = get_roi_mask(img)
    mask= cv2.bitwise_and(roi_mask.astype(np.uint8),mask.astype(np.uint8))
    cv2.imwrite('morph_map.jpg', mask*255)
    edges = cv2.bitwise_or(edges,edges,mask=mask)
    # cv2.namedWindow('edges_with_Morph',cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('edges_with_Morph',980,980)
    # cv2.imshow('edges_with_Morph',edges)
    cv2.imwrite('edges_after_Morph.jpg', edges)
    # bboxes drawing 
    for i,group in enumerate(grouped_xyxys):
        color = COLORS[i]
        for x1,y1,x2,y2 in group.astype(np.int_):
            p1= x1,y1
            p2= x2,y2
            cv2.rectangle(img,p1,p2,color,5,0)

    # Orthgonal Lines
    linesP= cv2.HoughLinesP(edges,rho=rho,theta=theta,threshold= threshold,minLineLength=minLineLength,maxLineGap=maxLineGap)
    # lines = cv2.HoughLines(edges,rho,theta,threshold) 
    # orthgonal_lines = degree_filter(lines,5)
    orthgonal_linesP = degree_filter(linesP,5)
    for x0,y0,x1,y1 in orthgonal_linesP.squeeze():
        cv2.line(img,(x0,y0),(x1,y1),(50,127,0),1,0)
    # cv2.namedWindow('houghlines',cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('houghlines',980,980)
    # cv2.imshow('houghlines',img)

    # cv2.waitKey()
    # cv2.destroyAllWindows()








    # # Use rho,theta and bboxes to draw segment within panel 
    # lines_segments = get_lines_segments_in_bboxes(orthgonal_lines,bboxes_orig_xyxys)
    # for x0,y0,x1,y1 in lines_segments.squeeze().astype(np.int_):
    #     p1 = (x0,y0)
    #     p2 = (x1,y1)
    #     cv2.line(img,p1,p2,GREEN,2,cv2.LINE_AA)
    # for rho,theta in orthgonal_lines.squeeze(axis=1):
    #     a= np.cos(theta)
    #     b= np.sin(theta)
    #     x0=int (a*rho)
    #     y0=int(b*rho)
    #     cv2.circle(img,(x0,y0),1,GREEN,10,-1)
    

