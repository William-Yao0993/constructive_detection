import cv2
import numpy as np
import matplotlib.pyplot as plt

inputImgHead = "rebar_"
inputImgTail = ".jpg"

outputImgHead = "_edges"
outputLineHead = "_lines"
outputAllLineHead = "_lines_all"
outputImgTail = ".png"

dataIndex = 1

minLineLength=50

def fitStraightLines(imagePath):

    print("processing data " + imagePath)
    # Detect edges
    img = cv2.imread(imagePath) 
    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(imggray, (5, 5), 0)
    edges = cv2.Canny(blurred,50,300)

    cv2.imwrite(outputName, edges)
    # Apply Hough Line Transform to detect lines
    #longLines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=minLineLength, maxLineGap=5)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=10)
    longLines = lines
    # Calculate the angles of the lines and find the most common angle
    angles = [np.arctan2(line[0][3] - line[0][1], line[0][2] - line[0][0]) for line in longLines]
    degreesAngles = np.degrees(angles) % 180
    mostCommonAngle = np.bincount(np.round(degreesAngles).astype(int)).argmax()
    print("most common angle " + str(mostCommonAngle))

    # Create a list to store angle differences between lines with most common angle and other lines
    angleDiffList = []

    # list to store possible candidates that are orthogonal to the most common angle, that are angles
    # within a threshold
    angleDiffOrthogonalCandidateList = []

    angles = [np.arctan2(line[0][3] - line[0][1], line[0][2] - line[0][0]) for line in lines]
    # Calculate the angle difference for each line and store in the list
    for angle in angles:
        degreesAngle = np.degrees(angle) % 180
        angleDiff = np.abs(degreesAngle - mostCommonAngle)
        angleDiffList.append(angleDiff)
        if angleDiff >50 and angleDiff<135:
            angleDiffOrthogonalCandidateList.append(angleDiff)
        #print("ori " + str(degreesAngle))
        #print("diff " + str(angleDiff))

    # Find the most common angle difference
    mostCommonDiff = 0
    if angleDiffOrthogonalCandidateList is not None:
        mostCommonDiff = np.bincount(np.round(angleDiffOrthogonalCandidateList).astype(int)).argmax()
    #print (np.bincount(np.round(angleDiffOrthogonalCandidateList).astype(int)))
    print("most common diff " + str(mostCommonDiff))
    
    # Draw the lines with the most common normal and orthogonal lines on a copy of the original image
    imageWithLines = np.copy(img)
    imageWithAllLines = np.copy(img)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(imageWithAllLines, (x1, y1), (x2, y2), (0, 0, 255), 2)
        angle = np.arctan2(y2 - y1, x2 - x1)
             
        degreesAngle = np.degrees(angle) % 180
        angleDiff = np.abs(degreesAngle - mostCommonAngle)
        if angleDiff < 5:
            cv2.line(imageWithLines, (x1, y1), (x2, y2), (0, 0, 255), 2)

    for line, angleDiff in zip(lines, angleDiffList):
        x1, y1, x2, y2 = line[0]
        if np.abs(angleDiff - mostCommonDiff) < 5:
            #print("drrawing most common diff line (" + str(x1) +","+str(y1)+"), (" + str(x2) + "," + str(y2) + ")")
            cv2.line(imageWithLines, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    cv2.imwrite(outputLinesName, imageWithLines)
    cv2.imwrite(outputAllLinesName, imageWithAllLines)
    print("Done with " + imagePath)
for i in range(9):
    inputName = inputImgHead + str(dataIndex) + inputImgTail
    outputName = str(dataIndex) + outputImgHead + outputImgTail
    outputLinesName = str(dataIndex) + outputLineHead + outputImgTail
    outputAllLinesName = str(dataIndex) + outputAllLineHead + outputImgTail
    fitStraightLines(inputName)
    dataIndex = dataIndex + 1
