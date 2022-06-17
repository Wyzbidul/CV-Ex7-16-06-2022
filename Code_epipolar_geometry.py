#####################################################################################################################################
#	PROGRAM BY : MC IATRIDES
#	LAST UPDATE : 16-06-2022
#	TITLE : Exercise #7 - (16-06-2022)
#   SUBTITLE : Epipolar Geometry Rectified
#	REDACTED FOR : COMPUTER VISION
#####################################################################################################################################

##### PACKAGES ######################################################################################################################
from cv2 import STEREO_BM_PREFILTER_NORMALIZED_RESPONSE
from numpy import *
import cv2 as cv
from matplotlib.pyplot import *
#####################################################################################################################################

###### FUNCTIONS ####################################################################################################################
def drawlines(img1, img2, lines, pts1, pts2):
    '''img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2= cv.cvtColor(img2,cv.COLOR_GRAY2BGR)

    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])

        img1 = cv.line(img1, (x0,y0), (x1,y1), color, 1)
        img1 = cv.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv.circle(img2, tuple(pt2), 5, color, -1)
    
    return img1, img2

def distance(point, line):
    '''Calculates the distance between a point and the corresponding epiline'''
    a,b,c = line[0], line[1], line[2] 
    x,y = point[0], point[1] 
    dist = abs(a*x+b*y+c)/sqrt(a**2+b**2)

    return dist

#####################################################################################################################################

###### ANALYSIS PART ################################################################################################################
print('START TESTS')

img1 = cv.imread('left.jpg',0)      #queryimage # left image
img2 = cv.imread('right.jpg',0)     #trainimage # right image

#Create SIFT descriptor and use with FLANN based matcher and ratio test
sift = cv.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

pts1 = []
pts2 = []

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

#Find the list of best matches to form Fundamental Matrix
pts1 = int32(array(pts1))
pts2 = int32(array(pts2))

F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)

# We select only onlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)

img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1, F)
lines2 = lines2.reshape(-1,3)

img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)

#Calculate the homographies between the two images onlier points
_, H1, H2 = cv.stereoRectifyUncalibrated(pts1, pts2, F, img1.shape)

#Wrap the images to obtain a rectified viewpoint using the homographies
img_rect1 = cv.warpPerspective(img5, H1, img1.shape)
img_rect2 = cv.warpPerspective(img3, H2, img2.shape)

#Accuracy of the F-Matrix
error1 = []
error2 = []

for k in range(len(pts1)):
    error1.append(distance(pts1[k], lines1[k]))
    error2.append(distance(pts2[k], lines2[k]))

#Plot distances error
n = len(error1)
X,Y,Y_mean = [],[],[]

for i in range(n): 
    X.append(i)
    Y.append(abs(error1[i]-error2[i]))
    Y_mean.append(mean(Y))

figure(figsize=(20,5))
subplot(121), plot(X, Y), plot(X, Y_mean, label="Mean = " + str(Y_mean[0]))

#Plot result images without rectification
figure(figsize=(20,20))
subplot(121), axis("off"), imshow(img5)
subplot(122), axis("off"), imshow(img3)

#Plot result images with rectification
figure(figsize=(20,20))
subplot(121), axis("off"), imshow(img_rect1)
subplot(122), axis("off"), imshow(img_rect2)
show()

#Disparity calculations
#Calculations only function on gray images
img_disp1 = cv.cvtColor(img_rect1,cv.COLOR_BGR2GRAY)
img_disp2 = cv.cvtColor(img_rect2,cv.COLOR_BGR2GRAY)

stereo = cv.StereoBM_create(numDisparities=160, blockSize=15)
disparity = stereo.compute(img_disp1,img_disp2)

#Plot disparity
figure(figsize=(10,10))
imshow(disparity,'gray')
axis('off')
show()

#Save result images
cv.imwrite('result_left.jpg', img5)
cv.imwrite('result_right.jpg', img3)

cv.imwrite('result_left_rect.jpg', img_rect1)
cv.imwrite('result_right_rect.jpg', img_rect2)

cv.imwrite('disparity.jpg', disparity)

print('END TESTS')
#####################################################################################################################################
