import cv2
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, inf
from scipy.spatial.distance import euclidean

def irisLocalization(img):
    """
        1.Project the image in the vertical and horizontal
        direction to approximately estimate the center
        coordinates (Xp; Yp) of the pupil. Since the pupil is
        generally darker than its surroundings, the coordinates
        corresponding to the minima of the two
        projection profiles are considered as the center
        coordinates of the pupil.
        2. Binarize a 120 * 120 region centered at the point
        Xp; Yp by adaptively selecting a reasonable threshold
        using the gray-level histogram of this region. The
        centroid of the resulting binary region is considered
        as a more accurate estimate of the pupil coordinates.
        In this binary region, we can also roughly compute
        the radius of the pupil
        3. Calculate the exact parameters of these two circles
        using edge detection (Canny operator in experiments)
        and Hough transform in a certain region
        determined by the center of the pupil.
        
        In experiments, we perform the second step twice 
        for a reasonably accurate estimate. 
    Args:
        img ([type]): This will reduce the region for edge
        detection and the search space of Hough transform and,
        thus, result in lower computational demands.
    """
    
    # Project the image in the vertical and horizontal
    # direction to approximately estimate the center
    # coordinates Xp; Yp of the pupil
    xp, yp = getPupilCentroid(img)
    # print(xp,yp)
    
    # 2. Binarize a 120 X 120 region centered at the point
    # Xp; Yp by adaptively selecting a reasonable threshold
    # >>>>>> using the gray-level histogram of this region. 
    # The centroid of the resulting binary region is considered
    # as a more accurate estimate of the pupil coordinates.
    xp, yp = adaptiveCentroid(img,xp,yp,size=60)
    # cimg = img.copy()
    # cv2.circle(cimg,(xp,yp),rp,(255,0,0),2)
    # plt.imshow(cimg,cmap="gray")
    # plt.show()

    # In this binary region, we can also roughly compute
    # the radius of the pupil
    cimg = img.copy()
    x0, x1,y0, y1  = getCorner(xp, yp, half_width=60)
    try:
        # region = cv2.cvtColor(img[x0:x1,y0:y1],cv2.COLOR_GRAY2BGR)
        _, region = cv2.threshold(img[y0:y1,x0:x1], 60, 255, cv2.THRESH_BINARY)
    except:
        xp = int(img.shape[0]/2)
        yp = int(img.shape[1]/2)
        # region = cv2.cvtColor(img[yp-60:yp+60,xp-60:xp+60],cv2.COLOR_GRAY2BGR)
        _, region = cv2.threshold(img[yp-60:yp+60,xp-60:xp+60], 60, 255, cv2.THRESH_BINARY)
    
    # Calculate the exact parameters of these two circles
    # using edge detection (Canny operator in experiments) 
    # and Hough transform in a certain region
    # determined by the center of the pupil.
    
    edges = cv2.Canny(region,100,200)
    minR = 0
    maxR = 0
    inner_circle = cv2.HoughCircles(edges,
                                    cv2.HOUGH_GRADIENT, 
                                    1, 250,
                                    param1=30, param2=10,
                                    minRadius=minR, maxRadius=maxR)
    # print(type(inner_circle))
    try:
        inner_circle = np.uint16(np.around(inner_circle))
    except:
        try:
            print(x0,x1,y0,y1)
            region = cv2.cvtColor(img[yp-60:yp+60,xp-60:xp+60],cv2.COLOR_GRAY2BGR)
        except:
            xp = int(img.shape[0]/2)
            yp = int(img.shape[1]/2)
            region = cv2.cvtColor(img[yp-60:yp+60,xp-60:xp+60],cv2.COLOR_GRAY2BGR)
        edges = cv2.Canny(region,100,200)
        inner_circle = cv2.HoughCircles(edges,
                                        cv2.HOUGH_GRADIENT, 
                                        1, 250,
                                        param1=30, param2=10,
                                        minRadius=minR, maxRadius=maxR)
    # print(inner_circle)

    
    for i in inner_circle[0, :]:
        # draw the outer circle
        i[0] += max(xp-60,0)
        i[1] += max(yp-60,0)
        cv2.circle(cimg, (int(i[0]), int(i[1])), int(i[2]), (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(cimg, (int(i[0]), int(i[1])), 2, (0, 0, 255), 3)

    # plt.imshow(cimg)
    # plt.show()

    edges = cv2.Canny(img,20,30)
    minR = 98
    maxR = minR + 20
    outer_circle = cv2.HoughCircles(edges,
                                    cv2.HOUGH_GRADIENT, 
                                    1, 250,
                                    param1=30, param2=10,
                                    minRadius=minR, maxRadius=maxR)

    outer_circle = np.uint16(np.around(outer_circle))

    try:
        flag = (sqrt(((outer_circle.flatten()- \
                    inner_circle.flatten())**2).sum()) \
                > 0.6*outer_circle.flatten()[-1])
    except:
        flag = False
        
    if flag:
        outer_circle[0,0,:2] = inner_circle[0,0,:2]
        outer_circle[0,0,-1] = inner_circle[0,0,-1]+55
    
    for i in outer_circle[0, :]:
        # draw the outer circle
        cv2.circle(cimg, (int(i[0]), int(i[1])), int(i[2]), (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(cimg, (int(i[0]), int()), 2, (0, 0, 255), 3)

    # plt.imshow(cimg)
    # plt.show()
    
    return inner_circle, outer_circle


def getCorner(x, y, half_width=60):
    x0, x1, y0, y1 = x - half_width, x + half_width, y - half_width, y + half_width
    x0 = x0 if x0 >= 0 else 0
    x1 = x1 if x1 <= 280 else 280
    y0 = y0 if y0 >= 0 else 0
    y1 = y1 if y1 <= 320 else 320
    return x0, x1, y0, y1



def projectImg(img,axis):
    try:
        return np.sum(img, axis=axis)
    except:
        pass

def getPupilCentroid(img):
    x_projection = projectImg(img=img,axis=0)
    y_projection = projectImg(img=img,axis=1)
    
    xp = np.argmin(x_projection)
    yp = np.argmin(y_projection)
    
    return xp, yp
    
def adaptiveCentroid(img,xp,yp,size=60):
    for _ in range(2):
        region_ = img[yp-size:yp+size,xp-size:xp+size].copy()
        _, img_binary = cv2.threshold(region_,64,65,cv2.THRESH_BINARY)
        xp_,yp_ = getPupilCentroid(img_binary)
        xp = xp_ + (xp - size)
        yp = yp_ + (yp - size)
        # p = region_.copy()
        # cv2.circle(p,(xp_,yp_),1,(255,0,0),2)
        # print(xp,yp)
        # plt.imshow(p,cmap="gray")
        # plt.show()
        
    # print(xp_,yp_)
    # rp = radiusCalc(img_binary)
    # print(rp)
    # draw = cv2.circle(img_binary[yp_-size:yp_+size,xp_-size:xp_+size],(xp_,yp_),rp,(255,0,0),2)
    # plt.imshow(draw)
    # plt.title("adaptive final")
    # plt.show()
    
    # xp = min(319,max(0,xp))
    # yp = min(279,max(0,yp))   
    return xp,yp
        

def radiusCalc(img_binary):
    img_binary_ = np.where(img_binary > 1, 0, 1)
    # print(img_binary_)
    diameter = max(projectImg(img_binary_, axis=0).max(), projectImg(img_binary_, axis=1).max())
    return int(0.5 * diameter)
    
    
# def hough(img,xp,yp, minR = 0, maxR = 0,size = 60):
#     # refer to "https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html"
#     cimg = img[yp-size:yp+size,xp-size:xp+size].copy()
#     mask = cv2.inRange(cimg,0,70)
#     masked = cv2.bitwise_and(cimg,mask)
#     edges = cv2.Canny(cimg,100,220)
    
#     plt.subplot(121),plt.imshow(masked,cmap = 'gray')
#     plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#     plt.subplot(122),plt.imshow(edges,cmap = 'gray')
#     plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
#     plt.show()
    
#     # refer to "https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghcircles/py_houghcircles.html"
#     circles = cv2.HoughCircles(
#         edges,
#         cv2.HOUGH_GRADIENT,
#         10,250,
#         param1=50, param2=30,
#         minRadius=minR, maxRadius=maxR
#     )
    
#     circles = np.uint16(np.around(circles))
#     circles = circles.tolist()
#     if len(circles[0]) > 1:
#         # find the best circle by the shortest distance to the approx.
#         min_dst=inf
#         for i in circles[0]:
#             #find the circle whose center is closest to the approx center found above
#             dst = euclidean((xp,yp), (i[0],i[1]))
#             if dst<min_dst:
#                 min_dst=dst
#                 circle = i
#     else:
#         circle = circles[0][0]
        
#     print("new circle",circle)

#     # draw the outer circle
#     cv2.circle(img,(circle[0]+xp-size,circle[1]+yp-size),circle[2],(0,255,0),2)
#     # draw the center of the circle
#     cv2.circle(img,(circle[0]+xp-size,circle[1]+yp-size),2,(0,0,255),3)

#     plt.imshow(img)
#     plt.title("inner circle")
#     plt.show()
    
#     return circle