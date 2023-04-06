import numpy as np
from math import pi, exp, cos,sqrt
from scipy.signal import convolve2d


params = {
    "channel_1":{
        "delta_x": 3,
        "delta_y": 1.5
    },
    "channel_2":{
        "delta_x": 4.5,
        "delta_y": 1.5
    }
}

# 3.3.1 Spatial Filters

def Mi(x,y,f,i=1,theta=None):
    # Calcualte M1(x;y;f) or M2(x;y;f)
    if i == 1:
        return cos(2*pi*f*sqrt(x**2+y**2))
    else:
        return cos(2*pi*f*(x*cos(theta)+y*cos(theta)))

def G(x,y,delta_x,delta_y,f,i=1,theta=None):
    # Calcualte G(x;y;f)
    return (1/(2*pi*delta_x*delta_y))*exp(-.5*(x**2/delta_x**2 + y**2/delta_y**2)) * Mi(x,y,f,i,theta)


# As mentioned earlier, local details of the iris generally
# spread along the radial direction, so information density in
# the angular direction corresponding to the horizontal
# direction in the normalized image is higher than that in
# other directions, which is validated by our experimental
# results in Section 4.3. 
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Use M1 and horizontal direction

def spatialFilter(delta_x,delta_y):
    # Generate 8*8 spatial filter matrix
    f = 1/delta_y
    kernal = np.zeros((8,8))
    for i in range(8):
        for j in range(8):
            kernal[i,j] = G(x=j-4,y=i-4,delta_x=delta_x,delta_y=delta_y,f=f)
    return kernal

# 3.3.2 Feature Vector
# According to the above scheme, filtering the ROI (48 X 512)
def ROI(img):
    # Extarct the region of interest (ROI)
    return img[:48,:]


def featureVector(img1,img2,step=8):
    #  Filtering the ROI (48 * 512) with the defined multichannel spatial filters,
    #  Compute the mean m and the average absolute deviation σ of each filtered block as feature values,
    #  Arrange resulting feature values to a 1D feature vector.

    vectors = []
    # nrow,ncol = tuple(map(lambda x: int(x/step), img1.shape))
    nrow,ncol = img1.shape
    # print(nrow,ncol)
    for x in np.arange(0, nrow, step):
        for y in np.arange(0, ncol, step):
            product1 = img1[x:(x + step), y:(y + step)]
            vectors.extend([np.mean(np.absolute(product1)), np.mean(np.absolute(product1 - np.mean(np.absolute(product1))))])
            product2 = img2[x:(x + step), y:(y + step)]
            vectors.extend([np.mean(np.absolute(product2)), np.mean(np.absolute(product2 - np.mean(np.absolute(product2))))])
    return vectors


def featureExtraction(img):
    sfilter1 = spatialFilter(delta_x=params["channel_1"]["delta_x"],delta_y=params["channel_1"]["delta_y"])
    sfilter2 = spatialFilter(delta_x=params["channel_2"]["delta_x"],delta_y=params["channel_2"]["delta_y"])
    
    roi = ROI(img)
    img_sfilter1 = convolve2d(roi,sfilter1,mode="same")
    img_sfilter2 = convolve2d(roi,sfilter2,mode="same")
    vectors = featureVector(img_sfilter1,img_sfilter2)
    
    return vectors
