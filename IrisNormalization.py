import numpy as np
from math import pi,sin,cos,floor
import math

"""Refer to [https://journals.sagepub.com/doi/pdf/10.1177/1729881417703931]
"""

def irisNormalization(img:np.ndarray,inner_circle:np.ndarray,outer_circle:np.ndarray,offset=0):
    '''
    Project the original iris from a Cartesian coordinate system into a doubly dimensionless pseudopolar coordinate.
    Counterclockwise unwrap the iris ring to a rectangular block, and normalizes irises of different size to the same size.

    Args:
        img: image to be processed
        inner_circle: detected inner_circle parameters from localization
        outer_circle: detected outer_circle parameters from localization
        offset: offset determines the starting point while unwrapping the iris ring

    Returns:
        img_normalized: a normalized iris image
    '''

    M,N = 64, 512
    # create a placeholder for normalized image
    img_normalized = np.zeros((M,N))

    try:
        [xp_,yp_,rp] = inner_circle.astype(int).flatten()
        [xi_,yi_,ri] = outer_circle.astype(int).flatten()
    except:
        [xp_,yp_,rp] = inner_circle.astype(int).flatten()[:3]
        [xi_,yi_,ri] = outer_circle.astype(int).flatten()[:3]

    for X in range(N):
        for Y in range(M):
            theta = 2*pi*(X/N) + offset
            if theta > 2*pi:
                theta -= 2*pi
            xp,yp = unwrap(xp_,yp_,rp,theta)
            xi,yi = unwrap(xi_,yi_,ri,theta)

            x = int(xp + (xi - xp)*(Y/M))
            y = int(yp + (yi - yp)*(Y/M))
            try:
                img_normalized[Y,X] = img[y,x]
            except:
                continue

    return img_normalized


# def irisNormalization(img, inner_circle:np.ndarray,outer_circle:np.ndarray,offset=0):
#     M, N = 64, 512
#     # create a placeholder for normalized image
#     img_normalized = np.zeros((M, N))
#     # xc_inner, yc_inner, rp, xc_outer, yc_outer, ri

#     try:
#         [xc_inner, yc_inner, rp] = inner_circle.astype(int).flatten()
#         [xc_outer, yc_outer, ri] = outer_circle.astype(int).flatten()
#     except:
#         [xc_inner, yc_inner, rp] = inner_circle.astype(int).flatten()[:3]
#         [xc_outer, yc_outer, ri] = outer_circle.astype(int).flatten()[:3]


#     for X in range(N):
#         for Y in range(M):
#             theta = 2 * math.pi * (X / N)

#             xp = xc_inner - rp * np.cos(theta)
#             yp = yc_inner + rp * np.sin(theta)

#             xi = xc_outer - ri * np.cos(theta)
#             yi = yc_outer + ri * np.sin(theta)

#             x = math.floor(xp + ((xi - xp)) * (Y / M))
#             y = math.floor(yp + ((yi - yp)) * (Y / M))
#             img_normalized[Y, X] = img[y, x]

#     return img_normalized


def unwrap(x,y,r,theta):
    # this method normalizes irises of different size to the same size.
    # Similar to this scheme, we counterclockwise unwrap the iris
    # ring to a rectangular block with a fixed size
    x += r*cos(theta)
    y += r*sin(theta)
    return x,y