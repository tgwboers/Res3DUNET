import geodesic_distance
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt
def test_geodesic_distance2d():
    I = np.asarray(Image.open('./data/img2d.png').convert('L'), np.float32)
    S = np.zeros_like(I, np.uint8)
    S[100][100] = 1
    t0 = time.time()
    D1 = geodesic_distance.geodesic2d_fast_marching(I,S)
    t1 = time.time()
    D2 = geodesic_distance.geodesic3d_raster_scan(I,S,np.float32(0),np.float32(1))
    dt1 = t1 - t0
    dt2 = time.time() - t1
    print( "runtime(s) of fast marching {}".format(dt1))
    print( "runtime(s) of raster  scan  {}".format(dt2))
    plt.subplot(1,3,1); plt.imshow(I)
    plt.subplot(1,3,2); plt.imshow(D1)
    plt.subplot(1,3,3); plt.imshow(D2)
    plt.show()


if __name__ == '__main__':
    test_geodesic_distance2d()