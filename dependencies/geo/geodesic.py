#python setup.py build
#python setup.py install

#python
import geodesic_distance
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
I = np.float32(scipy.misc.ascent())/255.
Cost = nd.gaussian_gradient_magnitude(I,1.)
Seed = np.zeros_like(Cost)
Seed[300,300]=1
A=geodesic_distance.geodesic2d_raster_scan(Cost,np.uint8(Seed))

plt.imshow(A)
plt.show()

I3 = np.stack([I[::2,::2],I[::-2,::2]],2)
I3=np.concatenate([I3,I3[:,:,::-1]],2)
I3=np.concatenate([I3,I3[:,:,::-1]],2)
I3=np.concatenate([I3,I3[:,:,::-1]],2)
I3=np.concatenate([I3,I3[:,:,::-1]],2)
I3=np.concatenate([I3,I3[:,:,::-1]],2)

Cost3 = nd.gaussian_gradient_magnitude(I3,1.)
Seed3 = np.zeros_like(Cost3)
Seed3[150,150,32]=1
A3=geodesic_distance.geodesic3d_raster_scan(Cost3,np.uint8(Seed3),1.,4.)
plt.imshow(A3[:,:,32])
plt.show()
plt.imshow(A3[:,64,:])
plt.show()
