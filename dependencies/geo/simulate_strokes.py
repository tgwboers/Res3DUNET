import scipy.ndimage
import numpy as np
#import geodesic_distance
from skimage import measure
from scipy.sparse import csgraph
from scipy.sparse import csr_matrix
import nibabel
from matplotlib import pyplot


ref = nibabel.load(r'E:\ClusterMirror\deeplearning\TensorFlow\NiftyNet\NiftyNet\data\abdominal_ct\with_seg\1_Label.nii')
seg = nibabel.load(r'E:\ClusterMirror\deeplearning\TensorFlow\NiftyNet\NiftyNet\output\dense_vnet_abdominal_ct_cv\1__niftynet_out.nii.gz')
LUT=np.zeros([256],dtype=np.int)
LUT[[1,2,3,4,5,6,7,8]]=[[1,2,3,4,5,6,7,8]]
ref_img = LUT[ref.get_data()]
seg_img = LUT[seg.get_data()]

def path(predecessor,idx_src, idx_dest):
  path = []
  while idx_dest!=idx_src:
    if idx_dest<0:
     raise ValueError('no path found')
    path.append(idx_dest)
    idx_dest = predecessor[idx_dest]
  path.append(idx_src)
  return path[::-1]

dist=np.zeros_like(ref_img).astype(float)
for label in np.unique(ref_img):
 err = np.logical_and(ref_img==label, seg_img!=label)
 distL = scipy.ndimage.morphology.distance_transform_edt(err,return_distances = True)
 distL[distL<1.01]=0.
 dist=np.maximum(distL,dist)

prob = dist/np.sum(dist)
cost = np.exp(-dist/np.max(dist))

index_image = np.reshape(range(dist.size),dist.shape)
weight=[]
row_ind=[]
col_ind=[]
for dx1, dx2, dx3, dx4 in [(0,-1,1,None),(0,None,0,None),(1,None,0,-1)]:
 for dy1, dy2, dy3, dy4 in [(0,-1,1,None),(0,None,0,None),(1,None,0,-1)]:
  if dx1==0 and dx3==0 and dy1==0 and dy3==0:
   continue
  src=index_image[dx1:dx2,dy1:dy2]
  trg=index_image[dx3:dx4,dy3:dy4]
  cost_trg = cost[dx3:dx4,dy3:dy4]
  mask = ref_img[dx1:dx2,dy1:dy2,:]==ref_img[dx3:dx4,dy3:dy4,:]
  row_ind.extend(src[mask])
  col_ind.extend(trg[mask])
  weight.extend(cost_trg[mask])

graph = csr_matrix( (weight, (row_ind,col_ind)), (dist.size,dist.size))

cdf = np.cumsum(np.reshape(prob,[-1]))
cdf_samples = np.random.uniform(0.,1.,[100])
starting_indices = np.searchsorted(cdf,cdf_samples)
starting_points = list(zip(*np.unravel_index(starting_indices, dist.shape)))
label_specific_starting_points = []
for label in np.unique(ref_img):
 cdf = np.cumsum(np.reshape(prob*(ref_img==label),[-1]))
 cdf_samples = np.random.uniform(0.,1.,[10])*cdf[-1]
 starting_indices = np.searchsorted(cdf,cdf_samples)
 label_specific_starting_points.extend(list(zip(*np.unravel_index(starting_indices, dist.shape))))

import time
label_cnt=[0]*16
for seed in starting_points+label_specific_starting_points:
  st=time.time()
  label = ref_img[seed]
  
  err2= np.logical_and(ref_img[:,:,seed[2]]==label, seg_img[:,:,seed[2]]!=label)
  dist2 = scipy.ndimage.morphology.distance_transform_edt(err2,return_distances = True)
  print(time.time()-st)
  prob2 = dist2/np.sum(dist2)
  cc2 = measure.label(err2)
  print(time.time()-st)
  end_point_prob = prob2 * (cc2 == cc2[seed[0],seed[1]])
  end_point_cdf = np.cumsum(np.reshape(end_point_prob,[-1]))
  end_point_cdf_sample = np.random.uniform(0.,1.,[1])*end_point_cdf[-1]
  end_point_index = np.searchsorted(end_point_cdf,end_point_cdf_sample)
  end_point = list(zip(*np.unravel_index(end_point_index, dist2.shape)))[0]
  print(time.time()-st)
    
  graph_start_index = index_image[seed]
  graph_end_index = index_image[end_point[0],end_point[1],seed[2]]
  D=csgraph.shortest_path(graph, indices = [graph_start_index,graph_end_index],return_predecessors=True)
  print(time.time()-st)
  stroke = path(D[1][0,:],graph_start_index,graph_end_index)
  print(time.time()-st)
  stroke_points = list(zip(*np.unravel_index(stroke, index_image.shape)))
  label_cnt[label]+=1

