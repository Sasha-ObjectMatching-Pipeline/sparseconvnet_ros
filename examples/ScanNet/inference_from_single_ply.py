import torch
import sparseconvnet as scn
from unet_modular import Model
import sys
import NYU40_colors
from plyfile import PlyData
import math
import numpy as np

num_classes = 40

# VALID_CLAS_IDS have been mapped to the range {0,1,...,19}
remapper=np.ones(150)*(-100)
print("Semantic segmentation with {0} classes".format(num_classes))
if num_classes is 20:
    # VALID_CLAS_IDS have been mapped to the range {0,1,...,19}
    for i,x in enumerate([1,2,3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39]):
        remapper[x]=i
if num_classes is 21:
    # VALID_CLAS_IDS have been mapped to the range {0,1,...,19}
    for i,x in enumerate([1,2,3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39,40]):
        remapper[x]=i


def visualize(ids, mesh_file, output_file):
    if not output_file.endswith('.ply'):
        sys.stderr.write('ERROR: ' + 'output file must be a .ply file' + '\n')
    colors = NYU40_colors.create_color_palette_legend()
    num_colors = len(colors)
    with open(mesh_file, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        if num_verts != len(ids):
           sys.stderr.write('ERROR: ' + '#predicted labels = ' + str(len(ids)) + 'vs #mesh vertices = ' + str(num_verts))
        # *_vh_clean_2.ply has colors already
        for i in range(num_verts):
            if ids[i]+1 >= num_colors:
               sys.stderr.write('ERROR: ' + 'found predicted label ' + str(ids[i]) + ' not in nyu40 label set')
            color = colors[ids[i]+1]
            plydata['vertex']['red'][i] = color[0]
            plydata['vertex']['green'][i] = color[1]
            plydata['vertex']['blue'][i] = color[2]
    plydata.write(output_file)



#ply_file ='/home/edith/Software/SparseConvNet/examples/ScanNet/test/GH25_office_ElasticFusion_rotated.ply'
ply_file='/home/edith/Downloads/ETH_change_detection_ds/icra_2017_change_detection/living_room/complete_mesh/observation_0_aligned.ply'
exp_name='unet_scale20_m16_rep1_noResidualBlocks_' + str(num_classes) +'classes/unet_scale20_m16_rep1_noResidualBlocks'
scale = 20
full_scale=4096
val_reps=1

a= PlyData().read(ply_file)
v=np.array([list(x) for x in a.elements[0]])    #elements[0] stores all vertices with [x,y,z,r,g,b,alpha]
v = v.astype(np.float32)
coords=np.ascontiguousarray(v[:,:3]-v[:,:3].mean(0))
colors=np.ascontiguousarray(v[:,3:6])/127.5-1
fake_labels = np.zeros(len(coords))
#torch.save((coords,colors,w),fn[:-4]+'.pth')    #saves for each vertex the coordinates, color and label


locs=[]
a = coords
b = colors
c = fake_labels
m=np.eye(3)
m[0][0]*=np.random.randint(0,2)*2-1
m*=scale
theta=np.random.rand()*2*math.pi
m=np.matmul(m,[[math.cos(theta),math.sin(theta),0],[-math.sin(theta),math.cos(theta),0],[0,0,1]])
a=np.matmul(a,m)+full_scale/2+np.random.uniform(-2,2,3)
m=a.min(0)
M=a.max(0)
q=M-m
offset=-m+np.clip(full_scale-M+m-0.001,0,None)*np.random.rand(3)+np.clip(full_scale-M+m+0.001,None,0)*np.random.rand(3)
a+=offset
idxs=(a.min(1)>=0)*(a.max(1)<full_scale)    #only those points which have coords >= 0 and < full_scale are kept
a=a[idxs]
b=b[idxs]
c=c[idxs]
a=torch.from_numpy(a).long()

#we do not need that, with the last columnt you can specify the sample you want to add the point to
#locs = torch.cat([a,torch.LongTensor(a.shape[0],1).fill_(0)],1) #the fourth column is 0, why?
locs = a    #long
feats = torch.from_numpy(b).float()     #float
labels = torch.from_numpy(c)            #double
point_ids = torch.from_numpy(np.nonzero(idxs)[0])   #long
batch = {'x': [locs,feats], 'y': labels.long(), 'id': 0, 'point_ids': point_ids} #this is a dictionary

use_cuda = torch.cuda.is_available()
unet=Model()
if use_cuda:
    unet=unet.cuda()

training_epoch=scn.checkpoint_restore(unet,exp_name,'unet',use_cuda)
if training_epoch is 1:
    print("Training epoch is 1. The model probably couldn't be loaded.")
print('#classifer parameters', sum([x.nelement() for x in unet.parameters()]))

with torch.no_grad():
    unet.eval()
    scn.forward_pass_multiplyAdd_count = 0
    scn.forward_pass_hidden_states = 0
    store=torch.zeros(len(labels), num_classes)
    for rep in range(1, 1 + val_reps):
        if use_cuda:
            batch['x'][1] = batch['x'][1].cuda()
        predictions = unet(batch['x'])
        store.index_add_(0, batch['point_ids'], predictions.cpu())

labels = store.max(1)[1].numpy()
if num_classes != 40:
    for i, l in enumerate(labels):
        labels[i] = np.where(remapper == l)[0][0] - 1

pth_save = ply_file[:-4] + '_pred_legend_' + str(num_classes) + '.ply'
visualize(labels, ply_file, pth_save)


