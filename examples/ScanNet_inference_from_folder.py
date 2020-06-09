import torch
import sparseconvnet as scn
from ScanNet.unet_modular import Model
import sys
from ScanNet import NYU40_colors
from plyfile import PlyData, PlyElement
import numpy as np
from scipy.special import entr
import glob
import os
from numpy.lib.recfunctions import merge_arrays

num_classes = 21

# VALID_CLASS_IDS have been mapped to the range {0,1,...,19}
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
        vert = plydata['vertex'] #same as plydata.elements[0]
        faces = plydata.elements[1]
        num_verts = plydata['vertex'].count
        if num_verts != len(ids):
           sys.stderr.write('ERROR: ' + '#predicted labels = ' + str(len(ids)) + 'vs #mesh vertices = ' + str(num_verts))
        # *_vh_clean_2.ply has colors already, save label id instead of alpha value
        for i in range(num_verts):
            if ids[i]+1 >= num_colors:
               sys.stderr.write('ERROR: ' + 'found predicted label ' + str(ids[i]) + ' not in nyu40 label set')
            color = colors[ids[i]+1]
            plydata['vertex']['red'][i] = color[0]
            plydata['vertex']['green'][i] = color[1]
            plydata['vertex']['blue'][i] = color[2]
        #add the label field to the ply file
        #check if element alpha exists
        props = [p.name for p in vert.properties]
        if 'alpha' not in props:
            # Create the new vertex data with appropriate dtype
            a = merge_arrays([vert.data, np.zeros(len(vert.data), [('alpha', 'u1')])], flatten=True)
            # Recreate the PlyElement instance
            v = PlyElement.describe(a, 'vertex')
            # Recreate the PlyData instance
            plydata = PlyData([v], text=True)
            vert = plydata['vertex']  # same as plydata.elements[0]

        (x, y, z, r, g, b, alpha) = (vert[t] for t in ('x', 'y', 'z', 'red', 'green', 'blue', 'alpha'))
        new_data = np.column_stack((x,y,z,r,g,b,ids))
        points_tuple = list([tuple(row) for row in new_data])
        new_points = np.core.records.fromrecords(points_tuple,
                                                 names='x,y,z,red,green,blue,label',
                                                 formats='f4,f4,f4,u1,u1,u1,u1')
        el = PlyElement.describe(new_points, 'vertex')
    PlyData([el, faces], text=False).write(output_file)

def saveConfToFile(store, coords):
    store = store.numpy()

    file = open("coordsConf.txt", "w")
    for p in range(len(coords)):
        file.write(str(coords[p][0]) + " " + str(coords[p][1]) + " " + str(coords[p][2]))
        softmax = np.exp(store[p]) / np.sum(np.exp(store[p]))
        for i in range(len(store[p])):
            label_id = np.where(remapper == i)[0][0]
            file.write(" " + str(label_id) + " " + str(softmax[i]))
        entropy = entr(softmax).sum(axis=0)
        #file.write(" entropy " + str(entropy))
        file.write("\n")
    file.close()

#ply_folder= '/mnt/fe59de27-965c-4dbe-aae9-e6ee6173bb7c/Datasets/ChangeDetectionDatasetEdith/20190729_Full_dataset/InputScenes/'
#ply_folder='/mnt/fe59de27-965c-4dbe-aae9-e6ee6173bb7c/Datasets/icra_2017_change_detection_ETH/living_room/complete_mesh/transformed/'
ply_folder='/media/edith/Sasha1/Edith_Datasets/ChangeDetectionDatasetEdith/GH30_office/ScalableFusion/InputScenes/'
result_folder = ply_folder + "/3D_SemSeg_Results/"
if not os.path.exists(result_folder):
	os.makedirs(result_folder)
scenes = glob.glob(ply_folder + "/*.ply")

dir = '/usr/mount/v4rtemp/el/SparseConvNet/'
exp_name='unet_scale50_m32_rep2_ResBlocksTrue_classes' + str(num_classes) + '/unet_scale50_m32_rep2_ResBlocksTrue_classes' + str(num_classes)
scale = 50
full_scale=4096
ep=512

use_cuda = torch.cuda.is_available()
unet=Model(num_classes)

if use_cuda:
    unet=unet.cuda()

training_epoch=scn.checkpoint_restore(unet, dir + exp_name,'unet',use_cuda, ep)
if training_epoch is 1:
    print("Training epoch is 1. The model probably couldn't be loaded.")
print('#classifer parameters', sum([x.nelement() for x in unet.parameters()]))

for file in scenes:
    a= PlyData().read(file)
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
    ##########################################################
    # m=np.eye(3)
    # m[0][0]*=np.random.randint(0,2)*2-1 #ouput is either 1 or -1
    # m*=scale
    # theta=np.random.rand()*2*math.pi
    # m=np.matmul(m,[[math.cos(theta),math.sin(theta),0],[-math.sin(theta),math.cos(theta),0],[0,0,1]])
    # a=np.matmul(a,m)+full_scale/2+np.random.uniform(-2,2,3)
    # m=a.min(0)
    # M=a.max(0)
    # q=M-m
    # offset=-m+np.clip(full_scale-M+m-0.001,0,None)*np.random.rand(3)+np.clip(full_scale-M+m+0.001,None,0)*np.random.rand(3)
    ##############################################################
    m = np.eye(3)
    m *= scale
    a = np.matmul(a, m) + full_scale / 2
    m = a.min(0)
    M = a.max(0)
    q = M - m
    offset = -m + np.clip(full_scale - M + m - 0.001, 0, None) + np.clip(full_scale - M + m + 0.001, None, 0)
    a += offset

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



    with torch.no_grad():
        unet.eval()
        scn.forward_pass_multiplyAdd_count = 0
        scn.forward_pass_hidden_states = 0
        store=torch.zeros(len(labels), num_classes)
        if use_cuda:
           batch['x'][1] = batch['x'][1].cuda()
        predictions = unet(batch['x'])
        store.index_add_(0, batch['point_ids'], predictions.cpu())

    labels = store.max(1)[1].numpy()
    if num_classes != 40:
        for i, l in enumerate(labels):
            labels[i] = np.where(remapper == l)[0][0] - 1

    pth_save = result_folder + os.path.basename(file)[:-4] + '_pred_legend_' + str(num_classes) + '.ply'
    #saveConfToFile(store, coords)
    visualize(labels, file, pth_save)
