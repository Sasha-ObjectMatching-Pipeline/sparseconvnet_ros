import torch
import sparseconvnet as scn
from unet_modular import Model
import sys
import NYU40_colors
from plyfile import PlyData
import glob
import data_modular
import os
import numpy as np


# VALID_CLAS_IDS have been mapped to the range {0,1,...,19}

pth_dir = '/usr/mount/v4rtemp/el/SparseConvNet/'
#pth_dir=''

remapper=np.ones(150)*(-100)
for i,x in enumerate([1,2,3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39]):
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


data_modular.batch_size = 1
data_loader = data_modular.load_val_data(pth_dir + 'val_20classes/')

use_cuda = torch.cuda.is_available()
exp_name='unet_scale20_m16_rep1_noResidualBlocks_20classes/unet_scale20_m16_rep1_noResidualBlocks_20classes'
num_classes = 20

unet=Model()
if use_cuda:
    unet=unet.cuda()

training_epoch=scn.checkpoint_restore(unet, exp_name,'unet',use_cuda)
print('training epoch: ' + str(training_epoch) + ' #classifer parameters', sum([x.nelement() for x in unet.parameters()]))

with torch.no_grad():
    unet.eval()
    scn.forward_pass_multiplyAdd_count = 0
    scn.forward_pass_hidden_states = 0
    store=torch.zeros(data_modular.valOffsets[-1], num_classes)   #valOffsets is the number of all verteces of all scenes
    for rep in range(1, 1 + data_modular.val_reps):
        for i, batch in enumerate(data_loader):    #one batch can contain several scenes
            if use_cuda:
                batch['x'][1] = batch['x'][1].cuda()
            predictions = unet(batch['x'])
            store.index_add_(0, batch['point_ids'], predictions.cpu())

labels = store.max(1)[1].numpy()

pth_files = sorted(glob.glob(pth_dir + 'val_20classes/' + '*.pth'))
ply_path = '/usr/mount/v4rtemp/datasets/ScanNet/ScanNetV2/scans/'
pth_save = pth_dir + 'unet_scale20_m16_rep1_noResidualBlocks_20classes/val_results/legend_labels/'
#pth_save = pth_dir + 'val/legend_color/'
for batch in data_loader:
    print(batch['id'])
    scene_ids = batch['id']  # all ids of this batch
    for s_id in scene_ids:
        f = os.path.basename(pth_files[s_id])
        ply_folder = f[0:12] + '/'   #get only the scene_name
        ply_file = ply_folder[:-1] + '_vh_clean_2.ply'
        #get the data for each scene
        start_idx = data_modular.valOffsets[s_id]
        end_idx = data_modular.valOffsets[s_id+1]
        scene_labels = labels[start_idx:end_idx]

        if num_classes == 20:
            for i,l in enumerate(scene_labels):
                scene_labels[i] = np.where(remapper==l)[0][0] - 1

        if not os.path.isfile(pth_save + ply_file[:-4]+'_pred.ply'):
            print(pth_save + ply_file[:-4]+'_pred.ply')
            visualize(scene_labels, ply_path + ply_folder + ply_file, pth_save + ply_file[:-4]+'_pred.ply')


