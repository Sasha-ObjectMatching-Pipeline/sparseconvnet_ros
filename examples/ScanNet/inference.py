import torch, data, iou
import sparseconvnet as scn
from unet import Model
import sys
import NYU40_colors
from plyfile import PlyData
import glob

def visualize(ids, mesh_file, output_file):
    if not output_file.endswith('.ply'):
        sys.stderr.write('ERROR: ' + 'output file must be a .ply file' + '\n')
    colors = NYU40_colors.create_color_palette()
    num_colors = len(colors)
    with open(mesh_file, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        if num_verts != len(ids):
           sys.stderr.write('ERROR: ' + '#predicted labels = ' + str(len(ids)) + 'vs #mesh vertices = ' + str(num_verts))
        # *_vh_clean_2.ply has colors already
        for i in range(num_verts):
            if ids[i] >= num_colors:
               sys.stderr.write('ERROR: ' + 'found predicted label ' + str(ids[i]) + ' not in nyu40 label set')
            color = colors[ids[i]]
            plydata['vertex']['red'][i] = color[0]
            plydata['vertex']['green'][i] = color[1]
            plydata['vertex']['blue'][i] = color[2]
    plydata.write(output_file)

use_cuda = torch.cuda.is_available()
exp_name='unet_scale20_m16_rep1_notResidualBlocks'

unet=Model()
if use_cuda:
    unet=unet.cuda()

training_epoch=scn.checkpoint_restore(unet,exp_name,'unet',use_cuda)
print('#classifer parameters', sum([x.nelement() for x in unet.parameters()]))

store=torch.zeros(data.valOffsets[-1],20)   #valOffsets is the number of all verteces of all scenes
for rep in range(1, 1 + data.val_reps):
    for i, batch in enumerate(data.val_data_loader):    #one batch can contain several scenes
        if use_cuda:
            batch['x'][1] = batch['x'][1].cuda()
        predictions = unet(batch['x'])
        store.index_add_(0, batch['point_ids'], predictions.cpu())

labels = store.max(1)[1].numpy()

files = sorted(glob.glob('val/*vh_clean_2.ply'))

for batch in data.val_data_loader:
    print(batch['id'])
    scene_ids = batch['id']  # all ids of this batch
    for s_id in scene_ids:
        f = files[s_id]
        #get the data for each scene
        start_idx = data.valOffsets[s_id]
        end_idx = data.valOffsets[s_id+1]
        labels = labels[start_idx:end_idx]
        visualize(labels, f, f[:-4]+'_pred.ply')


