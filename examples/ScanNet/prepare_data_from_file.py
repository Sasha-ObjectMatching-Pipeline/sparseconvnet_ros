# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import plyfile, numpy as np, multiprocessing as mp, torch
from itertools import repeat

dataset_dir = '/usr/mount/v4rtemp/datasets/ScanNet/ScanNetV2/scans/'
txt_dir = '/usr/mount/v4rtemp/el/SparseConvNet/'
txt_files = ['scannetv2_train.txt','scannetv2_val.txt']


# Map relevant classes to {0,1,...,19}, and ignored classes to -100
remapper=np.ones(150)*(-100)
for i,x in enumerate([1,2,3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39,40]):
    remapper[x]=i
#for i,x in enumerate(np.array(range(1,41))):   #use that for 40 classes
#    remapper[x]=i

def func(fn, ft):
    fn1 = dataset_dir +fn + '/'+ fn + '_vh_clean_2.ply'
    fn2 = fn1[:-3]+'labels.ply'
    a=plyfile.PlyData().read(fn1)
    v=np.array([list(x) for x in a.elements[0]])    #elements[0] stores all vertices with [x,y,z,r,g,b,alpha]
    coords=np.ascontiguousarray(v[:,:3]-v[:,:3].mean(0))
    colors=np.ascontiguousarray(v[:,3:6])/127.5-1
    a=plyfile.PlyData().read(fn2)
    w = remapper[np.array(a.elements[0]['label'])]
    save_path = txt_dir + ft + '_21classes/' + fn +'.pth'
    #save_path = txt_dir + ft + '_40classes/' + fn + '.pth'
    torch.save((coords,colors,w),save_path)    #saves for each vertex the coordinates, color and label
    print(fn1, fn2)

for file in txt_files:
    path = txt_dir + file
    file_type = 'train' if 'train' in file else 'val'
    #for each line of path-file get the data, process it and store it in the train/val folder
    with open(path) as f:
        lines = [line.rstrip('\n') for line in f]
        p = mp.Pool(processes=mp.cpu_count())
        p.starmap(func, zip(lines, repeat(file_type)))
        p.close()
        p.join()
