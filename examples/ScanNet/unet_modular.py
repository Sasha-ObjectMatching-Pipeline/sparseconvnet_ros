# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This is a copy of unet.py, but it uses data_modular.py. That means that loading the train or val data has to be
# triggered manually. Otherwise those data get loaded automatically whenever you want to model t

# Options
m = 32 # 16 or 32
residual_blocks=True #True or False
block_reps = 2 #Conv block repetition factor: 1 or 2


import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sparseconvnet as scn
import time
from . import iou
from . import data_modular



class Model(nn.Module):
    def __init__(self, num_classes, m=m, residual_blocks=residual_blocks, block_reps=block_reps):
        nn.Module.__init__(self)
        self.sparseModel = scn.Sequential().add(
           scn.InputLayer(data_modular.dimension,data_modular.full_scale, mode=4)).add(
           scn.SubmanifoldConvolution(data_modular.dimension, 3, m, 3, False)).add(
               scn.UNet(data_modular.dimension, block_reps, [m, 2*m, 3*m, 4*m, 5*m, 6*m, 7*m], residual_blocks)).add(
           scn.BatchNormReLU(m)).add(
           scn.OutputLayer(data_modular.dimension))
        self.linear = nn.Linear(m, num_classes)
    def forward(self,x):
        x=self.sparseModel(x)
        x=self.linear(x)
        return x


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    dir = '/usr/mount/v4rtemp/el/SparseConvNet/'    #where the trained models get stored
    num_classes = 21
    exp_name = 'unet_scale40_m32_rep2_ResBlocksTrue_classes' + str(num_classes)
    restore_epoch = 256 #this is optional

    unet=Model(num_classes)
    if use_cuda:
        unet=unet.cuda()

    train_data_loader = data_modular.load_train_data(dir + 'train_' + str(num_classes) +'classes/')
    val_data_loader = data_modular.load_val_data(dir + 'val_' + str(num_classes) + 'classes/')

    training_epochs=512
    training_epoch=scn.checkpoint_restore(unet,dir+exp_name,'unet',use_cuda, restore_epoch)
    optimizer = optim.Adam(unet.parameters())
    print('#classifer parameters', sum([x.nelement() for x in unet.parameters()]))

    for epoch in range(training_epoch, training_epochs+1):
        unet.train()
        stats = {}
        scn.forward_pass_multiplyAdd_count=0
        scn.forward_pass_hidden_states=0
        start = time.time()
        train_loss=0
        for i,batch in enumerate(train_data_loader):
            optimizer.zero_grad()
            if use_cuda:
                batch['x'][1]=batch['x'][1].cuda()
                batch['y']=batch['y'].cuda()
            predictions=unet(batch['x'])
            loss = torch.nn.functional.cross_entropy(predictions,batch['y'])
            train_loss+=loss.item()
            loss.backward()
            optimizer.step()
        print(epoch,'Train loss',train_loss/(i+1), 'MegaMulAdd=',scn.forward_pass_multiplyAdd_count/len(data_modular.train)/1e6, 'MegaHidden',scn.forward_pass_hidden_states/len(data_modular.train)/1e6,'time=',time.time() - start,'s')
        scn.checkpoint_save(unet,dir+exp_name,'unet',epoch, use_cuda)

        if scn.is_power2(epoch):
            with torch.no_grad():
                unet.eval()
                store=torch.zeros(data_modular.valOffsets[-1], num_classes)
                scn.forward_pass_multiplyAdd_count=0
                scn.forward_pass_hidden_states=0
                start = time.time()
                for rep in range(1,1+data_modular.val_reps):
                    for i,batch in enumerate(val_data_loader):
                        if use_cuda:
                            batch['x'][1]=batch['x'][1].cuda()
                            batch['y']=batch['y'].cuda()
                        predictions=unet(batch['x'])
                        store.index_add_(0,batch['point_ids'],predictions.cpu())
                    print(epoch,rep,'Val MegaMulAdd=',scn.forward_pass_multiplyAdd_count/len(data_modular.val)/1e6, 'MegaHidden',scn.forward_pass_hidden_states/len(data_modular.val)/1e6,'time=',time.time() - start,'s')
                    iou.evaluate(store.max(1)[1].numpy(),data_modular.valLabels)
