I added are pre-trained models for 20,21, and 40 classes using the training data from ScanNet. 

*inference_from_single_ply.py* applies the model to a single ply file saved on disk. You have to specify the ply-file path (ply_file) and the path to the model (exp_name). This method saves a labeled ply-file in the same folder as the input file.

*inference.py* reads all pth-files stored in a provided folder (data_loader) and saves for each scene a labeled ply-file.

For the mentioned scripts I changed unet.py and data.py. They are now more modular and more flexible to use. Start *data_modular.py* when you want to train a new model. In that file you can specify the scale. *unet_modular.py* creates a model with specified parameters.

Make sure that you comment/uncomment the class label definition in *iou.py* depending on the amount of classes you train for.

[ScanNet](http://www.scan-net.org/)
-------

To train a small U-Net with 5cm-cubed sparse voxels:

1. Download [ScanNet](http://www.scan-net.org/) files
2. [Split](https://github.com/ScanNet/ScanNet/tree/master/Tasks/Benchmark) the files *vh_clean_2.ply and *_vh_clean_2.labels.ply files into 'train/' and 'val/' folders
3. Run 'pip install plyfile'
4. Run 'python prepare_data.py'
5. Run 'python unet.py'

You can train a bigger/more accurate network by changing `m` / `block_reps` / `residual_blocks` / `scale` / `val_reps` in unet.py / data.py, e.g.
```
m=32 # Wider network
block_reps=2 # Deeper network
residual_blocks=True # ResNet style basic blocks
scale=50 # 1/50 m = 2cm voxels
val_reps=3 # Multiple views at test time
batch_size=5 # Fit in 16GB of GPU memory
```
