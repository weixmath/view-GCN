# Pytorch code for view-GCN.

Xin Wei, Ruixuan Yu and Jian Sun. View-GCN: View-based Graph Convolutional Network for 3D Shape Analysis. CVPR, accepted, 2020. [[pdf]](http://gr.xjtu.edu.cn/c/document_library/get_file?folderId=1401787&name=DLFE-129432.pdf)


# Requiement

This code is tested on Python 3.6 and Pytorch 1.0 + 

# Training
## Dataset

First download the 20 views ModelNet40 dataset provided by   and put it under 'data'

## Command for training:
>python train.py -name view-gcn -num_models 0 -weight_decay 0.001 -num_views 20 -cnn_name resnet18

The code is heavily borrowed from [mvcnn_new](https://github.com/jongchyisu/mvcnn_pytorch).
