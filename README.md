# Pytorch code for view-GCN [CVPR2020], view-GCN++ [TPAMI 2023].

Xin Wei, Ruixuan Yu and Jian Sun. **View-GCN: View-based Graph Convolutional Network for 3D Shape Analysis**. CVPR, accepted, 2020. [[pdf]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Wei_View-GCN_View-Based_Graph_Convolutional_Network_for_3D_Shape_Analysis_CVPR_2020_paper.pdf) [[supp]](http://openaccess.thecvf.com/content_CVPR_2020/supplemental/Wei_View-GCN_View-Based_Graph_CVPR_2020_supplemental.pdf)

Xin Wei, Ruixuan Yu and Jian Sun. **Learning view-based graph convolutional network for multi-view 3d shape analysis**. IEEE TPAMI, 2023.

## Citation
If you find our work useful in your research, please consider citing:
```
@InProceedings{Wei_2020_CVPR,
author = {Wei, Xin and Yu, Ruixuan and Sun, Jian},
title = {View-GCN: View-Based Graph Convolutional Network for 3D Shape Analysis},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

```
@ARTICLE{9947327,
  author={Wei, Xin and Yu, Ruixuan and Sun, Jian},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Learning View-Based Graph Convolutional Network for Multi-View 3D Shape Analysis}, 
  year={2023},
  volume={45},
  number={6},
  pages={7525-7541}}
```
## Training

### Requiement

This code is tested on Python 3.6 and Pytorch 1.0 + 

### Dataset

First download the 20 views ModelNet40 dataset provided by [[rotationnet]](https://github.com/kanezaki/pytorch-rotationnet) and put it under `data`

`https://drive.google.com/file/d/1Z8UphI48B9KUJ9zhIhcgXaRCzZPIlztb/view?usp=sharing`

Rotated-ModelNet40 dataset: ``

Aligned-ScanObjectNN dataset: `https://drive.google.com/file/d/1ihR6Fv88-6FOVUWdfHVMfDbUrx2eIPpR/view?usp=sharing`

Rotated-ScanObjectNN dataset: `https://drive.google.com/file/d/1GCwgrfbO_uO3Qh9UNPWRCuz2yr8UyRRT/view?usp=sharing`



### Command for training:

`python train.py -name view-gcn -num_models 0 -weight_decay 0.001 -num_views 20 -cnn_name resnet18`

The code is heavily borrowed from [[mvcnn-new]](https://github.com/jongchyisu/mvcnn_pytorch).

We also provide a [trained view-GCN network](https://drive.google.com/file/d/1qkltpvabunsI7frVRSEC9lP2xDP6cDj3/view?usp=sharing) achieving 97.6% accuracy on ModelNet40.

`https://drive.google.com/file/d/1qkltpvabunsI7frVRSEC9lP2xDP6cDj3/view?usp=sharing`

## Reference
Asako Kanezaki, Yasuyuki Matsushita and Yoshifumi Nishida. RotationNet: Joint Object Categorization and Pose Estimation Using Multiviews from Unsupervised Viewpoints. CVPR, 2018.

Jong-Chyi Su, Matheus Gadelha, Rui Wang, and Subhransu Maji. A Deeper Look at 3D Shape Classifiers. Second Workshop on 3D Reconstruction Meets Semantics, ECCV, 2018.

