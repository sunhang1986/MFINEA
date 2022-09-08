# MFINEA
Multi-level Features Interaction and Efficient Non-local Information Enhanced Channel Attention for Single Image Dehazing
==
Created by Bohui Li, [Hang Sun](https://github.com/sunhang1986), Zhiping Dan from Department of Computer and Information, China Three Gorges University.

Introduction
--
 Our paper proposes an MFINEA dehazing network. The proposed network includes a multi-level features interaction module and a efficient non-local information enhanced channel attention module.

Prerequisites
--
+ Pytorch 1.8.0
+ Python 3.7.1
+ CUDA 11.7
+ Ubuntu 18.04

Test
--
The [Download](https://sites.google.com/view/reside-dehaze-datasets) path of RESIDE dataset . the [Download](https://labsites.rochester.edu/gsharma/research/computer-vision/hazerd/) path of Hazerd dataset . the [Download](https://data.vision.ee.ethz.ch/cvl/ntire18//o-haze/) path of Ohaze dataset . 

Test the model on *RESIDE*:

` python   test.py   --cuda --gpus 0,1 --test --test_ori --test_path test_imgpath --Gx1_model_path premodel/epoch_ots_Gx1.pth` 

Test the model on *Hazerd*:

` python   test.py   --cuda --gpus 0,1 --test --test_ori --test_path test_imgpath --Gx1_model_path premodel/epoch_hazerd_Gx1.pth` 

Test the model on *Ohaze*:

` python   test.py   --cuda --gpus 0,1 --test --test_ori --test_path test_imgpath --Gx1_model_path premodel/epoch_ohaze_Gx1.pth` 


