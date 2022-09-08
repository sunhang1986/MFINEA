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
The [Download](https://sites.google.com/view/reside-dehaze-datasets) path of RESIDE dataset . the [Download](https://data.vision.ee.ethz.ch/cvl/ntire19//dense-haze/) path of Densehaze dataset. the [Download](https://data.vision.ee.ethz.ch/cvl/ntire21/) path of Nhhaze dataset . The [Download](https://www.dropbox.com/s/k2i3p7puuwl2g59/Haze1k.zip?dl=0) path of haze1k dataset

Test the model on *RESIDE*:
` python Test.py --input_dir "test_dataset_path" --result_dir "save_path" --weight "model_path" --gpus "1" `



