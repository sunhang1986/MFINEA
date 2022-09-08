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

Test:
--
The [Download](https://sites.google.com/view/reside-dehaze-datasets) path of RESIDE dataset . the [Download](https://data.vision.ee.ethz.ch/cvl/ntire19//dense-haze/) path of Densehaze dataset. the [Download](https://data.vision.ee.ethz.ch/cvl/ntire21/) path of Nhhaze dataset . The [Download](https://www.dropbox.com/s/k2i3p7puuwl2g59/Haze1k.zip?dl=0) path of haze1k dataset. You can [Download](https://pan.baidu.com/s/1yE7I83yCbEriL9cWdl2K6Q) the pre-training model through Baidu Netdisk.The extract the code is lo0v.


Test the model:
--
` python Test.py --input_dir "test_dataset_path" --result_dir "save_path" --weight "model_path" --gpus "1" `

Cal PSNR and SSIM:
--
` python cal_psnr_ssim.py --test_dir "the name of datasets" --input_imgs "results_path" --gt_imgs "gt_path" `
