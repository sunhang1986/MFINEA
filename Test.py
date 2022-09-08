import numpy as np
import os
import argparse
from tqdm import tqdm
import math
import torchvision.transforms as tt
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from Model import Base_Model
import utils
from DataSet import RESIDE_Test
from skimage import img_as_ubyte


parser = argparse.ArgumentParser(description='Image Deraining using MPRNet')
parser.add_argument('--input_dir', default='/data/sh_data/lbh/Test_ssim/thick', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
parser.add_argument('--weights', default='/data/sh_data/lbh/network/checkpoints/thick_fog/model_best.pth', type=str, help='Path to weights')
parser.add_argument('--gpus', default='1', type=str, help='CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

def padding_image(image, h, w):
    assert h >= image.size(2)
    assert w >= image.size(3)
    padding_top = (h - image.size(2)) // 2
    padding_down = h - image.size(2) - padding_top
    padding_left = (w - image.size(3)) // 2
    padding_right = w - image.size(3) - padding_left
    out = torch.nn.functional.pad(image, (padding_left, padding_right, padding_top,padding_down), mode='reflect')
    return out, padding_left, padding_left + image.size(3), padding_top,padding_top + image.size(2)


G = Base_Model(3,3)


utils.load_checkpoint(G , args.weights)
print("use weight:",args.weights)

G.cuda()
G = nn.DataParallel(G)
G.eval()

# 准备测试集
datasets =['outdoor']


for dataset in datasets:

    rgb_dir_test = os.path.join(args.input_dir, dataset)
    test_dataset = RESIDE_Test(rgb_dir_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False,
                             pin_memory=True)

    result_dir = os.path.join(args.result_dir, dataset)
    utils.mkdir(result_dir)

    with torch.no_grad():
        for ii, data_test in enumerate(tqdm(test_loader), 0):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            input_ = data_test[0].cuda()
            filenames = data_test[1]

            h,w = input_.shape[2],input_.shape[3]

            max_h = int(math.ceil(h / 4)) * 4
            max_w = int(math.ceil(w / 4)) * 4
            input_, ori_left, ori_right, ori_top, ori_down = padding_image(input_, max_h, max_w)


            restored = G(input_)
            restored = restored.data[:, :, ori_top:ori_down, ori_left:ori_right]
            restored = torch.clamp(restored, 0, 1)


            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()

            for batch in range(len(restored)):
                restored_img = img_as_ubyte(restored[batch])
                if datasets == ['outdoor']:
                    utils.save_img((os.path.join(result_dir, filenames[batch] + '.png')), restored_img)
                elif datasets == ['dense_outdoor']:
                    utils.save_img((os.path.join(result_dir, filenames[batch])), restored_img)
                elif datasets == ['test']:
                    utils.save_img((os.path.join(result_dir, filenames[batch])), restored_img)



