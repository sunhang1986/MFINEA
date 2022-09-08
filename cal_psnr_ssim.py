import numpy as np
import math
import cv2
import os
import numpy
from scipy.ndimage import gaussian_filter
from PIL import Image 
from scipy.signal import convolve2d
import argparse

from numpy.lib.stride_tricks import as_strided as ast

parser = argparse.ArgumentParser(description='For Test')
parser.add_argument('--test_dir', default='outdoor', type=str)
parser.add_argument('--input_imgs', default="/data/sh_data/lbh/network/results/test/",type=str)
parser.add_argument('--gt_imgs', default="/data/sh_data/lbh/Ex_tra/base_eca/data/test/gt/",type=str)
args = parser.parse_args()
data_name = args.test_dir



def psnr(img1, img2):
   mse = np.mean( (img1/255. - img2/255.) ** 2)
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def block_view(A, block=(3, 3)):
    shape = (A.shape[0]// block[0], A.shape[1]// block[1])+ block
    strides = (block[0]* A.strides[0], block[1]* A.strides[1])+ A.strides
    print(shape)
    print(strides)
    return ast(A, shape= shape, strides= strides)

def ssim(img1, img2, C1=0.01**2, C2=0.03**2):
    bimg1 = block_view(img1, (4,4))
    bimg2 = block_view(img2, (4,4))
    s1  = numpy.sum(bimg1, (-1, -2))
    s2  = numpy.sum(bimg2, (-1, -2))
    ss  = numpy.sum(bimg1*bimg1, (-1, -2)) + numpy.sum(bimg2*bimg2, (-1, -2))
    s12 = numpy.sum(bimg1*bimg2, (-1, -2))

    vari = ss - s1*s1 - s2*s2
    covar = s12 - s1*s2

    ssim_map =  (2*s1*s2 + C1) * (2*covar + C2) / ((s1*s1 + s2*s2 + C1) * (vari + C2))
    return numpy.mean(ssim_map)

 
def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
 
def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)
 
def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
 
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")
 
    M, N = im1.shape
    C1 = (k1*L)**2
    C2 = (k2*L)**2
    window = matlab_style_gauss2D(shape=(win_size,win_size), sigma=1.5)
    window = window/np.sum(np.sum(window))
 
    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)
 
    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1*im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2*im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1*im2, window, 'valid') - mu1_mu2
 
    ssim_map = ((2*mu1_mu2+C1) * (2*sigmal2+C2)) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))
 
    return np.mean(np.mean(ssim_map))

# # outdoor
# deal_img_path = r"/data/sh_data/lbh/network/results/outdoor/"
# gt_img_path = r"/data/sh_data/lbh/DATASETS/Test_DATASET/outdoor/gt/"

# dense
# deal_img_path = r"/data/sh_data/lbh/network/results/dense_outdoor/"
# gt_img_path = r"/data/sh_data/lbh/DATASETS/Test_DATASET/dense_outdoor/gt/"

# nhhaze
deal_img_path = args.input_imgs
gt_img_path = args.gt_imgs

# deal_img_path = r"/data/sh_data/lbh/network/results/test/"
# gt_img_path = r"/data/sh_data/lbh/Ex_tra/base_eca/data/test/gt/"



# thinfog
# deal_img_path = r"/data/sh_data/lbh/network/results/test/"
# gt_img_path = r"/data/sh_data/lbh/Test_ssim/thin/test/target/"

# moderatefog
# deal_img_path = r"/data/sh_data/lbh/network/results/test/"
# gt_img_path = r"/data/sh_data/lbh/Test_ssim/zhong/test/target/"

# thickfog
# deal_img_path = r"/data/sh_data/lbh/network/results/test/"
# gt_img_path = r"/data/sh_data/lbh/Test_ssim/thick/test/target/"

deal_img_list = os.listdir(deal_img_path)
gt_img_list = os.listdir(gt_img_path)

print("测试数据集长度：", len(deal_img_list))
print("gt数据集长度:", len(gt_img_list))

value_psnr_list = []
value_ssim_list = []

if data_name == "outdoor":
    for i, filename in enumerate(deal_img_list):

        # 计算PSNR
        img1 = cv2.imread(os.path.join(deal_img_path, filename))
        name = filename.split("_",1)[0] + ".png"
        img2 = cv2.imread(os.path.join(gt_img_path, name))
        print(os.path.join(deal_img_path, filename))
        print(os.path.join(gt_img_path, name))
        value_psnr = psnr(img1, img2)
        value_psnr_list.append(value_psnr)
        print(value_psnr)

        # 计算SSIM
        img3 = Image.open(os.path.join(deal_img_path, filename))
        name = filename.split("_",1)[0] + ".png"
        img4 = Image.open(os.path.join(gt_img_path, name))
        value_ssim = compute_ssim(np.array(img3.convert('L')), np.array(img4.convert('L')))
        value_ssim_list.append(value_ssim)
        print(value_ssim)
        print("-------------------------------------------------------")

elif data_name == "densehaze":

    for i, filename in enumerate(gt_img_list):
        img1 = cv2.imread(os.path.join(gt_img_path, filename))
        print(os.path.join(gt_img_path, filename))
        print(os.path.join(deal_img_path, filename))
        img2 = cv2.imread(os.path.join(deal_img_path,filename))

        value_psnr = psnr(img1, img2)
        value_psnr_list.append(value_psnr)
        print(value_psnr)

        img3 = Image.open(os.path.join(gt_img_path, filename))
        img4 = Image.open(os.path.join(deal_img_path, filename))

        value_ssim = compute_ssim(np.array(img3.convert('L')), np.array(img4.convert('L')))
        value_ssim_list.append(value_ssim)
        print(value_ssim)

        print("-------------------------------------------------------")

elif data_name == "nhhaze":

    for i, filename in enumerate(gt_img_list):
        # /data/sh_data/lbh/network/results/test
        img1 = cv2.imread(os.path.join(gt_img_path, filename))
        print(os.path.join(gt_img_path, filename))
        print(os.path.join(deal_img_path, filename))
        img2 = cv2.imread(os.path.join(deal_img_path,filename))

        value_psnr = psnr(img1, img2)
        value_psnr_list.append(value_psnr)
        print(value_psnr)

        img3 = Image.open(os.path.join(gt_img_path, filename))
        img4 = Image.open(os.path.join(deal_img_path, filename))

        value_ssim = compute_ssim(np.array(img3.convert('L')), np.array(img4.convert('L')))
        value_ssim_list.append(value_ssim)
        print(value_ssim)

        print("-------------------------------------------------------")

elif data_name == "thinfog":

    for i, filename in enumerate(gt_img_list):
        img1 = cv2.imread(os.path.join(gt_img_path, filename))
        name = filename.split("-", 1)[0]
        img2 = cv2.imread(os.path.join(deal_img_path, "{}-inputs.png".format(name)))

        print(gt_img_path + filename)
        print(deal_img_path + filename)
        value_psnr = psnr(img1, img2)
        value_psnr_list.append(value_psnr)
        print(value_psnr)

        img3 = Image.open(os.path.join(gt_img_path, filename))
        name = filename.split("-", 1)[0]
        img4 = Image.open(os.path.join(deal_img_path, "{}-inputs.png".format(name)))

        value_ssim = compute_ssim(np.array(img3.convert('L')), np.array(img4.convert('L')))
        value_ssim_list.append(value_ssim)
        print(value_ssim)

        print("-------------------------------------------------------")

elif data_name == "moderatefog":

    for i, filename in enumerate(gt_img_list):
        img1 = cv2.imread(os.path.join(gt_img_path, filename))
        name = filename.split("-", 1)[0]
        img2 = cv2.imread(os.path.join(deal_img_path, "{}-inputs.png".format(name)))

        print(gt_img_path + filename)
        print(deal_img_path + filename)
        value_psnr = psnr(img1, img2)
        value_psnr_list.append(value_psnr)
        print(value_psnr)

        img3 = Image.open(os.path.join(gt_img_path, filename))
        name = filename.split("-", 1)[0]
        img4 = Image.open(os.path.join(deal_img_path, "{}-inputs.png".format(name)))

        value_ssim = compute_ssim(np.array(img3.convert('L')), np.array(img4.convert('L')))
        value_ssim_list.append(value_ssim)
        print(value_ssim)

        print("-------------------------------------------------------")


elif data_name == "thickfog":

    for i, filename in enumerate(gt_img_list):
        img1 = cv2.imread(os.path.join(gt_img_path, filename))
        name = filename.split("-", 1)[0]
        img2 = cv2.imread(os.path.join(deal_img_path, "{}-inputs.png".format(name)))

        print(gt_img_path + filename)
        print(deal_img_path + filename)
        value_psnr = psnr(img1, img2)
        value_psnr_list.append(value_psnr)
        print(value_psnr)

        img3 = Image.open(os.path.join(gt_img_path, filename))
        name = filename.split("-", 1)[0]
        img4 = Image.open(os.path.join(deal_img_path, "{}-inputs.png".format(name)))

        value_ssim = compute_ssim(np.array(img3.convert('L')), np.array(img4.convert('L')))
        value_ssim_list.append(value_ssim)
        print(value_ssim)

        print("-------------------------------------------------------")



sum_value = 0
for i in value_psnr_list:
    sum_value = sum_value + i
print(sum_value)
print(len(value_psnr_list))
print("PSNR:",sum_value/len(value_psnr_list))
print("-----------------------------------------------")
sum_value = 0
for i in value_ssim_list:
    sum_value = sum_value + i
print(sum_value)
print(len(value_ssim_list))
print("SSIM:",sum_value/len(value_ssim_list))

