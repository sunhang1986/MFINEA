import os
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from pdb import set_trace as stx
import random
import torchvision.transforms as tfs

class RESIDE_Dataset(Dataset):
    def __init__(self, path, crop_size, format='.jpg'):
        super(RESIDE_Dataset, self).__init__()
        self.size = crop_size

        self.haze_imgs_dir = os.listdir(os.path.join(path, 'hazy'))
        self.haze_imgs = [os.path.join(path, 'hazy', img) for img in self.haze_imgs_dir]

        self.clear_dir = os.path.join(path,'clear')
        self.format = format

    def __len__(self):
        return len(self.haze_imgs)

    def __getitem__(self, index):

        size = self.size

        haze = Image.open(self.haze_imgs[index]).convert("RGB")

        img = self.haze_imgs[index]
        id = img.split('/')[-1].split('_')[0]

        clear_id = os.path.join(self.clear_dir, (id + self.format))
        clear = Image.open(clear_id).convert("RGB")


        w, h = clear.size
        padw = size - w if w < size else 0
        padh = size - h if h < size else 0

        if padw!=0 or padh!=0:
            haze = TF.pad(haze, (0,0,padw,padh), padding_mode='reflect')
            clear = TF.pad(clear, (0,0,padw,padh), padding_mode='reflect')

        haze = TF.to_tensor(haze)
        clear = TF.to_tensor(clear)

        hh, ww = clear.shape[1], clear.shape[2]

        rr = random.randint(0, hh - size)
        cc = random.randint(0, ww - size)
        aug = random.randint(0, 8)

        haze = haze[:, rr:rr + size, cc:cc + size]
        clear = clear[:, rr:rr + size, cc:cc + size]


        if aug==1:
            haze = haze.flip(1)
            clear = clear.flip(1)
        elif aug==2:
            haze = haze.flip(2)
            clear = clear.flip(2)
        elif aug==3:
            haze = torch.rot90(haze,dims=(1,2))
            clear = torch.rot90(clear,dims=(1,2))
        elif aug==4:
            haze = torch.rot90(haze,dims=(1,2), k=2)
            clear = torch.rot90(clear,dims=(1,2), k=2)
        elif aug==5:
            haze = torch.rot90(haze,dims=(1,2), k=3)
            clear = torch.rot90(clear,dims=(1,2), k=3)
        elif aug==6:
            haze = torch.rot90(haze.flip(1),dims=(1,2))
            clear = torch.rot90(clear.flip(1),dims=(1,2))
        elif aug==7:
            haze = torch.rot90(haze.flip(2),dims=(1,2))
            clear = torch.rot90(clear.flip(2),dims=(1,2))

        haze = tfs.Normalize(mean=[0.5, 0.5, 0.55], std=[0.5, 0.5, 0.5])(haze)

        return  clear, haze


class RESIDE_Test(Dataset):
    def __init__(self, path, crop_size = 256, format='.jpg'):
        super(RESIDE_Test, self).__init__()

        self.haze_name = sorted(os.listdir(os.path.join(path,'hazy')))
        self.haze_imgs = [os.path.join(path, "hazy", x) for x in self.haze_name]

        self.format = format
        self.size = crop_size

    def __getitem__(self, index):

        haze = Image.open(self.haze_imgs[index]).convert("RGB")
        haze_imgs = TF.to_tensor(haze)

        file_name = self.haze_imgs[index]
        filename = os.path.basename(file_name.split('/')[-1]).split('.jpg')[0]
        # haze_imgs = tfs.Normalize(mean=[0.5, 0.5, 0.55], std=[0.5, 0.5, 0.5])(haze_imgs)

        return haze_imgs, filename

    def __len__(self):
        return len(self.haze_imgs)

class RESIDE_Val(Dataset):
    def __init__(self, path, format =".png"):
        super(RESIDE_Val, self).__init__()

        self.haze_name =  sorted(os.listdir(os.path.join(path,'hazy')))
        self.haze_imgs = [os.path.join(path, "hazy", x) for x in self.haze_name]

        self.format = format
        self.clear_dir = os.path.join(path, 'gt')

    def __getitem__(self, index):

        haze = Image.open(self.haze_imgs[index]).convert("RGB")
        haze_imgs = TF.to_tensor(haze)
        img = self.haze_imgs[index]
        id = img.split('/')[-1].split('_')[0]

        # 对应的清晰图片路劲
        clear_id = os.path.join(self.clear_dir, (id + self.format))
        clear = Image.open(clear_id).convert("RGB")
        clear = TF.to_tensor(clear)
        haze_imgs = tfs.Normalize(mean=[0.5, 0.5, 0.55], std=[0.5, 0.5, 0.5])(haze_imgs)
        return clear,haze_imgs

    def __len__(self):
        return len(self.haze_imgs)

