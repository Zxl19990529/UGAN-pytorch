import torch
import torch.nn as nn 
from torch.nn import Conv2d,LeakyReLU,BatchNorm2d, ConvTranspose2d,ReLU
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2,os


def get_transforms():
    transform = transforms.Compose([
        transforms.ToTensor(),# H,W,C -> C,H,W && [0,255] -> [0,1]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) #[0,1] -> [-1,1]
    ])
    return transform


class MYDataSet(Dataset):
    def __init__(self,src_data_path,dst_data_path):
        self.train_A_imglist = self.get_imglist(src_data_path)
        self.train_B_imglist = self.get_imglist(dst_data_path)
        self.transform = get_transforms()
    
    def get_imglist(self,img_dir):
        img_name_list = sorted(os.listdir(img_dir))
        img_list = []
        for img_name in img_name_list:
            img_path = os.path.join(img_dir,img_name)
            img_list.append(img_path)
        return img_list
    def __len__(self):
        return len(self.train_A_imglist)
    def __getitem__(self,index):
        train_A_img_path = self.train_A_imglist[index]
        train_B_img_path = self.train_B_imglist[index]

        train_A_img = cv2.imread(train_A_img_path)
        train_B_img = cv2.imread(train_B_img_path)

        train_A_tensor = self.transform(train_A_img)
        train_B_tensor = self.transform(train_B_img)

        return [train_A_tensor,train_B_tensor]

