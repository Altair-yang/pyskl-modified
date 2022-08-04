# Author: Yang jie
# Date  :
import argparse
import os

import torch

from pyskl.datasets import build_dataset
from mmcv import Config


def parse_args():
    parser = argparse.ArgumentParser(description='Train a recognizer')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--test-last',
        action='store_true',
        help='whether to test the checkpoint after training')
    parser.add_argument(
        '--test-best',
        action='store_true',
        help='whether to test the best checkpoint (if applicable) after training')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--launcher',
        choices=['pytorch', 'slurm'],
        default='pytorch',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def Tensor2Image (input_tensor):
    from torchvision import transforms

    ToPIL = transforms.ToPILImage()
    image = input_tensor.cpu().clone()  # clone the tensor
    image = ToPIL(image)
    return image


args = parse_args()
cfg = Config.fromfile(args.config)

datasets = [build_dataset(cfg.data.test)]
dataset = datasets[0]

sample1 = dataset[6]
test_img,label = sample1['imgs'],sample1['label']   # 1,17,48,56,56

from adaptive_sampliing_test import *
clip_len = 36
new_img = sample_merge(test_img , clip_len)

image = test_img.squeeze(0)  # 17,48,56,56
image = image.sum(axis=0, keepdim=False)  # 48,56,56 合并骨架点维度

# temp_img = torch.zeros(48 , 56 , 56)
# temp_img[1:] = image[:-1]
#
# diff = temp_img[1:].view(47,56*56) - image[1:].view(47,56*56)
# diff = diff.sum(dim=1)  # 差异值
# diff = torch.abs(diff)  # 差异值取绝对值
# print(diff)
# print(torch.sort(diff))  # 打印

imgs = []
for i in range(clip_len):

    img = Tensor2Image(image[i])
    imgs.append(img)
    img.save('./img/img{}.jpg'.format(i),)

imgs[0].save('./img/kp.gif', save_all=True, append_images=imgs, duration=2)
