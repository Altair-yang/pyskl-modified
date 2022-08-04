# Author: Yang jie
# Date  : 07.29

import torch
import numpy as np
from scipy.spatial import distance

# 用于3D热图进行帧采样，可以最开始不进行帧采样，在生成热图后，再进行帧采样
def sample_merge(x, clip_len):  # full_segment, n_segment -> num_frames = 60, clip_len = 48
    n_batch, c, t, h, w = x.size()  # x.shape = (8, 17, 60, 56, 56)

    x = x.transpose(1, 2)  # n_batch ,c, t, h, w -> n_batch ,t, c, h, w
    x = x.reshape(n_batch, t, -1)  # n_batch ,c, t, h, w -> n_batch ,t, -1   (8, 60, 17*56*56) Todo:关节点叠加后再拉直

    merged_x = slope_selection(x[0], clip_len)  # 对第1个样本进行聚合，n_segment为设定好的聚类数
    merged_x = merged_x[None, :, :]
    for batch in range(1, n_batch):
        tmp = slope_selection(x[batch], clip_len)
        merged_x = torch.cat([merged_x, tmp[None, :, :]], dim=0)  # n_batch ,t, c*h*w
    # print('merged_x size:', merged_x.size())
    merged_nt = n_batch * clip_len
    merged_x = merged_x.reshape(n_batch, clip_len, c, h, w)
    merged_x = merged_x.transpose(1, 2)

    return merged_x

def slope_selection(x, clip_len, print_cluster=False):
    # x.shape = (t,c*h*w)
    # 如果只输出1帧，直接把每一帧的值相加，再做平均
    if clip_len == 1:
        return torch.sum(x[:], 0, keepdim=True) / x.shape[0]

    numpy_x = x.clone()         # [60, 17*56*56]
    numpy_x = numpy_x.detach().cpu().numpy()

    # 滑动累计分布函数放在slope这个列表里
    slope = []
    # sample_num 表示帧数
    sample_num = numpy_x.shape[0]
    # 循环计算每2帧之间的汉明距离，放到slope列表里
    for j in range(sample_num-1):
        hamming = distance.hamming(np.sign(numpy_x[j]), np.sign(numpy_x[j+1]))   #2值化，使用numpy自带的sign函数；汉明距离，使用scipy自带的hamming函数
        slope.append(hamming)
    # print('slope:', slope)
    # partition 这里先对所有的帧之间的汉明距离进行从小到大排序，然后找到最大的n-1个点，partition最后得到的就是梯度最大的几个点
    partition = sorted(range(len(slope)), key=lambda i: slope[i])[-(clip_len-1):]
    partition.sort()

    if print_cluster:
        print('partition:', partition)
    # if partition[0] > 2:
    #     mid = partition[0] // 2
    #     new_x = torch.sum(x[mid-1:mid+1], 0, keepdim=True) / 2
    # else:
    # 计算分割出的第1段：第1帧到第partition[0]+1帧所有帧直接相加，再除以帧数
    new_x = torch.sum(x[0:partition[0]+1], 0, keepdim=True) / (partition[0]+1)
    # print([0, partition[0]+1])
    # 循环计算中间14段最后的输出（均为相加做平均）
    for c in range(clip_len-2):
        # if (partition[c+1] - partition[c]) > 2:
        #     mid = (partition[c+1]+partition[c]) // 2
        #     tmp = torch.sum(x[mid-1:mid + 1], 0, keepdim=True) / 2
        #     new_x = torch.cat((new_x, tmp), dim=0)
        # else:
        tmp = torch.sum(x[partition[c]+1:partition[c+1]+1], 0, keepdim=True) / (partition[c+1]-partition[c])
        new_x = torch.cat((new_x, tmp), dim=0)
    # if x[partition[(n_segment-1)-1]+1:].size(0) > 2:
    #     mid = partition[(n_segment-1)-1] + 1 + x[partition[(n_segment-1)-1]+1:].size(0) // 2
    #     tmp = torch.sum(x[mid-1:mid+1], 0, keepdim=True) / 2
    # else:
    # tmp表示最后1段
    tmp = torch.sum(x[partition[clip_len-2]+1:], 0, keepdim=True) / x[partition[(clip_len-1)-1]+1:].size(0)
    # new_x此时的维度为 [16,9]
    new_x = torch.cat((new_x, tmp), dim=0)
    # print(torch.from_numpy(new_x).float().cuda().size())
    return new_x

def accumulated_selection(x, n_segment, print_cluster=False):
    numpy_x = x.clone()
    numpy_x = numpy_x.detach().cpu().numpy()
    hamming = distance.hamming(np.sign(numpy_x[0]), np.sign(numpy_x[1]))
    accum_dis = [0, hamming]
    sample_num = numpy_x.shape[0]
    for j in range(1, sample_num - 1):
        hamming = distance.hamming(np.sign(numpy_x[j]), np.sign(numpy_x[j + 1]))
        accum_dis.append(hamming + accum_dis[-1])
    dis_index = accum_dis[-1] / n_segment

    if dis_index == 0:
        return x[:n_segment]

    cnt = 1
    clus = []
    clus_set = []
    new_x = None
    for k in range(sample_num):
        if accum_dis[k] <= dis_index * cnt:
            clus.append(k)
        else:
            if new_x is None:
                # if len(clus) > 4:
                #     mid = len(clus) // 4
                #     new_x = torch.sum(x[clus[mid-1]:clus[mid+2] + 1], 0, keepdim=True) / 4
                # else:
                new_x = torch.sum(x[clus[0]:clus[-1] + 1], 0, keepdim=True) / len(clus)
            else:
                # if len(clus) > 4:
                #     mid = len(clus) // 4
                #     tmp = torch.sum(x[clus[mid-1]:clus[mid+2] + 1], 0, keepdim=True) / 4
                # else:
                tmp = torch.sum(x[clus[0]:clus[-1] + 1], 0, keepdim=True) / len(clus)
                new_x = torch.cat((new_x, tmp), dim=0)
            if print_cluster:
                clus_set.append(clus)
            clus = []
            cnt += 1
            clus.append(k)
    while new_x.size(0) < n_segment:
        # tmp = torch.sum(x[clus[0]:clus[-1] + 1], 0, keepdim=True) / len(clus)
        new_x = torch.cat((new_x, tmp), dim=0)

    if print_cluster:
        clus_set.append(list(range(clus_set[-1][-1], 32)))
        return new_x, clus_set

    return new_x

if __name__ == "__main__":
    # 进行了聚合
    print('start')
    num_frames = 60
    clip_len = 48
    x = torch.rand((60,56*56))
    x = accumulated_selection(x,48)
    #merged_x = sample_merge(x,clip_len)
    #print(merged_x.shape)