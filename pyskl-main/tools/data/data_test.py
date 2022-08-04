# Author: Yang jie
# Date  : 22.07.27

import mmcv
import torch

test = torch.load('latest.pth')

ann_file = './nturgbd/ntu60_hrnet.pkl'
# ann_file = './gym/gym_hrnet.pkl'
# ann_file = 'D:\\1_工作学习\\4_code\\Code\\pyskl-main\\pyskl-main\\tools\\work_dirs\\posec3d\\slowonly_r50_gym\\joint\\result.pkl'
data = mmcv.load(ann_file)
ann = data['annotations']
ann_0 = ann[0]
kp = ann_0['keypoint']
kp_t1 = kp[0,0,:,:] # t1时刻的坐标，(17,2)

x,y=kp_t1[:,0] ,kp_t1[:,1]

from matplotlib import pyplot as plt

fig,ax=plt.subplots()
ax.scatter(x,y,c='r')

for i in range(17):
    ax.annotate(i,(x[i],y[i]))
plt.show()

pass
