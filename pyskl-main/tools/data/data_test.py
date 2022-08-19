# Author: Yang jie
# Date  : 22.07.27

import mmcv
import torch

# ann_file = './nturgbd/ntu60_hrnet.pkl'
ann_file = './gym/gym_hrnet.pkl'
data = mmcv.load(ann_file)
ann = data['annotations']

num_1,num_2,num_3 = 0, 0, 0
for i in range(len(ann)):
    if ann[i]['total_frames'] < 48:
        num_1 += 1
    elif 48 <= ann[i]['total_frames'] < 48*2:
        num_2 += 1
    else:
        num_3 += 1
print(num_1, num_2, num_3)
# 19437 7569 1999

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
