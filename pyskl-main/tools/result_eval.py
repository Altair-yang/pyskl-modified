# Author: Yang jie
# Date  :
import argparse
import os
import numpy as np
import tqdm
import os.path as osp
import time

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv import Config, load
from mmcv.cnn import fuse_conv_bn
from mmcv.engine import multi_gpu_test
from mmcv.fileio.io import file_handlers
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from pyskl.datasets import build_dataloader, build_dataset
from pyskl.models import build_model
from pyskl.utils import cache_checkpoint, mc_off, mc_on, test_port


# 解决KeyError: 'RANK'
os.environ['RANK'] = '0'
# ValueError: Error initializing torch.distributed using env:// rendezvous:
# environment variable MASTER_ADDR expected, but not set
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5678'


def parse_args():
    parser = argparse.ArgumentParser(
        description='pyskl test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('-C', '--checkpoint', help='checkpoint file',
                        default='D:\\1_工作学习\\4_code\\Code\\pyskl-main\\pyskl-main\\tools\\work_dirs\\posec3d\\slowonly_r50_gym\\joint\\slowonly_r50_u48_240e_gym_keypoint-b07a98a0.pth')
    parser.add_argument(
        '--out',
        default=None,
        help='output result file in pkl/yaml/json format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        default=['top_k_accuracy', 'mean_class_accuracy'],
        help='evaluation metrics, which depends on the dataset, e.g.,'
        ' "top_k_accuracy", "mean_class_accuracy" for video dataset')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple workers')
    parser.add_argument(
        '--average-clips',
        choices=['score', 'prob', None],
        default=None,
        help='average type when averaging test clips')
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

args = parse_args()
cfg = Config.fromfile(args.config)

dataset = build_dataset(cfg.data.test, dict(test_mode=True))

# Load eval_config from cfg
eval_cfg = cfg.get('evaluation', {})
result_file = './result.pkl'
outputs = mmcv.load(result_file)

# class_num = 99
# err_list = np.zeros(99, dtype=int)
# pred = np.argmax(outputs, axis=1)
# for i in tqdm.tqdm(range(len(outputs))):
#     label = dataset[i]['label']
#     if label != pred[i] :
#         err_list[label] += 1
# print("acc：", err_list.sum()/8521)
# print(err_list)


if eval_cfg:
    eval_res = dataset.evaluate(outputs, **eval_cfg)
    for name, val in eval_res.items():
        print(f'{name}: {val:.04f}')