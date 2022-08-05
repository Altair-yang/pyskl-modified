#!/usr/bin/env bash

# $1 代表传入的第1个参数
CONFIG=$1
CHECKPOINT=$2
GPUS=$3

# ${file:-my.file.txt} ：假如$file没有设定或为空值，则使用my.file.txt作传回值。(非空值时不作处理)
# :-是一种赋值方式
PORT=${PORT:-29500}

# $()和``的作用一致，都是用来做命令替换用，一般用于将命令返回的结果传递给变量
# dirname $0 获取当前Shell程序的路径
MKL_SERVICE_FORCE_INTEL=1 PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

# Arguments starting from the forth one are captured by ${@:4}
# ${@:4} 表示第4个及以后的参数
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG -C $CHECKPOINT --launcher pytorch ${@:4}
