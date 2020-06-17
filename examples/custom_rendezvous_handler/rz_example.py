import os
import sys
import argparse
import torchelastic
print(torchelastic.__version__)

"""
>>> python -m torchelastic.distributed.launch \
        --nnodes=1 \
        --nproc_per_node=1 \
        --rdzv_id=003 \
        --rdzv_backend=etcd \
        --rdzv_endpoint=worker-0:2379 \
        main.py \
        --arch resnet18 \
        --epochs 20 \
        --batch-size 32 \
        --dist-backend gloo
        /mnt/c/Users/t-viabey/Documents/data/imagenet/tiny-imagenet-200
"""

parser = argparse.ArgumentParser(description="PyTorch Elastic Custom Handler")
parser.add_argument(
    "-d",
    "--data",
    default=None,
    type=str,
    metavar="DATA",
    help="path to data",
)

args = parser.parse_args()

if not os.path.exists(args.data):
    raise ValueError("Data Path {} doesn not exist".format(args.data))

print("Data Dir : {}".format(args.data))


