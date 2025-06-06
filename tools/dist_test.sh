# set -x
#!/usr/bin/env bash
CONFIG=$1
CHECKPOINT=$2
GPUS=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
# PORT=${PORT:-29550}
PORT=${PORT:-$((RANDOM%20001+20000))} #再也不用设定PORT了
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

export PYTHONPATH=$PYTHONPATH:./gen-efficientnet-pytorch

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.run \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch \
    ${@:4}
