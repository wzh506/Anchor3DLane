CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-49511}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"} #原来127.0.0.1
# export PYTHONPATH=$PYTHONPATH:./gen-efficientnet-pytorch
export PYTHONPATH=$PYTHONPATH:./gen-efficientnet-pytorch
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.run \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train_dist.py \
    $CONFIG \
    --launcher pytorch ${@:3}
