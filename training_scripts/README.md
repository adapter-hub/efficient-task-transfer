# Intermediate Task Transfer

## Setup

As described in the main README.

## Sequential Adapter Fine-Tuning

Transfer all source tasks in `task_map.json` to the given target task:

```bash
export RUN_CONFIG_DIR="/path/to/run_configs"
export DEFAULT_TASK_MAP="/path/to/task_map.json"

export JOB_ID=1
export TARGET_TASK="rte"
export TARGET_TRAIN_SIZE=-1  # -1 for full dataset

python run_transfer.py \
    --id $JOB_ID \
    --train_size $TARGET_TRAIN_SIZE \
    --learning_rate 1e-4 \
    --restarts 5 \
    $TARGET_TASK
```

## Adapter Fusion

Fuse the pre-trained adapter given in `SOURCE_TASKS` for training on the given target task:

```bash
export RUN_CONFIG_DIR="/path/to/run_configs"
export DEFAULT_TASK_MAP="/path/to/task_map.json"

export JOB_ID=1
export SOURCE_TASKS="mnli,qnli,snli"
export TARGET_TASK="rte"
export TARGET_TRAIN_SIZE=-1  # -1 for full dataset

python run_fusion.py \
    --id $JOB_ID \
    --train_size $TARGET_TRAIN_SIZE \
    --learning_rate 5e-5 \
    --restarts 5 \
    --source_tasks $SOURCE_TASKS \
    $TARGET_TASK
```
