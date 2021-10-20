# Intermediate Task Selection

## Setup

As described in the main README.

## Feature embeddings (TextEmb, TaskEmb, SEmb)

Compute SEmb embeddings for all source tasks in `task_map.json`:

```bash
export RUN_CONFIG_DIR="/path/to/run_configs"
export DEFAULT_TASK_MAP="/path/to/task_map.json"

export MODEL_NAME="sentence-transformers/roberta-base-nli-stsb-mean-tokens"
export OUTPUT_DIR="output_semb"

python run_embeddings.py \
    --semb \
    --model_name $MODEL_NAME \
    --source_tasks \
    -o $OUTPUT_DIR
```

Compute TextEmb (TaskEmb) embeddings for all source tasks in `task_map.json`:

```bash
export RUN_CONFIG_DIR="/path/to/run_configs"
export DEFAULT_TASK_MAP="/path/to/task_map.json"

export MODEL_TYPE="roberta"
export MODEL_NAME="roberta-base"
export OUTPUT_DIR="output_textemb_roberta"

python run_embeddings.py \
    --textemb \
    --model_type $MODEL_TYPE \
    --model_name $MODEL_NAME \
    --source_tasks \
    -o $OUTPUT_DIR
```

Replace `--source_tasks` with `--target_tasks` for target tasks or give custom tasks using `--tasks task1,task2,...`.

## Few-shot fine-tuning (FSFT, FS-TaskEmb)

Perform and evaluate few-shot fine-tuning from all source tasks in `task_map.json` towards the specified target task(s):

```bash
export RUN_CONFIG_DIR="/path/to/run_configs"
export DEFAULT_TASK_MAP="/path/to/task_map.json"

export MODEL_TYPE="roberta"
export MODEL_NAME="roberta-base"
export TARGET_TASKS="rte"
export OUTPUT_DIR="output_fs_roberta"

# Checkpoint fine-tuning steps
python run_few_shot.py checkpoint \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL_NAME \
    --target_tasks $TARGET_TASKS \
    -o $OUTPUT_DIR

# Evaluate performance & compute TaskEmb
python run_few_shot.py probe \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL_NAME \
    --target_tasks $TARGET_TASKS \
    -o $OUTPUT_DIR
```

## Encoder features (for proxy models)

Compute encoder features for the given target tasks when transferring each pre-trained adapter in `task_map.json`:

```bash
export RUN_CONFIG_DIR="/path/to/run_configs"
export DEFAULT_TASK_MAP="/path/to/task_map.json"

export MODEL_TYPE="roberta"
export MODEL_NAME="roberta-base"
export TARGET_TASKS="boolq,rotten_tomatoes,rte,stsb"
export OUTPUT_DIR="output_fs_roberta"

python run_model_features.py \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL_NAME \
    --target_tasks $TARGET_TASKS \
    -o $OUTPUT_DIR
```
