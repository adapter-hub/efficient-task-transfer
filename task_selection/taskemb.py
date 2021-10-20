# coding=utf-8
"""
Compute TaskEmb for classification/ regression/ question-answering tasks (Vu et al., 2020).
Adapted from https://github.com/tuvuumass/task-transferability.
"""

import argparse
import logging
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.distributions.normal import Normal
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import AutoTokenizer, PreTrainedModel
from utils import MODEL_TYPE_MAP

from itrain import DatasetManager, QADatasetManager, RunArguments
from itrain.arguments import DatasetArguments
from itrain.datasets import DATASET_MANAGER_CLASSES
from itrain.datasets.tagging import TaggingDatasetManager
from itrain.runner import set_seed


logger = logging.getLogger(__name__)


@dataclass
class TaskEmbArguments:
    """
    ## Task embeddings
    parser.add_argument("--num_softmax_classifiers", default=1, type=int,
                        help="Number of softmax classifiers on top of Bert's output.")
    parser.add_argument("--pow", type=float, default=2.0,
                        help="Return features to the power pow.")
    parser.add_argument("--feature_type", default='grads', type=str,
                        help="The type of the features selected in ['grads', 'weights']")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--retain_gradients", default=True, type=eval,
                        help="Whether to retain gradients at each layer output of the feature extractor.")
    parser.add_argument("--do_pooling", default=True, type=eval,
                        help="Whether to pool the feature extractor.")
    parser.add_argument("--use_labels", default=True, type=eval,
                        help="Whether to use training labels or sample from the model's predictive distribution \n"
                             "pθ(y|xn), e.g., to compute the theoretical Fisher information.")
    parser.add_argument("--num_trials_for_FIM", type=int, default=100,
                        help="Number of trials to sample from the model's predictive distribution pθ(y|xn).")
    parser.add_argument("--FIM_scale", type=float, default=0.25,
                        help="Standard deviation of the distribution used to compute the theoretical FIM.")
    parser.add_argument("--finetune_classifier", default=False, type=eval,
                        help="Whether to fine-tune the final classifier.")
    parser.add_argument("--finetune_feature_extractor", default=False, type=eval,
                        help="Whether to fine-tune the feature extractor.")
    """

    num_softmax_classifiers: int = 1
    pow: float = 2.0
    feature_type: str = "grads"
    retain_gradients: bool = True
    do_pooling: bool = True
    use_labels: bool = True
    num_trials_for_FIM: int = 100
    FIM_scale: float = 0.25


def compute_Fisher(args, model, input_mask, total_tokens):
    outputs = {}

    base_model = model.base_model
    for name, parameter in base_model.named_parameters():
        if parameter.requires_grad:
            score = parameter.grad if args.feature_type == "grads" else parameter
            if score is not None and name not in outputs:
                score = score ** args.pow
                outputs[name] = score
    # activations
    for key in ["multihead_output", "layer_output"]:
        model_outputs = base_model._get_model_outputs(key=key)
        for i in range(base_model.config.num_hidden_layers):
            name = "encoder.layer.{}.{}".format(i, key)
            model_outputs_i = model_outputs[i].grad if args.feature_type == "grads" else model_outputs[i]

            if model_outputs_i is not None:
                score = torch.einsum(
                    "ijk,ij->ijk", [model_outputs_i, input_mask.float()]  # batch_size x max_seq_length x hidden_size
                )  # batch_size x max_seq_length
                if score is not None and name not in outputs:
                    score = score.sum(0).sum(0)
                    score = score ** args.pow
                    # normalize
                    score = score / total_tokens
                    outputs[name] = score
    # RoBERTa doesn't use a pooler by default
    # # cls output
    # name = "cls_output"
    # score = (
    #     base_model._get_model_outputs(key=name).grad
    #     if args.feature_type == "grads"
    #     else base_model._get_model_outputs(key=name)
    # )  # batch_size x hidden_size

    # if score is not None and name not in outputs:
    #     score = score.sum(0)
    #     score = score ** args.pow
    #     # normalize
    #     score = score / total_tokens
    #     outputs[name] = score

    # task-specific layer
    for name, parameter in model.named_parameters():
        if model.config.model_type not in name:
            score = parameter.grad if args.feature_type == "grads" else parameter
            if score is not None and name not in outputs:
                score = score ** args.pow
                outputs[name] = score

    return outputs


def compute_Fisher_no_labels_cr(args, model, input_mask, logits, num_labels):
    total_tokens = input_mask.float().detach().sum().data

    if num_labels == 1:
        #  We are doing regression
        if args.num_softmax_classifiers > 1:
            raise ValueError("Not implemented.")
        else:
            normal = Normal(logits, scale=torch.tensor([args.FIM_scale]).to(logits.device))
            num_trials = 0
            sampled_logits = None
            while num_trials < args.num_trials_for_FIM:
                sample = normal.sample(sample_shape=(1,)).squeeze(0)
                sample.to(logits.device)
                if ((sample >= 0.0) & (sample <= 5.0)).sum().item() == logits.size(0):
                    num_trials += 1
                    if sampled_logits is None:
                        sampled_logits = sample
                    else:
                        sampled_logits = torch.cat((sampled_logits, sample), dim=1)
            sampled_logits = -((sampled_logits - logits) ** 2) / (2 * (args.FIM_scale ** 2))
            sampled_logits = sampled_logits.sum(0).sum(0) / sampled_logits.numel()

            model.zero_grad()
            # if args.finetune_classifier:
            #     sampled_logits.backward(retain_graph=True)
            # else:
            sampled_logits.backward()
            outputs = compute_Fisher(args, model, input_mask, total_tokens)
    else:
        #  We are doing classification
        if args.num_softmax_classifiers > 1:
            raise ValueError("Not implemented.")
        else:
            softmax_logits = torch.softmax(logits, dim=1)  # batch_size x num_labels
            sampled_indices = torch.multinomial(softmax_logits, args.num_trials_for_FIM, True)
            log_softmax_logits = torch.log(softmax_logits)
            sampled_log_softmax_logits = torch.gather(log_softmax_logits, dim=1, index=sampled_indices)

            sampled_log_softmax_logits = sampled_log_softmax_logits.sum(0).sum(0) / sampled_log_softmax_logits.numel()

            model.zero_grad()
            # if args.finetune_classifier:
            #     sampled_log_softmax_logits.backward(retain_graph=True)
            # else:
            sampled_log_softmax_logits.backward()
            outputs = compute_Fisher(args, model, input_mask, total_tokens)
    return outputs


def compute_Fisher_no_labels_qa(args, model, input_mask, start_logits, end_logits):
    total_tokens = input_mask.float().detach().sum().data

    if args.num_softmax_classifiers > 1:
        raise ValueError("Not implemented.")
    else:
        # start_logits: batch_size x max_seq_length
        # end_logits: batch_size x max_seq_length
        start_softmax_logits = torch.softmax(start_logits, dim=1)
        end_softmax_logits = torch.softmax(end_logits, dim=1)
        sampled_start_indices = torch.multinomial(start_softmax_logits, args.num_trials_for_FIM, True)
        sampled_end_indices = torch.multinomial(end_softmax_logits, args.num_trials_for_FIM, True)
        log_start_softmax_logits = torch.log(start_softmax_logits)
        log_end_softmax_logits = torch.log(end_softmax_logits)

        sampled_log_start_softmax_logits = torch.gather(log_start_softmax_logits, dim=1, index=sampled_start_indices)
        sampled_log_start_softmax_logits = (
            sampled_log_start_softmax_logits.sum(0).sum(0) / sampled_log_start_softmax_logits.numel()
        )

        sampled_log_end_softmax_logits = torch.gather(log_end_softmax_logits, dim=1, index=sampled_end_indices)
        sampled_log_end_softmax_logits = (
            sampled_log_end_softmax_logits.sum(0).sum(0) / sampled_log_end_softmax_logits.numel()
        )

        sampled_log_softmax_logits = (sampled_log_start_softmax_logits + sampled_log_end_softmax_logits) / 2.0

        model.zero_grad()
        # if args.finetune_classifier:
        #     sampled_log_softmax_logits.backward(retain_graph=True)
        # else:
        sampled_log_softmax_logits.backward()
        outputs = compute_Fisher(args, model, input_mask, total_tokens)
    return outputs


def compute_Fisher_with_labels(args, model, input_mask, loss):
    total_tokens = input_mask.float().detach().sum().data

    model.zero_grad()
    loss.backward()

    outputs = compute_Fisher(args, model, input_mask, total_tokens)
    return outputs


def compute_taskemb(
    args: TaskEmbArguments, dataset_manager: DatasetManager, model: PreTrainedModel, run_args: RunArguments
):
    """ Feed task data through the model """

    train_dataloader = DataLoader(
        dataset_manager.train_split,
        batch_size=run_args.batch_size,
        sampler=dataset_manager.train_sampler(),
        collate_fn=dataset_manager.collate_fn,
    )
    model.eval()

    logger.info("***** Compute TaskEmb *****")
    logger.info("Num batches = %d", len(train_dataloader))
    logger.info("Batch size = %d", run_args.batch_size)

    total_num_examples = 0
    model.zero_grad()
    train_iterator = trange(int(run_args.num_train_epochs), desc="Epoch", disable=False)
    global_feature_dict = {}
    for _ in train_iterator:
        num_examples = 0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        for step, inputs in enumerate(epoch_iterator):
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(run_args.device)

            outputs = model(**inputs)
            loss = outputs[0]

            input_mask = inputs["attention_mask"]
            input_mask = input_mask.view(-1, input_mask.size(-1))

            if not args.use_labels:
                if isinstance(dataset_manager, QADatasetManager):
                    start_logits, end_logits = outputs[1], outputs[2]
                    feature_dict = compute_Fisher_no_labels_qa(args, model, input_mask, start_logits, end_logits)
                else:
                    logits = outputs[1]
                    feature_dict = compute_Fisher_no_labels_cr(
                        args, model, input_mask, logits, dataset_manager.num_labels
                    )
            else:
                feature_dict = compute_Fisher_with_labels(args, model, input_mask, loss)

            if len(global_feature_dict) == 0:
                for key in feature_dict:
                    global_feature_dict[key] = feature_dict[key].detach().cpu().numpy()
            else:
                for key in feature_dict:
                    global_feature_dict[key] += feature_dict[key].detach().cpu().numpy()

            model.zero_grad()
            num_examples += inputs["input_ids"].size(0)
        total_num_examples += num_examples

    # Normalize
    for key in global_feature_dict:
        global_feature_dict[key] = global_feature_dict[key] / total_num_examples

    return global_feature_dict


def run_taskemb(args, data_args=None, emb_args=None, run_args=None, seed=42):
    emb_args = emb_args or TaskEmbArguments()

    run_args = run_args or RunArguments(
        batch_size=16,
        num_train_epochs=1,
    )

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger.warning("Device: %s", run_args.device)

    # Dataset
    data_args = data_args or DatasetArguments(dataset_name=args["dataset"], task_name=args["dataset_task"])
    dataset_manager: DatasetManager = DATASET_MANAGER_CLASSES[data_args.dataset_name](data_args)

    # Create output directory if needed
    task_output_dir = os.path.join(args["output_dir"], dataset_manager.name)
    if not os.path.exists(task_output_dir):
        os.makedirs(task_output_dir)
    # skip computation if already exists
    elif len(os.listdir(task_output_dir)) > 0 and not args["overwrite"]:
        logger.info("Output already exists, skipping {0}".format(dataset_manager.name))
        return

    # Set seed
    set_seed(seed)

    # Model config
    model_config_class, model_with_heads_class = MODEL_TYPE_MAP[args["model_type"]]
    model_config = model_config_class.from_pretrained(
        args.get("model_dir", None) or args["model_name"],
        num_softmax_classifiers=emb_args.num_softmax_classifiers,
        retain_gradients=emb_args.retain_gradients,
        do_pooling=emb_args.do_pooling,
    )
    if isinstance(dataset_manager, TaggingDatasetManager):
        add_prefix_space = True
    else:
        add_prefix_space = False
    tokenizer = AutoTokenizer.from_pretrained(
        args["model_name"],
        use_fast=args.get("fast_tokenizer", False),
        add_prefix_space=add_prefix_space,
    )
    dataset_manager.tokenizer = tokenizer

    # Load
    if args.get("adapter_dir", None) and os.path.exists(args["adapter_dir"]):
        model = model_with_heads_class.from_pretrained(args["model_name"], config=model_config)
        model.load_adapter(args["adapter_dir"], load_as=dataset_manager.name)
        model.set_active_adapters([dataset_manager.name])
        # model.train_adapter([dataset_manager.name])
    elif args.get("model_dir", None) and os.path.exists(args["model_dir"]):
        model = model_with_heads_class.from_pretrained(args["model_dir"], config=model_config)
    else:
        raise ValueError("Either adapter_dir or model_dir is required and must be a valid path.")

    dataset_manager.load_and_preprocess()
    model.to(run_args.device)

    feature_dict = compute_taskemb(emb_args, dataset_manager, model, run_args)

    for key in feature_dict:
        np.save(os.path.join(task_output_dir, "{}.npy".format(key)), feature_dict[key])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default=None, type=str, required=True)
    parser.add_argument("--dataset_task", default=None, type=str)
    parser.add_argument("--model_type", default="roberta", type=str)
    parser.add_argument("--model_name", default="roberta-base", type=str)
    parser.add_argument("--output_dir", default="output/taskemb", type=str)
    parser.add_argument("--overwrite", action="store_true", default=False)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--adapter_dir", default=None, type=str)
    group.add_argument("--model_dir", default=None, type=str)

    args = parser.parse_args()

    run_taskemb(vars(args))


if __name__ == "__main__":
    main()
