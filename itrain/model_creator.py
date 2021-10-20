from typing import Mapping

from transformers import (
    AdapterConfig,
    AdapterType,
    AutoConfig,
    AutoModel,
    AutoModelForMultipleChoice,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelWithHeads,
    AutoTokenizer,
)

from .arguments import ModelArguments
from .datasets import DatasetManager
from .ext.modeling.dependency_parsing import BiaffineParsingHead


HEAD_TO_CLASSIC_MODEL_MAP = {
    "classification": AutoModelForSequenceClassification,
    "tagging": AutoModelForTokenClassification,
    "multiple_choice": AutoModelForMultipleChoice,
    "question_answering": AutoModelForQuestionAnswering,
}


def create_tokenizer(args: ModelArguments, **kwargs):
    return AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        use_fast=args.use_fast_tokenizer,
        **kwargs,
    )


def register_heads(model):
    model.register_custom_head("dependency_parsing", BiaffineParsingHead)


# TODO check full fine-tuning with flex heads
def create_model(args: ModelArguments, manager: DatasetManager, use_classic_model_class=False):
    head_config = manager.get_prediction_head_config()
    num_labels = (
        head_config["num_choices"] if head_config["head_type"] == "multiple_choice" else head_config["num_labels"]
    )
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        **manager.get_model_config_kwargs(),
    )

    if use_classic_model_class:
        head_type = head_config["head_type"]
        if args.drop_model_head:
            base_model = AutoModel.from_pretrained(args.model_name_or_path, config=config)
            model = HEAD_TO_CLASSIC_MODEL_MAP[head_type].from_config(config)
            model.base_model = base_model
        else:
            model = HEAD_TO_CLASSIC_MODEL_MAP[head_type].from_pretrained(args.model_name_or_path, config=config)
    else:
        model = AutoModelWithHeads.from_pretrained(args.model_name_or_path, config=config)
    register_heads(model)

    # load pre-trained adapters
    if args.load_adapters is not None:
        adapter_config = AdapterConfig.load(
            args.adapter_config,
            non_linearity=args.adapter_non_linearity,
            reduction_factor=args.adapter_reduction_factor,
        )
        if isinstance(args.load_adapters, Mapping):
            for name, adapter in args.load_adapters.items():
                model.load_adapter(adapter, config=adapter_config, load_as=name)
        else:
            for adapter in args.load_adapters:
                model.load_adapter(adapter, config=adapter_config)

    # if args.train_adapter and args.train_adapter_fusion is not None:
    #     raise ValueError("train_adapter cannot be set together with train_adapter_fusion.")
    # setup adapters
    if args.train_adapter and args.train_adapter_fusion is None:
        # if adapter was already loaded, train the loaded adapter
        if manager.name not in model.config.adapters.adapters:
            adapter_config = AdapterConfig.load(args.adapter_config)
            model.add_adapter(manager.name, AdapterType.text_task, config=adapter_config)
        model.train_adapter([manager.name])
    elif args.train_adapter_fusion is not None:
        fusion_adapters = args.train_adapter_fusion.split(",")
        model.add_fusion(fusion_adapters)
        model.train_fusion(fusion_adapters)
    # drop the last layer of AF, this tends to achieve better results
    if args.train_adapter_fusion is not None and args.drop_last_fusion_layer:
        del model.base_model.encoder.layer[11].output.adapter_fusion_layer[args.train_adapter_fusion]
        for name in args.load_adapters:
            del model.base_model.encoder.layer[11].output.layer_text_task_adapters[name]
    # unfreeze fused adapters
    if args.train_adapter_fusion is not None and args.train_adapter:
        # HACK: this only works for BERT-based models
        model.base_model.encoder.enable_adapters(args.train_adapter_fusion, True, True)

    if not use_classic_model_class:
        model.add_prediction_head_from_config(
            manager.name,
            head_config,
            overwrite_ok=True,
        )
        model.active_head = manager.name

    return model
