# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, Optional

import torch

from ...data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.misc import calculate_tps
from ...extras.packages import is_transformers_version_greater_than
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push, create_ref_model
from .metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor
from .trainer import CustomSeq2SeqTrainer


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = get_logger(__name__)


def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    ref_model = None
    if finetuning_args.use_asft_loss:
        ref_model = create_ref_model(model_args, finetuning_args)

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        neat_packing=data_args.neat_packing,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )

    # Metric utils
    metric_module = {}
    if model_args.use_kt:
        if training_args.predict_with_generate:
            raise NotImplementedError("`predict_with_generate` is not supported in KTransformers SFT yet.")
        elif finetuning_args.compute_accuracy:
            raise NotImplementedError("`compute_accuracy` is not supported in KTransformers SFT yet.")

    if training_args.predict_with_generate:
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    elif finetuning_args.compute_accuracy:
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)

    # Compatible with Transformers v4 and Transformers v5
    if is_transformers_version_greater_than("4.58.0"):
        extra_ids = getattr(tokenizer, "additional_special_tokens_ids", None)
        if not isinstance(extra_ids, list):
            extra_special_tokens = getattr(tokenizer, "_extra_special_tokens", [])
            string_tokens = [str(t) for t in extra_special_tokens]
            extra_ids = tokenizer.convert_tokens_to_ids(string_tokens)
        all_eos_ids = [tokenizer.eos_token_id] + [i for i in extra_ids if i != -1]
        unique_eos_ids = list(dict.fromkeys(all_eos_ids))
        gen_kwargs["eos_token_id"] = unique_eos_ids
    else:
        gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

    # ===== Attach binary classification head if enabled =====
    cls_head_state = {}
    if finetuning_args.use_cls_head:
        from ...model.cls_head import attach_cls_head

        # Find the un-wrapped base model (LoRA wraps it once or twice)
        base_model = model
        while hasattr(base_model, "base_model") and base_model.base_model is not base_model:
            base_model = base_model.base_model
        if hasattr(base_model, "model") and not isinstance(base_model.model, torch.nn.Linear):
            base_model = base_model.model

        # Hidden size: Qwen3.5 stores it under text_config
        hidden_size = getattr(getattr(base_model, "config", None), "text_config", None)
        hidden_size = hidden_size.hidden_size if hidden_size is not None else base_model.config.hidden_size

        head = attach_cls_head(
            base_model,
            hidden_size=hidden_size,
            dropout=finetuning_args.cls_head_dropout,
            dtype=torch.bfloat16,
        )
        device = next(base_model.parameters()).device
        head.to(device)

        # CRITICAL: mirror init_adapter's fp32 upcast for trainable params.
        # cls_head was attached AFTER init_adapter ran, so it stays in bf16
        # by default. Training trainable params in bf16 causes Adam updates
        # (~lr * grad ≈ 1e-5) to underflow → cls_head never actually learns.
        if not finetuning_args.pure_bf16 and not finetuning_args.use_badam:
            for p in head.parameters():
                p.data = p.data.to(torch.float32)
            logger.info_rank0("Upcasted cls_head params to float32 (mirrors init_adapter).")

        logger.info_rank0(
            "Attached BinaryClassificationHead: hidden={}, params={:,}".format(
                hidden_size, sum(p.numel() for p in head.parameters())
            )
        )

        # Resolve token IDs for the two classes
        pos_ids = tokenizer.encode(finetuning_args.cls_positive_text, add_special_tokens=False)
        neg_ids = tokenizer.encode(finetuning_args.cls_negative_text, add_special_tokens=False)
        if not pos_ids or not neg_ids or pos_ids[0] == neg_ids[0]:
            raise ValueError(
                f"Invalid class token mapping: positive='{finetuning_args.cls_positive_text}'->{pos_ids}, "
                f"negative='{finetuning_args.cls_negative_text}'->{neg_ids}. "
                "Each text must tokenize to >= 1 token, and the first tokens must differ."
            )
        cls_head_state["positive_token_id"] = pos_ids[0]
        cls_head_state["negative_token_id"] = neg_ids[0]
        cls_head_state["base_model"] = base_model
        logger.info_rank0(
            "Class token IDs: positive(label=1) {} ('{}')  |  negative(label=0) {} ('{}')".format(
                pos_ids[0], finetuning_args.cls_positive_text,
                neg_ids[0], finetuning_args.cls_negative_text,
            )
        )

    # Initialize our Trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        gen_kwargs=gen_kwargs,
        ref_model=ref_model,
        **dataset_module,
        **tokenizer_module,
        **metric_module,
    )

    # Inject cls_head context into trainer
    if finetuning_args.use_cls_head:
        trainer.cls_head = cls_head_state["base_model"].cls_head
        trainer.positive_token_id = cls_head_state["positive_token_id"]
        trainer.negative_token_id = cls_head_state["negative_token_id"]

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            keys = ["loss"]
            if isinstance(dataset_module.get("eval_dataset"), dict):
                keys += sum(
                    [[f"eval_{key}_loss", f"eval_{key}_accuracy"] for key in dataset_module["eval_dataset"].keys()], []
                )
            else:
                keys += ["eval_loss", "eval_accuracy"]

            plot_loss(training_args.output_dir, keys=keys)

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.warning_rank0_once("Batch generation can be very slow. Consider using `scripts/vllm_infer.py` instead.")
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(dataset_module["eval_dataset"], predict_results, generating_args.skip_special_tokens)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
