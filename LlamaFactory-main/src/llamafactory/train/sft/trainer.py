# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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

import json
import os
from functools import partial
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
from torch.nn import functional as F
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...model.cls_head import focal_loss
from ..callbacks import SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, patch_accelerator_for_fp8, verify_fp8_status
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, ModelArguments, TrainingArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        ref_model: Optional["torch.nn.Module"] = None,
        **kwargs,
    ) -> None:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
        # Configure FP8 environment if enabled
        training_args: TrainingArguments = kwargs.get("args")
        if training_args.fp8:
            configure_fp8_environment(training_args)
            if getattr(training_args, "fp8_backend", "auto") == "te":
                patch_accelerator_for_fp8()

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        self.ref_model = ref_model

        if ref_model is not None:
            from trl.models.utils import prepare_deepspeed, prepare_fsdp

            if getattr(self.accelerator.state, "deepspeed_plugin", None) is not None:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif getattr(self.accelerator.state, "fsdp_plugin", None) is not None:
                if self.accelerator.is_fsdp2:
                    from accelerate.utils.fsdp_utils import fsdp2_prepare_model

                    self.ref_model = fsdp2_prepare_model(self.accelerator, self.ref_model)
                else:
                    self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

        elif finetuning_args.use_eaft_loss:
            from ..trainer_utils import eaft_loss_func

            self.compute_loss_func = lambda outputs, labels, num_items_in_batch=None: eaft_loss_func(
                outputs, labels, num_items_in_batch, finetuning_args.eaft_alpha
            )
        elif finetuning_args.use_asft_loss:
            from ..trainer_utils import asft_loss_func

            self.compute_loss_func = partial(
                asft_loss_func,
                asft_alpha=finetuning_args.asft_alpha,
            )

        if training_args.fp8 and hasattr(self, "accelerator"):  # verify FP8 status after trainer initialization
            verify_fp8_status(self.accelerator, training_args)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        # Branch 1: Binary classification head (skip CE over full vocab)
        if getattr(self.finetuning_args, "use_cls_head", False) and hasattr(self, "cls_head"):
            return self._compute_loss_cls_head(model, inputs, *args, **kwargs)

        # Branch 2: Existing ASFT path
        if self.finetuning_args.use_asft_loss:
            with torch.no_grad():
                ref_outputs = self.ref_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                )
                ref_logits = ref_outputs.logits
            outputs = model(**inputs)
            return self.compute_loss_func(outputs, inputs["labels"], ref_logits)

        # Branch 3: Default LM SFT
        return super().compute_loss(model, inputs, *args, **kwargs)

    def _compute_loss_cls_head(self, model, inputs, *args, **kwargs):
        """Binary classification on hidden state at answer position.

        Skips the full-vocab CE entirely to save memory.
        """
        return_outputs = kwargs.get("return_outputs", False)

        labels = inputs.pop("labels")  # [B, T]
        # Don't request LM logits unless we need LM regularization
        need_logits = self.finetuning_args.cls_lm_loss_weight > 0
        forward_kwargs = dict(inputs)
        forward_kwargs["output_hidden_states"] = True
        forward_kwargs["return_dict"] = True
        # Tell model to skip its internal loss (we don't pass labels)
        outputs = model(**forward_kwargs)

        # Find answer position: first token where label != IGNORE_INDEX
        is_answer = labels != IGNORE_INDEX                 # [B, T]
        has_answer = is_answer.any(dim=-1)                 # [B]
        answer_pos = is_answer.int().argmax(dim=-1)        # [B] - first True
        # If a row has no answer, fallback to seq_len-1 (will be masked out)
        seq_len = labels.size(1)
        answer_pos = torch.where(
            has_answer, answer_pos, torch.full_like(answer_pos, seq_len - 1)
        )

        # Hidden state right BEFORE first answer token
        # (in causal LM, h[i-1] is what produces logits for token at position i)
        hidden_pos = (answer_pos - 1).clamp(min=0)         # [B]

        last_hidden = outputs.hidden_states[-1]            # [B, T, D]
        B = last_hidden.size(0)
        idx = torch.arange(B, device=last_hidden.device)
        h_for_cls = last_hidden[idx, hidden_pos]           # [B, D]

        # Forward classification head (cast to head's dtype)
        head_dtype = next(self.cls_head.parameters()).dtype
        cls_logits = self.cls_head(h_for_cls.to(head_dtype))  # [B, 2]

        # Extract binary labels from token at answer_pos.
        # Map: positive_token_id -> 1, negative_token_id -> 0; anything else falls back to 0.
        first_answer_token = labels.gather(1, answer_pos.unsqueeze(1)).squeeze(1)  # [B]
        is_positive = first_answer_token == self.positive_token_id
        is_negative = first_answer_token == self.negative_token_id
        cls_labels = is_positive.long()                                            # [B]
        # Rows whose first answer token is neither class: drop from loss
        valid_class = is_positive | is_negative                                    # [B]
        has_answer = has_answer & valid_class

        # Drop rows without valid answer
        if (~has_answer).any():
            valid = has_answer
            if valid.sum() == 0:
                # Edge case: empty batch - return zero loss requiring grad
                zero = (cls_logits.sum() * 0.0)
                if return_outputs:
                    outputs.loss = zero
                    return zero, outputs
                return zero
            cls_logits = cls_logits[valid]
            cls_labels = cls_labels[valid]

        # Focal loss
        loss = focal_loss(
            cls_logits,
            cls_labels,
            alpha=self.finetuning_args.cls_focal_alpha,
            gamma=self.finetuning_args.cls_focal_gamma,
        )

        # Optional: blend in LM CE as light regularization
        lm_w = self.finetuning_args.cls_lm_loss_weight
        if lm_w > 0 and need_logits and getattr(outputs, "logits", None) is not None:
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)).float(),
                shift_labels.view(-1),
                ignore_index=IGNORE_INDEX,
            )
            loss = loss + lm_w * lm_loss

        if return_outputs:
            outputs.cls_logits = cls_logits.detach()
            outputs.cls_labels = cls_labels.detach()
            outputs.loss = loss
            return loss, outputs
        return loss

    @override
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        super()._save(output_dir=output_dir, state_dict=state_dict)
        # Save cls_head separately (PEFT only saves LoRA + modules_to_save)
        if getattr(self.finetuning_args, "use_cls_head", False) and hasattr(self, "cls_head"):
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            cls_head_path = os.path.join(output_dir, "cls_head.bin")
            head = self.cls_head
            torch.save({k: v.detach().cpu() for k, v in head.state_dict().items()}, cls_head_path)
            meta = {
                "task": "binary_classification",
                "label_map": {
                    "0": self.finetuning_args.cls_negative_text,
                    "1": self.finetuning_args.cls_positive_text,
                },
                "positive_token_id": int(self.positive_token_id),
                "negative_token_id": int(self.negative_token_id),
                "positive_text": self.finetuning_args.cls_positive_text,
                "negative_text": self.finetuning_args.cls_negative_text,
                "hidden_size": head.dense.in_features,
                "dropout": head.dropout.p if isinstance(head.dropout, torch.nn.Dropout) else 0.0,
            }
            with open(os.path.join(output_dir, "cls_head_meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            logger.info_rank0(f"Saved cls_head to {cls_head_path}")

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        input_ids_column = dataset["input_ids"]
        try:
            input_ids_list = input_ids_column.to_pylist()
        except AttributeError:
            input_ids_list = list(input_ids_column)

        decoded_inputs = self.processing_class.batch_decode(input_ids_list, skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
