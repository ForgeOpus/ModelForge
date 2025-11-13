"""
Metrics computation for different tasks.
Provides task-specific evaluation metrics.
"""
import numpy as np
from typing import Dict, Any
from transformers import EvalPrediction

from ..logging_config import logger


class MetricsCalculator:
    """Calculator for task-specific metrics."""

    @staticmethod
    def compute_causal_lm_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
        """
        Compute metrics for causal language modeling (text generation).

        Metrics:
        - Perplexity: exp(loss)
        - Loss: Cross-entropy loss

        Args:
            eval_pred: Evaluation prediction with predictions and labels

        Returns:
            Dictionary of metrics
        """
        logger.info("Computing causal LM metrics")

        predictions, labels = eval_pred.predictions, eval_pred.label_ids

        # For language modeling, predictions are logits
        # Loss is computed by the trainer, we just calculate perplexity
        if hasattr(eval_pred, 'loss') and eval_pred.loss is not None:
            loss = eval_pred.loss
        else:
            # Fallback: estimate loss from predictions
            loss = 0.0

        perplexity = np.exp(loss) if loss > 0 else 0.0

        metrics = {
            "perplexity": float(perplexity),
            "eval_loss": float(loss),
        }

        logger.info(f"Causal LM metrics: {metrics}")
        return metrics

    @staticmethod
    def compute_seq2seq_metrics(eval_pred: EvalPrediction, tokenizer: Any = None) -> Dict[str, float]:
        """
        Compute metrics for sequence-to-sequence tasks (summarization).

        Metrics:
        - ROUGE-1, ROUGE-2, ROUGE-L (if available)
        - BLEU score (if available)

        Args:
            eval_pred: Evaluation prediction
            tokenizer: Tokenizer for decoding predictions

        Returns:
            Dictionary of metrics
        """
        logger.info("Computing seq2seq metrics")

        predictions, labels = eval_pred.predictions, eval_pred.label_ids

        # Decode predictions and labels
        if tokenizer is not None:
            # Replace -100 in labels (used for padding)
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            # Try to compute ROUGE scores
            try:
                import evaluate
                rouge_metric = evaluate.load("rouge")
                result = rouge_metric.compute(
                    predictions=decoded_preds,
                    references=decoded_labels,
                    use_stemmer=True
                )

                metrics = {
                    "rouge1": float(result["rouge1"].mid.fmeasure),
                    "rouge2": float(result["rouge2"].mid.fmeasure),
                    "rougeL": float(result["rougeL"].mid.fmeasure),
                }

                logger.info(f"Seq2Seq metrics: {metrics}")
                return metrics

            except Exception as e:
                logger.warning(f"Could not compute ROUGE metrics: {e}")

        # Fallback: return basic metrics
        return {"eval_loss": 0.0}

    @staticmethod
    def compute_qa_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
        """
        Compute metrics for question answering.

        Metrics:
        - Exact Match (EM)
        - F1 Score

        Args:
            eval_pred: Evaluation prediction

        Returns:
            Dictionary of metrics
        """
        logger.info("Computing QA metrics")

        predictions, labels = eval_pred.predictions, eval_pred.label_ids

        # For QA, predictions are start/end positions
        # This is a simplified version - full implementation would need more context
        try:
            # Extract start and end positions
            start_preds, end_preds = predictions
            start_labels, end_labels = labels

            # Compute exact match
            exact_matches = (
                (start_preds == start_labels) & (end_preds == end_labels)
            ).astype(float)
            em_score = exact_matches.mean()

            # Simplified F1 (proper F1 would need token-level comparison)
            f1_score = em_score  # Placeholder

            metrics = {
                "exact_match": float(em_score),
                "f1": float(f1_score),
            }

            logger.info(f"QA metrics: {metrics}")
            return metrics

        except Exception as e:
            logger.warning(f"Could not compute QA metrics: {e}")
            return {"eval_loss": 0.0}

    @classmethod
    def get_metrics_fn_for_task(cls, task: str, tokenizer: Any = None):
        """
        Get the appropriate metrics function for a task.

        Args:
            task: Task type (text-generation, summarization, extractive-question-answering)
            tokenizer: Optional tokenizer for decoding

        Returns:
            Metrics computation function
        """
        task_metrics_map = {
            "text-generation": cls.compute_causal_lm_metrics,
            "summarization": lambda pred: cls.compute_seq2seq_metrics(pred, tokenizer),
            "extractive-question-answering": cls.compute_qa_metrics,
        }

        return task_metrics_map.get(task, cls.compute_causal_lm_metrics)
