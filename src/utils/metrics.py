"""
Surprisal Metrics Calculator
============================
Computes Negative Log-Likelihood (NLL) and Perplexity using local Hugging Face models.
"""

import math
from typing import Optional, List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config


class SurprisalCalculator:
    """
    Calculator for computing surprisal (NLL) scores using a local causal language model.

    S_raw(u_t) = -log P_LLM(u_t | C_t, Q_{t-1})
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the surprisal calculator.

        Args:
            model_id: Hugging Face model ID (default: from config)
            device: Device to use ('cuda', 'cpu', or 'auto')
        """
        self.model_id = model_id or config.SURPRISAL_MODEL_ID
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self._model = None
        self._tokenizer = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Lazy load the model and tokenizer."""
        if self._loaded:
            return

        print(f"Loading surprisal model: {self.model_id}...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
        ).to(self.device)
        self._model.eval()

        # Set pad token if not exists
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._loaded = True
        print(f"Surprisal model loaded on {self.device}")

    def calculate_nll(
        self,
        user_input: str,
        context: str = ""
    ) -> float:
        """
        Calculate the Negative Log-Likelihood (NLL) of user input given context.

        S_raw(u_t) = -log P(u_t | C_t)

        Args:
            user_input: The user's input text u_t
            context: The retrieved context C_t (can be empty)

        Returns:
            NLL score (higher = more surprising)
        """
        self._ensure_loaded()

        # Construct the full prompt
        if context:
            full_text = f"Context: {context}\nUser: {user_input}"
            prefix = f"Context: {context}\nUser: "
        else:
            full_text = f"User: {user_input}"
            prefix = "User: "

        with torch.no_grad():
            # Tokenize full text
            full_encoding = self._tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(self.device)

            # Tokenize prefix to find where user input starts
            prefix_encoding = self._tokenizer(
                prefix,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            prefix_len = prefix_encoding.input_ids.shape[1]

            # Get model outputs
            outputs = self._model(
                input_ids=full_encoding.input_ids,
                attention_mask=full_encoding.attention_mask
            )

            # Get logits and compute NLL only for user input tokens
            logits = outputs.logits[0]  # (seq_len, vocab_size)

            # Shift logits and labels for causal LM loss
            shift_logits = logits[prefix_len - 1:-1]  # Predict user input tokens
            shift_labels = full_encoding.input_ids[0, prefix_len:]  # User input token IDs

            if len(shift_labels) == 0:
                return 0.0

            # Compute cross-entropy loss (NLL)
            loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
            nll = loss_fn(shift_logits, shift_labels)

            return float(nll.item())

    def calculate_perplexity(
        self,
        user_input: str,
        context: str = ""
    ) -> float:
        """
        Calculate perplexity (exponential of NLL).

        PPL = exp(NLL)
        """
        nll = self.calculate_nll(user_input, context)
        return math.exp(nll)

    def batch_calculate_nll(
        self,
        inputs: List[str],
        contexts: Optional[List[str]] = None
    ) -> List[float]:
        """
        Calculate NLL for multiple inputs.

        Args:
            inputs: List of user inputs
            contexts: Optional list of contexts (same length as inputs)

        Returns:
            List of NLL scores
        """
        if contexts is None:
            contexts = [""] * len(inputs)

        results = []
        for user_input, context in zip(inputs, contexts):
            nll = self.calculate_nll(user_input, context)
            results.append(nll)

        return results


class MockSurprisalCalculator:
    """Mock calculator for testing without loading models."""

    def calculate_nll(self, user_input: str, context: str = "") -> float:
        """Return a mock NLL based on simple heuristics."""
        # Simple heuristic: longer inputs and less context overlap = higher surprisal
        input_len = len(user_input.split())
        context_len = len(context.split()) if context else 0

        # Check for keyword overlap
        input_words = set(user_input.lower().split())
        context_words = set(context.lower().split()) if context else set()
        overlap = len(input_words & context_words)

        # Base surprisal + adjustments
        base = 2.0
        length_factor = 0.1 * input_len
        context_factor = -0.2 * overlap if context_len > 0 else 0.5

        return max(0.1, base + length_factor + context_factor)

    def calculate_perplexity(self, user_input: str, context: str = "") -> float:
        return math.exp(self.calculate_nll(user_input, context))

    def batch_calculate_nll(
        self,
        inputs: List[str],
        contexts: Optional[List[str]] = None
    ) -> List[float]:
        if contexts is None:
            contexts = [""] * len(inputs)
        return [self.calculate_nll(i, c) for i, c in zip(inputs, contexts)]


# Singleton instance
_calculator: Optional[SurprisalCalculator] = None


def get_surprisal_calculator(use_mock: bool = False) -> SurprisalCalculator:
    """Get the surprisal calculator instance."""
    global _calculator

    if use_mock or config.USE_MOCK_LLM:
        return MockSurprisalCalculator()

    if _calculator is None:
        _calculator = SurprisalCalculator()
    return _calculator
