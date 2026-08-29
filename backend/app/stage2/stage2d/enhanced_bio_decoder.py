"""
enhanced_bio_decoder.py

Stage 2D — Enhanced BIO Sequence Decoder & Subword Reconstructor

Provides strict syntactic validation and reconstruction for scientific NER spans:
  1. Transition matrix constraints (prevents orphaned I- tags)
  2. WordPiece token alignment & subword stitching
  3. Punctuation sanitization (stripping boundary brackets, periods, commas)
  4. Dual-level confidence computation (token-level & entity-level)
  5. Exact character-offset mapping back to source document.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class EnhancedBIODecoder:
    """
    Decodes raw model logits / BIO sequences into validated, clean entity spans.
    """

    def __init__(self, id2label: Dict[int, str]):
        self.id2label = id2label

    def decode_token_predictions(
        self,
        token_ids: List[int],
        probs: List[float],
        offset_mapping: List[Tuple[int, int]],
        sentence: str,
        sentence_offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Decodes a sequence of predicted token label IDs and probabilities into valid entity spans.
        """
        raw_labels = [self.id2label.get(tid, "O") for tid in token_ids]
        n_tokens = len(raw_labels)

        # 1. Enforce strict BIO grammar transitions
        sanitized_labels = self._enforce_bio_grammar(raw_labels)

        spans: List[Dict[str, Any]] = []
        i = 1  # Skip [CLS] token at index 0

        while i < n_tokens - 1:  # Skip [SEP] token at the end
            label = sanitized_labels[i]
            if label == "O" or offset_mapping[i][0] == offset_mapping[i][1]:
                i += 1
                continue

            if label.startswith("B-"):
                entity_type = label[2:]
                span_token_start = i
                span_token_end = i
                token_probs = [float(probs[i])]

                j = i + 1
                while j < n_tokens - 1:
                    next_label = sanitized_labels[j]
                    if next_label == f"I-{entity_type}":
                        span_token_end = j
                        token_probs.append(float(probs[j]))
                        j += 1
                    else:
                        break

                # Extract character slice from sentence
                char_start_local = offset_mapping[span_token_start][0]
                char_end_local = offset_mapping[span_token_end][1]

                raw_span_text = sentence[char_start_local:char_end_local]
                clean_text, start_adj, end_adj = self._sanitize_span_text(raw_span_text)

                if clean_text and len(clean_text) >= 2:
                    global_start = sentence_offset + char_start_local + start_adj
                    global_end = global_start + len(clean_text)
                    mean_conf = sum(token_probs) / len(token_probs)

                    spans.append({
                        "text": clean_text,
                        "entity_type": entity_type,
                        "start_char": global_start,
                        "end_char": global_end,
                        "entity_confidence": round(mean_conf, 4),
                        "token_confidences": [round(p, 4) for p in token_probs],
                        "bio_tag": label,
                    })

                i = j
            else:
                i += 1

        return spans

    def _enforce_bio_grammar(self, labels: List[str]) -> List[str]:
        """
        Enforces valid BIO transition rules: an I-X tag is only permitted
        if immediately preceded by B-X or I-X.
        """
        corrected = list(labels)
        for idx in range(len(corrected)):
            lbl = corrected[idx]
            if lbl.startswith("I-"):
                etype = lbl[2:]
                prev_lbl = corrected[idx - 1] if idx > 0 else "O"
                if prev_lbl not in [f"B-{etype}", f"I-{etype}"]:
                    # Convert orphaned I- into B-
                    corrected[idx] = f"B-{etype}"
        return corrected

    def _sanitize_span_text(self, text: str) -> Tuple[str, int, int]:
        """
        Strips leading/trailing punctuation and boundary whitespace, returning offset adjustments.
        """
        clean = text.strip()
        start_trim = len(text) - len(text.lstrip())

        # Strip boundary punctuation: . , ; : ( ) [ ] " '
        punc_strip = re.sub(r"^[\s\.,;:()\[\]\"\'\-]+|[\s\.,;:()\[\]\"\'\-]+$", "", clean)
        if not punc_strip:
            return "", 0, 0

        start_adj = start_trim + clean.find(punc_strip)
        end_adj = len(text) - (start_adj + len(punc_strip))

        return punc_strip, start_adj, end_adj
