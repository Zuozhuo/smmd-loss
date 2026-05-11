from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Protocol

import torch


class TokenizerLike(Protocol):
    """
    Minimal tokenizer protocol.

    This module only relies on:
      - vocab_size
      - encode(text, add_special_tokens=False) -> List[int]

    Any HuggingFace-style tokenizer typically satisfies this.
    """

    vocab_size: int

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        ...


@dataclass(frozen=True)
class NumericTokenIndex:
    """
    Indexing of the numeric sub-vocabulary V_num used by SMMD.

    Paper notation:
      - V_num: a numeric sub-vocabulary (subset of V)
      - π: a bijection from V_num to {0, ..., N-1}
      - v_i: parsed numeric value associated with index i

    This class stores a practical instantiation of (V_num, π, {v_i}).

    In our experiments, we often choose V_num as the set of *standalone integer tokens*
    in a bounded range (e.g., {0,...,999}). This corresponds to the "Practical note on K"
    discussion: N is small (10 for digit-tokenizers, 1000 for 0..999 integer tokens),
    so K can be precomputed per tokenizer and reused cheaply.
    """

    values: List[int]                 # [N] numeric values v_i (here: integers)
    token_ids: torch.LongTensor       # [N] token IDs of V_num in π-order
    token_is_numeric: torch.BoolTensor  # [|V|] True iff token_id ∈ V_num
    token_to_index: torch.LongTensor    # [|V|] π(token_id) for tokens in V_num

    @property
    def size(self) -> int:
        return int(self.token_ids.numel())


def build_integer_vnum_from_tokenizer(tokenizer: TokenizerLike, max_value: int = 999) -> NumericTokenIndex:
    """
    Build V_num as standalone integer tokens within [0, max_value].

    This is a restricted instantiation of Algorithm 1 in the paper:
      - Paper (Algorithm 1): parse all tokens in V via deterministic numeric parsing (float casting).
      - Here (engineering choice): probe integers 0..max_value and keep those represented by a single token.

    Rationale:
      - Many LLM tokenizers are digit-level (V_num={0..9}), or contain many single-token integers (0..999).
      - Keeping V_num small makes K and L precomputation trivial and stable during training.
    """
    num_token_map: Dict[int, int] = {}
    for v in range(int(max_value) + 1):
        ids = tokenizer.encode(str(v), add_special_tokens=False)
        if len(ids) == 1:
            num_token_map[v] = ids[0]

    values = sorted(num_token_map.keys())
    if len(values) == 0:
        raise ValueError(
            f"[SMMD] No standalone integer tokens found in 0..{max_value} for this tokenizer."
        )

    token_ids = torch.tensor([num_token_map[v] for v in values], dtype=torch.long)

    vocab_size = int(tokenizer.vocab_size)
    token_is_numeric = torch.zeros(vocab_size, dtype=torch.bool)
    token_to_index = torch.zeros(vocab_size, dtype=torch.long)

    for idx, v in enumerate(values):
        tid = num_token_map[v]
        token_is_numeric[tid] = True
        token_to_index[tid] = idx

    return NumericTokenIndex(
        values=values,
        token_ids=token_ids,
        token_is_numeric=token_is_numeric,
        token_to_index=token_to_index,
    )
