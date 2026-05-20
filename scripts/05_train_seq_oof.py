#!/usr/bin/env python3
"""Stage 05 (OPTIONAL / stretch goal): frame-level sequence branch OOF.

NOT IMPLEMENTED in v0.1. The pooled-table + IRT-joint stack is the priority.
Once stage 06 OOF QWK stabilizes, this script will produce an extra OOF view
that aggregates frame-level SSL embeddings via attention pooling + a small
session-level Transformer, feeding the same IRT joint head.

To run this you'll need:
  - load_sequence(...) from adodas/data/feature_io.py (already available)
  - a SequenceDataset that emits (B, T, D) padded tensors per session
  - a small (1-2 layer) Transformer over frame tokens → 4 session tokens
    → 1 participant token → IRTHead

The pre-extracted SSL features at 25 Hz mean a 30-sec session is ~750 frames;
fully attention is fine on a single GPU.
"""
from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--paths-config", default="configs/paths.yaml")
    args = p.parse_args()
    raise NotImplementedError(
        "Sequence branch is a stretch goal. Land stage 06 meta-blend first; "
        "if OOF QWK plateaus and there is time before 2026-06-01, implement "
        "this. See /Users/ericzhang/.claude/plans/woolly-sprouting-dragonfly.md "
        "for the design notes."
    )


if __name__ == "__main__":
    main()
