#!/usr/bin/env python3
"""
N-Gram Anomaly Detector
=======================
Detects anomalous strings by comparing their n-grams against a Bloom filter
built from a training corpus. Strings with many unseen n-grams are flagged.

Usage:
    python ngram_anomaly_detector.py --help
    python ngram_anomaly_detector.py train  --input train.txt --model model.pkl
    python ngram_anomaly_detector.py detect --input test.txt  --model model.pkl
    python ngram_anomaly_detector.py demo
"""

import argparse
import hashlib
import json
import math
import os
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# Bloom Filter
# ---------------------------------------------------------------------------

class BloomFilter:
    """
    A space-efficient probabilistic set membership structure.

    Uses k independent hash functions over a bit array of size m.
    False positives are possible; false negatives are not.

    Parameters
    ----------
    capacity : int
        Expected number of elements to insert.
    error_rate : float
        Desired false-positive probability (e.g. 0.01 for 1 %).
    """

    def __init__(self, capacity: int, error_rate: float = 0.01):
        if not (0 < error_rate < 1):
            raise ValueError("error_rate must be between 0 and 1 exclusive.")
        if capacity < 1:
            raise ValueError("capacity must be at least 1.")

        self.capacity = capacity
        self.error_rate = error_rate

        # Optimal bit-array size and number of hash functions
        self.m = self._optimal_m(capacity, error_rate)
        self.k = self._optimal_k(self.m, capacity)

        self._bits = bytearray(math.ceil(self.m / 8))
        self._count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, item: str) -> None:
        """Insert *item* into the filter."""
        for bit in self._hash_positions(item):
            self._bits[bit >> 3] |= 1 << (bit & 7)
        self._count += 1

    def __contains__(self, item: str) -> bool:
        """Return True if *item* is *possibly* in the filter."""
        return all(
            self._bits[bit >> 3] & (1 << (bit & 7))
            for bit in self._hash_positions(item)
        )

    @property
    def count(self) -> int:
        """Approximate number of items inserted."""
        return self._count

    @property
    def memory_bytes(self) -> int:
        """Memory used by the bit array (bytes)."""
        return len(self._bits)

    def estimated_false_positive_rate(self) -> float:
        """Current estimated FP rate given items inserted so far."""
        return (1 - math.exp(-self.k * self._count / self.m)) ** self.k

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _hash_positions(self, item: str) -> list[int]:
        """Return k bit positions for *item*."""
        h1 = int(hashlib.md5(item.encode(), usedforsecurity=False).hexdigest(), 16)
        h2 = int(hashlib.sha1(item.encode(), usedforsecurity=False).hexdigest(), 16)
        return [(h1 + i * h2) % self.m for i in range(self.k)]

    @staticmethod
    def _optimal_m(n: int, p: float) -> int:
        """Optimal bit-array size for n elements and false-positive rate p."""
        return max(1, int(-n * math.log(p) / (math.log(2) ** 2)))

    @staticmethod
    def _optimal_k(m: int, n: int) -> int:
        """Optimal number of hash functions."""
        return max(1, round((m / n) * math.log(2)))

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "error_rate": self.error_rate,
            "m": self.m,
            "k": self.k,
            "count": self._count,
            "bits": self._bits.hex(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BloomFilter":
        obj = cls.__new__(cls)
        obj.capacity = d["capacity"]
        obj.error_rate = d["error_rate"]
        obj.m = d["m"]
        obj.k = d["k"]
        obj._count = d["count"]
        obj._bits = bytearray.fromhex(d["bits"])
        return obj


# ---------------------------------------------------------------------------
# N-gram helpers
# ---------------------------------------------------------------------------

def extract_ngrams(text: str, n: int) -> Iterator[str]:
    """Yield all character n-grams of length *n* from *text*."""
    for i in range(len(text) - n + 1):
        yield text[i : i + n]


def ngram_set(text: str, n: int) -> set[str]:
    return set(extract_ngrams(text, n))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class NGramModel:
    """Trained n-gram anomaly detector."""

    n: int
    bloom: BloomFilter
    threshold: float          # anomaly score above which a string is flagged
    total_ngrams_trained: int = 0
    training_strings: int = 0
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @classmethod
    def train(
        cls,
        texts: list[str],
        n: int = 3,
        bloom_capacity: int | None = None,
        error_rate: float = 0.01,
        threshold: float = 0.3,
    ) -> "NGramModel":
        """
        Build a model from *texts*.

        Parameters
        ----------
        texts        : Training corpus (list of strings).
        n            : N-gram length.
        bloom_capacity: Bloom filter capacity. Defaults to unique n-gram estimate.
        error_rate   : Bloom filter false-positive rate.
        threshold    : Fraction of unseen n-grams above which a string is anomalous.
        """
        print(f"[train] Extracting {n}-grams from {len(texts):,} strings …")
        t0 = time.perf_counter()

        all_ngrams: set[str] = set()
        for text in texts:
            all_ngrams.update(ngram_set(text, n))

        unique = len(all_ngrams)
        capacity = bloom_capacity or max(unique, 1000)
        bloom = BloomFilter(capacity, error_rate)

        for ng in all_ngrams:
            bloom.add(ng)

        elapsed = time.perf_counter() - t0
        print(
            f"[train] Done in {elapsed:.2f}s | "
            f"unique n-grams: {unique:,} | "
            f"Bloom size: {bloom.memory_bytes / 1024:.1f} KB | "
            f"estimated FP rate: {bloom.estimated_false_positive_rate():.4%}"
        )

        return cls(
            n=n,
            bloom=bloom,
            threshold=threshold,
            total_ngrams_trained=unique,
            training_strings=len(texts),
            metadata={
                "training_time_s": round(elapsed, 4),
                "bloom_capacity": capacity,
                "error_rate": error_rate,
            },
        )

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def score(self, text: str) -> float:
        """
        Return the anomaly score for *text* in [0, 1].

        Score = fraction of the string's n-grams absent from the Bloom filter.
        A score of 0 means all n-grams were seen during training (normal).
        A score of 1 means none were seen (maximally anomalous).
        """
        grams = list(extract_ngrams(text, self.n))
        if not grams:
            # String shorter than n — cannot score; treat as suspicious
            return 1.0
        unseen = sum(1 for g in grams if g not in self.bloom)
        return unseen / len(grams)

    def is_anomalous(self, text: str) -> tuple[bool, float]:
        """Return (anomalous, score) for *text*."""
        s = self.score(text)
        return s >= self.threshold, s

    def analyze(self, text: str) -> dict:
        """Return a detailed breakdown for *text*."""
        grams = list(extract_ngrams(text, self.n))
        if not grams:
            return {
                "text": text,
                "score": 1.0,
                "anomalous": True,
                "total_ngrams": 0,
                "unseen_ngrams": 0,
                "unseen_examples": [],
                "reason": "String shorter than n — no n-grams extracted.",
            }

        unseen = [g for g in grams if g not in self.bloom]
        score = len(unseen) / len(grams)
        # Deduplicate but keep order for display
        seen_set: set[str] = set()
        unique_unseen: list[str] = []
        for g in unseen:
            if g not in seen_set:
                seen_set.add(g)
                unique_unseen.append(g)

        return {
            "text": text,
            "score": round(score, 4),
            "anomalous": score >= self.threshold,
            "total_ngrams": len(grams),
            "unseen_ngrams": len(unseen),
            "unseen_examples": unique_unseen[:10],
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        data = {
            "n": self.n,
            "threshold": self.threshold,
            "total_ngrams_trained": self.total_ngrams_trained,
            "training_strings": self.training_strings,
            "metadata": self.metadata,
            "bloom": self.bloom.to_dict(),
        }
        with open(path, "wb") as fh:
            pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[save] Model written to {path} ({path.stat().st_size / 1024:.1f} KB)")

    @classmethod
    def load(cls, path: str | Path) -> "NGramModel":
        path = Path(path)
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        bloom = BloomFilter.from_dict(data["bloom"])
        return cls(
            n=data["n"],
            bloom=bloom,
            threshold=data["threshold"],
            total_ngrams_trained=data["total_ngrams_trained"],
            training_strings=data["training_strings"],
            metadata=data.get("metadata", {}),
        )

    def summary(self) -> str:
        lines = [
            "=" * 55,
            " N-Gram Anomaly Detector — Model Summary",
            "=" * 55,
            f"  N-gram length       : {self.n}",
            f"  Anomaly threshold   : {self.threshold:.0%} unseen",
            f"  Training strings    : {self.training_strings:,}",
            f"  Unique n-grams      : {self.total_ngrams_trained:,}",
            f"  Bloom filter size   : {self.bloom.memory_bytes / 1024:.1f} KB",
            f"  Bloom k (hashes)    : {self.bloom.k}",
            f"  Bloom FP rate (est.): {self.bloom.estimated_false_positive_rate():.4%}",
            "=" * 55,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Batch detection helpers
# ---------------------------------------------------------------------------

def detect_from_file(
    model: NGramModel,
    path: str | Path,
    output_path: str | Path | None = None,
    verbose: bool = False,
) -> list[dict]:
    """
    Score every non-empty line in *path* and return results.
    Optionally write a JSON report to *output_path*.
    """
    path = Path(path)
    results = []
    anomalies = 0

    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip()]

    print(f"[detect] Scoring {len(lines):,} strings …")
    t0 = time.perf_counter()

    for text in lines:
        row = model.analyze(text)
        results.append(row)
        if row["anomalous"]:
            anomalies += 1
            if verbose:
                print(
                    f"  ANOMALY (score={row['score']:.2%}) | "
                    f"unseen n-grams sample: {row['unseen_examples'][:3]} | "
                    f"{text[:80]}"
                )

    elapsed = time.perf_counter() - t0
    print(
        f"[detect] Done in {elapsed:.2f}s | "
        f"anomalies: {anomalies}/{len(lines)} "
        f"({anomalies/max(len(lines),1):.1%})"
    )

    if output_path:
        output_path = Path(output_path)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"[detect] Report written to {output_path}")

    return results


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

DEMO_TRAIN = [
    # English prose snippets
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "To be or not to be, that is the question.",
    "All that glitters is not gold.",
    "It was the best of times, it was the worst of times.",
    "Call me Ishmael.",
    "It is a truth universally acknowledged.",
    "Happy families are all alike.",
    "In the beginning God created the heavens and the earth.",
    "The only way out is through.",
    "Ask not what your country can do for you.",
    "We hold these truths to be self-evident.",
    "Four score and seven years ago our fathers brought forth.",
    "I have a dream that one day this nation will rise up.",
    "That's one small step for man, one giant leap for mankind.",
    "The internet is becoming the town square for the global village.",
    "In the middle of difficulty lies opportunity.",
    "Life is what happens when you are busy making other plans.",
    "The greatest glory in living lies not in never falling.",
    "The way to get started is to quit talking and begin doing.",
    "If life were predictable it would cease to be life.",
    "If you look at what you have in life, you will always have more.",
    "If you set your goals ridiculously high and it's a failure.",
    "Life is not measured by the number of breaths we take.",
    "You only live once but if you do it right once is enough.",
]

DEMO_TEST = [
    # Normal — similar English text
    ("Normal", "The sun also rises over the distant mountains."),
    ("Normal", "Knowledge is the beginning of all wisdom."),
    # Mildly unusual — mixed or formal language
    ("Borderline", "Henceforth the aforementioned parties shall be bound."),
    # Anomalous — random characters / code / foreign script
    ("Anomalous", "xQ9#fP!z2@Lk^mV7&wR0*"),
    ("Anomalous", "SELECT * FROM users WHERE 1=1; DROP TABLE users;--"),
    ("Anomalous", "乗客の皆様、まもなく終点に到着いたします。"),
    ("Anomalous", "\\x00\\x01\\x02\\x03\\x04\\x05\\x06\\x07\\x08\\x09"),
    ("Anomalous", "Zq7!@#$%^&*()_+{}|:<>?`~[];',./"),
]


def run_demo() -> None:
    print("\n" + "=" * 55)
    print(" N-GRAM ANOMALY DETECTOR — DEMO")
    print("=" * 55)

    # Train
    model = NGramModel.train(
        texts=DEMO_TRAIN,
        n=3,
        error_rate=0.001,
        threshold=0.30,
    )
    print()
    print(model.summary())

    # Detect
    print("\n--- Detection results ---\n")
    header = f"{'Label':<12} {'Score':>7}  {'Flag':<9}  Text"
    print(header)
    print("-" * 70)
    for label, text in DEMO_TEST:
        result = model.analyze(text)
        flag = "⚠ ANOMALY" if result["anomalous"] else "  normal "
        display = text[:55] + ("…" if len(text) > 55 else "")
        print(f"{label:<12} {result['score']:>6.1%}  {flag}  {display}")
        if result["anomalous"] and result["unseen_examples"]:
            sample = result["unseen_examples"][:5]
            print(f"{'':12}   unseen n-grams sample: {sample}")

    print("\nDemo complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="N-gram anomaly detector backed by a Bloom filter.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- train ---
    p_train = sub.add_parser("train", help="Train a model from a text file.")
    p_train.add_argument("--input", "-i", required=True, help="Training corpus (one string per line).")
    p_train.add_argument("--model", "-m", default="model.pkl", help="Output model file [model.pkl].")
    p_train.add_argument("--n", type=int, default=3, help="N-gram length [3].")
    p_train.add_argument("--error-rate", type=float, default=0.01, help="Bloom filter FP rate [0.01].")
    p_train.add_argument("--threshold", type=float, default=0.30,
                         help="Fraction of unseen n-grams to flag as anomalous [0.30].")
    p_train.add_argument("--capacity", type=int, default=None,
                         help="Override Bloom filter capacity (defaults to unique n-gram count).")

    # --- detect ---
    p_det = sub.add_parser("detect", help="Score strings from a file.")
    p_det.add_argument("--input", "-i", required=True, help="Input file (one string per line).")
    p_det.add_argument("--model", "-m", default="model.pkl", help="Trained model file [model.pkl].")
    p_det.add_argument("--output", "-o", default=None, help="JSON report output path.")
    p_det.add_argument("--verbose", "-v", action="store_true", help="Print anomalies to stdout.")
    p_det.add_argument("--threshold", type=float, default=None,
                       help="Override the model's anomaly threshold.")

    # --- score (single string) ---
    p_sc = sub.add_parser("score", help="Score a single string.")
    p_sc.add_argument("text", help="String to score.")
    p_sc.add_argument("--model", "-m", default="model.pkl", help="Trained model file [model.pkl].")

    # --- info ---
    p_info = sub.add_parser("info", help="Print model summary.")
    p_info.add_argument("--model", "-m", default="model.pkl", help="Model file.")

    # --- demo ---
    sub.add_parser("demo", help="Run a self-contained demo (no files needed).")

    return parser


def cmd_train(args: argparse.Namespace) -> None:
    path = Path(args.input)
    with open(path, encoding="utf-8", errors="replace") as fh:
        texts = [ln.rstrip("\n") for ln in fh if ln.strip()]
    if not texts:
        sys.exit("Training file is empty.")
    model = NGramModel.train(
        texts=texts,
        n=args.n,
        bloom_capacity=args.capacity,
        error_rate=args.error_rate,
        threshold=args.threshold,
    )
    print()
    print(model.summary())
    model.save(args.model)


def cmd_detect(args: argparse.Namespace) -> None:
    model = NGramModel.load(args.model)
    if args.threshold is not None:
        model.threshold = args.threshold
    detect_from_file(
        model,
        path=args.input,
        output_path=args.output,
        verbose=args.verbose,
    )


def cmd_score(args: argparse.Namespace) -> None:
    model = NGramModel.load(args.model)
    result = model.analyze(args.text)
    flag = "ANOMALY" if result["anomalous"] else "normal"
    print(f"Score : {result['score']:.4%}  [{flag}]")
    print(f"Total n-grams  : {result['total_ngrams']}")
    print(f"Unseen n-grams : {result['unseen_ngrams']}")
    if result["unseen_examples"]:
        print(f"Unseen sample  : {result['unseen_examples']}")


def cmd_info(args: argparse.Namespace) -> None:
    model = NGramModel.load(args.model)
    print(model.summary())


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "train": cmd_train,
        "detect": cmd_detect,
        "score": cmd_score,
        "info": cmd_info,
        "demo": lambda _: run_demo(),
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
