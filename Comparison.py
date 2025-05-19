#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code.py : interactively choose n for n-gram analysis
"""
import argparse
import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


# ──────────── n-gram utilities ────────────────────────────────────────────
def tokenize(text: str) -> List[str]:
    return WORD_RE.findall(text.lower())


def ngrams(tokens: List[str], n: int) -> Iterable[Tuple[str, ...]]:
    return zip(*(tokens[i:] for i in range(n)))


def build_counter(text: str, n_set: List[int]) -> Counter:  # ★変更
    tokens = tokenize(text)
    ctr = Counter()
    for n in n_set:  # ★変更
        ctr.update(ngrams(tokens, n))
    return ctr


def top_items(counter: Counter, limit: int) -> List[Tuple[Tuple[str, ...], int]]:
    return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]


def tup2str(ng: Tuple[str, ...]) -> str:
    return " ".join(ng)


# ──────────── comparison logic ────────────────────────────────────────────
def incremental_overlap_detail(
    left: List[Tuple[Tuple[str, ...], int]],
    right: List[Tuple[Tuple[str, ...], int]],
    step: int,
    limit: int,
) -> List[Tuple[int, int, List[Tuple[str, ...]]]]:
    details = []
    for start in range(0, limit, step):
        l_slice = {ng for ng, _ in left[start : start + step]}
        r_slice = {ng for ng, _ in right[start : start + step]}
        shared = sorted(l_slice & r_slice)
        details.append((start, len(shared), shared))
    return details


def pairwise_best_match(
    ranked: Dict[str, List[Tuple[Tuple[str, ...], int]]], step: int, limit: int
) -> Tuple[Tuple[str, str], int]:
    best_pair, best_score = None, -1
    for a, b in combinations(ranked.keys(), 2):
        total = sum(
            c
            for _, c, _ in incremental_overlap_detail(ranked[a], ranked[b], step, limit)
        )
        if total > best_score:
            best_pair, best_score = (a, b), total
    return best_pair, best_score


# ──────────── CLI & main ──────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="n-gram overlap (interactive n choice)")
    p.add_argument("-s", "--step", type=int, default=20, help="window step size")
    p.add_argument("-m", "--max", dest="limit", type=int, default=200, help="top limit")
    p.add_argument("files", nargs="+", help="text files (≥2)")
    return p.parse_args()


def read_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        sys.exit(f"Error reading {path}: {e}")


def print_ranking(name: str, ranking: List[Tuple[Tuple[str, ...], int]]) -> None:
    print(f"\n--- {name} : top {len(ranking)} ---")
    step = 20   # ブロックを何単語ごとに設定するか
    for block_start in range(0, len(ranking), step):
        for i, (ng, freq) in enumerate(ranking[block_start:block_start + step], block_start + 1):
            print(f"{i:>3}. {tup2str(ng):<40} {freq}")
        print()  # ブロックごとに改行を挿入


def print_overlap(
    a: str,
    b: str,
    detail: List[Tuple[int, int, List[Tuple[str, ...]]]],
    step: int,
    limit: int,
) -> None:
    print(f"\n=== Overlap {a} vs {b}  [step {step}, top {limit}] ===")
    for start, cnt, shared in detail:
        end = min(start + step, limit)
        print(f"{start+1:>3}-{end:>3}: {cnt} shared")
        if shared:
            print("       ", ", ".join(tup2str(ng) for ng in shared))


def ask_n_gram() -> List[int]:  # ★追加
    ans = input(
        "Analyse which n-gram length? " "(1=uni, 2=bi, 3=tri, etc.  Enter=1-3 all): "
    ).strip()
    if not ans:
        return [1, 2, 3]
    try:
        n = int(ans)
        if n < 1:
            raise ValueError
        return [n]
    except ValueError:
        sys.exit("Please enter a positive integer for n.")


def main() -> None:
    args = parse_args()
    if len(args.files) < 2:
        sys.exit("Need ≥2 files.")

    n_set = ask_n_gram()  # ★対話で取得

    # ランキング
    ranked: Dict[str, List[Tuple[Tuple[str, ...], int]]] = {}
    for fp in args.files:
        counter = build_counter(read_file(Path(fp)), n_set)  # ★変更
        ranked[fp] = top_items(counter, args.limit)
        print_ranking(fp, ranked[fp])

    if len(args.files) == 2:
        a, b = args.files
        det = incremental_overlap_detail(ranked[a], ranked[b], args.step, args.limit)
        print_overlap(a, b, det, args.step, args.limit)
        return

    # 3 本以上
    totals = {
        (a, b): sum(
            c
            for _, c, _ in incremental_overlap_detail(
                ranked[a], ranked[b], args.step, args.limit
            )
        )
        for a, b in combinations(args.files, 2)
    }
    print("\n=== Pairwise total shared n-grams ===")
    for (a, b), t in totals.items():
        print(f"{a} & {b}: {t}")
    (best_a, best_b), best_val = max(totals.items(), key=lambda kv: kv[1])
    print(f"\n>>> Best match: {best_a} & {best_b} (total {best_val})")

    det_best = incremental_overlap_detail(
        ranked[best_a], ranked[best_b], args.step, args.limit
    )
    print_overlap(best_a, best_b, det_best, args.step, args.limit)


if __name__ == "__main__":
    main()
