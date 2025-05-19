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


def print_ranking(
    name: str,
    ranking: List[Tuple[Tuple[str, ...], int]],
    shared_map: Dict[int, List[Tuple[str, ...]]],
    step: int
) -> None:
    print(f"\n--- {name} : top {len(ranking)} ---")
    for i in range(0, len(ranking), step):
        block = ranking[i:i+step]
        for j, (ng, freq) in enumerate(block, start=i+1):
            print(f"{j:>3}. {tup2str(ng):<40} {freq}")

        shared = shared_map.get(i, [])
        print(f"\n[ {len(shared)} shared ]")
        if shared:
            print(f"     → Shared n-grams ({i+1}-{i+len(block)}): ", end="")
            print(", ".join(tup2str(ng) for ng in shared))
        print("\n")


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

    n_set = ask_n_gram()

    ranked: Dict[str, List[Tuple[Tuple[str, ...], int]]] = {}
    raw: Dict[str, str] = {fp: read_file(Path(fp)) for fp in args.files}
    for fp in args.files:
        counter = build_counter(raw[fp], n_set)
        ranked[fp] = top_items(counter, args.limit)

    q_file = args.files[0]

    if len(args.files) == 2:
        a, b = args.files
        det = incremental_overlap_detail(ranked[a], ranked[b], args.step, args.limit)
        shared_map = {start: shared for start, _, shared in det}
        print_ranking(a, ranked[a], shared_map, args.step)
        print_ranking(b, ranked[b], shared_map, args.step)
        print_overlap(a, b, det, args.step, args.limit)
        return

    # 3つ以上のファイルがあるとき：Qテキストとのみ比較
    totals = {}
    detail_maps = {}
    for b in args.files[1:]:
        det = incremental_overlap_detail(ranked[q_file], ranked[b], args.step, args.limit)
        total = sum(c for _, c, _ in det)
        totals[(q_file, b)] = total
        detail_maps[b] = {start: shared for start, _, shared in det}

    print("\n=== Pairwise total shared n-grams with Q ===")
    for (a, b), t in totals.items():
        print(f"{a} & {b}: {t}")
    (best_a, best_b), best_val = max(totals.items(), key=lambda kv: kv[1])
    print(f"\n>>> Best match: {best_a} & {best_b} (total {best_val})")

    for fp in args.files:
        shared_map = detail_maps.get(fp, {}) if fp != q_file else {
            start: shared for start, _, shared in incremental_overlap_detail(
                ranked[q_file], ranked[best_b], args.step, args.limit
            )
        }

        print_ranking(fp, ranked[fp], shared_map, args.step)

    det_best = incremental_overlap_detail(ranked[best_a], ranked[best_b], args.step, args.limit)
    print_overlap(best_a, best_b, det_best, args.step, args.limit)


if __name__ == "__main__":
    main()
