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


def print_ranking_side_by_side(
    q_name: str,
    q_rank: List[Tuple[Tuple[str, ...], int]],
    k_name: str,
    k_rank: List[Tuple[Tuple[str, ...], int]],
    q_shared: Dict[int, List[Tuple[str, ...]]],
    k_shared: Dict[int, List[Tuple[str, ...]]],
    step: int,
    width: int = 40
) -> None:
    print(f"\n--- Side-by-side: {q_name} (Q) vs {k_name} (K) ---")
    print(f"{'Q Ranking':<{width}} {'K Ranking':<{width}}")
    print(f"{'-'*width} {'-'*width}")

    max_len = max(len(q_rank), len(k_rank))
    for i in range(0, max_len, step):
        q_block = q_rank[i:i+step]
        k_block = k_rank[i:i+step]
        for j in range(step):
            q_line = ""
            k_line = ""
            if j < len(q_block):
                ng_q, fq = q_block[j]
                q_line = f"{i+j+1:>3}. {tup2str(ng_q):<30} {fq:<4}"
            if j < len(k_block):
                ng_k, fk = k_block[j]
                k_line = f"{i+j+1:>3}. {tup2str(ng_k):<30} {fk:<4}"
            print(f"{q_line:<{width}} {k_line:<{width}}")

        # Shared info for this step
        q_shared_ngrams = q_shared.get(i, [])
        k_shared_ngrams = k_shared.get(i, [])

        shared = sorted(set(q_shared_ngrams) & set(k_shared_ngrams))
        print(f"\n[ {len(shared)} shared ]")
        if shared:
            print(f"     → Shared n-grams ({i+1}-{i+step}): ", end="")
            print(", ".join(tup2str(ng) for ng in shared))
        print("\n")


# ──────────── CLI & main ──────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="n-gram overlap (interactive n choice)")
    p.add_argument("-s", "--step", type=int, default=20, help="window step size")
    p.add_argument("-m", "--max", dest="limit", type=int, default=200, help="top limit")
    p.add_argument("--only-best", action="store_true", help="Show only question and best match")
    p.add_argument("folder", help="Folder containing the question and known text files")
    p.add_argument("question", help="Filename of the question text (must be in folder)")
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
    folder = Path(args.folder)
    q_path = folder / args.question
    if not q_path.exists():
        sys.exit(f"Question file {q_path} not found.")

    all_files = list(folder.glob("*.txt"))
    known_files = [fp for fp in all_files if fp != q_path]

    if len(known_files) < 1:
        sys.exit("Need at least one known file besides the question.")

    n_set = ask_n_gram()

    ranked: Dict[str, List[Tuple[Tuple[str, ...], int]]] = {}
    raw: Dict[str, str] = {str(fp): read_file(fp) for fp in [q_path] + known_files}
    for fp in [q_path] + known_files:
        counter = build_counter(raw[str(fp)], n_set)
        ranked[str(fp)] = top_items(counter, args.limit)

    q_file = str(q_path)

    totals = {}
    detail_maps = {}
    for b in known_files:
        b_file = str(b)
        det = incremental_overlap_detail(ranked[q_file], ranked[b_file], args.step, args.limit)
        total = sum(c for _, c, _ in det)
        totals[(q_file, b_file)] = total
        detail_maps[b_file] = {start: shared for start, _, shared in det}

    print("\n=== Pairwise total shared n-grams with Q ===")
    for (a, b), t in totals.items():
        print(f"{Path(a).name} & {Path(b).name}: {t}")
    (best_a, best_b), best_val = max(totals.items(), key=lambda kv: kv[1])
    print(f"\n>>> Best match: {Path(best_a).name} & {Path(best_b).name} (total {best_val})")

    if args.only_best:
        # サイドバイサイド出力用の shared_map を作成
        q_shared_map = {
            start: shared for start, _, shared in incremental_overlap_detail(
                ranked[best_a], ranked[best_b], args.step, args.limit
            )
        }
        k_shared_map = detail_maps[best_b]

        print_ranking_side_by_side(
            Path(best_a).name,
            ranked[best_a],
            Path(best_b).name,
            ranked[best_b],
            q_shared_map,
            k_shared_map,
            args.step
        )
    else:
        files_to_show = [str(q_path)] + [str(fp) for fp in known_files]
        for fp in files_to_show:
            shared_map = (
                detail_maps.get(fp, {}) if fp != q_file else {
                    start: shared for start, _, shared in incremental_overlap_detail(
                        ranked[q_file], ranked[best_b], args.step, args.limit
                    )
                }
            )
            print_ranking(Path(fp).name, ranked[fp], shared_map, args.step)

    det_best = incremental_overlap_detail(ranked[best_a], ranked[best_b], args.step, args.limit)
    print_overlap(Path(best_a).name, Path(best_b).name, det_best, args.step, args.limit)


if __name__ == "__main__":
    main()
