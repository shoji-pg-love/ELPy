#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code.py : 1 番目のデータセットを基準に、残りすべてのファイルを
          1 枚のテーブルで横並び比較（トップ 200 n-gram）
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


# ─────────────── n-gram utilities ──────────────────────────────────────────
def tokenize(text: str) -> List[str]:
    return WORD_RE.findall(text.lower())


def ngrams(tokens: List[str], n: int) -> Iterable[Tuple[str, ...]]:
    return zip(*(tokens[i:] for i in range(n)))


def build_counter(text: str, n_val: List[int]) -> Counter:
    toks = tokenize(text)
    ctr = Counter()
    for n in n_val:
        ctr.update(ngrams(toks, n))
    return ctr


def top_items(counter: Counter, limit: int) -> List[Tuple[Tuple[str, ...], int]]:
    # 頻度降順 → 辞書順
    return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]


def tup2str(ng: Tuple[str, ...]) -> str:  # ('machine','learning') → 'machine learning'
    return " ".join(ng)


# ─────────────── CLI helpers ───────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare 1st dataset with all others in a single table "
        "(top-N n-grams)"
    )
    p.add_argument(
        "-m", "--max", type=int, default=200, help="top-N n-grams to list (default 200)"
    )
    p.add_argument("files", nargs="+", help="text files (≥2)")
    return p.parse_args()


def read_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        sys.exit(f"Error reading {path}: {e}")


def ask_n_list() -> List[int]:
    ans = input("Analyse which n? (Enter=1-3): ").strip()
    if not ans:
        return [1, 2, 3]
    try:
        n = int(ans)
        if n < 1:
            raise ValueError
        return [n]
    except ValueError:
        sys.exit("n must be a positive integer.")


# ─────────────── table printer ─────────────────────────────────────────────
def print_table(
    files: List[str],
    rankings: Dict[str, List[Tuple[Tuple[str, ...], int]]],
    limit: int,
) -> None:
    col_width = 25  # 1 列の横幅（必要なら調整）
    base = files[0]
    headers = ["Rank", base] + files[1:]
    line_fmt = "{:>4}  " + "  ".join("{:<" + str(col_width) + "}" for _ in headers[1:])

    # ヘッダー
    print("\n" + line_fmt.format(*headers))
    print("-" * (4 + 2 + (col_width + 2) * (len(headers) - 1)))

    # 各順位
    for i in range(limit):
        row: List[str] = [str(i + 1)]
        for fp in files:
            try:
                row.append(tup2str(rankings[fp][i][0]))
            except IndexError:
                row.append("")
        print(line_fmt.format(*row))


# ─────────────── main ──────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    if len(args.files) < 2:
        sys.exit("Need ≥2 files.")

    n_list = ask_n_list()
    limit = args.max
    files = args.files
    base = files[0]

    # カウント → ランキング
    rankings: Dict[str, List[Tuple[Tuple[str, ...], int]]] = {}
    for fp in files:
        txt = read_file(Path(fp))
        rankings[fp] = top_items(build_counter(txt, n_list), limit)

    # テーブル表示
    print_table(files, rankings, limit)

    # 共有数とベストマッチ
    best_match, best_shared = None, -1
    base_set = {ng for ng, _ in rankings[base]}

    for fp in files[1:]:
        shared = len(base_set & {ng for ng, _ in rankings[fp]})
        print(f"\nShared with {fp}: {shared}")
        if shared > best_shared:
            best_match, best_shared = fp, shared

    if len(files) > 2:
        print(f"\n>>> Best match with {base}: {best_match}  (shared {best_shared})")


if __name__ == "__main__":
    main()
