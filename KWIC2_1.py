# Almost written by ChatGPT
import sys
import re
import spacy
from collections import defaultdict

RED = '\033[91m'
RESET = '\033[0m'

nlp = spacy.load("en_core_web_sm")  # 英語モデルを使用（適宜変更可）

# カラーの選択肢
COLORS = {
    'b': '\033[94m',  # blue
    'g': '\033[92m',  # green
    'r': '\033[91m',  # red
    'c': '\033[96m',  # cyan
    'm': '\033[95m',  # magenta
    'y': '\033[93m',  # yellow
    'k': '\033[90m',  # black (dark grey)
    'w': '\033[97m',  # white
}

# tokenize
def tokenize_with_punctuation(text):
    return re.findall(r"\w+|[^\w\s]", text)

# re tokenize
def reconstruct_text(tokens):
    result = ''
    for i, token in enumerate(tokens):
        if i > 0 and (token not in '.,;!?)]' and tokens[i - 1] not in '(['):
            result += ' '
        result += token
    result = re.sub(r'\s+', ' ', result)
    return result.strip()

# read a file
def read_file(filename):
    try:
        with open(filename, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return None

# 1. tokens search
def search_keyword(tokens, keyword_tokens):
    word_indices = [i for i, t in enumerate(tokens) if re.match(r"\w+", t)]
    word_tokens = [tokens[i] for i in word_indices]
    lower_word_tokens = [t.lower() for t in word_tokens]

    found_positions = []
    i = 0
    while i <= len(lower_word_tokens) - len(keyword_tokens):
        if lower_word_tokens[i:i+len(keyword_tokens)] == keyword_tokens:
            found_positions.append(i)
            i += len(keyword_tokens)
        else:
            i += 1
    return found_positions, word_indices

# 2. POS tags search
def search_by_pos(doc, pos_sequence):
    tokens = [token.text for token in doc]
    word_indices = [i for i, token in enumerate(doc) if not token.is_punct]
    pos_tags = [token.pos_ for token in doc if not token.is_punct]

    found_positions = []
    i = 0
    while i <= len(pos_tags) - len(pos_sequence):
        if pos_tags[i:i+len(pos_sequence)] == pos_sequence:
            found_positions.append(i)
            i += len(pos_sequence)
        else:
            i += 1
    return found_positions, word_indices, tokens

# 3. entity search
def search_by_entity(doc, entity_type):
    found_positions = []
    for ent in doc.ents:
        if ent.label_ == entity_type:
            found_positions.append((ent.start, ent.end))
    return found_positions, [token.text for token in doc]

# 1-1. tokens & sequentially
def highlight_results(tokens, found_positions, word_indices, keyword_tokens, highlight_color, window_size):
    for position in found_positions:
        start_idx = max(0, position - window_size)
        end_idx = min(len(word_indices), position + len(keyword_tokens) + window_size)
        context_word_indices = word_indices[start_idx:end_idx]

        token_start = context_word_indices[0]
        token_end = context_word_indices[-1] + 1
        context_tokens = tokens[token_start:token_end]

        highlight_word_indices = word_indices[position:position + len(keyword_tokens)]
        highlight_token_indices = set(highlight_word_indices)

        highlighted_tokens = []
        for j in range(token_start, token_end):
            token = tokens[j]
            if j in highlight_token_indices and token.lower() in keyword_tokens:
                highlighted_tokens.append(f"{highlight_color}{token}{RESET}")
            else:
                highlighted_tokens.append(token)

        print(reconstruct_text(highlighted_tokens))

# 1-2. tokens & by most frequent token
def highlight_results_sorted_by_next_token(tokens, found_positions, word_indices, keyword_tokens, highlight_color, window_size):
    next_word_map = defaultdict(list)

    for pos in found_positions:
        absolute_pos = word_indices[pos + len(keyword_tokens)] if (pos + len(keyword_tokens)) < len(word_indices) else None
        if absolute_pos is not None:
            next_token = tokens[absolute_pos].lower()
            next_word_map[next_token].append(pos)

    sorted_next_tokens = sorted(next_word_map.items(), key=lambda x: len(x[1]), reverse=True)

    for next_token, positions in sorted_next_tokens:
        print(f"\n--- Next token: '{next_token}' ({len(positions)} times) ---")
        for pos in positions:
            start_idx = max(0, pos - window_size)
            end_idx = min(len(word_indices), pos + len(keyword_tokens) + window_size)
            context_indices = word_indices[start_idx:end_idx]
            token_start = context_indices[0]
            token_end = context_indices[-1] + 1

            highlight_indices = set(word_indices[pos:pos + len(keyword_tokens)])

            highlighted_tokens = []
            for i in range(token_start, token_end):
                token = tokens[i]
                if i in highlight_indices and token.lower() in keyword_tokens:
                    highlighted_tokens.append(f"{highlight_color}{token}{RESET}")
                else:
                    highlighted_tokens.append(token)

            print(reconstruct_text(highlighted_tokens))

# 1-3. tokens & by most frequent POS tags
def highlight_results_sorted_by_next_pos(tokens, found_positions, word_indices, keyword_tokens, highlight_color, window_size, doc):
    next_pos_map = defaultdict(list)

    for pos in found_positions:
        # キーワード直後の絶対インデックス
        base = pos + len(keyword_tokens)
        if base < len(word_indices):
            abs_idx = word_indices[base]
            # SPACE, PUNCT, SYM タグをスキップして次の実語のPOSを取得
            look_idx = abs_idx
            while look_idx < len(doc) and doc[look_idx].pos_ in ("SPACE", "PUNCT", "SYM"):
                look_idx += 1
            if look_idx < len(doc):
                next_pos = doc[look_idx].pos_
                next_pos_map[next_pos].append(pos)

    # 頻度順にソート
    sorted_next = sorted(next_pos_map.items(), key=lambda x: len(x[1]), reverse=True)

    for next_pos, positions in sorted_next:
        print(f"\n--- Next POS: '{next_pos}' ({len(positions)} times) ---")
        for p in positions:
            start_idx = max(0, p - window_size)
            end_idx = min(len(word_indices), p + len(keyword_tokens) + window_size)
            context_indices = word_indices[start_idx:end_idx]
            token_start = context_indices[0]
            token_end = context_indices[-1] + 1

            highlight_indices = set(word_indices[p:p + len(keyword_tokens)])
            highlighted_tokens = []
            for i in range(token_start, token_end):
                token_text = tokens[i]
                if i in highlight_indices and token_text.lower() in keyword_tokens:
                    highlighted_tokens.append(f"{highlight_color}{token_text}{RESET}")
                else:
                    highlighted_tokens.append(token_text)
            print(reconstruct_text(highlighted_tokens))

# 2-1. POS tags & sequentially
def highlight_results_for_pos(doc, found_positions, word_indices, pos_sequence, highlight_color, window_size):
    for position in found_positions:
        start_idx = max(0, position - window_size)
        end_idx = min(len(word_indices), position + len(pos_sequence) + window_size)
        context_word_indices = word_indices[start_idx:end_idx]

        token_start = context_word_indices[0]
        token_end = context_word_indices[-1] + 1
        highlight_indices = set(word_indices[position:position + len(pos_sequence)])

        highlighted_tokens = []
        for i in range(token_start, token_end):
            token_text = doc[i].text.replace('\n', ' ')
            if i in highlight_indices:
                highlighted_tokens.append(f"{highlight_color}{token_text}{RESET}")
            else:
                highlighted_tokens.append(token_text)

        print(reconstruct_text(highlighted_tokens))

# 2-2. POS tags & by most frequent token
def highlight_results_for_pos_sorted_by_next_token(doc, found_positions, word_indices, pos_sequence, highlight_color, window_size):
    next_word_map = defaultdict(list)

    for pos in found_positions:
        next_pos_index = pos + len(pos_sequence)
        # word_indices 上で次のトークンが存在するかチェック
        if next_pos_index < len(word_indices):
            abs_idx = word_indices[next_pos_index]
            token_obj = doc[abs_idx]
            # 単語（英数字）トークンのみを対象
            if re.match(r"\w+", token_obj.text):
                next_token = token_obj.text.lower()
                next_word_map[next_token].append(pos)

    # 出現頻度順にソート
    sorted_next = sorted(next_word_map.items(), key=lambda x: len(x[1]), reverse=True)

    for next_token, positions in sorted_next:
        print(f"\n--- Next token: '{next_token}' ({len(positions)} times) ---")
        for pos in positions:
            start_idx = max(0, pos - window_size)
            end_idx = min(len(word_indices), pos + len(pos_sequence) + window_size)
            context_indices = word_indices[start_idx:end_idx]
            token_start = context_indices[0]
            token_end = context_indices[-1] + 1
            highlight_indices = set(word_indices[pos:pos + len(pos_sequence)])

            highlighted_tokens = []
            for i in range(token_start, token_end):
                token_text = doc[i].text.replace('\n', ' ')
                if i in highlight_indices:
                    highlighted_tokens.append(f"{highlight_color}{token_text}{RESET}")
                else:
                    highlighted_tokens.append(token_text)
            # 文脈を再構築して表示
            print(reconstruct_text(highlighted_tokens))

# 2-3. POS tags & by most frequent POS tags
def highlight_results_for_pos_sorted_by_next_pos(doc, found, word_idx, pos_seq, color, win):
    next_map = defaultdict(list)
    for pos in found:
        base = pos + len(pos_seq)
        if base < len(word_idx):
            idx = word_idx[base]
            while idx < len(doc) and doc[idx].pos_ in ("SPACE","PUNCT","SYM"): idx += 1
            if idx < len(doc): next_map[doc[idx].pos_].append(pos)
    for next_pos, poses in sorted(next_map.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"\n--- Next POS: '{next_pos}' ({len(poses)} times) ---")
        for p in poses:
            s = max(0,p-win); e = min(len(word_idx),p+len(pos_seq)+win)
            idxs = word_idx[s:e]; start, end = idxs[0], idxs[-1]+1
            highlight = set(word_idx[p:p+len(pos_seq)])
            out = []
            for i in range(start,end):
                txt = doc[i].text.replace('\n',' ')
                out.append(f"{color}{txt}{RESET}" if i in highlight else txt)
            print(reconstruct_text(out))

# 3-1. entity & sequentially
def highlight_results_for_ent(tokens, entity_ranges, highlight_color, window_size):
    word_indices = [i for i, token in enumerate(tokens) if re.match(r"\w+", token)]

    for start, end in entity_ranges:
        try:
            start_word_pos = word_indices.index(start)
            end_word_pos = word_indices.index(end - 1)
        except ValueError:
            continue

        context_start_pos = max(0, start_word_pos - window_size)
        context_end_pos = min(len(word_indices), end_word_pos + 1 + window_size)
        token_start = word_indices[context_start_pos]
        token_end = word_indices[context_end_pos - 1] + 1

        context_tokens = tokens[token_start:token_end]

        highlighted_tokens = []
        for i in range(token_start, token_end):
            if start <= i < end:
                highlighted_tokens.append(f"{highlight_color}{tokens[i]}{RESET}")
            else:
                highlighted_tokens.append(tokens[i])

        print(reconstruct_text(highlighted_tokens))

# 3-2. entity & by most frequent token
def highlight_results_fot_ent_sorted_by_next_token(tokens, entity_ranges, highlight_color, window_size):
    # 単語トークンのインデックスだけを抽出
    word_indices = [i for i, token in enumerate(tokens) if re.match(r"\w+", token)]
    next_token_map = defaultdict(list)

    # エンティティ後の "次の" 単語トークンを探す
    for start, end in entity_ranges:
        idx = end
        # エンティティ直後に続く最初のワードトークンを探索
        while idx < len(tokens) and not re.match(r"\w+", tokens[idx]):
            idx += 1
        if idx < len(tokens):
            next_tok = tokens[idx].lower()
            next_token_map[next_tok].append((start, end))

    # 出現頻度順にソート
    sorted_items = sorted(next_token_map.items(), key=lambda x: len(x[1]), reverse=True)

    for next_token, ranges in sorted_items:
        print(f"\n--- Next token: '{next_token}' ({len(ranges)} times) ---")
        for start, end in ranges:
            # エンティティ範囲をワードベースで再マッピング
            try:
                start_word_pos = word_indices.index(start)
                end_word_pos = word_indices.index(end - 1)
            except ValueError:
                continue

            # コンテキスト範囲をワードインデックスで設定
            context_start = max(0, start_word_pos - window_size)
            context_end = min(len(word_indices), end_word_pos + 1 + window_size)
            context_indices = word_indices[context_start:context_end]

            # トークンレベルの開始・終了位置
            token_start = context_indices[0]
            token_end = context_indices[-1] + 1

            # ハイライト表示
            highlighted = []
            for i in range(token_start, token_end):
                if start <= i < end:
                    highlighted.append(f"{highlight_color}{tokens[i]}{RESET}")
                else:
                    highlighted.append(tokens[i])
            print(reconstruct_text(highlighted))

# 3-3. entity & by most frequent POS tags
def highlight_results_for_ent_sorted_by_next_pos(tokens, ent_ranges, color, win, doc):
    next_map = defaultdict(list)
    word_idx = [i for i, t in enumerate(tokens) if re.match(r"\w+", t)]
    for (start,end) in ent_ranges:
        # 実スタート位置
        idx = end
        while idx < len(tokens) and not re.match(r"\w+", tokens[idx]):
            idx += 1
        # convert token idx to doc idx
        if idx < len(tokens):
            # find corresponding doc index
            doc_idx = None
            count = 0
            for j,tok in enumerate(doc):
                if tok.text == tokens[idx]:
                    doc_idx = j; break
            if doc_idx is not None:
                while doc_idx < len(doc) and doc[doc_idx].pos_ in ("SPACE","PUNCT","SYM"):
                    doc_idx += 1
                if doc_idx < len(doc):
                    next_map[doc[doc_idx].pos_].append((start,end))
    for next_pos, ranges in sorted(next_map.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"\n--- Next POS: '{next_pos}' ({len(ranges)} times) ---")
        for (start,end) in ranges:
            # ワードインデックスからcontext
            wpos = [i for i, t in enumerate(tokens) if re.match(r"\w+", t)]
            try:
                sp = wpos.index(start)
                ep = wpos.index(end-1)
            except ValueError:
                continue
            s = max(0,sp-win); e = min(len(wpos),ep+1+win)
            idxs = wpos[s:e]; st, ed = idxs[0], idxs[-1]+1
            out = []
            for i in range(st,ed):
                tok = tokens[i]
                if start <= i < end:
                    out.append(f"{color}{tok}{RESET}")
                else:
                    out.append(tok)
            print(reconstruct_text(out))

# main
def main():
    if len(sys.argv) < 4:
        print("Usage: python KWIC2_1.py sample.txt [mode: token|pos|ent] \"keyword/POS tag/entity type\"")
        return

    filename = sys.argv[1]
    mode = sys.argv[2]
    query = sys.argv[3]

    text = read_file(filename)
    if text is None:
        return

    doc = nlp(text)

    display_mode = input("Display mode (1 = sequentially, 2 = by most frequent token, 3 = by most frequent POS tags): ") or "1"
    color_choice = input(f"Color (default is red: ‘r’): ") or 'r'
    window_size = input(f"Window size (number of words displayed before and after, default is 5): ") or 5
    window_size = int(window_size)

    highlight_color = COLORS.get(color_choice, RED)

    if mode == "token":
        keyword_tokens = query.lower().split()
        tokens = tokenize_with_punctuation(text)
        found_positions, word_indices = search_keyword(tokens, keyword_tokens)
        print(f"\nResults: {len(found_positions)}\n")
        if found_positions:
            if display_mode == "2":
                highlight_results_sorted_by_next_token(tokens, found_positions, word_indices, keyword_tokens, highlight_color, window_size)
            elif display_mode == "3":
                highlight_results_sorted_by_next_pos(tokens, found_positions, word_indices, keyword_tokens, highlight_color, window_size, doc)
            else:
                highlight_results(tokens, found_positions, word_indices, keyword_tokens, highlight_color, window_size)
        else:
            print("No matches found.")

    elif mode == "pos":
        pos_tags = query.strip().upper().split()
        found_positions, word_indices, _ = search_by_pos(doc, pos_tags)
        print(f"\nResults: {len(found_positions)}\n")
        if found_positions:
            if display_mode == "2":
                highlight_results_for_pos_sorted_by_next_token(doc, found_positions, word_indices, pos_tags, highlight_color, window_size)
            elif display_mode == "3":
                highlight_results_for_pos_sorted_by_next_pos(doc, found_positions, word_indices, pos_tags, highlight_color, window_size)
            else:
                highlight_results_for_pos(doc, found_positions, word_indices, pos_tags, highlight_color, window_size)
        else:
            print("No matches found.")

    elif mode == "ent":
        entity_type = query.strip().upper()
        found_ranges, tokens = search_by_entity(doc, entity_type)
        print(f"\nResults: {len(found_ranges)}\n")
        if found_ranges:
            if display_mode == "2":
                highlight_results_fot_ent_sorted_by_next_token(tokens, found_ranges, highlight_color, window_size)
            elif display_mode == "3":
                highlight_results_for_ent_sorted_by_next_pos(tokens, found_ranges, highlight_color, window_size, doc)
            else:
                highlight_results_for_ent(tokens, found_ranges, highlight_color, window_size)
        else:
            print("No matches found.")

    else:
        print("Invalid mode. Choose from: token, pos, ent")

if __name__ == "__main__":
    main()