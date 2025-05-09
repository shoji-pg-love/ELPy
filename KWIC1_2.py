# Almost written by ChatGPT
import sys
import re
import spacy

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

# 1. トークン化（記号も含む）
def tokenize_with_punctuation(text):
    return re.findall(r"\w+|[^\w\s]", text)

# 2. キーワード検索（トークン単位）
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

# 3. POSタグによる検索
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

# 4. エンティティ検索（例：ORG, PERSON など）
def search_by_entity(doc, entity_type):
    found_positions = []
    for ent in doc.ents:
        if ent.label_ == entity_type:
            found_positions.append((ent.start, ent.end))
    return found_positions, [token.text for token in doc]

# 5. トークンを再構築（空白や記号の扱いを調整）
def reconstruct_text(tokens):
    result = ''
    for i, token in enumerate(tokens):
        if i > 0 and (token not in '.,;!?)]' and tokens[i - 1] not in '(['):
            result += ' '
        result += token
    return result.strip()

# 6. ハイライト表示（トークン検索・POSタグ検索用）
def highlight_results(tokens, found_positions, word_indices, keyword_tokens, highlight_color, window_size):
    for position in found_positions:
        start_idx = max(0, position - window_size)  # ウィンドウサイズに基づいて前のトークンを表示
        end_idx = min(len(word_indices), position + len(keyword_tokens) + window_size)  # ウィンドウサイズに基づいて後のトークンを表示
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

# POSタグによる検索結果のハイライト表示
def highlight_results_for_pos(doc, found_positions, word_indices, pos_sequence, highlight_color, window_size):
    for position in found_positions:
        start_idx = max(0, position - window_size)  # ウィンドウサイズに基づいて前のトークンを表示
        end_idx = min(len(word_indices), position + len(pos_sequence) + window_size)  # ウィンドウサイズに基づいて後のトークンを表示
        context_word_indices = word_indices[start_idx:end_idx]

        token_start = context_word_indices[0]
        token_end = context_word_indices[-1] + 1

        highlight_indices = set(word_indices[position:position + len(pos_sequence)])

        highlighted_tokens = []
        for i in range(token_start, token_end):
            token_text = doc[i].text.replace('\n', '')  # 空白にせず、改行だけ除去
            if i in highlight_indices:
                highlighted_tokens.append(f"{highlight_color}{token_text}{RESET}")
            else:
                highlighted_tokens.append(token_text)

        print(reconstruct_text(highlighted_tokens))

# 7. エンティティ結果のハイライト表示（エンティティの最初と最後の単語を別行で表示）
def highlight_entity_results(tokens, entity_ranges, highlight_color, window_size):
    # 単語インデックス（句読点などを除いたインデックス）を作成
    word_indices = [i for i, token in enumerate(tokens) if re.match(r"\w+", token)]

    for start, end in entity_ranges:
        # エンティティの先頭が word_indices の中で何番目かを取得
        try:
            start_word_pos = word_indices.index(start)
            end_word_pos = word_indices.index(end - 1)
        except ValueError:
            # 句読点などで index が見つからない場合はスキップ
            continue

        # 表示範囲の単語インデックスを決定
        context_start_pos = max(0, start_word_pos - window_size)
        context_end_pos = min(len(word_indices), end_word_pos + 1 + window_size)

        # 実際のトークンインデックスを取得
        token_start = word_indices[context_start_pos]
        token_end = word_indices[context_end_pos - 1] + 1  # +1で inclusive に

        # ハイライト用にトークンを整形
        context_tokens = tokens[token_start:token_end]

        highlighted_tokens = []
        for i in range(token_start, token_end):
            if start <= i < end:
                highlighted_tokens.append(f"{highlight_color}{tokens[i]}{RESET}")
            else:
                highlighted_tokens.append(tokens[i])

        print(reconstruct_text(highlighted_tokens))

# 8. ファイル読み込み
def read_file(filename):
    try:
        with open(filename, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return None

# 9. メイン処理
def main():
    if len(sys.argv) < 4:
        print("Usage: python KWIC1_2.py sample.txt [mode: token|pos|ent] \"keyword/POS tag/entity type\"")
        return

    filename = sys.argv[1]
    mode = sys.argv[2]
    query = sys.argv[3]

    text = read_file(filename)
    if text is None:
        return

    doc = nlp(text)

    # カラーとウィンドウサイズの入力を促す
    color_choice = input(f"What color do you want? (Default is red: ‘r’): ") or 'r'
    window_size = input(f"Window size (number of words displayed before and after, default is 5): ") or 5
    window_size = int(window_size)

    highlight_color = COLORS.get(color_choice, RED)

    if mode == "token":
        keyword_tokens = query.lower().split()
        tokens = tokenize_with_punctuation(text)
        found_positions, word_indices = search_keyword(tokens, keyword_tokens)
        if found_positions:
            highlight_results(tokens, found_positions, word_indices, keyword_tokens, highlight_color, window_size)  # window_size を追加
        else:
            print("No matches found.")

    elif mode == "pos":
        pos_tags = query.strip().upper().split()
        found_positions, word_indices, _ = search_by_pos(doc, pos_tags)
        if found_positions:
            highlight_results_for_pos(doc, found_positions, word_indices, pos_tags, highlight_color, window_size)  # window_size を追加
        else:
            print("No matches found.")

    elif mode == "ent":
        entity_type = query.strip().upper()
        found_ranges, tokens = search_by_entity(doc, entity_type)
        if found_ranges:
            highlight_entity_results(tokens, found_ranges, highlight_color, window_size)  # window_size を追加
        else:
            print("No matches found.")

    else:
        print("Invalid mode. Choose from: token, pos, ent")


if __name__ == "__main__":
    main()
