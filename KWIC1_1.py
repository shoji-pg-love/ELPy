# Almost written by ChatGPT
import sys
import re

RED = '\033[91m'
RESET = '\033[0m'

# 1. 文章をトークンに分割する関数
def tokenize_with_punctuation(text):
    return re.findall(r"\w+|[^\w\s]", text)

# 2. 検索関数（キーワードが一致する箇所を見つける）
def search_keyword(tokens, keyword_tokens):
    word_indices = [i for i, t in enumerate(tokens) if re.match(r"\w+", t)]
    word_tokens = [tokens[i] for i in word_indices]
    lower_word_tokens = [t.lower() for t in word_tokens]

    found_positions = []
    i = 0
    while i <= len(lower_word_tokens) - len(keyword_tokens):
        if lower_word_tokens[i:i+len(keyword_tokens)] == keyword_tokens:
            found_positions.append(i)
            i += len(keyword_tokens)  # キーワード分スキップ
        else:
            i += 1
    return found_positions, word_indices

# 3. コンマやピリオドなど記号を適切に処理する関数
def reconstruct_text(tokens):
    result = ''
    for i, token in enumerate(tokens):
        if i > 0:
            prev = tokens[i - 1]
            if (token not in '.,;!?)]' and prev not in '(['):
                result += ' '
        result += token
    return result

# 4. 検索結果をハイライトして表示する関数
def highlight_results(tokens, found_positions, word_indices, keyword_tokens):
    for position in found_positions:
        start_idx = max(0, position - 5)
        end_idx = min(len(word_indices), position + len(keyword_tokens) + 5)
        context_word_indices = word_indices[start_idx:end_idx]

        token_start = context_word_indices[0]
        token_end = context_word_indices[-1] + 1
        context_tokens = tokens[token_start:token_end]

        highlight_word_indices = word_indices[position:position+len(keyword_tokens)]
        highlight_token_indices = set(highlight_word_indices)

        highlighted_tokens = []
        for j in range(token_start, token_end):
            token = tokens[j]
            if j in highlight_token_indices and token.lower() in keyword_tokens:
                highlighted_tokens.append(f"{RED}{token}{RESET}")
            else:
                highlighted_tokens.append(token)

        print(reconstruct_text(highlighted_tokens))

# 5. ファイルの入力を受け取る関数
def read_file(filename):
    try:
        with open(filename, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return None

# 6. メイン関数（これまでの関数をまとめたもの）
def main():
    if len(sys.argv) != 3:
        print("Usage: python KWIC1_1.py sample.txt \"(キーワード)\"")
        return

    filename = sys.argv[1]
    keyword_phrase = sys.argv[2].lower()
    keyword_tokens = keyword_phrase.split()

    text = read_file(filename)
    if text is None:
        return

    tokens = tokenize_with_punctuation(text)

    # キーワード位置を検索
    found_positions, word_indices = search_keyword(tokens, keyword_tokens)

    if found_positions:
        highlight_results(tokens, found_positions, word_indices, keyword_tokens)
    else:
        print("No matches found.")

if __name__ == "__main__":
    main()
