import sqlite3
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import matplotlib as mpl
from matplotlib.font_manager import fontManager

DB_FILE = "hakka_vocab_quiz.db"
JSON_FILE = "hakka_vocab_data.json"

def initialize_database():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # 建立詞彙表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vocab (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word TEXT NOT NULL,
            pinyin TEXT NOT NULL,
            example TEXT NOT NULL,
            last_review TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            s_value REAL DEFAULT 2.0,
            incorrect_count INTEGER DEFAULT 0
        )
    """)

    # 建立填空題目表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS quiz (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word_id INTEGER NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            options TEXT NOT NULL,
            FOREIGN KEY (word_id) REFERENCES vocab (id)
        )
    """)

    # 建立答錯記錄表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS incorrect_answers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word_id INTEGER NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            selected_option TEXT NOT NULL,
            FOREIGN KEY (word_id) REFERENCES vocab (id)
        )
    """)

    conn.commit()
    conn.close()
def load_json_data():
    """從 JSON 檔案載入詞彙數據"""
    if not os.path.exists(JSON_FILE):
        print(f"無法找到 JSON 檔案：{JSON_FILE}")
        return []

    with open(JSON_FILE, "r", encoding="utf-8") as file:
        try:
            data = json.load(file)
            return data
        except json.JSONDecodeError as e:
            print(f"JSON 檔案解析失敗：{e}")
            return []


def populate_vocab_and_quiz():
    """將 JSON 資料填充至資料庫"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # 檢查是否需要填充資料
    cursor.execute("SELECT COUNT(*) FROM vocab")
    vocab_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM quiz")
    quiz_count = cursor.fetchone()[0]

    if vocab_count > 0 and quiz_count > 0:
        print("資料庫已初始化，跳過填充資料。")
        conn.close()
        return

    # 載入 JSON 資料
    data = load_json_data()
    if not data:
        print("無法填充資料，因為 JSON 檔案是空的或解析失敗。")
        conn.close()
        return

    # 插入詞彙資料
    vocab_data = [(item["word"], item["pinyin"], item["example"]) for item in data]
    cursor.executemany("""
        INSERT INTO vocab (word, pinyin, example) VALUES (?, ?, ?)
    """, vocab_data)

    # 插入 quiz 資料
    cursor.execute("SELECT id, word, pinyin, example FROM vocab")
    vocab_records = cursor.fetchall()

    quiz_data = []
    for record in vocab_records:
        word_id, word, pinyin, example = record
        question = example.replace(word, "_____")
        options = random.sample([r[1] for r in vocab_records if r[1] != word], 3)
        options.append(word)
        random.shuffle(options)
        options_str = ";".join(options)
        quiz_data.append((word_id, question, word, options_str))

    cursor.executemany("""
        INSERT INTO quiz (word_id, question, answer, options) VALUES (?, ?, ?, ?)
    """, quiz_data)

    conn.commit()
    conn.close()
    print("資料庫初始化完成。")


def log_incorrect_answer(word_id, selected_option):
    """記錄答錯的題目"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO incorrect_answers (word_id, selected_option)
        VALUES (?, ?)
    """, (word_id, selected_option))
    conn.commit()
    conn.close()


def update_review(word_id, performance):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("SELECT s_value, incorrect_count FROM vocab WHERE id = ?", (word_id,))
    s_value, incorrect_count = cursor.fetchone()

    if performance == "good":
        s_value *= 1.05
    elif performance == "poor":
        s_value *= 0.95
        incorrect_count += 1

    last_review_time = datetime.now()
    cursor.execute("""
        UPDATE vocab
        SET last_review = ?, s_value = ?, incorrect_count = ?
        WHERE id = ?
    """, (last_review_time, s_value, incorrect_count, word_id))

    print(f"單字 ID: {word_id} 更新成功！最後複習時間: {last_review_time}, 新的 s_value: {s_value}, 錯誤次數: {incorrect_count}")

    conn.commit()
    conn.close()


def start_quiz():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # 查詢包含拼音的 quiz 資料
    cursor.execute("""
        SELECT q.question, q.answer, q.options, v.id, v.pinyin
        FROM quiz q
        JOIN vocab v ON q.word_id = v.id
        ORDER BY RANDOM() LIMIT 5
    """)
    quizzes = cursor.fetchall()
    conn.close()

    if quizzes:
        print("\n=== 開始測驗 ===")
        for quiz in quizzes:
            question, answer, options_str, word_id, answer_pinyin = quiz
            # 解析選項
            options = options_str.split(";")
            # 取得每個選項的拼音
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            option_with_pinyin = []
            for option in options:
                cursor.execute("SELECT word, pinyin FROM vocab WHERE word = ?", (option,))
                result = cursor.fetchone()
                if result:
                    option_with_pinyin.append(f"{result[0]} ({result[1]})")
                else:
                    option_with_pinyin.append(option)  # 若找不到，僅顯示單字
            conn.close()

            print(f"\n題目: {question}")
            for i, option in enumerate(option_with_pinyin, start=1):
                print(f"{chr(64 + i)}. {option}")

            # 使用者回答
            user_input = input("\n請選擇答案 (A/B/C/D)：").strip().upper()
            if user_input in "ABCD":
                selected_index = ord(user_input) - 65
                selected_option = options[selected_index]
                if selected_option == answer:
                    print("答對了！")
                    update_review(word_id, "good")
                else:
                    print(f"答錯了！正確答案是：{answer} ({answer_pinyin})")
                    update_review(word_id, "poor")
                    log_incorrect_answer(word_id, selected_option)
            else:
                print("無效的選擇，跳過該題。")


def fetch_for_review():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("SELECT id, word, pinyin, last_review, s_value, incorrect_count FROM vocab")
    words = cursor.fetchall()

    words_to_review = []
    for word_id, word, pinyin, last_review, s_value, incorrect_count in words:
        last_review_date = datetime.strptime(last_review.split('.')[0], "%Y-%m-%d %H:%M:%S")
        days_elapsed = (datetime.now() - last_review_date).total_seconds() / (3600 * 24)
        #print(days_elapsed)
        retention = np.exp(-days_elapsed / s_value)

        if retention < 1.02 and incorrect_count > 0:
            words_to_review.append((word_id, f"{word} ({pinyin})", retention, incorrect_count))

    conn.close()
    return sorted(words_to_review, key=lambda x: (-x[3], x[2]))


def review_words():
    words_to_review = fetch_for_review()
    if not words_to_review:
        print("目前沒有需要複習的單字。")
        return

    print("\n需要複習的單字：")
    for word_id, word, retention, incorrect_count in words_to_review:
        print(f"單字: {word} (記憶保留率: {retention:.2f}, 錯誤次數: {incorrect_count})")

    plot_all_retention_curves(words_to_review)


def plot_all_retention_curves(words_to_review):
    plt.figure(figsize=(12, 8))
    days = np.arange(0, 15, 1)
    line_styles = ['-', '--', '-.', ':']
    colors = plt.cm.tab20(np.linspace(0, 1, len(words_to_review)))

    for idx, (word_id, word, retention, _) in enumerate(words_to_review):
        values = [np.exp(-day / retention) for day in days]
        plt.plot(
            days,
            values,
            label=word,
            linestyle=line_styles[idx % len(line_styles)],
            color=colors[idx],
            alpha=0.7
        )

    plt.axhline(0.5, color='r', linestyle='--', label="50% 保留率")
    plt.title("艾賓浩斯遺忘曲線 - 多單字")
    plt.xlabel("天數")
    plt.ylabel("記憶保留率")
    plt.legend(loc="upper right", fontsize=9, ncol=2)
    plt.grid()
    plt.tight_layout()
    plt.show()


def view_incorrect_answers():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT ia.timestamp, v.word, v.pinyin, ia.selected_option
        FROM incorrect_answers ia
        JOIN vocab v ON ia.word_id = v.id
        ORDER BY ia.timestamp DESC
    """)
    records = cursor.fetchall()
    conn.close()

    if not records:
        print("目前沒有答錯記錄。")
    else:
        print("\n=== 答錯記錄 ===")
        for timestamp, word, pinyin, selected_option in records:
            print(f"{timestamp} - 單字: {word} ({pinyin})，錯誤選項: {selected_option}")


def SettingFont():
    fontManager.addfont("FontAndTool/ChineseFont.ttf")
    mpl.rc('font', family="ChineseFont")

def show_today_review_words():
    today_review_words = get_today_review_words()
    if not today_review_words:
        print("今天沒有需要複習的單字！")
        return

    print("\n=== 今日需要複習的單字 ===")
    for word_id, word, pinyin, retention, incorrect_count in today_review_words:
        print(f"單字: {word} ({pinyin}) - 記憶保留率: {retention:.2f}, 錯誤次數: {incorrect_count}")

    # 提示使用者開始複習
    print("\n建議您進行複習，可以通過選單中的『複習單字』開始測驗！")
def main_menu():
    while True:
        print("\n=== 主選單 ===")
        print("1. 開始測驗")
        print("2. 複習單字")
        print("3. 查看答錯記錄")
        print("4. 查看今日需要複習的單字")
        print("5. 離開")

        choice = input("請選擇操作：").strip()
        if choice == "1":
            start_quiz()
        elif choice == "2":
            review_words()
        elif choice == "3":
            view_incorrect_answers()
        elif choice == "4":
            show_today_review_words()
        elif choice == "5":
            print("再見！")
            break
        else:
            print("無效的選擇，請重新輸入。")

def get_today_review_words():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # 查詢所有單字的記憶保留率和錯誤次數
    cursor.execute("SELECT id, word, pinyin, last_review, s_value, incorrect_count FROM vocab")
    words = cursor.fetchall()
    conn.close()

    # 計算需要複習的單字
    today_review_words = []
    for word_id, word, pinyin, last_review, s_value, incorrect_count in words:
        last_review_date = datetime.strptime(last_review.split('.')[0], "%Y-%m-%d %H:%M:%S")
        days_elapsed = (datetime.now() - last_review_date).total_seconds() / (3600 * 24)

        retention = np.exp(-days_elapsed / s_value)
        if retention < 0.5 and incorrect_count > 0:  # 記憶保留率低於 50% 或有答錯記錄
            today_review_words.append((word_id, word, pinyin, retention, incorrect_count))

    return today_review_words
if __name__ == "__main__":
    SettingFont()
    initialize_database()
    populate_vocab_and_quiz()
    main_menu()