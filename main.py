import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

# 1. TEXT CLEANING FUNCTION
def clean_text(text):
    """
    Hàm làm sạch văn bản: chuyển thành chữ thường, xóa ký tự đặc biệt và khoảng trắng thừa.
    """
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    # Loại bỏ các dấu câu, ký tự đặc biệt (giữ lại chữ cái, số và khoảng trắng)
    text = re.sub(r'[^\w\s]', '', text)
    # Gom nhiều khoảng trắng liên tiếp thành 1 khoảng trắng
    text = re.sub(r'\s+', ' ', text)
    return text

print(">>> Reading data...")
# Đọc file, bỏ qua dòng rác đầu tiên (skiprows=1) và dùng phân cách ';'
raw = pd.read_csv("dataset.csv", sep=";", skiprows=1)

print(">>> Columns found:", raw.columns.tolist())

# Xử lý text cho cả 2 cột (sentence1 và sentence2)
print(">>> Cleaning text...")
raw["sentence1"] = raw["sentence1"].apply(clean_text)
raw["sentence2"] = raw["sentence2"].apply(clean_text)

# (Tùy chọn) Loại bỏ các dòng bị rỗng hoàn toàn sau khi clean
raw = raw[(raw["sentence1"] != "") & (raw["sentence2"] != "")]


# 3. PREPARE TRAINING DATA
X = raw[["sentence1", "sentence2"]]
y = raw["label"]

# Chia dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n>>> Training samples: {len(X_train)}")
print(f">>> Test samples: {len(X_test)}")

# 4. ENCODING USING SENTENCE TRANSFORMERS
print("\n>>> Loading SentenceTransformer model (This might take a moment)...")
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

def encode_pairs(df):
    """
    Hàm mã hóa 2 câu văn thành 1 vector duy nhất đại diện cho mối quan hệ của chúng.
    """
    emb1 = model.encode(df["sentence1"].tolist(), convert_to_numpy=True)
    emb2 = model.encode(df["sentence2"].tolist(), convert_to_numpy=True)
    
    # Nối 2 vector và kèm theo độ chênh lệch tuyệt đối giữa chúng.
    # Đây là phương pháp tối ưu hơn hẳn so với việc chỉ cộng (emb1 + emb2)
    combined_emb = np.concatenate([emb1, emb2, np.abs(emb1 - emb2)], axis=1)
    
    return combined_emb

print("\n>>> Encoding training data...")
X_train_emb = encode_pairs(X_train)

print(">>> Encoding testing data...")
X_test_emb = encode_pairs(X_test)

# 5. TRAIN THE CLASSIFIER
print("\n>>> Training Logistic Regression classifier...")
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_emb, y_train)

print(">>> Model training completed!")

# 6. EVALUATION
print("\n>>> Evaluating model on test data...")
y_pred = clf.predict(X_test_emb)

acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\n>>> Accuracy: {acc:.4f}")
print(">>> Classification Report:\n")
print(report)

# 7. REAL-TIME DETECTION SYSTEM
print("\n" + "="*55)
print(">>> HỆ THỐNG PHÁT HIỆN TƯƠNG ĐỒNG (DETECTION SYSTEM) <<<")
print("="*55)

def detect_similarity(sent1, sent2):
    """
    Hệ thống nhận đầu vào là 2 câu mới tinh, tự động làm sạch, 
    mã hóa và đưa ra kết quả phát hiện.
    """
    # 1. Làm sạch văn bản như lúc Train
    c_sent1 = clean_text(sent1)
    c_sent2 = clean_text(sent2)
    
    # 2. Mã hóa bằng SentenceTransformer
    emb1 = model.encode([c_sent1], convert_to_numpy=True)
    emb2 = model.encode([c_sent2], convert_to_numpy=True)
    
    # 3. Nối vector y hệt như công thức lúc nãy
    combined_emb = np.concatenate([emb1, emb2, np.abs(emb1 - emb2)], axis=1)
    
    # 4. Dự đoán bằng mô hình đã học
    prediction = clf.predict(combined_emb)[0]
    probabilities = clf.predict_proba(combined_emb)[0] # Lấy % độ tự tin
    
    # 5. In kết quả
    print(f"\n[Câu 1]: {sent1}")
    print(f"[Câu 2]: {sent2}")
    
    if prediction == 1:
        print(f"🚨 [PHÁT HIỆN]: HAI CÂU TƯƠNG ĐỒNG / TRÙNG Ý NGHĨA!")
        print(f"   (Độ tin cậy của AI: {probabilities[1]*100:.2f}%)")
    else:
        print(f"✅ [PHÁT HIỆN]: HAI CÂU KHÁC NHAU!")
        print(f"   (Độ tin cậy của AI: {probabilities[0]*100:.2f}%)")

# --- CHẠY THỬ HỆ THỐNG ---

# Thử nghiệm 1: Hai câu có vẻ giống nhau
cau_a = "A man is playing a guitar."
cau_b = "The guy is strumming an acoustic guitar."
detect_similarity(cau_a, cau_b)

# Thử nghiệm 2: Hai câu hoàn toàn khác nhau
cau_c = "I love drinking coffee in the morning."
cau_d = "The stock market is going down today."
detect_similarity(cau_c, cau_d)

# (TÙY CHỌN) Bỏ comment đoạn dưới đây nếu bạn muốn TỰ GÕ câu từ bàn phím

# while True:
#     print("\n--- Nhập 'q' để thoát ---")
#     s1 = input("Nhập câu 1: ")
#     if s1.lower() == 'q': break
#     s2 = input("Nhập câu 2: ")
#     if s2.lower() == 'q': break
#     detect_similarity(s1, s2)
# ============================================================
# 8. XUẤT KẾT QUẢ KIỂM TRA ĐẠO VĂN RA FILE CSV
# ============================================================
print("\n" + "="*55)
print(">>> CHẠY KIỂM TRA ĐẠO VĂN HÀNG LOẠT CHO TOÀN BỘ FILE <<<")
print("="*55)

# 1. Tạo một bản sao của dữ liệu gốc để lưu kết quả
export_df = raw.copy()

# 2. Lấy vector của toàn bộ dữ liệu gốc (đã được làm sạch ở bước trên)
print(">>> Đang mã hóa toàn bộ dữ liệu...")
all_emb = encode_pairs(export_df)

# 3. Cho AI dự đoán toàn bộ
print(">>> AI đang kiểm tra trùng lặp...")
predictions = clf.predict(all_emb)
probabilities = clf.predict_proba(all_emb)

# 4. Thêm cột kết quả vào bảng dữ liệu
export_df['AI_Prediction'] = predictions
export_df['Confidence_Score(%)'] = np.max(probabilities, axis=1) * 100
export_df['Result_Text'] = export_df['AI_Prediction'].apply(lambda x: "Đạo văn/Trùng lặp" if x == 1 else "Khác nhau")

# 5. Lưu ra file CSV mới
output_filename = "plagiarism_results.csv"
# Lưu bằng dấu tab (\t) hoặc dấu phẩy (,) tùy bạn. Ở đây mình dùng dấu phẩy cho chuẩn quốc tế
export_df.to_csv(output_filename, index=False, encoding='utf-8')

print(f"\n✅ ĐÃ XONG! Toàn bộ kết quả đã được lưu vào file: '{output_filename}'")
print("Bạn có thể mở file này bằng Excel để xem chi tiết.")
#Visualization
import matplotlib.pyplot as plt
import seaborn as sns