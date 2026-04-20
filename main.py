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
    Clean text: convert to lowercase, remove special characters, and trim extra spaces.
    """
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    # Remove punctuation and special characters (keep letters, numbers, spaces)
    text = re.sub(r'[^\w\s]', '', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    return text

print(">>> Reading data...")
# Read file, skip first noisy row, separator = ';'
raw = pd.read_csv("dataset.csv", sep=";", skiprows=1)

print(">>> Columns found:", raw.columns.tolist())

# Clean both text columns
print(">>> Cleaning text...")
raw["sentence1"] = raw["sentence1"].apply(clean_text)
raw["sentence2"] = raw["sentence2"].apply(clean_text)

# (Optional) Remove empty rows after cleaning
raw = raw[(raw["sentence1"] != "") & (raw["sentence2"] != "")]


# 3. PREPARE TRAINING DATA
X = raw[["sentence1", "sentence2"]]
y = raw["label"]

# Split train/test
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
    Encode two sentences into a single feature vector representing their relationship.
    """
    emb1 = model.encode(df["sentence1"].tolist(), convert_to_numpy=True)
    emb2 = model.encode(df["sentence2"].tolist(), convert_to_numpy=True)
    
    # Combine embeddings + absolute difference
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
print(">>> REAL-TIME SIMILARITY DETECTION SYSTEM <<<")
print("="*55)

def detect_similarity(sent1, sent2):
    """
    Real-time detector: clean, encode, predict similarity.
    """
    # 1. Clean text
    c_sent1 = clean_text(sent1)
    c_sent2 = clean_text(sent2)
    
    # 2. Encode
    emb1 = model.encode([c_sent1], convert_to_numpy=True)
    emb2 = model.encode([c_sent2], convert_to_numpy=True)
    
    # 3. Combine vectors
    combined_emb = np.concatenate([emb1, emb2, np.abs(emb1 - emb2)], axis=1)
    
    # 4. Predict
    prediction = clf.predict(combined_emb)[0]
    probabilities = clf.predict_proba(combined_emb)[0]
    
    # 5. Print result
    print(f"\n[Sentence 1]: {sent1}")
    print(f"[Sentence 2]: {sent2}")
    
    if prediction == 1:
        print(f"🚨 [DETECTED]: SENTENCES ARE SIMILAR / SAME MEANING!")
        print(f"   (AI Confidence: {probabilities[1]*100:.2f}%)")
    else:
        print(f"✅ [DETECTED]: SENTENCES ARE DIFFERENT!")
        print(f"   (AI Confidence: {probabilities[0]*100:.2f}%)")

# --- TEST SYSTEM ---

# Test case 1: similar
cau_a = "A man is playing a guitar."
cau_b = "The guy is strumming an acoustic guitar."
detect_similarity(cau_a, cau_b)

# Test case 2: different
cau_c = "I love drinking coffee in the morning."
cau_d = "The stock market is going down today."
detect_similarity(cau_c, cau_d)

# OPTIONAL manual input
# while True:
#     print("\n--- Enter 'q' to quit ---")
#     s1 = input("Sentence 1: ")
#     if s1.lower() == 'q': break
#     s2 = input("Sentence 2: ")
#     if s2.lower() == 'q': break
#     detect_similarity(s1, s2)

# ============================================================
# 8. EXPORT BULK CHECK RESULTS TO CSV
# ============================================================
print("\n" + "="*55)
print(">>> RUNNING BULK PLAGIARISM CHECK FOR ENTIRE FILE <<<")
print("="*55)

# 1. Copy dataset
export_df = raw.copy()

# 2. Encode all data
print(">>> Encoding all data...")
all_emb = encode_pairs(export_df)

# 3. Predict
print(">>> AI checking similarity...")
predictions = clf.predict(all_emb)
probabilities = clf.predict_proba(all_emb)

# 4. Add results
export_df['AI_Prediction'] = predictions
export_df['Confidence_Score(%)'] = np.max(probabilities, axis=1) * 100
export_df['Result_Text'] = export_df['AI_Prediction'].apply(
    lambda x: "Plagiarized / Similar" if x == 1 else "Different"
)

# 5. Save CSV
output_filename = "plagiarism_results.csv"
export_df.to_csv(output_filename, index=False, encoding='utf-8')

print(f"\n✅ DONE! Results saved to: '{output_filename}'")
print("You can open this file in Excel to view details.")

