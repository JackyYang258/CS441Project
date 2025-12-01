from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def load_iclr_dataset(years=(2022, 2023)):
    """Load and merge ICLR datasets for the specified years."""
    data_dir = Path("ICLR_Dataset")
    frames = []
    for year in years:
        path = data_dir / f"ICLR{year}" / "dataset.tsv"
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset file: {path}")
        yearly_df = pd.read_csv(path, sep="\t")
        yearly_df["year"] = year
        frames.append(yearly_df)
    merged = pd.concat(frames, ignore_index=True)
    merged["text"] = merged["title"].astype(str) + " " + merged["abstract"].astype(str)
    merged["label"] = merged["decision"].apply(lambda x: 1 if "Accept" in str(x) else 0)
    return merged


# 1. Load Data (ICLR 2022 + 2023)
df = load_iclr_dataset()

# Split Training / Validation / Testing
X_train_val, X_test, y_train_val, y_test = train_test_split(
    df["text"], df["label"], test_size=0.1, random_state=42, stratify=df["label"]
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.1111, random_state=42, stratify=y_train_val
)

# 3. Vectorization (Turn text into numbers)
# max_features=5000 means we only look at the top 5000 most frequent words
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)

# 4. Model Training (Logistic Regression)
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train_vec, y_train)

# 5. Evaluation
val_predictions = model.predict(X_val_vec)
test_predictions = model.predict(X_test_vec)

print("Validation Accuracy:", accuracy_score(y_val, val_predictions))
print("\nValidation Report:\n", classification_report(y_val, val_predictions))

print("\nTest Accuracy:", accuracy_score(y_test, test_predictions))
print("\nTest Report:\n", classification_report(y_test, test_predictions))

# 6. Interpretability 
feature_names = vectorizer.get_feature_names_out()
coefs = model.coef_[0]
# Sort words by their weight
sorted_indices = coefs.argsort()

print("\nTop 10 words predicting REJECT:", [feature_names[i] for i in sorted_indices[:10]])
print("Top 10 words predicting ACCEPT:", [feature_names[i] for i in sorted_indices[-10:]])
