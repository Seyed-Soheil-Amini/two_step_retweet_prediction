import pandas as pd
import numpy as np

from sklearn.svm import LinearSVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from scipy.sparse import hstack, csr_matrix
from sklearn.decomposition import TruncatedSVD

# =========================
# 1. Load Data
# =========================
df = pd.read_csv("train.csv") 

# =========================
# 2. Create Binary Label (Stage 1)
# =========================
threshold = df["retweets_count"].median()
df["y_retweet"] = (df["retweets_count"] > threshold).astype(int)

# =========================
# 3. Feature Engineering
# =========================

french_stopwords = [
    "le","la","les","de","des","du","un","une","et","en","à","pour","sur","dans"
]

tfidf = TfidfVectorizer(
    max_features=2000,      # ↓ خیلی مهم
    ngram_range=(1, 1),     # unigram فقط
    stop_words=french_stopwords,
    dtype=np.float32        # ↓ نصف RAM
)

X_text = tfidf.fit_transform(df["text"].astype(str))

df["hour"] = pd.to_datetime(df["timestamp"], unit="ms").dt.hour
df["log_followers"] = np.log1p(df["followers_count"])
df["engagement_ratio"] = df["favorites_count"] / (df["followers_count"] + 1)

# --- User & Meta Features
meta_features = [
    "favorites_count",
    "followers_count",
    "statuses_count",
    "friends_count",
    "log_followers",
    "engagement_ratio",
    "verified"
]

X_meta = csr_matrix(df[meta_features].fillna(0).values, dtype=np.float32)
X_time = csr_matrix(df[["hour"]].values, dtype=np.float32)

X = hstack([X_text, X_meta, X_time])

# =========================
# 4. Train/Test Split
# =========================
y_stage1 = df["y_retweet"].values
y_stage2 = np.log1p(df["retweets_count"].values)

X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    X, y_stage1, y_stage2, test_size=0.2, random_state=42
)


# =========================
# 5. Scaling
# =========================
svd = TruncatedSVD(n_components=200, random_state=42)

X_train_red = svd.fit_transform(X_train)
X_test_red = svd.transform(X_test)

# =========================
# 6. Stage 1: Dynamic Naive Bayes (Probabilistic)
# =========================
nb = GaussianNB()
nb.fit(X_train_red, y1_train)

# Probability of retweet
p_train = nb.predict_proba(X_train_red)[:, 1]
p_test = nb.predict_proba(X_test_red)[:, 1]

# =========================
# 7. Stage 2: SVR (Regression)
# =========================

# Use only tweets predicted as retweetable
train_tau = np.percentile(p_train, 68)
# train_tau = np.mean(p_train) + 0.5*np.std(p_train)
test_tau = np.percentile(p_test, 68)
# test_tau = np.mean(p_test) + 0.5*np.std(p_test)

mask_train = p_train >= train_tau
mask_test = p_test >= test_tau

X_train_svr = np.hstack([
    X_train_red[mask_train],
    (p_train[mask_train] ** 2)[:, None]
])

X_test_svr = np.hstack([
    X_test_red[mask_test],
    (p_test[mask_test] ** 2)[:, None]
])


y_train_svr = y2_train[mask_train]
y_test_svr = y2_test[mask_test]


svr = LinearSVR(C=1.0, random_state=42)
svr.fit(X_train_svr, y2_train[mask_train])

svr = SVR(kernel="rbf", C=10, epsilon=0.01)
svr.fit(X_train_svr, y_train_svr)
pred = svr.predict(X_test_svr)
mae = mean_absolute_error(y_test_svr, pred)

y_pred_real = np.expm1(pred)
y_test_real = np.expm1(y_test_svr)

#این خطا نمی تواند معیار خوبی باشد زیرا چند توییت با تعداد بازنشر بسیار بالا میانگین خطا را زیر سوال می برند 
# مثلا 130 خطا برای توییت های با تعداد بازنشر 30000 بسیار نا چیز است ولی برای توییت های نرمال که تعداد بازنشر بین 100 تا 1000 دارند این عدد مهم است.
# ولی نمی توان به این عدد استناد کرد
print("MAE real:", mean_absolute_error(y_test_real, y_pred_real))
# این خطا برای 50 درصد داده ها این تعدا خطا بازنشر دارد
print("Median AE:", np.median(np.abs(y_test_real - y_pred_real)))

print("MAE (Stage 2 Retweet Count Prediction):", mae)