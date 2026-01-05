# ==========================================
# 0. Imports
# ==========================================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from scipy.sparse import hstack, csr_matrix

# ==========================================
# 1. Load & Sort Data
# ==========================================
df = pd.read_csv("train.csv")
df = df.sort_values("timestamp").reset_index(drop=True)

# ==========================================
# 2. Labels
# ==========================================
threshold = df["retweets_count"].median()
df["y_retweet"] = (df["retweets_count"] > threshold).astype(int)

y_stage1 = df["y_retweet"].values
y_stage2 = np.log1p(df["retweets_count"].values)

# ==========================================
# 3. Feature Engineering
# ==========================================
french_stopwords = ["le","la","les","de","des","du","un","une","et","en","à","pour","sur","dans"]

tfidf = TfidfVectorizer(
    max_features=2000,
    stop_words=french_stopwords,
    dtype=np.float32
)

X_text = tfidf.fit_transform(df["text"].astype(str))

df["hour"] = pd.to_datetime(df["timestamp"], unit="ms").dt.hour
df["log_followers"] = np.log1p(df["followers_count"])
df["engagement_ratio"] = df["favorites_count"] / (df["followers_count"] + 1)

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

# ==========================================
# 4. Temporal Cross-Validation
# ==========================================
N_FOLDS = 5
fold_size = X.shape[0] // (N_FOLDS + 1)

SEQ_LEN = 13
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

mae_log_list = []
mae_real_list = []
median_real_list = []

# ==========================================
# Helper functions
# ==========================================
def make_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

class LSTMModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.sigmoid(self.fc(h[-1]))

# ==========================================
# 5. Cross-Validation Loop
# ==========================================
for fold in range(N_FOLDS):
    print(f"\n===== Fold {fold+1}/{N_FOLDS} =====")

    train_end = (fold + 1) * fold_size
    test_end = train_end + fold_size

    X_train, X_test = X[:train_end], X[train_end:test_end]
    y1_train, y1_test = y_stage1[:train_end], y_stage1[train_end:test_end]
    y2_train, y2_test = y_stage2[:train_end], y_stage2[train_end:test_end]

    # --- SVD
    svd = TruncatedSVD(n_components=200, random_state=42)
    X_train_red = svd.fit_transform(X_train)
    X_test_red = svd.transform(X_test)

    # --- LSTM Stage 1
    X_lstm, y_lstm = make_sequences(X_train_red, y1_train, SEQ_LEN)

    model = LSTMModel(X_lstm.shape[2]).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    X_tensor = torch.tensor(X_lstm, dtype=torch.float32).to(DEVICE)
    y_tensor = torch.tensor(y_lstm, dtype=torch.float32).to(DEVICE)

    for _ in range(5):
        optimizer.zero_grad()
        preds = model(X_tensor).squeeze()
        loss = criterion(preds, y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        p_train = model(X_tensor).cpu().numpy().flatten()
        X_test_seq, _ = make_sequences(X_test_red, y1_test, SEQ_LEN)
        p_test = model(torch.tensor(X_test_seq, dtype=torch.float32).to(DEVICE)).cpu().numpy().flatten()

    # --- Stage 2 SVR
    tau = np.percentile(p_train, 68)

    mask_train = p_train >= tau
    mask_test = p_test >= tau

    X_train_svr = np.hstack([X_train_red[SEQ_LEN:][mask_train], p_train[mask_train][:, None]])
    X_test_svr = np.hstack([X_test_red[SEQ_LEN:][mask_test], p_test[mask_test][:, None]])

    y_train_svr = y2_train[SEQ_LEN:][mask_train]
    y_test_svr = y2_test[SEQ_LEN:][mask_test]

    svr = SVR(kernel="rbf", C=10, epsilon=0.01)
    svr.fit(X_train_svr, y_train_svr)

    pred_log = svr.predict(X_test_svr)

    # --- Metrics
    y_pred_real = np.expm1(pred_log)
    y_test_real = np.expm1(y_test_svr)

    mae_log = mean_absolute_error(y_test_svr, pred_log)
    mae_real = mean_absolute_error(y_test_real, y_pred_real)
    median_real = np.median(np.abs(y_test_real - y_pred_real))

    mae_log_list.append(mae_log)
    mae_real_list.append(mae_real)
    median_real_list.append(median_real)

    print(f"MAE (log): {mae_log:.4f}")
    print(f"MAE (real): {mae_real:.2f}")
    print(f"Median AE (real): {median_real:.2f}")

# ==========================================
# 6. Final CV Results
# ==========================================
print("\n===== FINAL CROSS-VALIDATION RESULTS =====")
print(f"Mean MAE (log): {np.mean(mae_log_list):.4f} ± {np.std(mae_log_list):.4f}")
print(f"Mean MAE (real): {np.mean(mae_real_list):.2f} ± {np.std(mae_real_list):.2f}")
print(f"Mean Median AE (real): {np.mean(median_real_list):.2f}")
