from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── register Chinese fonts ───────────────────────────────────────────────────
FONT_SIMHEI = "C:/Windows/Fonts/simhei.ttf"
FONT_MSYH   = "C:/Windows/Fonts/msyh.ttc"
FONT_MSYHBD = "C:/Windows/Fonts/msyhbd.ttc"

pdfmetrics.registerFont(TTFont("SimHei",   FONT_SIMHEI))
pdfmetrics.registerFont(TTFont("MSYH",     FONT_MSYH,   subfontIndex=0))
pdfmetrics.registerFont(TTFont("MSYHBold", FONT_MSYHBD, subfontIndex=0))

OUTPUT = "C:/Users/Admin/Documents/GitHub/fre-gy-7773-mlfe/ML_Code_Cheatsheet.pdf"

# ── colours ──────────────────────────────────────────────────────────────────
C_DARK   = colors.HexColor("#1e2a3a")
C_BLUE   = colors.HexColor("#2563eb")
C_LIGHT  = colors.HexColor("#eff6ff")
C_BORDER = colors.HexColor("#bfdbfe")
C_WARN   = colors.HexColor("#fef3c7")
C_WARN_B = colors.HexColor("#f59e0b")

doc = SimpleDocTemplate(
    OUTPUT, pagesize=A4,
    leftMargin=1.8*cm, rightMargin=1.8*cm,
    topMargin=2*cm,    bottomMargin=2*cm,
)
W = A4[0] - 3.6*cm

styles = getSampleStyleSheet()

def S(name, **kw):
    base = kw.pop("parent", "Normal")
    return ParagraphStyle(name, parent=styles[base], **kw)

# styles that support Chinese
title_style = S("MyTitle",  fontName="MSYHBold", fontSize=22,
                textColor=colors.white, spaceAfter=6, alignment=TA_CENTER)
sub_style   = S("MySub",    fontName="MSYH",     fontSize=11,
                textColor=colors.HexColor("#93c5fd"), alignment=TA_CENTER)
h1_style    = S("MyH1",     fontName="MSYHBold", fontSize=14,
                textColor=C_BLUE, spaceBefore=14, spaceAfter=4)
h2_style    = S("MyH2",     fontName="MSYHBold", fontSize=11,
                textColor=C_DARK, spaceBefore=8,  spaceAfter=2)
body_style  = S("MyBody",   fontName="MSYH",     fontSize=9,
                leading=14, spaceAfter=4)
code_style  = S("MyCode",   fontName="Courier",  fontSize=8,
                leading=11, textColor=colors.HexColor("#1e293b"))
tip_style   = S("MyTip",    fontName="MSYH",     fontSize=8.5,
                leading=13, textColor=colors.HexColor("#92400e"))
tbl_k_style = S("TblK",     fontName="MSYHBold", fontSize=8.5)
tbl_v_style = S("TblV",     fontName="MSYH",     fontSize=8.5)

# ── helpers ──────────────────────────────────────────────────────────────────
def code_block(text):
    p = Paragraph(
        text.strip("\n").replace("\n","<br/>").replace(" ","&nbsp;"),
        code_style
    )
    t = Table([[p]], colWidths=[W])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), C_LIGHT),
        ("BOX",           (0,0),(-1,-1), 0.5, C_BORDER),
        ("LEFTPADDING",   (0,0),(-1,-1), 8),
        ("RIGHTPADDING",  (0,0),(-1,-1), 8),
        ("TOPPADDING",    (0,0),(-1,-1), 6),
        ("BOTTOMPADDING", (0,0),(-1,-1), 6),
    ]))
    return t

def tip_block(text):
    p = Paragraph(text, tip_style)
    t = Table([[p]], colWidths=[W])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), C_WARN),
        ("BOX",           (0,0),(-1,-1), 0.8, C_WARN_B),
        ("LEFTPADDING",   (0,0),(-1,-1), 8),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
    ]))
    return t

def banner(title_text, sub_text):
    t = Table([[Paragraph(title_text, title_style)],
               [Paragraph(sub_text,   sub_style)]],
              colWidths=[W])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), C_DARK),
        ("TOPPADDING",    (0,0),(-1,-1), 14),
        ("BOTTOMPADDING", (0,-1),(-1,-1), 14),
        ("LEFTPADDING",   (0,0),(-1,-1), 10),
    ]))
    return t

def section(title):
    return [HRFlowable(width=W, thickness=1.5, color=C_BLUE, spaceAfter=2),
            Paragraph(title, h1_style)]

def subsection(title):
    return [Paragraph(title, h2_style)]

def body(text):
    return Paragraph(text, body_style)

# ── story ────────────────────────────────────────────────────────────────────
story = []

story.append(banner(
    "ML for Finance — Code Cheat Sheet",
    "FRE-GY 7773 | 考试速查手册"
))
story.append(Spacer(1, 12))

# ══════════════════════════════════════════════════════════════════════════════
# 1. 线性回归
# ══════════════════════════════════════════════════════════════════════════════
story += section("1. 线性回归 (Linear Regression)")

story += subsection("1-A  sklearn — 最常用")
story.append(code_block("""\
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)          # X must be 2D: shape (n_samples, n_features)
model.intercept_         # beta_0 (intercept)
model.coef_              # beta_1, beta_2, ... (coefficients)
model.predict(X_new)     # predictions"""))
story.append(Spacer(1,6))

story += subsection("1-B  statsmodels — 含完整统计输出")
story.append(code_block("""\
import statsmodels.api as sm

X_sm     = sm.add_constant(X)         # add intercept column to X
results  = sm.OLS(y, X_sm).fit()
print(results.summary())              # R2, t/F stats, p-value, etc.
results.params                        # coefficients (incl. intercept)
results.rsquared                      # R^2
results.resid                         # residuals
sm.graphics.tsa.plot_acf(results.resid, lags=20)  # ACF of residuals"""))
story.append(Spacer(1,6))

story += subsection("1-C  正规方程 (Normal Equation)")
story.append(code_block("""\
# Method 1: pseudo-inverse (recommended, handles singular matrices)
beta = np.linalg.pinv(X_b) @ y         # X_b already has intercept column

# Method 2: solve (faster than inv)
beta = np.linalg.solve(X_b.T @ X_b, X_b.T @ y)

# Add intercept column manually
X_b = np.column_stack([np.ones(n), X])
# or via sklearn:
from sklearn.preprocessing import add_dummy_feature
X_b = add_dummy_feature(X)"""))
story.append(Spacer(1,6))

story += subsection("1-D  1D 手动公式")
story.append(code_block("""\
beta_1 = np.cov(X, Y, bias=True)[0, 1] / np.var(X, ddof=0)
beta_0 = np.mean(Y) - beta_1 * np.mean(X)"""))
story.append(Spacer(1,6))

story += subsection("1-E  R-squared (决定系数)")
story.append(code_block("""\
ss_res = np.sum((y - y_hat) ** 2)      # residual sum of squares
ss_tot = np.sum((y - y.mean()) ** 2)   # total sum of squares
r2     = 1 - ss_res / ss_tot

# or directly with sklearn:
r2 = model.score(X_test, y_test)"""))
story.append(Spacer(1,8))

# ══════════════════════════════════════════════════════════════════════════════
# 2. 数据预处理
# ══════════════════════════════════════════════════════════════════════════════
story += section("2. 数据预处理")

story += subsection("2-A  Train / Test 分割")
story.append(code_block("""\
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)"""))
story.append(Spacer(1,6))

story += subsection("2-B  标准化 (StandardScaler)")
story.append(code_block("""\
from sklearn.preprocessing import StandardScaler

scaler     = StandardScaler().fit(X_train)   # fit ONLY on training set!
X_train_s  = scaler.transform(X_train)
X_test_s   = scaler.transform(X_test)        # use train stats for test

# equivalent to: (X - scaler.mean_) / scaler.scale_"""))
story.append(Spacer(1,4))
story.append(tip_block(
    "<b>关键原则：</b> StandardScaler 的 fit 只能用训练集，"
    "transform 分别应用于训练集和测试集，避免数据泄露 (data leakage)。"
    "Ridge / Lasso 必须先标准化，否则正则化效果不公平。"
))
story.append(Spacer(1,8))

# ══════════════════════════════════════════════════════════════════════════════
# 3. Ridge & Lasso
# ══════════════════════════════════════════════════════════════════════════════
story += section("3. Ridge & Lasso 正则化")

story += subsection("3-A  基本用法")
story.append(code_block("""\
from sklearn.linear_model import Ridge, Lasso

ridge = Ridge(alpha=1.0).fit(X_train_s, y_train)
lasso = Lasso(alpha=0.1).fit(X_train_s, y_train)

ridge.coef_       # Ridge: coefficients shrink but stay non-zero (L2)
lasso.coef_       # Lasso: some coefficients become exactly 0 (L1, sparse)
ridge.intercept_"""))
story.append(Spacer(1,6))

story += subsection("3-B  交叉验证选 alpha — 推荐用 Pipeline")
story.append(code_block("""\
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
import numpy as np

cv     = KFold(n_splits=5, shuffle=True, random_state=42)
alphas = np.logspace(-4, 1, 20)

pipeline = Pipeline([
    ("scaler",    StandardScaler()),
    ("regressor", RidgeCV(alphas=alphas, cv=cv)),   # or LassoCV
])
pipeline.fit(X_train, y_train)

best_alpha = pipeline.named_steps["regressor"].alpha_
test_mse   = mean_squared_error(y_test, pipeline.predict(X_test))"""))
story.append(Spacer(1,6))

story += subsection("3-C  多项式回归 (Polynomial Regression)")
story.append(code_block("""\
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    StandardScaler(),
    Ridge(alpha=1.0),
)
model.fit(X, y)
model.predict(X_new)"""))
story.append(Spacer(1,6))

story += subsection("3-D  学习曲线 (Bias-Variance 诊断)")
story.append(code_block("""\
from sklearn.model_selection import learning_curve

train_sizes, train_scores, valid_scores = learning_curve(
    model, X, y,
    train_sizes=np.linspace(0.01, 1.0, 40),
    cv=5,
    scoring="neg_root_mean_squared_error",
)
train_errors = -train_scores.mean(axis=1)
valid_errors = -valid_scores.mean(axis=1)
# Two curves close together  -> high bias (underfitting)
# Large gap between curves   -> high variance (overfitting)"""))
story.append(Spacer(1,8))

# ══════════════════════════════════════════════════════════════════════════════
# 4. 评估指标
# ══════════════════════════════════════════════════════════════════════════════
story += section("4. 评估指标")

story.append(code_block("""\
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

# MSE / RMSE
mse  = mean_squared_error(y_test, y_hat)

# Cross-validated RMSE
cv_rmse = -cross_val_score(
    model, X, y, cv=5, scoring="neg_root_mean_squared_error"
).mean()

# R^2 (closer to 1 = better)
r2 = model.score(X_test, y_test)"""))
story.append(Spacer(1,8))

# ══════════════════════════════════════════════════════════════════════════════
# 5. 分类
# ══════════════════════════════════════════════════════════════════════════════
story += section("5. 分类 (Classification)")

story += subsection("5-A  Logistic Regression")
story.append(code_block("""\
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X_train, y_train)
clf.predict(X_test)          # class labels
clf.predict_proba(X_test)    # class probabilities
clf.score(X_test, y_test)    # accuracy
clf.coef_                    # coefficients
clf.intercept_"""))
story.append(Spacer(1,6))

story += subsection("5-B  混淆矩阵与分类报告")
story.append(code_block("""\
from sklearn.metrics import confusion_matrix, classification_report

y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
# Layout:   [[TN,  FP],
#            [FN,  TP]]

print(classification_report(y_test, y_pred))  # precision/recall/F1"""))
story.append(Spacer(1,6))

story += subsection("5-C  Precision / Recall / F1")
story.append(code_block("""\
from sklearn.metrics import precision_score, recall_score, f1_score

# Precision = TP / (TP + FP)  -- minimise false alarms (e.g. spam filter)
precision_score(y, y_pred)

# Recall    = TP / (TP + FN)  -- minimise missed detections (e.g. disease)
recall_score(y, y_pred)

# F1 = harmonic mean of Precision and Recall
f1_score(y, y_pred)"""))
story.append(Spacer(1,6))

story += subsection("5-D  ROC 曲线 & AUC")
story.append(code_block("""\
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_predict

# Get decision scores (not class labels)
y_scores = cross_val_predict(clf, X, y, cv=3, method="decision_function")

fpr, tpr, thresholds = roc_curve(y, y_scores)
auc = roc_auc_score(y, y_scores)   # closer to 1 = better"""))
story.append(Spacer(1,6))

story += subsection("5-E  交叉验证评估")
story.append(code_block("""\
from sklearn.model_selection import cross_val_score, cross_val_predict

scores = cross_val_score(clf, X, y, cv=10, scoring="accuracy")
print(scores.mean(), scores.std())

# Out-of-fold predictions (for confusion matrix — avoids test leakage)
y_pred_cv = cross_val_predict(clf, X_train, y_train, cv=3)
cm = confusion_matrix(y_train, y_pred_cv)"""))
story.append(Spacer(1,8))

# ══════════════════════════════════════════════════════════════════════════════
# 6. 梯度下降
# ══════════════════════════════════════════════════════════════════════════════
story += section("6. 梯度下降 (Gradient Descent)")

story += subsection("6-A  Batch GD — 每次用全量数据")
story.append(code_block("""\
eta      = 0.1        # learning rate
n_epochs = 1000
theta    = rng.standard_normal(2)   # random initialisation

for epoch in range(n_epochs):
    gradients = (2 / m) * X_b.T @ (X_b @ theta - y)
    theta     = theta - eta * gradients"""))
story.append(Spacer(1,6))

story += subsection("6-B  Stochastic GD (SGD) — 每次用 1 个样本")
story.append(code_block("""\
def learning_schedule(t, t0=5, t1=50):
    return t0 / (t + t1)              # decaying learning rate

theta = rng.standard_normal((2, 1))
for epoch in range(n_epochs):
    for i in range(m):
        idx = rng.integers(m)
        xi, yi  = X_b[idx:idx+1], y[idx:idx+1]
        grad    = xi.T @ (xi @ theta - yi)   # NOTE: do NOT divide by m
        eta     = learning_schedule(epoch * m + i)
        theta   = theta - eta * grad"""))
story.append(Spacer(1,6))

story += subsection("6-C  Mini-batch GD — 每次用小批量")
story.append(code_block("""\
batch_size = 20
for epoch in range(n_epochs):
    idx_perm  = rng.permutation(m)            # shuffle each epoch
    X_s, y_s  = X_b[idx_perm], y[idx_perm]
    for i in range(0, m, batch_size):
        xi    = X_s[i:i+batch_size]
        yi    = y_s[i:i+batch_size]
        grad  = xi.T @ (xi @ theta - yi) / batch_size
        theta = theta - eta * grad"""))
story.append(Spacer(1,8))

# ══════════════════════════════════════════════════════════════════════════════
# 7. MLE
# ══════════════════════════════════════════════════════════════════════════════
story += section("7. 最大似然估计 (MLE)")

story.append(code_block("""\
# Univariate Normal MLE
mean_mle = x.mean()                        # unbiased & MLE
var_mle  = np.var(x, ddof=0)              # MLE (biased)
var_unb  = np.var(x, ddof=1)              # unbiased estimate

# Multivariate Normal MLE
mean_mle = X.mean(axis=0)                  # shape (p,)
cov_mle  = np.cov(X, rowvar=False, ddof=0) # MLE (biased)
# equivalent to:
cov_mle  = (X - mean_mle).T @ (X - mean_mle) / n

# Bernoulli MLE (coin toss)
p_mle = tosses.sum() / len(tosses)"""))
story.append(Spacer(1,8))

# ══════════════════════════════════════════════════════════════════════════════
# 8. 金融应用
# ══════════════════════════════════════════════════════════════════════════════
story += section("8. 金融应用 (CAPM / yfinance)")

story.append(code_block("""\
import yfinance as yf
import statsmodels.api as sm

# Download price data
data = yf.download(["AAPL", "^GSPC", "^IRX"],
                   start="2015-01-01", auto_adjust=True)["Close"]

# Daily returns
ret       = data.pct_change().dropna()
rf_daily  = data["^IRX"] / 100 / 252       # annualised -> daily

# Excess returns
asset_excess = ret["AAPL"]  - rf_daily
mkt_excess   = ret["^GSPC"] - rf_daily

# CAPM OLS:  R_i - R_f = alpha + beta*(R_m - R_f) + eps
X_sm    = sm.add_constant(mkt_excess)
results = sm.OLS(asset_excess, X_sm).fit()
alpha, beta = results.params
print(results.summary())

# Rolling beta (252-day window)
beta_roll = (
    ret["AAPL"].rolling(252).cov(ret["^GSPC"])
    / ret["^GSPC"].rolling(252).var()
)"""))
story.append(Spacer(1,8))

# ══════════════════════════════════════════════════════════════════════════════
# 9. 速记要点
# ══════════════════════════════════════════════════════════════════════════════
story += section("9. 速记要点")

tips = [
    ("X.reshape(-1, 1)",
     "1D 数组转 2D，sklearn 要求输入为 2D"),
    ("sm.add_constant(X)",
     "statsmodels 在 X 左侧添加截距列"),
    ("np.linalg.pinv vs np.linalg.inv",
     "pinv 更稳定，能处理奇异矩阵，推荐优先使用"),
    ("ddof=0 vs ddof=1",
     "ddof=0 是 MLE 有偏估计；ddof=1 是无偏估计"),
    ("混淆矩阵布局",
     "cm[0,0]=TN, cm[0,1]=FP, cm[1,0]=FN, cm[1,1]=TP"),
    ("Precision = TP/(TP+FP)",
     "关注假正例 — 垃圾邮件过滤器场景"),
    ("Recall = TP/(TP+FN)",
     "关注假负例 — 疾病漏诊场景"),
    ("Ridge (L2)",
     "系数缩小但不为 0，适合多重共线性"),
    ("Lasso (L1)",
     "系数可变为 0，实现特征选择 (稀疏解)"),
    ("learning_curve",
     "两曲线靠近 = 欠拟合 (高 bias)；gap 大 = 过拟合 (高 variance)"),
    ("cross_val_predict",
     "获取 out-of-fold 预测，用于混淆矩阵，避免测试集泄露"),
    ("Pipeline",
     "Scaler + Model 合并为一体，自动防止数据泄露"),
]

rows = []
for k, v in tips:
    rows.append([
        Paragraph(k, tbl_k_style),
        Paragraph(v, tbl_v_style)
    ])

tbl = Table(rows, colWidths=[W*0.38, W*0.62])
tbl.setStyle(TableStyle([
    ("ROWBACKGROUNDS", (0,0),(-1,-1),
     [colors.white, colors.HexColor("#f0f9ff")]),
    ("BOX",       (0,0),(-1,-1), 0.5, C_BORDER),
    ("INNERGRID", (0,0),(-1,-1), 0.3, C_BORDER),
    ("TOPPADDING",    (0,0),(-1,-1), 5),
    ("BOTTOMPADDING", (0,0),(-1,-1), 5),
    ("LEFTPADDING",   (0,0),(-1,-1), 7),
]))
story.append(tbl)

# ── build ─────────────────────────────────────────────────────────────────────
doc.build(story)
print(f"PDF saved -> {OUTPUT}")
