# %% [markdown]
# # Аналіз помилок: де моделі помиляються і чому
#
# Працює на вже збережених файлах у папці `cache/` (нічого не треба довчати):
# * ознаки ResNet18 → логістична регресія (поновлюється за секунди),
# * HOG + колір → логістична регресія (поновлюється за ~30 с),
# * CNN з нуля → беремо збережені ваги `cnn_best.pt`.
#
# Спочатку перевіряємо, що цифри збігаються з таблицею `results_v2.csv`
# (так видно, що аналізуємо саме ті моделі).
#
# Запускай секції по черзі. Запускати треба з тієї ж папки, де лежить `cache/`.

# %%
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from skimage.color import rgb2hsv

CACHE = Path("cache")
N_CLASSES = 102
rng = np.random.default_rng(0)


def load_imgs(split):
    d = np.load(CACHE / f"flowers_{split}_128.npz")
    return d["x"], d["y"]


def load_feats(split):
    d = np.load(CACHE / f"v2_resnet18_{split}.npz")
    return d["x"], d["y"]


x_tr, y_tr = load_imgs("train")
x_va, y_va = load_imgs("val")
x_te, y_te = load_imgs("test")
print("train", x_tr.shape, "val", x_va.shape, "test", x_te.shape)
counts = np.bincount(y_te, minlength=N_CLASSES)
print(f"Фото на клас у test: від {counts.min()} до {counts.max()} (у train по 10 на клас)")

scores = {}      # назва моделі -> оцінки класів на test, форма (n_test, 102)

# %% [markdown]
# ## 1. Відновлюємо три моделі

# %%
# --- Модель 3: ResNet18 + логістична регресія ---
f_tr, fy_tr = load_feats("train")
f_va, _ = load_feats("val")
f_te, fy_te = load_feats("test")
assert np.array_equal(fy_te, y_te) and np.array_equal(fy_tr, y_tr), "порядок міток не збігається"
sc = StandardScaler().fit(f_tr)
clf = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(f_tr), y_tr)
scores["ResNet18 + LogReg"] = clf.decision_function(sc.transform(f_te))
print("ResNet18 готова")

# %%
# --- Модель 1: HOG + колір (HSV) + логістична регресія ---
def hog_features(x):
    return np.array([hog(im.mean(axis=2), orientations=9, pixels_per_cell=(16, 16),
                         cells_per_block=(2, 2)) for im in x], dtype=np.float32)


def color_features(x):
    out = []
    for im in x:
        hsv = rgb2hsv(im)
        h2d, _, _ = np.histogram2d(hsv[..., 0].ravel(), hsv[..., 1].ravel(),
                                   bins=(18, 6), range=((0, 1), (0, 1)))
        h2d = h2d.ravel() / h2d.sum()
        grid = im.reshape(4, 32, 4, 32, 3).mean(axis=(1, 3)).ravel() / 255.0
        out.append(np.concatenate([h2d, grid]))
    return np.array(out, dtype=np.float32)


H = [hog_features(a) for a in (x_tr, x_te)]
Cc = [color_features(a) for a in (x_tr, x_te)]
sh = StandardScaler().fit(H[0])
sc2 = StandardScaler().fit(Cc[0])
X = [np.hstack([sh.transform(h), 3.0 * sc2.transform(c)]) for h, c in zip(H, Cc)]
clf_h = LogisticRegression(C=0.01, max_iter=1000).fit(X[0], y_tr)
scores["HOG + колір"] = clf_h.decision_function(X[1])
print("HOG + колір готова")

# %%
# --- Модель 2: CNN із збережених ваг ---
def cnn_scores():
    import torch
    import torch.nn as nn
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    class SmallCNN(nn.Module):             # та сама архітектура, що у версії 2
        def __init__(self, n_classes=102, w=(48, 96, 192, 384)):
            super().__init__()
            def block(i, o):
                return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
            self.features = nn.Sequential(block(3, w[0]), block(w[0], w[1]),
                                          block(w[1], w[2]), block(w[2], w[3]))
            self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                      nn.Dropout(0.4), nn.Linear(w[3], n_classes))

        def forward(self, x):
            return self.head(self.features(x))

    model = SmallCNN().to(device)
    model.load_state_dict(torch.load("cnn_best.pt", map_location=device))
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x_te), 128):
            xb = np.ascontiguousarray(x_te[i:i + 128, 8:120, 8:120])
            t = torch.from_numpy(xb).to(device).permute(0, 3, 1, 2).float() / 255
            out.append(model((t - 0.5) / 0.5).float().cpu())
    return torch.cat(out).numpy()


try:
    scores["CNN з нуля"] = cnn_scores()
    print("CNN готова")
except Exception as e:                      # напр., немає файлу cnn_best.pt
    print("CNN пропущена:", repr(e))

# %% [markdown]
# ## 2. Перевірка: чи збігаються цифри з таблицею результатів?
# Очікувано (з `results_v2.csv`): HOG+колір ≈ 0.330, CNN ≈ 0.559, ResNet18 ≈ 0.831.

# %%
preds = {name: s.argmax(1) for name, s in scores.items()}
for name, p in preds.items():
    print(f"{name:<20} top-1 на test = {(p == y_te).mean():.4f}")


def per_class_acc(p):
    return np.array([(p[y_te == c] == c).mean() for c in range(N_CLASSES)])


pc = pd.DataFrame({name: per_class_acc(p) for name, p in preds.items()})
pc["n_test"] = counts
print("\nСередня точність по класах (macro) проти звичайної (micro):")
for name, p in preds.items():
    print(f"{name:<20} macro {pc[name].mean():.3f} | micro {(p == y_te).mean():.3f}")

# %% [markdown]
# ## 3. Які класи найважчі?
# Чи важкі класи одні й ті самі для всіх моделей?

# %%
order = pc[list(scores)[-1]].sort_values().index     # сортуємо за останньою моделлю (ResNet18)
best_model = "ResNet18 + LogReg"
order = pc[best_model].sort_values().index

print(f"10 найважчих класів для {best_model} (номер класу, точність, фото в test):")
print(pc.loc[order[:10], [best_model, "n_test"]].round(3).to_string())
print(f"\nКласів, які {best_model} розпізнає без жодної помилки: {(pc[best_model] == 1).sum()} із {N_CLASSES}")

print("\nКореляція складності класів між моделями (1 = важкі для однієї моделі важкі й для іншої):")
print(pc[list(scores)].corr().round(2).to_string())

fig, ax = plt.subplots(figsize=(11, 4))
for name in scores:
    ax.plot(range(N_CLASSES), pc.loc[order, name].values, marker=".", lw=1, label=name)
ax.set_xlabel(f"класи, відсортовані від найважчих до найлегших для {best_model}")
ax.set_ylabel("точність по класу")
ax.grid(alpha=.3); ax.legend(); plt.tight_layout()
plt.savefig("error_per_class.png", dpi=150); plt.show()

# %% [markdown]
# ## 4. Що з чим плутають?
# Пари «справжній клас → що передбачила модель», які трапляються найчастіше.

# %%
def top_confusions(p, k=10):
    wrong = p != y_te
    return Counter(zip(y_te[wrong], p[wrong])).most_common(k)


for name, p in preds.items():
    print(f"\n{name}: найчастіші помилки")
    rows = [{"справжній": a, "передбачено": b, "разів": n, "% класу": round(100 * n / counts[a], 1)}
            for (a, b), n in top_confusions(p)]
    print(pd.DataFrame(rows).to_string(index=False))

# %% [markdown]
# ## 5. Дивимось на самі фото
# Верхній ряд кожної пари: справжній клас, нижній: клас, який передбачила модель.
# Якщо квітки виглядають схожими, помилку легко зрозуміти.

# %%
p = preds[best_model]
pairs = [pair for pair, _ in top_confusions(p, 4)]
fig, axes = plt.subplots(len(pairs) * 2, 6, figsize=(11, 3.7 * len(pairs)))
for r, (a, b) in enumerate(pairs):
    for row_off, c, tag in ((0, a, "справжній"), (1, b, "передбачено")):
        idx = rng.choice(np.where(y_te == c)[0], 6, replace=False)
        for col, i in enumerate(idx):
            ax = axes[2 * r + row_off, col]
            ax.imshow(x_te[i]); ax.axis("off")
            if col == 0:
                ax.set_title(f"{tag}: клас {c}", fontsize=9, loc="left")
plt.tight_layout(); plt.savefig("error_confusions.png", dpi=130); plt.show()

# %%
wrong_idx = np.where(p != y_te)[0]
pick = rng.choice(wrong_idx, 12, replace=False)
fig, axes = plt.subplots(2, 6, figsize=(13, 5.5))
for ax, i in zip(axes.flat, pick):
    ax.imshow(x_te[i]); ax.axis("off")
    ax.set_title(f"справжній {y_te[i]}\nпередбачено {p[i]}", fontsize=9)
plt.suptitle(f"Випадкові помилки моделі {best_model}")
plt.tight_layout(); plt.savefig("error_examples.png", dpi=130); plt.show()

# %% [markdown]
# ## Як це описати у звіті
# * Які класи найважчі і чи збігаються вони для різних моделей (кореляція в розділі 3).
# * Які пари класів плутають, чи схожі вони візуально (колір, форма, фон).
# * Чи залежить помилка від кількості фото класу у test (див. `n_test`).
# * Чим відрізняються помилки простої моделі (що бачить лише форму й колір) від помилок ResNet.
# * Назви класів у torchvision не збережені; за потреби їх можна взяти з опису датасету
#   (наприклад, з TensorFlow Datasets) і підставити замість номерів.
