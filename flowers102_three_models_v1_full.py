# %% [markdown]
# # Flowers-102: три моделі різної складності (для MacBook M2 Air, 8 ГБ)
#
# Задача: за фото квітки визначити, який це із 102 видів.
#
# | № | Модель | Ідея |
# |---|---|---|
# | 1 | **Проста** | Ручні ознаки (HOG + кольорова гістограма) + логістична регресія |
# | 2 | **Середня** | Невелика згорткова мережа (CNN), навчена з нуля, з аугментацією |
# | 3 | **Складна** | Попередньо навчений ResNet18 (transfer learning): ознаки з ImageNet + логістична регресія |
#
# Правила чесного порівняння: усі моделі **навчаються на `train`**, гіперпараметри/кращу епоху
# обираємо за **`val`**, а **`test` рахуємо один раз** наприкінці.
#
# %%
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import Flowers102
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog

QUICK = False
PRETRAINED = True 
SEED = 42

EPOCHS = 3 if QUICK else 50
BATCH = 32

np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Пристрій:", device)

DATA = Path("data")
CACHE = Path("cache")
CACHE.mkdir(exist_ok=True)
results = []


def top_k_acc(scores, y, k):
    """Частка прикладів, де правильний клас серед k найкращих."""
    topk = np.argsort(-scores, axis=1)[:, :k]
    return float((topk == y[:, None]).any(axis=1).mean())


def add_result(name, val_acc, test_scores, y_test, seconds):
    results.append({
        "модель": name,
        "val_top1": round(val_acc, 4),
        "test_top1": round(top_k_acc(test_scores, y_test, 1), 4),
        "test_top5": round(top_k_acc(test_scores, y_test, 5), 4),
        "час, с": round(seconds, 1),
    })
    print(pd.DataFrame(results).to_string(index=False))

# %% [markdown]
# ## 1. Дані

# %%
def load_split(split, size=128):
    f = CACHE / f"flowers_{split}_{size}.npz"
    if f.exists():
        d = np.load(f)
        return d["x"], d["y"]
    ds = Flowers102(root=DATA, split=split, download=True)
    xs, ys = [], []
    for img, y in ds:
        xs.append(np.asarray(img.convert("RGB").resize((size, size))))
        ys.append(y)
    x, y = np.stack(xs), np.array(ys)
    np.savez(f, x=x, y=y)
    return x, y


t0 = time.time()
x_tr, y_tr = load_split("train")
x_va, y_va = load_split("val")
x_te, y_te = load_split("test")
if QUICK:
    x_te, y_te = x_te[:600], y_te[:600]
print(f"train {x_tr.shape}, val {x_va.shape}, test {x_te.shape}  ({time.time() - t0:.0f} с)")
print("Класів:", len(np.unique(y_tr)), "| зображень на клас у train:", np.bincount(y_tr).min(), "-", np.bincount(y_tr).max())

# Приклади зображень
fig, axes = plt.subplots(2, 6, figsize=(14, 5))
for ax, i in zip(axes.flat, np.random.choice(len(x_tr), 12, replace=False)):
    ax.imshow(x_tr[i]); ax.set_title(f"клас {y_tr[i]}"); ax.axis("off")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 2. Модель 1 (проста): HOG + кольорова гістограма + логістична регресія
# **HOG** описує форму (напрямки країв), **гістограма кольору** описує колір пелюсток.

# %%
def handcrafted_features(x):
    out = []
    for im in x:
        gray = im.mean(axis=2)
        h = hog(gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2))
        col = np.concatenate([np.histogram(im[..., c], bins=16, range=(0, 255))[0] for c in range(3)]).astype(float)
        col /= col.sum()
        out.append(np.concatenate([h, col]))
    return np.array(out, dtype=np.float32)


t0 = time.time()
f_tr, f_va, f_te = (handcrafted_features(a) for a in (x_tr, x_va, x_te))
scaler = StandardScaler().fit(f_tr)
f_tr, f_va, f_te = scaler.transform(f_tr), scaler.transform(f_va), scaler.transform(f_te)
print("Ознак на зображення:", f_tr.shape[1], f"({time.time() - t0:.0f} с)")

best = (-1, None, None)
for C in [0.001, 0.01, 0.1]:
    clf = LogisticRegression(C=C, max_iter=500).fit(f_tr, y_tr)
    acc = clf.score(f_va, y_va)
    print(f"C={C}: val acc = {acc:.3f}")
    if acc > best[0]:
        best = (acc, C, clf)
val_acc, C, clf = best
add_result(f"1. HOG + LogReg (C={C})", val_acc, clf.decision_function(f_te), y_te, time.time() - t0)

# %% [markdown]
# ## 3. Модель 2 (середня): власна CNN з нуля
# Чотири згорткові блоки (Conv → BatchNorm → ReLU → MaxPool). Аугментація: випадкове
# обрізання та віддзеркалення.
# Кращу епоху обираємо за `val`.

# %%
class SmallCNN(nn.Module):
    def __init__(self, n_classes=102):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.3), nn.Linear(256, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def augment(xb):                      
    out = np.empty((len(xb), 112, 112, 3), np.uint8)
    for i in range(len(xb)):
        a, b = np.random.randint(0, 17, 2)
        im = xb[i, a:a + 112, b:b + 112]
        out[i] = im[:, ::-1] if np.random.rand() < 0.5 else im
    return out


def center(xb):
    return np.ascontiguousarray(xb[:, 8:120, 8:120])


def to_tensor(xb):
    t = torch.from_numpy(xb).to(device).permute(0, 3, 1, 2).float() / 255
    return (t - 0.5) / 0.5


@torch.no_grad()
def predict_scores(model, x, bs=128):
    model.eval()
    out = [model(to_tensor(center(x[i:i + bs]))).float().cpu() for i in range(0, len(x), bs)]
    return torch.cat(out).numpy()


t0 = time.time()
model = SmallCNN().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
steps = EPOCHS * int(np.ceil(len(x_tr) / BATCH))
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

best_val, best_state, history = 0.0, None, []
for ep in range(EPOCHS):
    model.train()
    perm = np.random.permutation(len(x_tr))
    for i in range(0, len(perm), BATCH):
        idx = perm[i:i + BATCH]
        if len(idx) < 2:
            continue
        xb = to_tensor(augment(x_tr[idx]))
        yb = torch.from_numpy(y_tr[idx]).to(device)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        sched.step()
    val_acc = float((predict_scores(model, x_va).argmax(1) == y_va).mean())
    history.append(val_acc)
    if val_acc > best_val:
        best_val = val_acc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    print(f"епоха {ep + 1:>2}/{EPOCHS}  loss {loss.item():.3f}  val acc {val_acc:.3f}  ({time.time() - t0:.0f} с)")

model.load_state_dict(best_state)
add_result("2. Мала CNN (з нуля)", best_val, predict_scores(model, x_te), y_te, time.time() - t0)

plt.plot(range(1, len(history) + 1), history); plt.xlabel("епоха"); plt.ylabel("val accuracy")
plt.title("CNN: точність на val"); plt.grid(alpha=.3); plt.show()

# %% [markdown]
# ## 4. Модель 3 (складна): ResNet18 з ImageNet + логістична регресія (transfer learning)

# %%
def extract_resnet_features(split):
    f = CACHE / f"resnet18_{split}.npz"
    if f.exists():
        d = np.load(f)
        return d["x"], d["y"]
    tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    ds = Flowers102(root=DATA, split=split, download=True, transform=tf)
    loader = DataLoader(ds, batch_size=BATCH, num_workers=0) 
    net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if PRETRAINED else None)
    net.fc = nn.Identity()
    net = net.to(device).eval()
    xs, ys = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xs.append(net(xb.to(device)).float().cpu().numpy())
            ys.append(yb.numpy())
            if len(xs) % 20 == 0:
                print(f"  {split}: {len(xs) * BATCH} зображень...")
    x, y = np.concatenate(xs), np.concatenate(ys)
    np.savez(f, x=x, y=y)
    return x, y


t0 = time.time()
r_tr, _ = extract_resnet_features("train")
r_va, _ = extract_resnet_features("val")
r_te, _ = extract_resnet_features("test")
if QUICK:
    r_te = r_te[:600]
print(f"Ознаки ResNet18 готові ({time.time() - t0:.0f} с)")

sc = StandardScaler().fit(r_tr)
r_tr, r_va, r_te = sc.transform(r_tr), sc.transform(r_va), sc.transform(r_te)
best = (-1, None, None)
for C in [0.001, 0.01, 0.1, 1.0]:
    clf = LogisticRegression(C=C, max_iter=1000).fit(r_tr, y_tr)
    acc = clf.score(r_va, y_va)
    print(f"C={C}: val acc = {acc:.3f}")
    if acc > best[0]:
        best = (acc, C, clf)
val_acc, C, clf = best
add_result(f"3. ResNet18 (ImageNet) + LogReg (C={C})", val_acc, clf.decision_function(r_te), y_te, time.time() - t0)

# %% [markdown]
# ## 5. Порівняння

# %%
df = pd.DataFrame(results)
df.to_csv("results_v1_full.csv", index=False)
print(df.to_string(index=False))

ax = df.set_index("модель")[["test_top1", "test_top5"]].plot.bar(figsize=(9, 4), rot=15)
ax.set_ylabel("точність на test"); ax.set_ylim(0, 1); ax.grid(axis="y", alpha=.3)
plt.title("Flowers-102: порівняння моделей"); plt.tight_layout(); plt.savefig("comparison_v1_full.png", dpi=150); plt.show()
