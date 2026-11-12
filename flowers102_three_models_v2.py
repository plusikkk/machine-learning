# %% [markdown]
# # Flowers-102: три моделі, версія 2 (покращення)
#
# **Що змінилося порівняно з версією 1**
# * **Модель 1 (HOG):** кольорові ознаки у просторі HSV + «де який колір» (сітка 4×4), ширший
#   підбір параметра C і таблиця «що дає форма, що дає колір» (ablation).
# * **Модель 2 (CNN):** сильніша аугментація (випадкове обрізання з масштабуванням, поворот,
#   зміна яскравості/кольору), трохи ширша мережа, 120 епох. Найкращі ваги зберігаються в `cnn_best.pt`.
# * **Модель 3:** додано ResNet50 для порівняння з ResNet18.
# * Час тепер порівнянний: для ResNet він включає витяг ознак навіть коли вони беруться з кешу.


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
from torchvision.transforms import v2
from torchvision.datasets import Flowers102
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from skimage.color import rgb2hsv

QUICK = False
PRETRAINED = True
SEED = 42

EPOCHS = 3 if QUICK else 120
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


def fit_best_lr(Xtr, ytr, Xva, yva, Cs, max_iter=1000):
    """Логістична регресія: перебираємо C, обираємо найкращий за val."""
    best = (-1, None, None)
    for C in Cs:
        clf = LogisticRegression(C=C, max_iter=max_iter).fit(Xtr, ytr)
        acc = clf.score(Xva, yva)
        print(f"   C={C}: val acc = {acc:.3f}")
        if acc > best[0]:
            best = (acc, C, clf)
    return best

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


x_tr, y_tr = load_split("train")
x_va, y_va = load_split("val")
x_te, y_te = load_split("test")
if QUICK:
    x_te, y_te = x_te[:600], y_te[:600]
print(f"train {x_tr.shape}, val {x_va.shape}, test {x_te.shape}")

# %% [markdown]
# ## 2. Модель 1 (проста): форма (HOG) + колір (HSV) + логістична регресія
# Колір у просторі **HSV** (відтінок, насиченість) для квітів інформативніший за сирі RGB.
# Додатково беремо середній колір у сітці 4×4: так модель бачить, де в кадрі який колір

# %%
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


t0 = time.time()
H = [hog_features(a) for a in (x_tr, x_va, x_te)]
Cc = [color_features(a) for a in (x_tr, x_va, x_te)]
feat_time = time.time() - t0
print(f"Ознаки: HOG {H[0].shape[1]}, колір {Cc[0].shape[1]}  ({feat_time:.0f} с)")

sh = StandardScaler().fit(H[0]); H = [sh.transform(a) for a in H]
sc = StandardScaler().fit(Cc[0]); Cc = [sc.transform(a) for a in Cc]
W = 3.0
variants = {
    "лише форма (HOG)": H,
    "лише колір (HSV)": Cc,
    "форма + колір": [np.hstack([h, W * c]) for h, c in zip(H, Cc)],
}

Cs = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.1]
best_all = (-1, None, None, None)
ablation = []
for name, (a_tr, a_va, a_te) in variants.items():
    print(name)
    acc, C, clf = fit_best_lr(a_tr, y_tr, a_va, y_va, Cs)
    ablation.append({"ознаки": name, "val_top1": round(acc, 4),
                     "test_top1": round(float(clf.score(a_te, y_te)), 4), "C": C})
    if acc > best_all[0]:
        best_all = (acc, C, clf, a_te, name)
print("\nЩо дає форма, а що колір:")
print(pd.DataFrame(ablation).to_string(index=False))

acc, C, clf, a_te, name = best_all
add_result(f"1. HOG+HSV LogReg [{name}] (C={C})", acc, clf.decision_function(a_te), y_te,
           time.time() - t0)

# %% [markdown]
# ## 3. Модель 2 (середня): CNN з нуля, сильніша аугментація

# %%
class SmallCNN(nn.Module):
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


aug_tf = v2.Compose([
    v2.RandomResizedCrop(112, scale=(0.4, 1.0), antialias=True),
    v2.RandomHorizontalFlip(),
    v2.RandomRotation(20),
    v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.03),
])


def finish(t):                         
    return (t.to(device).float() / 255 - 0.5) / 0.5


def prep_train(xb):                    
    return finish(torch.stack([aug_tf(torch.from_numpy(im).permute(2, 0, 1)) for im in xb]))


def prep_eval(xb):                    
    return finish(torch.from_numpy(np.ascontiguousarray(xb[:, 8:120, 8:120])).permute(0, 3, 1, 2))


@torch.no_grad()
def predict_scores(model, x, bs=128):
    model.eval()
    return torch.cat([model(prep_eval(x[i:i + bs])).float().cpu() for i in range(0, len(x), bs)]).numpy()


t0 = time.time()
model = SmallCNN().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=5e-2)
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
        xb = prep_train(x_tr[idx])
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
    if (ep + 1) % 5 == 0 or ep == 0:
        print(f"епоха {ep + 1:>3}/{EPOCHS}  loss {loss.item():.3f}  val acc {val_acc:.3f}  "
              f"(найкраще {best_val:.3f}, {time.time() - t0:.0f} с)")

model.load_state_dict(best_state)
torch.save({k: v.cpu() for k, v in best_state.items()}, "cnn_best.pt")
add_result("2. CNN v2 (з нуля, сильна аугментація)", best_val,
           predict_scores(model, x_te), y_te, time.time() - t0)

plt.plot(range(1, len(history) + 1), history); plt.xlabel("епоха"); plt.ylabel("val accuracy")
plt.title("CNN v2: точність на val"); plt.grid(alpha=.3); plt.show()

# %% [markdown]
# ## 4. Модель 3 (складна): ResNet18 і ResNet50 з ImageNet + логістична регресія
# Те саме, що в версії 1, плюс більша мережа ResNet50 для порівняння.

# %%
def extract_features(arch, split):
    f = CACHE / f"v2_{arch}_{split}.npz"
    if f.exists():
        d = np.load(f)
        return d["x"], d["y"], float(d["sec"])
    tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    ds = Flowers102(root=DATA, split=split, download=True, transform=tf)
    loader = DataLoader(ds, batch_size=BATCH, num_workers=0)
    if arch == "resnet18":
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if PRETRAINED else None)
    else:
        net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if PRETRAINED else None)
    net.fc = nn.Identity()
    net = net.to(device).eval()
    t = time.time()
    xs, ys = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xs.append(net(xb.to(device)).float().cpu().numpy())
            ys.append(yb.numpy())
            if len(xs) % 25 == 0:
                print(f"  {arch} {split}: {len(xs) * BATCH} зображень...")
    sec = time.time() - t
    x, y = np.concatenate(xs), np.concatenate(ys)
    np.savez(f, x=x, y=y, sec=sec)
    return x, y, sec


for arch, label in [("resnet18", "3a. ResNet18"), ("resnet50", "3b. ResNet50")]:
    t0 = time.time()
    (r_tr, _, s1), (r_va, _, s2), (r_te, _, s3) = (extract_features(arch, s) for s in ("train", "val", "test"))
    extract_sec = s1 + s2 + s3
    if QUICK:
        r_te = r_te[:600]
    sc_r = StandardScaler().fit(r_tr)
    r_tr, r_va, r_te = sc_r.transform(r_tr), sc_r.transform(r_va), sc_r.transform(r_te)
    print(label)
    acc, C, clf = fit_best_lr(r_tr, y_tr, r_va, y_va, [0.001, 0.01, 0.1, 1.0])
    add_result(f"{label} (ImageNet) + LogReg (C={C})", acc, clf.decision_function(r_te), y_te,
               extract_sec + (time.time() - t0))

# %% [markdown]
# ## 5. Порівняння

# %%
df = pd.DataFrame(results)
df.to_csv("results_v2.csv", index=False)
print(df.to_string(index=False))

ax = df.set_index("модель")[["test_top1", "test_top5"]].plot.bar(figsize=(10, 4.5), rot=15)
ax.set_ylabel("точність на test"); ax.set_ylim(0, 1); ax.grid(axis="y", alpha=.3)
plt.title("Flowers-102: порівняння моделей (v2)"); plt.tight_layout()
plt.savefig("comparison_v2.png", dpi=150); plt.show()
