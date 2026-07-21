# ics_federated_experiment.py (FIXED for Windows multiprocessing & pickling)
import os, time, math, random, json, multiprocessing
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset, TensorDataset

# ----------------- Multiprocessing start method for Windows -----------------
try:
    multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass
# RemappedSubset: wraps a dataset and remaps label values to 0..(K-1)
class RemappedSubset(torch.utils.data.Dataset):
    def __init__(self, ds, indices, label_map=None):
        """
        ds: original dataset (indexable)
        indices: list of indices to include
        label_map: dict {original_label: new_label} — will be auto-built if None
        """
        self.ds = Subset(ds, indices)
        # If underlying ds uses (img, label) pairs, derive label_map if not provided
        if label_map is None:
            # collect labels in this subset
            labels = []
            for _, y in self.ds:
                try:
                    labels.append(int(y))
                except:
                    labels.append(int(y.item()))
            uniq = sorted(set(labels))
            self.label_map = {old: new for new, old in enumerate(uniq)}
        else:
            self.label_map = label_map

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, y = self.ds[idx]
        try:
            oy = int(y)
        except:
            oy = int(y.item())
        new_label = self.label_map[oy]
        return x, new_label

# ----------------- CONFIG -----------------
CONFIG = {
    "datasets": ["MNIST", "EMNIST", "CIFAR10"],   # full run
    "num_clients": 15,       
    "rounds": 20,             # recommended for full run
    "local_epochs": 1,
    "batch_size": 64,
    "attack_ratio": 0.2,
    "attack_type": "targeted_label_flip",
    "target_label": 1,
    "S_per_class": 20,        # higher, better ICS estimation
    "mi_fgsm_steps": 10,
    "mi_fgsm_alpha": 0.5,
    "mi_fgsm_eps": 4/255,
    "ics_weights": [0.45, 0.25, 0.15, 0.10, 0.05],
    "ema_alpha": 0.3,
    "tau": 5.0,
    "seed": 42,
    "save_dir": "C:\\Users\\laiba\\ICS_results\\final_full_run",
    "use_rep": False,          # keep off for CIFAR to avoid slow training
    "use_amp": True,
    "verbose": True
}

# ------------------------------------------

# reproducibility
random.seed(CONFIG["seed"]); np.random.seed(CONFIG["seed"]); torch.manual_seed(CONFIG["seed"])

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, "torch:", torch.__version__, "cuda:", torch.version.cuda)

os.makedirs(CONFIG["save_dir"], exist_ok=True)

# ----------------- Model -----------------
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32*8*8, 128)
        self.fc2 = nn.Linear(128, num_classes)
    def forward(self, x, return_feat=False):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        feat = F.relu(self.fc1(x))
        out = self.fc2(feat)
        if return_feat:
            return out, feat
        return out

# ----------------- Top-level PoisonedWrapper (picklable) -----------------
class PoisonedWrapper(torch.utils.data.Dataset):
    def __init__(self, ds, attack_type, target_label, classes, strength=1.0):
        self.ds = ds
        self.attack_type = attack_type
        self.target_label = target_label
        self.classes = classes
        self.strength = strength
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        x, y = self.ds[idx]
        # handle y types
        try:
            yy = int(y)
        except:
            yy = int(y.item())
        if self.attack_type == "random_flip":
            if random.random() < self.strength:
                new = random.choice([i for i in range(self.classes) if i != yy])
                return x, new
        elif self.attack_type == "targeted_label_flip":
            if random.random() < self.strength:
                return x, self.target_label
        return x, yy

def apply_poison_to_client_dataset(client_dataset, attack_type, target_label, classes, strength=1.0):
    return PoisonedWrapper(client_dataset, attack_type, target_label, classes, strength)

# ----------------- Dataset helpers -----------------
def get_dataset(name, num_samples_per_class=200):
    """
    Returns:
      ds_small_mapped : RemappedSubset where labels are remapped to 0..C-1
      C               : number of unique classes in the original dataset
      in_ch           : input channels (1 or 3)
      label_map       : dict mapping original_label -> new_label
    """
    transform = T.Compose([T.ToTensor(), T.Resize(32)])
    try:
        if name == "MNIST":
            ds = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
            in_ch = 1
        elif name == "EMNIST":
            ds = torchvision.datasets.EMNIST(root="./data", split='balanced', train=True, download=True, transform=transform)
            in_ch = 1
        elif name == "CIFAR10":
            ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
            in_ch = 3
        else:
            raise ValueError("Unknown dataset")

        # compute all labels and unique classes from the full dataset
        labels_all = np.array([int(x[1]) for x in ds])
        uniq_all = np.unique(labels_all)
        C = int(len(uniq_all))

        # build a global label_map for the dataset so labels map to 0..C-1
        label_map = {int(old): int(new) for new, old in enumerate(sorted(uniq_all.tolist()))}

        # choose up to num_samples_per_class per original label to form a small subset
        indices = []
        for orig_label in sorted(uniq_all.tolist()):
            idx = np.where(labels_all == orig_label)[0][:num_samples_per_class]
            indices.extend(idx.tolist())

        # Wrap with RemappedSubset using the precomputed label_map
        ds_small_mapped = RemappedSubset(ds, indices=indices, label_map=label_map)

        return ds_small_mapped, C, in_ch, label_map

    except Exception as e:
        print("Dataset load failed, synthesizing small dataset:", e)
        C = 47 if name == "EMNIST" else 10
        in_ch = 3
        imgs = torch.randn(C * num_samples_per_class, in_ch, 32, 32)
        labels = torch.arange(C).repeat_interleave(num_samples_per_class)
        # build a simple dataset and label_map identity
        ds_synth = TensorDataset(imgs, labels)
        label_map = {int(i): int(i) for i in range(C)}
        ds_small_mapped = RemappedSubset(ds_synth, indices=list(range(len(ds_synth))), label_map=label_map)
        return ds_small_mapped, C, in_ch, label_map

def partition_label_skew(dataset, num_clients, classes):
    labels = np.array([int(y) for _, y in dataset])
    idx_by_label = {c: np.where(labels==c)[0].tolist() for c in range(classes)}
    clients_idx = [[] for _ in range(num_clients)]
    for i in range(num_clients):
        dominant = np.random.choice(range(classes), size=2, replace=False).tolist()
        for c in range(classes):
            take = 3
            if c in dominant: take = 15
            available = idx_by_label.get(c, [])
            selected = available[:take]
            idx_by_label[c] = available[take:]
            clients_idx[i].extend(selected)
    leftover=[]
    for c,lst in idx_by_label.items():
        leftover.extend(lst)
    for i, idx in enumerate(leftover):
        clients_idx[i % num_clients].append(idx)
    client_datasets = [Subset(dataset, ids) for ids in clients_idx]
    return client_datasets

# ----------------- Adversarial MI-FGSM (server-side) -----------------
def mi_fgsm_generate(model, images, target_labels, steps=10, alpha=1.0, eps=8/255):
    model.eval()
    images = images.clone().detach().to(device)
    orig = images.clone().detach()
    momentum = torch.zeros_like(images, device=device)
    images.requires_grad = True
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(steps):
        outputs = model(images)
        loss = loss_fn(outputs, target_labels.to(device))
        loss.backward()
        grad = images.grad.data
        grad_norm = grad / (torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True) + 1e-8)
        momentum = 0.9 * momentum + grad_norm
        images = images + alpha * momentum.sign()
        images = torch.max(torch.min(images, orig + eps), orig - eps).detach()
        images.requires_grad = True
    return images.detach()

# ----------------- ICS computation -----------------
def softmax_probs_gpu(model, x):
    with torch.no_grad():
        logits = model(x.to(device))
        probs = F.softmax(logits, dim=1)
    return probs.cpu()

def entropy_of_probs(probs):
    eps=1e-8
    ent = - (probs * torch.log(probs+eps)).sum(dim=1)
    return ent

def compute_ics_matrix(client_models, adversarial_by_class, classes, use_rep=False):
    num_clients = len(client_models)
    OC = np.zeros((num_clients, classes))
    CONF = np.zeros((num_clients, classes))
    ENT = np.zeros((num_clients, classes))
    CONS = np.zeros((num_clients, classes))
    REP = np.zeros((num_clients, classes)) if use_rep else None

    for i, cm in enumerate(client_models):
        cm.to(device).eval()
        for c in range(classes):
            A = adversarial_by_class[c].to(device)
            probs = softmax_probs_gpu(cm, A)  # CPU tensor returned
            preds = probs.argmax(dim=1).numpy()
            OC[i,c] = (preds == c).mean()
            CONF[i,c] = probs[:, c].mean().item()
            ENT[i,c] = entropy_of_probs(probs).mean().item() / math.log(max(classes,2))
            # consistency
            A_pert = (A + torch.randn_like(A, device=device) * 0.01).clamp(0,1)
            probs_pert = softmax_probs_gpu(cm, A_pert)
            preds_pert = probs_pert.argmax(dim=1).numpy()
            CONS[i,c] = (preds == preds_pert).mean()
            # rep (optional)
            if use_rep:
                _, feat = cm(A.to(device), return_feat=True)
                feat = feat.detach().cpu().numpy()
                REP[i,c] = np.mean(np.linalg.norm(feat, axis=1))
    w = CONFIG["ics_weights"]
    if use_rep:
        w1,w2,w3,w4,w5 = w
        ICS = w1*OC + w2*CONF + w3*(1-ENT) + w4*CONS + w5*REP
    else:
        w1,w2,w3,w4 = w[0], w[1], w[2], w[3]
        ICS = w1*OC + w2*CONF + w3*(1-ENT) + w4*CONS
    return {"OC":OC,"CONF":CONF,"ENT":ENT,"CONS":CONS,"REP":REP,"ICS":ICS}

# ----------------- Utility: convert ICS -> per-class weights -----------------
def ics_to_alpha(ICS_matrix, tau=5.0):
    exp_mat = np.exp(ICS_matrix * tau)
    alpha = exp_mat / (exp_mat.sum(axis=0, keepdims=True) + 1e-12)
    alpha_mean = alpha.mean(axis=1)
    return alpha, alpha_mean

# ----------------- Aggregation: selective_aggregate (per-class final-layer + shared layers) -----------------
# ----------------- Aggregation: selective_aggregate (fixed dtype handling) -----------------
def selective_aggregate(client_state_dicts, alpha_per_client_per_class, model_template):
    """
    client_state_dicts: list of state_dict() from each client (torch tensors)
    alpha_per_client_per_class: numpy array shape (num_clients, num_classes) with weights
    model_template: a model instance whose fc2 layer defines class dimension
    Returns: aggregated state_dict (same keys as client_state_dicts[0])
    """
    num_clients = len(client_state_dicts)
    C = model_template.fc2.out_features

    # Prepare float accumulators (use float32 for accumulation)
    agg_state = {}
    orig_dtypes = {}
    for k, v in client_state_dicts[0].items():
        orig_dtypes[k] = v.dtype
        agg_state[k] = torch.zeros_like(v, dtype=torch.float32)

    # Aggregate final linear (fc2) per-class (row-wise) using alpha_{i,c}
    for c in range(C):
        for i in range(num_clients):
            # convert client's tensor to float for accumulation
            w_row = client_state_dicts[i]['fc2.weight'][c].to(torch.float32) * float(alpha_per_client_per_class[i, c])
            agg_state['fc2.weight'][c] += w_row
        # aggregate bias for class c
        for i in range(num_clients):
            agg_state['fc2.bias'][c] += client_state_dicts[i]['fc2.bias'][c].to(torch.float32) * float(alpha_per_client_per_class[i, c])

    # Shared layers: use per-client mean weight across classes
    alpha_mean = alpha_per_client_per_class.mean(axis=1)  # numpy array
    for name in list(agg_state.keys()):
        if name.startswith('fc2'):
            continue
        for i in range(num_clients):
            agg_state[name] += client_state_dicts[i][name].to(torch.float32) * float(alpha_mean[i])

    # Convert aggregated tensors back to original dtypes
    final_state = {}
    for k, acc in agg_state.items():
        orig_dtype = orig_dtypes[k]
        # if original dtype is integer, round before casting to preserve reasonable values
        if orig_dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            final_state[k] = acc.round().to(dtype=orig_dtype)
        else:
            final_state[k] = acc.to(dtype=orig_dtype)
    return final_state

# ----------------- Train & eval helpers -----------------
def local_train(model, dataset, epochs, batch_size, lr=0.01, use_amp=False, in_ch=3):
    model = deepcopy(model)
    model.to(device).train()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type=='cuda') else None
    for ep in range(epochs):
        for xb,yb in loader:
            if in_ch==3 and xb.dim()==3:
                xb = xb.unsqueeze(1).repeat(1,3,1,1)
            if in_ch==3 and xb.size(1)==1:
                xb = xb.repeat(1,3,1,1)
            xb = xb.to(device, non_blocking=True); yb = yb.to(device)
            opt.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    logits = model(xb)
                    loss = F.cross_entropy(logits, yb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
                loss.backward(); opt.step()
    return model.state_dict()

def evaluate_global(model, test_loader, in_ch=3):
    model.to(device).eval()
    correct=0; total=0
    class_correct = defaultdict(int); class_total=defaultdict(int)
    with torch.no_grad():
        for xb,yb in test_loader:
            if in_ch==3 and xb.size(1)==1:
                xb = xb.repeat(1,3,1,1)
            xb=xb.to(device); yb=yb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1)
            correct += (preds==yb).sum().item()
            total += yb.size(0)
            for t,p in zip(yb.cpu().numpy(), preds.cpu().numpy()):
                class_total[int(t)] += 1
                if int(t)==int(p): class_correct[int(t)] += 1
    global_acc = correct/total if total>0 else 0.0
    per_class_acc = {c:(class_correct.get(c,0)/class_total.get(c,1)) for c in class_total.keys()}
    return global_acc, per_class_acc

# ----------------- Main experiment loop -----------------
def run_experiment_for_dataset(name):
    # get remapped training set and label_map
    ds_train, C, in_ch, label_map = get_dataset(name, num_samples_per_class=200)

    # Prepare test dataset and remap its labels using the same label_map when possible
    transform_test = T.Compose([T.ToTensor(), T.Resize(32)])
    if name == "MNIST":
        test_raw = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform_test)
    elif name == "EMNIST":
        test_raw = torchvision.datasets.EMNIST(root="./data", split='balanced', train=False, download=True, transform=transform_test)
    elif name == "CIFAR10":
        test_raw = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    else:
        test_raw = None

    if test_raw is not None:
        # Build test indices = all indices of test_raw (we remap labels using label_map if original labels are present)
        test_indices = list(range(len(test_raw)))
        # If test set contains labels not present in train label_map, we will build a fallback mapping
        # but prefer to reuse the training label_map for consistency
        # We'll create a safe label_map_test where missing labels are ignored or mapped if present.
        test_labels_all = np.array([int(x[1]) for x in test_raw])
        missing = set(np.unique(test_labels_all.tolist())) - set(label_map.keys())
        if len(missing) > 0:
            # Extend label_map to include any missing labels (append at the end)
            start = max(label_map.values()) + 1
            for j, orig in enumerate(sorted(list(missing))):
                label_map[int(orig)] = start + j
            C = max(C, max(label_map.values()) + 1)
        test_ds = RemappedSubset(test_raw, indices=test_indices, label_map=label_map)
    else:
        test_ds = None

    # Now classes should equal C (mapped label space)
    classes = int(C)
    client_datasets = partition_label_skew(ds_train, CONFIG["num_clients"], classes)

    num_malicious = max(1, int(CONFIG["num_clients"] * CONFIG["attack_ratio"]))
    malicious_indices = set(random.sample(range(CONFIG["num_clients"]), num_malicious))

    global_model = SimpleCNN(in_channels=in_ch, num_classes=classes).to(device)
    global_state = deepcopy(global_model.state_dict())

    logs = {"round": [], "global_acc": [], "ASR": []}
    ICS_history = []
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True) if test_ds else None
    ema_ics = np.zeros((CONFIG["num_clients"], classes))

    for r in range(1, CONFIG["rounds"] + 1):
        if CONFIG["verbose"]:
            print(f"[{name}] Round {r}/{CONFIG['rounds']}")
        gm = SimpleCNN(in_channels=in_ch, num_classes=classes).to(device)
        gm.load_state_dict(global_state)
        adversarial_by_class = {}

        # safe: load all data (remapped) with num_workers=0
        all_loader = DataLoader(ds_train, batch_size=len(ds_train), shuffle=False, num_workers=0)
        all_x, all_y = next(iter(all_loader))
        # all_y is already remapped to 0..classes-1

        for c in range(classes):
            idxs = np.where(np.array(all_y) != c)[0]
            if len(idxs) == 0:
                seed = torch.randn(CONFIG["S_per_class"], in_ch, 32, 32).to(device)
                targets = torch.full((CONFIG["S_per_class"],), c, dtype=torch.long).to(device)
                adv = mi_fgsm_generate(gm, seed, targets, steps=CONFIG["mi_fgsm_steps"],
                                       alpha=CONFIG["mi_fgsm_alpha"], eps=CONFIG["mi_fgsm_eps"])
                adversarial_by_class[c] = adv.cpu()
            else:
                pick = idxs[:CONFIG["S_per_class"]]
                seed = all_x[pick].clone()
                if in_ch == 3 and seed.size(1) == 1:
                    seed = seed.repeat(1, 3, 1, 1)
                targets = torch.full((seed.size(0),), c, dtype=torch.long)
                adv = mi_fgsm_generate(gm, seed.to(device), targets.to(device), steps=CONFIG["mi_fgsm_steps"],
                                       alpha=CONFIG["mi_fgsm_alpha"], eps=CONFIG["mi_fgsm_eps"])
                adversarial_by_class[c] = adv.cpu()

        client_models = []
        client_states = []
        for i in range(CONFIG["num_clients"]):
            ds_i = client_datasets[i]
            if i in malicious_indices:
                ds_i = apply_poison_to_client_dataset(ds_i, CONFIG["attack_type"], CONFIG["target_label"], classes, strength=1.0)
            cm = SimpleCNN(in_channels=in_ch, num_classes=classes).to(device)
            cm.load_state_dict(global_state)
            state = local_train(cm, ds_i, CONFIG["local_epochs"], CONFIG["batch_size"], use_amp=CONFIG["use_amp"], in_ch=in_ch)
            client_models.append(cm)
            client_states.append(state)

        ics_comps = compute_ics_matrix(client_models, adversarial_by_class, classes, use_rep=CONFIG["use_rep"])
        ICS = ics_comps["ICS"]
        ema_ics = CONFIG["ema_alpha"] * ICS + (1 - CONFIG["ema_alpha"]) * ema_ics
        ICS_history.append(deepcopy(ema_ics))
        alpha, alpha_mean = ics_to_alpha(ema_ics, tau=CONFIG["tau"])

        agg_state = selective_aggregate(client_states, alpha, SimpleCNN(in_channels=in_ch, num_classes=classes))
        global_state = agg_state
        global_model.load_state_dict(global_state)

        if test_loader is not None:
            gacc, per_c = evaluate_global(global_model, test_loader, in_ch=in_ch)
        else:
            gacc, per_c = 0.0, {}
        logs["round"].append(r)
        logs["global_acc"].append(gacc)

        if CONFIG["attack_type"].startswith("targeted"):
            total_other = 0; succ = 0
            for xb, yb in test_loader:
                if in_ch == 3 and xb.size(1) == 1:
                    xb = xb.repeat(1, 3, 1, 1)
                xb, yb = xb.to(device), yb.to(device)
                preds = global_model(xb).argmax(dim=1)
                for t, p in zip(yb.cpu().numpy(), preds.cpu().numpy()):
                    if int(t) != CONFIG["target_label"]:
                        total_other += 1
                        if int(p) == CONFIG["target_label"]:
                            succ += 1
            asr = succ / total_other if total_other > 0 else 0.0
        else:
            asr = 0.0
        logs["ASR"].append(asr)

        if CONFIG["verbose"]:
            print(f"Round {r}: global_acc={gacc:.4f}, ASR={asr:.4f}, avg_ICS={np.mean(ema_ics):.4f}")

    df = pd.DataFrame(logs)
    fname = os.path.join(CONFIG["save_dir"], f"{name}_logs.csv")
    df.to_csv(fname, index=False)
    last_ics = ICS_history[-1]
    plt.figure(figsize=(10, 3))
    plt.imshow(last_ics, aspect='auto')
    plt.title(f"ICS heatmap final — {name}")
    plt.xlabel("Class"); plt.ylabel("Client"); plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["save_dir"], f"{name}_ICS_heatmap.png"), dpi=300)
    plt.close()
    plt.figure()
    plt.plot(df["round"], df["global_acc"], marker='o')
    plt.title(f"Global accuracy vs rounds — {name}")
    plt.xlabel("Round"); plt.ylabel("Global Accuracy")
    plt.grid(True); plt.savefig(os.path.join(CONFIG["save_dir"], f"{name}_convergence.png"), dpi=300); plt.close()

    print(f"Saved logs to {fname} and plots to {CONFIG['save_dir']}")
    return df, ICS_history

# ----------------- Run full experiments for datasets in CONFIG -----------------
def main():
    results_summary = {}
    for ds in CONFIG["datasets"]:
        print("Starting experiment for dataset:", ds)
        df, ics_hist = run_experiment_for_dataset(ds)
        results_summary[ds] = {"logs":df, "ics_history":ics_hist}
    with open(os.path.join(CONFIG["save_dir"], "summary_meta.json"), "w") as f:
        json.dump({"config":CONFIG}, f, indent=2)
    print("All experiments done. Results in", CONFIG["save_dir"])

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
