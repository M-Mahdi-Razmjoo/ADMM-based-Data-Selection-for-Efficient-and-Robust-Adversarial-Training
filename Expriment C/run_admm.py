import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

BATCH_SIZE = 256
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
EPOCHS = 100
K_PERCENTAGE = 0.5
K_SAMPLES = int(BATCH_SIZE * K_PERCENTAGE)

EPSILON, ALPHA = 8/255, 2/255
PGD_STEPS_TRAIN, PGD_STEPS_EVAL = 10, 20

SEEDS = [41, 42, 43]
NUM_CLASSES = 100
METHOD_NAME = f"ADMM_discrete_CIFAR100_k{K_SAMPLES}_bs{BATCH_SIZE}"
OUT_DIR = "results_admm_runs"
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Method: {METHOD_NAME}, K={K_SAMPLES}, BATCH_SIZE={BATCH_SIZE}, seeds={SEEDS}")

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
test_dataset  = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print("Loaded CIFAR-100:", len(train_dataset), "train,", len(test_dataset), "test")

def project_shifted_lp_ball_torch(x, p=2, eps=1e-9):
    orig_shape = x.shape
    x = x.reshape(-1)
    shift_vec = 0.5 * torch.ones_like(x, device=x.device)
    shift_x = x - shift_vec
    normp_shift = torch.norm(shift_x, p=p)
    target = (x.numel() ** (1.0 / p)) / 2.0
    if normp_shift.item() < eps:
        return shift_vec.reshape(orig_shape)
    xp = (target / (normp_shift + 1e-20)) * shift_x + shift_vec
    return xp.reshape(orig_shape)

def project_cardinality_topk_torch(x, k):
    x_flat = x.reshape(-1)
    k = int(k)
    if k <= 0:
        return torch.zeros_like(x_flat).reshape(x.shape)
    _, idx = torch.topk(x_flat, k)
    y = torch.zeros_like(x_flat)
    y[idx] = 1.0
    return y.reshape(x.shape)

def admm_selection_discrete_torch(V, d, all_params=None, warm_x0=None, device=device):
    initial_params = {
        'stop_threshold':1e-4, 'gamma_val':1.0, 'rho_change_step':5,
        'max_iters':200, 'initial_rho':50.0, 'learning_fact':1.005,
        'projection_lp':2, 'eps':1e-9
    }
    if all_params is None: all_params = initial_params
    else:
        for k_param in initial_params:
            if k_param not in all_params: all_params[k_param] = initial_params[k_param]

    V = V.reshape(-1).to(device).float()
    n = V.numel()
    k_val = int(d)

    if warm_x0 is not None and warm_x0.shape[0] == n:
        x_sol = warm_x0.reshape(-1).to(device).float().clone()
    else:
        x_sol = torch.rand(n, device=device, dtype=torch.float32)

    y1, y2 = x_sol.clone(), x_sol.clone()
    z1, z2 = torch.zeros_like(y1), torch.zeros_like(y2)
    z3 = torch.zeros(1, device=device)
    rho1 = rho2 = rho3 = float(all_params['initial_rho'])
    gamma_val = float(all_params['gamma_val'])
    max_iters = int(all_params['max_iters'])
    stop_threshold = float(all_params['stop_threshold'])
    p = float(all_params['projection_lp'])
    learning_fact = float(all_params['learning_fact'])
    eps = float(all_params['eps'])
    u = torch.ones(n, device=device, dtype=torch.float32)

    for it in range(max_iters):
        y1 = project_cardinality_topk_torch(x_sol + z1 / rho1, k_val)
        y2 = project_shifted_lp_ball_torch(x_sol + z2 / rho2, p=p)
        q = (V - z1 - z2 - z3 * u) + rho1 * y1 + rho2 * y2 + rho3 * (d * u)
        alpha, beta = float(rho1 + rho2), float(rho3)
        denom = alpha * (alpha + beta * n) + 1e-30
        factor = beta / denom
        sum_q = torch.sum(q)
        x_new = q / alpha - factor * sum_q * u
        x_sol = x_new
        z1 += gamma_val * rho1 * (x_sol - y1)
        z2 += gamma_val * rho2 * (x_sol - y2)
        z3 += gamma_val * rho3 * (torch.sum(x_sol) - float(d))
        if (it + 1) % int(all_params['rho_change_step']) == 0:
            rho1 *= learning_fact
            rho2 *= learning_fact
            rho3 *= learning_fact
            gamma_val = max(gamma_val * 0.95, 1.0)
        norm_x = torch.norm(x_sol) if torch.norm(x_sol) > 0 else torch.tensor(1.0, device=device)
        res1 = torch.norm(x_sol - y1) / (norm_x + eps)
        res2 = torch.norm(x_sol - y2) / (norm_x + eps)
        if max(res1.item(), res2.item()) <= stop_threshold:
            break
    sel = torch.nonzero(y1.reshape(-1) >= 0.5, as_tuple=False).reshape(-1)
    if sel.numel() != k_val:
        _, sel = torch.topk(x_sol, k_val)
    return sel, x_sol.detach()

class ADMM_Discrete_Solver_Torch:
    def __init__(self, n, k, admm_params=None, device=device, use_warmstart=True):
        self.n = n
        self.k = int(k)
        self.device = device
        self.admm_params = admm_params if admm_params is not None else {}
        self.last_x = None
        self.use_warmstart = use_warmstart

    def solve(self, V):
        warm = self.last_x if (self.use_warmstart and self.last_x is not None and self.last_x.shape[0] == self.n) else None
        sel, x_sol = admm_selection_discrete_torch(V, d=self.k, all_params=self.admm_params, warm_x0=warm, device=self.device)
        self.last_x = x_sol.clone()
        return sel, x_sol
     
def pgd_attack(model, images, labels, epsilon, alpha, iters):
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    orig = images.clone().detach()
    for _ in range(iters):
        images.requires_grad = True
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()
        grad = images.grad
        if grad is None:
            break 
        images = images + alpha * grad.sign()
        eta = torch.clamp(images - orig, -epsilon, epsilon)
        images = torch.clamp(orig + eta, 0.0, 1.0).detach()
    return images

def compute_overlap_indices(losses, indices_a, indices_b):
    sa = set(indices_a.cpu().numpy().tolist())
    sb = set(indices_b.cpu().numpy().tolist())
    if len(sa) == 0: return 0.0
    return len(sa & sb) / float(len(sa))

def compute_selection_stability(list_of_selected_indices, k):
    if len(list_of_selected_indices) < 2: return 1.0
    scores = []
    prev = None
    for sel in list_of_selected_indices:
        s = set(sel.cpu().numpy().tolist())
        if prev is not None:
            scores.append(len(prev & s) / float(k))
        prev = s
    return float(np.mean(scores)) if len(scores) > 0 else 0.0

def evaluate_per_class(model, data_loader, attack_fn=None, num_classes=NUM_CLASSES):
    model.eval()
    correct_per_class = np.zeros(num_classes, dtype=np.int64)
    total_per_class = np.zeros(num_classes, dtype=np.int64)
    total_correct, total_samples = 0, 0
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        if attack_fn is not None:
            images = attack_fn(model, images, labels, EPSILON, ALPHA, PGD_STEPS_EVAL)
        with torch.no_grad():
            outs = model(images)
            _, preds = torch.max(outs, 1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            for c in range(num_classes):
                mask = (labels == c)
                if mask.sum() > 0:
                    correct_per_class[c] += (preds[mask] == c).sum().item()
                    total_per_class[c] += mask.sum().item()
    overall_acc = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0
    per_class_acc = [(100.0 * correct_per_class[c] / total_per_class[c]) if total_per_class[c] > 0 else None for c in range(num_classes)]
    return overall_acc, per_class_acc
     
def train_epoch_admm_discrete_collect(model, optimizer, data_loader, admm_solver, k):
    model.train()
    overlaps = []
    selected_history = []
    for clean_images, labels in data_loader:
        clean_images, labels = clean_images.to(device), labels.to(device)
        adv_images = pgd_attack(model, clean_images, labels, EPSILON, ALPHA, PGD_STEPS_TRAIN)
        combined_images = torch.cat([clean_images, adv_images], dim=0)
        combined_labels = torch.cat([labels, labels], dim=0)
        with torch.no_grad():
            outs = model(combined_images)
            losses = F.cross_entropy(outs, combined_labels, reduction='none')
        topk_idx = torch.topk(losses, k).indices
        sel_idx, _ = admm_solver.solve(losses)
        sel_idx = sel_idx.to(device)
        overlap = compute_overlap_indices(losses, topk_idx, sel_idx)
        overlaps.append(overlap)
        selected_history.append(sel_idx.clone().cpu())
        final_images = combined_images[sel_idx]
        final_labels = combined_labels[sel_idx]
        if final_images.size(0) > 0:
            optimizer.zero_grad()
            predictions = model(final_images)
            loss = F.cross_entropy(predictions, final_labels)
            loss.backward()
            optimizer.step()
    mean_overlap = float(np.mean(overlaps)) if len(overlaps)>0 else 0.0
    stability = compute_selection_stability(selected_history, k)
    return mean_overlap, stability

def run_experiment_seed(seed, method_name=METHOD_NAME, admm_params=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    print(f"\n=== ADMM RUN seed {seed} ===")
    model = models.resnet18(weights=None, num_classes=NUM_CLASSES).to(device)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0)
    admm_solver_local = ADMM_Discrete_Solver_Torch(n=2*BATCH_SIZE, k=K_SAMPLES, admm_params=admm_params, device=device, use_warmstart=True)

    history = {
        'epoch': [], 'std_acc': [], 'robust_acc': [], 'epoch_time': [], 'cumulative_time': [],
        'overlap': [], 'selection_stability': [], 'per_class_std_acc': [], 'per_class_robust_acc': []
    }

    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        mean_overlap, stability = train_epoch_admm_discrete_collect(model, optimizer, train_loader, admm_solver_local, K_SAMPLES)
        std_acc, per_class_std = evaluate_per_class(model, test_loader, attack_fn=None, num_classes=NUM_CLASSES)
        robust_acc, per_class_rob = evaluate_per_class(model, test_loader, attack_fn=pgd_attack, num_classes=NUM_CLASSES)
        scheduler.step()
        epoch_time = time.time() - t0
        cumulative_time = time.time() - start_time
        history['epoch'].append(epoch)
        history['std_acc'].append(std_acc)
        history['robust_acc'].append(robust_acc)
        history['epoch_time'].append(epoch_time)
        history['cumulative_time'].append(cumulative_time)
        history['overlap'].append(mean_overlap)
        history['selection_stability'].append(stability)
        history['per_class_std_acc'].append(per_class_std)
        history['per_class_robust_acc'].append(per_class_rob)
        print(f"Seed {seed} Epoch {epoch}/{EPOCHS} | Std: {std_acc:.2f}% | Robust: {robust_acc:.2f}% | Overlap: {mean_overlap:.3f} | Stability: {stability:.3f} | EpochTime: {epoch_time:.1f}s")

    out = {
        'experiment_name': f"{method_name}_seed{seed}",
        'hyperparameters': {
            'batch_size': BATCH_SIZE, 'learning_rate': LEARNING_RATE, 'epochs': EPOCHS,
            'k_percentage': K_PERCENTAGE, 'k_samples': K_SAMPLES, 'epsilon': EPSILON
        },
        'training_history': history,
        'final_summary': {
            'final_std_acc': history['std_acc'][-1] if history['std_acc'] else None,
            'final_robust_acc': history['robust_acc'][-1] if history['robust_acc'] else None,
            'total_training_time': history['cumulative_time'][-1] if history['cumulative_time'] else None
        }
    }
    seed_fname = os.path.join(OUT_DIR, f"{method_name}_seed{seed}.json")
    with open(seed_fname, 'w') as f:
        json.dump(out, f, indent=4)
    print("Saved seed results to", seed_fname)
    return seed_fname, out
     
seed_files = []
seed_outputs = []
admm_params = None

for s in SEEDS:
    fname, out = run_experiment_seed(s, method_name=METHOD_NAME, admm_params=admm_params)
    seed_files.append(fname)
    seed_outputs.append(out)

min_epochs = 0 if not seed_outputs else min(len(o['training_history']['epoch']) for o in seed_outputs)
metrics = ['std_acc','robust_acc','epoch_time','cumulative_time','overlap','selection_stability']
agg_history = {'epoch': list(range(1, min_epochs+1))}
for m in metrics:
    arr = np.array([o['training_history'][m][:min_epochs] for o in seed_outputs], dtype=float)
    agg_history[f'{m}_mean'] = list(np.nanmean(arr, axis=0))
    agg_history[f'{m}_std']  = list(np.nanstd(arr, axis=0, ddof=1))

per_class_std = np.array([o['training_history']['per_class_std_acc'][:min_epochs] for o in seed_outputs], dtype=float)
per_class_rob = np.array([o['training_history']['per_class_robust_acc'][:min_epochs] for o in seed_outputs], dtype=float)

aggregate_output = {
    'experiment_name': f"{METHOD_NAME}_aggregate",
    'seed_files': seed_files,
    'hyperparameters': {
        'batch_size': BATCH_SIZE, 'learning_rate': LEARNING_RATE, 'epochs': EPOCHS,
        'k_percentage': K_PERCENTAGE, 'k_samples': K_SAMPLES, 'epsilon': EPSILON
    },
    'training_history_aggregate': agg_history,
    'per_class_std_mean_per_epoch': np.nanmean(per_class_std, axis=0).tolist() if min_epochs > 0 else [],
    'per_class_std_std_per_epoch': np.nanstd(per_class_std, axis=0, ddof=1).tolist() if min_epochs > 0 else [],
    'per_class_robust_mean_per_epoch': np.nanmean(per_class_rob, axis=0).tolist() if min_epochs > 0 else [],
    'per_class_robust_std_per_epoch': np.nanstd(per_class_rob, axis=0, ddof=1).tolist() if min_epochs > 0 else [],
    'final_summary_aggregate': {
        'final_std_acc_mean': float(np.nanmean([o['final_summary']['final_std_acc'] for o in seed_outputs if o['final_summary']['final_std_acc'] is not None])),
        'final_std_acc_std' : float(np.nanstd([o['final_summary']['final_std_acc'] for o in seed_outputs if o['final_summary']['final_std_acc'] is not None], ddof=1)),
        'final_robust_acc_mean': float(np.nanmean([o['final_summary']['final_robust_acc'] for o in seed_outputs if o['final_summary']['final_robust_acc'] is not None])),
        'final_robust_acc_std' : float(np.nanstd([o['final_summary']['final_robust_acc'] for o in seed_outputs if o['final_summary']['final_robust_acc'] is not None], ddof=1)),
        'total_training_time_mean': float(np.nanmean([o['final_summary']['total_training_time'] for o in seed_outputs if o['final_summary']['total_training_time'] is not None])),
        'total_training_time_std' : float(np.nanstd([o['final_summary']['total_training_time'] for o in seed_outputs if o['final_summary']['total_training_time'] is not None], ddof=1))
    }
}

agg_fname = os.path.join(OUT_DIR, f"{METHOD_NAME}_aggregate.json")
with open(agg_fname, 'w') as f:
    json.dump(aggregate_output, f, indent=4)

print("Saved aggregate results to", agg_fname)
print("Seed files:", seed_files)