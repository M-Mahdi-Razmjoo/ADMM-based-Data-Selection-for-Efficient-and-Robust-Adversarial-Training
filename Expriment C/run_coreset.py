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

SEEDS = [42, 43, 44]
NUM_CLASSES = 100
METHOD_NAME_GRADMATCH = f"GradMatch_CIFAR100_k{K_SAMPLES}_bs{BATCH_SIZE}"
OUT_DIR = "results_gradmatch_runs"
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Method: {METHOD_NAME_GRADMATCH}, K={K_SAMPLES}, BATCH_SIZE={BATCH_SIZE}, seeds={SEEDS}")

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print("Loaded CIFAR-100:", len(train_dataset), "train,", len(test_dataset), "test")

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

def OrthogonalMP_REG_Parallel(A, b, tol=1E-4, nnz=None, positive=False, lam=1, device="cpu"):
    AT = torch.transpose(A, 0, 1)
    d, n = A.shape
    if nnz is None: nnz = n
    x = torch.zeros(n, device=device)
    resid = b.detach().clone()
    normb = b.norm().item() if b.norm().item() != 0 else 1.0
    indices = []
    A_i = None
    x_i = torch.zeros(0, device=device)
    for i in range(nnz):
        if resid.norm().item() / normb < tol: break
        projections = torch.matmul(AT, resid)
        abs_proj = torch.abs(projections)
        for idx in indices:
            abs_proj[idx] = -1.0
        index = int(torch.argmax(abs_proj).item())
        if index in indices: break
        indices.append(index)
        if len(indices) == 1:
            A_i = A[:, index].view(1, -1)
        else:
            A_i = torch.cat((A_i, A[:, index].view(1, -1)), dim=0)
        temp = torch.matmul(A_i, torch.transpose(A_i, 0, 1)) + lam * torch.eye(A_i.shape[0], device=device)
        rhs = torch.matmul(A_i, b).view(-1, 1)
        try:
            x_i_sol = torch.linalg.solve(temp, rhs)
        except Exception:
            try:
                x_i_sol, _ = torch.lstsq(rhs, temp)
                x_i_sol = x_i_sol[:temp.shape[0]]
            except Exception:
                x_i_sol = torch.matmul(torch.pinverse(temp), rhs)
        x_i = x_i_sol.view(-1)
        resid = b - torch.matmul(torch.transpose(A_i, 0, 1), x_i).view(-1)
    for i, index in enumerate(indices):
        if i < x_i.numel():
            x[index] = x_i[i]
    return x

def get_last_layer_embedding(net, x):
    out = net.conv1(x)
    out = net.bn1(out)
    out = net.relu(out)
    out = net.maxpool(out)
    out = net.layer1(out)
    out = net.layer2(out)
    out = net.layer3(out)
    out = net.layer4(out)
    out = net.avgpool(out)
    return out.view(out.size(0), -1)

def train_epoch_gradmatch_collect(model, optimizer, data_loader, k):
    model.train()
    selected_history = []
    for clean_images, labels in data_loader:
        clean_images, labels = clean_images.to(device), labels.to(device)
        adv_images = pgd_attack(model, clean_images, labels, EPSILON, ALPHA, PGD_STEPS_TRAIN)
        combined_images = torch.cat([clean_images, adv_images], dim=0)
        combined_labels = torch.cat([labels, labels], dim=0)
        embedding_dim = model.fc.in_features
        num_classes = model.fc.out_features
        features = get_last_layer_embedding(model, combined_images)
        outputs = model.fc(features)
        loss_per_sample = F.cross_entropy(outputs, combined_labels, reduction='none')
        loss_sum = loss_per_sample.sum()
        l0_grads = torch.autograd.grad(loss_sum, outputs, retain_graph=True)[0]
        l1_grads = (l0_grads.unsqueeze(2) * features.unsqueeze(1)).flatten(start_dim=1)
        batch_gradients = torch.cat((l1_grads, l0_grads), dim=1).detach()
        sum_of_gradients = torch.sum(batch_gradients, dim=0)
        weights = OrthogonalMP_REG_Parallel(torch.transpose(batch_gradients, 0, 1), sum_of_gradients, nnz=k, device=device)
        sel_idx = torch.nonzero(weights, as_tuple=False).squeeze(1)
        if sel_idx.dim() == 0: sel_idx = sel_idx.unsqueeze(0)
        if sel_idx.numel() == 0: sel_idx = torch.tensor([], dtype=torch.long, device=device)
        if sel_idx.numel() < k:
            num_to_add = int(k - sel_idx.numel())
            all_indices = list(range(batch_gradients.shape[0]))
            selected_list = [] if sel_idx.numel() == 0 else sel_idx.cpu().tolist()
            remaining_indices = list(set(all_indices) - set(selected_list))
            if len(remaining_indices) > 0:
                padding = np.random.choice(remaining_indices, min(num_to_add, len(remaining_indices)), replace=False)
                padding = torch.tensor(padding, device=device, dtype=torch.long)
                sel_idx = torch.cat([sel_idx, padding])
        sel_idx = torch.unique(sel_idx).long().to(device)
        if sel_idx.numel() > k: sel_idx = sel_idx[:k]
        elif sel_idx.numel() < k:
            needed = k - sel_idx.numel()
            all_indices = list(range(batch_gradients.shape[0]))
            selected_list = sel_idx.cpu().tolist()
            remaining_indices = list(set(all_indices) - set(selected_list))
            if len(remaining_indices) > 0:
                padding = np.random.choice(remaining_indices, min(needed, len(remaining_indices)), replace=False)
                padding = torch.tensor(padding, device=device, dtype=torch.long)
                sel_idx = torch.cat([sel_idx, padding])
            sel_idx = torch.unique(sel_idx).long().to(device)

        selected_history.append(sel_idx.clone().cpu())
        final_images = combined_images[sel_idx]
        final_labels = combined_labels[sel_idx]
        if final_images.size(0) > 0:
            optimizer.zero_grad()
            predictions = model(final_images)
            loss = F.cross_entropy(predictions, final_labels)
            loss.backward()
            optimizer.step()
    stability = compute_selection_stability(selected_history, k)
    return 1.0, stability

def run_experiment_seed_gradmatch(seed, method_name=METHOD_NAME_GRADMATCH):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    print(f"\n=== GRADMATCH RUN seed {seed} ===")
    model = models.resnet18(weights=None, num_classes=NUM_CLASSES).to(device)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0)

    history = {
        'epoch': [], 'std_acc': [], 'robust_acc': [], 'epoch_time': [], 'cumulative_time': [],
        'overlap': [], 'selection_stability': [], 'per_class_std_acc': [], 'per_class_robust_acc': []
    }

    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        mean_overlap, stability = train_epoch_gradmatch_collect(model, optimizer, train_loader, K_SAMPLES)
        std_acc, per_class_std = evaluate_per_class(model, test_loader, attack_fn=None)
        robust_acc, per_class_rob = evaluate_per_class(model, test_loader, attack_fn=pgd_attack)
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
        print(f"Seed {seed} Epoch {epoch}/{EPOCHS} | Std: {std_acc:.2f}% | Robust: {robust_acc:.2f}% | Stability: {stability:.3f} | EpochTime: {epoch_time:.1f}s")
    
    out = {
        'experiment_name': f"{method_name}_seed{seed}",
        'hyperparameters': {
            'batch_size': BATCH_SIZE, 'learning_rate': LEARNING_RATE, 'epochs': EPOCHS,
            'k_percentage': K_PERCENTAGE, 'k_samples': K_SAMPLES, 'epsilon': EPSILON
        },
        'training_history': history,
        'final_summary': {
            'final_std_acc': history['std_acc'][-1] if len(history['std_acc'])>0 else None,
            'final_robust_acc': history['robust_acc'][-1] if len(history['robust_acc'])>0 else None,
            'total_training_time': history['cumulative_time'][-1] if len(history['cumulative_time'])>0 else None
        }
    }
    seed_fname = os.path.join(OUT_DIR, f"{method_name}_seed{seed}.json")
    with open(seed_fname, 'w') as f: json.dump(out, f, indent=4)
    print("Saved GradMatch seed results to", seed_fname)
    return seed_fname, out

seed_files_gradmatch = []
seed_outputs_gradmatch = []

for s in SEEDS:
    fname, out = run_experiment_seed_gradmatch(s, method_name=METHOD_NAME_GRADMATCH)
    seed_files_gradmatch.append(fname)
    seed_outputs_gradmatch.append(out)

min_epochs = min(len(o['training_history']['epoch']) for o in seed_outputs_gradmatch)
metrics = ['std_acc', 'robust_acc', 'epoch_time', 'cumulative_time', 'overlap', 'selection_stability']
agg_history = {'epoch': list(range(1, min_epochs + 1))}
for m in metrics:
    arr = np.array([o['training_history'][m][:min_epochs] for o in seed_outputs_gradmatch], dtype=float)
    agg_history[m + '_mean'] = list(np.nanmean(arr, axis=0))
    agg_history[m + '_std'] = list(np.nanstd(arr, axis=0, ddof=1))
    
per_class_std = np.array([o['training_history']['per_class_std_acc'][:min_epochs] for o in seed_outputs_gradmatch], dtype=float)
per_class_rob = np.array([o['training_history']['per_class_robust_acc'][:min_epochs] for o in seed_outputs_gradmatch], dtype=float)

aggregate_output_gradmatch = {
    'experiment_name': METHOD_NAME_GRADMATCH + "_aggregate",
    'seed_files': seed_files_gradmatch,
    'hyperparameters': {
        'batch_size': BATCH_SIZE, 'learning_rate': LEARNING_RATE, 'epochs': EPOCHS,
        'k_percentage': K_PERCENTAGE, 'k_samples': K_SAMPLES, 'epsilon': EPSILON
    },
    'training_history_aggregate': agg_history,
    'per_class_std_mean_per_epoch': np.nanmean(per_class_std, axis=0).tolist(),
    'per_class_std_std_per_epoch': np.nanstd(per_class_std, axis=0, ddof=1).tolist(),
    'per_class_robust_mean_per_epoch': np.nanmean(per_class_rob, axis=0).tolist(),
    'per_class_robust_std_per_epoch': np.nanstd(per_class_rob, axis=0, ddof=1).tolist(),
    'final_summary_aggregate': {
        'final_std_acc_mean': float(np.nanmean([o['final_summary']['final_std_acc'] for o in seed_outputs_gradmatch])),
        'final_std_acc_std': float(np.nanstd([o['final_summary']['final_std_acc'] for o in seed_outputs_gradmatch], ddof=1)),
        'final_robust_acc_mean': float(np.nanmean([o['final_summary']['final_robust_acc'] for o in seed_outputs_gradmatch])),
        'final_robust_acc_std': float(np.nanstd([o['final_summary']['final_robust_acc'] for o in seed_outputs_gradmatch], ddof=1)),
        'total_training_time_mean': float(np.nanmean([o['final_summary']['total_training_time'] for o in seed_outputs_gradmatch])),
        'total_training_time_std': float(np.nanstd([o['final_summary']['total_training_time'] for o in seed_outputs_gradmatch], ddof=1))
    }
}

agg_fname_gradmatch = os.path.join(OUT_DIR, f"{METHOD_NAME_GRADMATCH}_aggregate.json")
with open(agg_fname_gradmatch, 'w') as f:
    json.dump(aggregate_output_gradmatch, f, indent=4)

print("Saved GradMatch aggregate results to", agg_fname_gradmatch)
print("GradMatch seed files:", seed_files_gradmatch)