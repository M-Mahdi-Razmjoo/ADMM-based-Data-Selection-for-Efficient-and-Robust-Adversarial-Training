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
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

SEEDS = [41, 42, 43]

torch.manual_seed(SEEDS[0])
np.random.seed(SEEDS[0])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEEDS[0])
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

NUM_CLASSES = 100
METHOD_NAME = f"LCSKM_CIFAR100_k{K_SAMPLES}_bs{BATCH_SIZE}"
OUT_DIR = "results_lcskm_runs"
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

def get_penultimate_features(model, x):
    model.eval()
    with torch.no_grad():
        out = model.conv1(x)
        out = model.bn1(out)
        out = model.relu(out)
        out = model.maxpool(out)
        out = model.layer1(out)
        out = model.layer2(out)
        out = model.layer3(out)
        out = model.layer4(out)
        out = model.avgpool(out)
        features = torch.flatten(out, 1)
    model.train()
    return features

def lcskm_selection(latent_features, k, num_clusters=NUM_CLASSES):
    numpy_features = latent_features.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto', algorithm='lloyd')
    kmeans.fit(numpy_features)
    centroids = kmeans.cluster_centers_
    distances = cdist(numpy_features, centroids)
    distances.sort(axis=1)
    metric = distances[:, 1] - distances[:, 0]
    selected_indices = np.argsort(metric)[:k]
    return torch.tensor(selected_indices, dtype=torch.long, device=latent_features.device)

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
        if grad is None: break
        images = images + alpha * grad.sign()
        eta = torch.clamp(images - orig, -epsilon, epsilon)
        images = torch.clamp(orig + eta, 0.0, 1.0).detach()
    return images

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

def train_epoch_lcskm(model, optimizer, data_loader, k):
    model.train()
    selected_history = []
    for clean_images, labels in data_loader:
        clean_images, labels = clean_images.to(device), labels.to(device)
        adv_images = pgd_attack(model, clean_images, labels, EPSILON, ALPHA, PGD_STEPS_TRAIN)
        combined_images = torch.cat([clean_images, adv_images], dim=0)
        combined_labels = torch.cat([labels, labels], dim=0)
        latent_features = get_penultimate_features(model, combined_images)
        sel_idx = lcskm_selection(latent_features, k)
        selected_history.append(sel_idx.clone().cpu())
        final_images = combined_images[sel_idx]
        final_labels = combined_labels[sel_idx]
        if final_images.size(0) > 0:
            optimizer.zero_grad()
            predictions = model(final_images)
            loss = F.cross_entropy(predictions, final_labels)
            loss.backward()
            optimizer.step()
    mean_overlap = 1.0
    stability = compute_selection_stability(selected_history, k)
    return mean_overlap, stability

def run_experiment_seed(seed, method_name=METHOD_NAME):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    print(f"\n=== LCS-KM RUN seed {seed} ===")
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
        mean_overlap, stability = train_epoch_lcskm(model, optimizer, train_loader, K_SAMPLES)
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
        print(f"Seed {seed} Epoch {epoch}/{EPOCHS} | Std: {std_acc:.2f}% | Robust: {robust_acc:.2f}% | Stability: {stability:.3f} | EpochTime: {epoch_time:.1f}s")

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

for s in SEEDS:
    fname, out = run_experiment_seed(s, method_name=METHOD_NAME)
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

print("\nSaved aggregate results to", agg_fname)
print("Seed files:", seed_files)