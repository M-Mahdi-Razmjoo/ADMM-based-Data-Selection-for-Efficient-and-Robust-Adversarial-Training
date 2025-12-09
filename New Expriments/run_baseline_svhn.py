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
from torch.optim.lr_scheduler import OneCycleLR

try:
    from autoattack import AutoAttack
    HAS_AUTOATTACK = True
except ImportError:
    print("AutoAttack package not found. Skipping AA evaluation.")
    HAS_AUTOATTACK = False

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
EPOCHS = 175
SCHEDULER_CYCLE_EPOCHS = 150

EPSILON, ALPHA = 8/255, 0.01
PGD_STEPS_TRAIN, PGD_STEPS_EVAL = 20, 20

SEEDS = [42, 43, 44]
NUM_CLASSES = 10
METHOD_NAME_ROBUST = f"RobustPGDAT_SVHN_bs{BATCH_SIZE}"
OUT_DIR = "results_svhn_baseline"
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Method: {METHOD_NAME_ROBUST}, Dataset: SVHN, BATCH_SIZE={BATCH_SIZE}, seeds={SEEDS}")

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform_train)
test_dataset  = datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print("Loaded SVHN:", len(train_dataset), "train,", len(test_dataset), "test")

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

def run_autoattack_evaluation(model, data_loader):
    if not HAS_AUTOATTACK:
        print("AutoAttack library missing. Skipping.")
        return 0.0
    
    print("\n------------------------------------------------")
    print("Running AutoAttack (Standard Evaluation)...")
    print("This might take a few minutes depending on GPU...")
    model.eval()

    l_x, l_y = [], []
    for x, y in data_loader:
        l_x.append(x)
        l_y.append(y)
    x_test = torch.cat(l_x, dim=0)
    y_test = torch.cat(l_y, dim=0)
    adversary = AutoAttack(model, norm='Linf', eps=EPSILON, version='standard')
    with torch.no_grad():
        x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=BATCH_SIZE)
    print("Calculating final robust accuracy from AutoAttack outputs...")
    dataset_adv = torch.utils.data.TensorDataset(x_adv, y_test)
    loader_adv = DataLoader(dataset_adv, batch_size=BATCH_SIZE, shuffle=False)
    
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader_adv:
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, pred = torch.max(out, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            
    final_acc = 100.0 * correct / total
    print(f"AutoAttack Final Accuracy: {final_acc:.2f}%")
    print("------------------------------------------------\n")
    return final_acc

def train_epoch_robust_collect(model, optimizer, scheduler, data_loader, current_epoch):
    model.train()
    for clean_images, labels in data_loader:
        clean_images, labels = clean_images.to(device), labels.to(device)
        adv_images = pgd_attack(model, clean_images, labels, EPSILON, ALPHA, PGD_STEPS_TRAIN)
        optimizer.zero_grad()
        predictions = model(adv_images)
        loss = F.cross_entropy(predictions, labels)
        loss.backward()
        optimizer.step()
        if current_epoch <= SCHEDULER_CYCLE_EPOCHS:
            scheduler.step()
    return 1.0, 1.0

def run_experiment_seed_robust(seed, method_name=METHOD_NAME_ROBUST):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"\n=== Robust PGD-AT SVHN RUN seed {seed} ===")
    model = models.resnet18(weights=None, num_classes=NUM_CLASSES).to(device)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = OneCycleLR(optimizer, max_lr=LEARNING_RATE, epochs=SCHEDULER_CYCLE_EPOCHS, steps_per_epoch=len(train_loader))
    
    history = {'epoch': [], 'std_acc': [], 'robust_acc': [], 'epoch_time': [], 'cumulative_time': [], 'overlap': [], 'selection_stability': [], 'per_class_std_acc': [], 'per_class_robust_acc': []}
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        mean_overlap, stability = train_epoch_robust_collect(model, optimizer, scheduler, train_loader, epoch)
        std_acc, per_class_std = evaluate_per_class(model, test_loader, attack_fn=None)
        robust_acc, per_class_rob = evaluate_per_class(model, test_loader, attack_fn=pgd_attack)
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
        
        print(f"Seed {seed} Epoch {epoch}/{EPOCHS} | Std: {std_acc:.2f}% | Robust: {robust_acc:.2f}% | Time: {epoch_time:.1f}s")
    
    aa_acc = run_autoattack_evaluation(model, test_loader)
    print(f"Final AutoAttack Accuracy: {aa_acc:.2f}%")

    out = {
        'experiment_name': f"{method_name}_seed{seed}", 
        'hyperparameters': {'batch_size': BATCH_SIZE, 'learning_rate': LEARNING_RATE, 'epochs': EPOCHS}, 
        'training_history': history, 
        'final_summary': {
            'final_std_acc': history['std_acc'][-1] if history['std_acc'] else None, 
            'final_robust_acc': history['robust_acc'][-1] if history['robust_acc'] else None, 
            'final_autoattack_acc': aa_acc,
            'total_training_time': history['cumulative_time'][-1] if history['cumulative_time'] else None
        }
    }
    
    seed_fname = os.path.join(OUT_DIR, f"{method_name}_seed{seed}.json")
    with open(seed_fname, 'w') as f:
        json.dump(out, f, indent=4)
    print("Saved seed results to", seed_fname)
    return seed_fname, out

seed_files, seed_outputs = [], []
for s in SEEDS:
    fname, out = run_experiment_seed_robust(s, method_name=METHOD_NAME_ROBUST)
    seed_files.append(fname)
    seed_outputs.append(out)

print("All seeds completed.")