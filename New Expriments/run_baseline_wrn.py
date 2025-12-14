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

BATCH_SIZE = 128
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
EPOCHS = 175
SCHEDULER_CYCLE_EPOCHS = 150

EPSILON, ALPHA = 8/255, 0.01
PGD_STEPS_TRAIN, PGD_STEPS_EVAL = 20, 20

SEEDS = [42, 43, 44]
NUM_CLASSES = 10
METHOD_NAME_ROBUST = f"RobustPGDAT_WRN34_CIFAR10_bs{BATCH_SIZE}"
OUT_DIR = "results_wrn_baseline"
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Method: {METHOD_NAME_ROBUST}, Model: WRN-34-10, BATCH_SIZE={BATCH_SIZE}, seeds={SEEDS}")

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth=34, num_classes=10, widen_factor=10, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print("Loaded CIFAR-10:", len(train_dataset), "train,", len(test_dataset), "test")
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
    print(f"\n=== Robust PGD-AT WRN-34 RUN seed {seed} ===")
    
    model = WideResNet(depth=34, num_classes=NUM_CLASSES, widen_factor=10).to(device)
    
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

if __name__ == "__main__":
    seed_files, seed_outputs = [], []
    for s in SEEDS:
        fname, out = run_experiment_seed_robust(s, method_name=METHOD_NAME_ROBUST)
        seed_files.append(fname)
        seed_outputs.append(out)
    min_epochs = 0 if not seed_outputs else min(len(o['training_history']['epoch']) for o in seed_outputs)
    metrics = ['std_acc', 'robust_acc', 'epoch_time', 'cumulative_time', 'overlap', 'selection_stability']
    agg_history = {'epoch': list(range(1, min_epochs + 1))}
    for m in metrics:
        if m in seed_outputs[0]['training_history']:
            arr = np.array([o['training_history'][m][:min_epochs] for o in seed_outputs], dtype=float)
            agg_history[f'{m}_mean'] = list(np.nanmean(arr, axis=0))
            agg_history[f'{m}_std'] = list(np.nanstd(arr, axis=0, ddof=1))
    final_summaries = [o['final_summary'] for o in seed_outputs]
    aggregate_output = {
        'experiment_name': f"{seed_outputs[0]['experiment_name'].split('_seed')[0]}_aggregate",
        'seed_files': seed_files,
        'hyperparameters': seed_outputs[0]['hyperparameters'],
        'training_history_aggregate': agg_history,
        'final_summary_aggregate': {
            'final_std_acc_mean': float(np.nanmean([s['final_std_acc'] for s in final_summaries if s['final_std_acc'] is not None])),
            'final_std_acc_std': float(np.nanstd([s['final_std_acc'] for s in final_summaries if s['final_std_acc'] is not None], ddof=1)),
            'final_robust_acc_mean': float(np.nanmean([s['final_robust_acc'] for s in final_summaries if s['final_robust_acc'] is not None])),
            'final_robust_acc_std': float(np.nanstd([s['final_robust_acc'] for s in final_summaries if s['final_robust_acc'] is not None], ddof=1)),
            'final_autoattack_acc_mean': float(np.nanmean([s.get('final_autoattack_acc', 0.0) for s in final_summaries])),
            'final_autoattack_acc_std': float(np.nanstd([s.get('final_autoattack_acc', 0.0) for s in final_summaries], ddof=1)),
            'total_training_time_mean': float(np.nanmean([s['total_training_time'] for s in final_summaries if s['total_training_time'] is not None])),
            'total_training_time_std': float(np.nanstd([s['total_training_time'] for s in final_summaries if s['total_training_time'] is not None], ddof=1))
        }
    }
    agg_fname = os.path.join(OUT_DIR, f"{aggregate_output['experiment_name']}.json")
    with open(agg_fname, 'w') as f:
        json.dump(aggregate_output, f, indent=4)
    print("\nSaved AGGREGATE results to", agg_fname)
    print("All seeds completed.")