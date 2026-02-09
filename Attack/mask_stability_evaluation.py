import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from collections import OrderedDict
import copy
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
from datetime import datetime

class Config:
    def __init__(self):
        self.method = 'fedmask'  # 'fedmask' or 'fedselect'
        self.dataset = 'cifar10'  # 'mnist', 'emnist', 'svhn', 'qmnist', 'cifar10'
        self.distribution_type = 'random'  # 'random' or 'dirichlet'
        self.num_clients = 10
        self.local_epochs = 50
        self.batch_size = 64
        self.lr = 0.01
        self.personalization_rate = 5.0
        self.seed = 42
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_evaluation_trials = 100

        # Random distribution settings
        self.total_samples_per_client = 200
        self.min_classes_per_client = 3
        self.max_classes_per_client = 3
        self.dominant_class_min_ratio = 0.7
        
        # Dirichlet distribution settings
        self.dirichlet_alpha = 0.1
        self.dirichlet_min_classes = 3
        self.dirichlet_total_samples = 200

        self.stability_metric = 'cosine'  # 'cosine' or 'hamming'

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Client_FedMask:
    def __init__(self, model_architecture, train_loader, config, in_channels=1, num_classes=10):
        self.model = model_architecture(in_channels=in_channels, num_classes=num_classes).to(config.device)
        # self.model = model_architecture(num_classes=num_classes).to(config.device)
        self.train_loader = train_loader
        self.config = config

    def generate_mask_with_tracking(self, initial_state_dict):
        self.model.load_state_dict(copy.deepcopy(initial_state_dict))
        optimizer = optim.SGD(self.model.parameters(), lr=self.config.lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        epoch_masks = []
        
        self.model.train()
        for epoch in range(self.config.local_epochs):
            if not self.train_loader or len(self.train_loader.dataset) == 0:
                continue

            for data, target in self.train_loader:
                data, target = data.to(self.config.device), target.to(self.config.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            current_state = self.model.state_dict()
            magnitudes = OrderedDict()
            mask = OrderedDict()
            
            for name in current_state.keys():
                mag = current_state[name].cpu().abs()
                magnitudes[name] = mag
                mask[name] = torch.zeros_like(mag)
            
            all_magnitudes = torch.cat([v.flatten() for v in magnitudes.values() if v is not None and v.numel() > 0])
            if all_magnitudes.numel() > 0:
                num_params_to_personalize = int(all_magnitudes.numel() * self.config.personalization_rate / 100)
                if 0 < num_params_to_personalize < all_magnitudes.numel():
                    threshold = torch.topk(all_magnitudes, num_params_to_personalize, largest=True).values.min()
                    for name in mask:
                        mask[name][magnitudes[name] >= threshold] = 1
            mask_flat = torch.cat([v.flatten() for v in mask.values()])
            epoch_masks.append(mask_flat)
        final_state_dict = self.model.state_dict()
        final_mask = self._compute_final_mask(final_state_dict)
        
        return final_mask, epoch_masks
    
    def _compute_final_mask(self, state_dict):
        magnitudes = OrderedDict()
        mask = OrderedDict()
        for name in state_dict.keys():
            mag = state_dict[name].cpu().abs()
            magnitudes[name] = mag
            mask[name] = torch.zeros_like(mag)
        
        all_magnitudes = torch.cat([v.flatten() for v in magnitudes.values() if v is not None and v.numel() > 0])
        if all_magnitudes.numel() == 0:
            return mask
        
        num_params_to_personalize = int(all_magnitudes.numel() * self.config.personalization_rate / 100)
        if 0 < num_params_to_personalize < all_magnitudes.numel():
            threshold = torch.topk(all_magnitudes, num_params_to_personalize, largest=True).values.min()
            for name in mask:
                mask[name][magnitudes[name] >= threshold] = 1
        return mask


class Client_FedSelect:
    def __init__(self, model_architecture, train_loader, config, in_channels=1, num_classes=10):
        self.model = model_architecture(in_channels=in_channels, num_classes=num_classes).to(config.device)
        # self.model = model_architecture(num_classes=num_classes).to(config.device)
        self.train_loader = train_loader
        self.config = config

    def generate_mask_with_tracking(self, initial_state_dict):
        self.model.load_state_dict(copy.deepcopy(initial_state_dict))
        optimizer = optim.SGD(self.model.parameters(), lr=self.config.lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        epoch_masks = []
        
        self.model.train()
        for epoch in range(self.config.local_epochs):
            if not self.train_loader or len(self.train_loader.dataset) == 0:
                continue

            for data, target in self.train_loader:
                data, target = data.to(self.config.device), target.to(self.config.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            current_state = self.model.state_dict()
            deltas = OrderedDict()
            mask = OrderedDict()
            
            for name in initial_state_dict.keys():
                delta = (initial_state_dict[name].cpu() - current_state[name].cpu()).abs()
                deltas[name] = delta
                mask[name] = torch.zeros_like(delta)
            
            all_deltas = torch.cat([v.flatten() for v in deltas.values() if v is not None and v.numel() > 0])
            if all_deltas.numel() > 0:
                num_params_to_personalize = int(all_deltas.numel() * self.config.personalization_rate / 100)
                if 0 < num_params_to_personalize < all_deltas.numel():
                    threshold = torch.topk(all_deltas, num_params_to_personalize, largest=True).values.min()
                    for name in mask:
                        mask[name][deltas[name] >= threshold] = 1

            mask_flat = torch.cat([v.flatten() for v in mask.values()])
            epoch_masks.append(mask_flat)

        final_state_dict = self.model.state_dict()
        final_mask = self._compute_final_mask(initial_state_dict, final_state_dict)
        
        return final_mask, epoch_masks
    
    def _compute_final_mask(self, initial_state_dict, final_state_dict):
        deltas = OrderedDict()
        mask = OrderedDict()
        for name in initial_state_dict.keys():
            delta = (initial_state_dict[name].cpu() - final_state_dict[name].cpu()).abs()
            deltas[name] = delta
            mask[name] = torch.zeros_like(delta)
        
        all_deltas = torch.cat([v.flatten() for v in deltas.values() if v is not None and v.numel() > 0])
        if all_deltas.numel() == 0:
            return mask
        
        num_params_to_personalize = int(all_deltas.numel() * self.config.personalization_rate / 100)
        if 0 < num_params_to_personalize < all_deltas.numel():
            threshold = torch.topk(all_deltas, num_params_to_personalize, largest=True).values.min()
            for name in mask:
                mask[name][deltas[name] >= threshold] = 1
        return mask

def load_dataset(config):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    if config.dataset == 'mnist':
        dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        num_classes = 10
        in_channels = 1
    elif config.dataset == 'emnist':
        dataset = datasets.EMNIST(root='./data', split='digits', train=True, download=True, transform=transform)
        num_classes = 10
        in_channels = 1
    elif config.dataset == 'svhn':
        dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        dataset.targets = dataset.labels
        num_classes = 10
        in_channels = 3
    elif config.dataset == 'qmnist':
        dataset = datasets.QMNIST(root='./data', train=True, download=True, transform=transform)
        num_classes = 10
        in_channels = 1
    elif config.dataset == 'cifar10':
        transform_cifar = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = datasets.CIFAR10(root='./data', train=True, transform=transform_cifar, download=True)
        num_classes = 10
        in_channels = 1   
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset}")
    
    return dataset, num_classes, in_channels

def generate_random_distribution(config):
    num_classes = np.random.randint(config.min_classes_per_client, config.max_classes_per_client + 1)
    classes = np.random.choice(range(config.num_clients), num_classes, replace=False)
    
    if num_classes == 1:
        return {classes[0]: config.total_samples_per_client}
    
    dominant_class = np.random.choice(classes)
    min_dominant_samples = int(config.total_samples_per_client * config.dominant_class_min_ratio)
    dominant_samples = np.random.randint(min_dominant_samples, config.total_samples_per_client + 1)
    
    remaining_samples = config.total_samples_per_client - dominant_samples
    other_classes = [c for c in classes if c != dominant_class]
    
    if remaining_samples > 0:
        proportions = np.random.dirichlet(np.ones(len(other_classes)))
        samples_for_others = (proportions * remaining_samples).astype(int)
        if len(other_classes) > 0:
            samples_for_others[-1] = remaining_samples - samples_for_others[:-1].sum()
    else:
        samples_for_others = [0] * len(other_classes)
        
    distribution = {dominant_class: dominant_samples}
    for cls, count in zip(other_classes, samples_for_others):
        distribution[cls] = count
        
    return distribution


def generate_dirichlet_distribution(config):
    num_classes = np.random.randint(config.dirichlet_min_classes, config.num_clients + 1)
    classes = np.random.choice(range(config.num_clients), num_classes, replace=False)
    
    proportions = np.random.dirichlet(alpha=[config.dirichlet_alpha] * num_classes)
    samples = (proportions * config.dirichlet_total_samples).astype(int)
    samples[-1] = config.dirichlet_total_samples - samples[:-1].sum()
    
    samples = np.maximum(samples, 1)
    if samples.sum() > config.dirichlet_total_samples:
        diff = samples.sum() - config.dirichlet_total_samples
        max_idx = np.argmax(samples)
        samples[max_idx] -= diff
    
    distribution = {int(cls): int(count) for cls, count in zip(classes, samples) if count > 0}
    
    return distribution

def calculate_mask_stability(epoch_masks, metric='cosine'):
    if len(epoch_masks) < 2:
        return 1.0
    
    similarities = []
    
    for i in range(len(epoch_masks) - 1):
        mask1 = epoch_masks[i].float()
        mask2 = epoch_masks[i + 1].float()
        
        if metric == 'cosine':
            sim = F.cosine_similarity(mask1, mask2, dim=0).item()
            similarities.append(sim)
        elif metric == 'hamming':
            hamming_dist = (mask1 != mask2).sum().item()
            total_bits = mask1.numel()
            similarity = 1.0 - (hamming_dist / total_bits)
            similarities.append(similarity)
    
    return np.mean(similarities)

def run_mask_stability_evaluation(config): 
    print("\n" + "="*80)
    print("MASK STABILITY EVALUATION")
    print("="*80)
    print(f"Method: {config.method.upper()}")
    print(f"Dataset: {config.dataset.upper()}")
    print(f"Distribution: {config.distribution_type.upper()}")
    print(f"Trials: {config.num_evaluation_trials}")
    print(f"Local Epochs: {config.local_epochs}")
    print(f"Stability Metric: {config.stability_metric.upper()}")
    print("="*80 + "\n")

    dataset, num_classes, in_channels = load_dataset(config)
    
    initial_model = SimpleCNN(in_channels=in_channels, num_classes=num_classes).to(config.device)
    initial_state_dict = copy.deepcopy(initial_model.state_dict())

    if config.method == 'fedmask':
        ClientClass = Client_FedMask
    elif config.method == 'fedselect':
        ClientClass = Client_FedSelect
    else:
        raise ValueError(f"Unsupported method: {config.method}")

    results = []
    
    for trial_idx in tqdm(range(config.num_evaluation_trials), desc="Evaluating"):
        if config.distribution_type == 'random':
            distribution = generate_random_distribution(config)
        elif config.distribution_type == 'dirichlet':
            distribution = generate_dirichlet_distribution(config)
        else:
            raise ValueError(f"Unsupported distribution type: {config.distribution_type}")

        indices = []
        for cls, num_samples in distribution.items():
            available_indices = np.where(np.array(dataset.targets) == cls)[0]
            chosen_indices = np.random.choice(available_indices, num_samples, replace=False)
            indices.append(chosen_indices)
        
        final_indices = np.concatenate(indices)
        train_loader = DataLoader(Subset(dataset, final_indices), batch_size=config.batch_size, shuffle=True)

        client = ClientClass(SimpleCNN, train_loader, config, in_channels=in_channels, num_classes=num_classes)
        final_mask, epoch_masks = client.generate_mask_with_tracking(initial_state_dict)

        stability = calculate_mask_stability(epoch_masks, metric=config.stability_metric)

        result = {
            'trial': trial_idx,
            'method': config.method,
            'dataset': config.dataset,
            'distribution_type': config.distribution_type,
            'distribution': json.dumps({int(k): int(v) for k, v in distribution.items()}),
            'num_classes': len(distribution),
            'stability': float(stability),
            'stability_metric': config.stability_metric
        }
        
        results.append(result)

    stabilities = [r['stability'] for r in results]
    avg_stability = np.mean(stabilities)
    std_stability = np.std(stabilities)
    min_stability = np.min(stabilities)
    max_stability = np.max(stabilities)

    print("\n" + "="*80)
    print("EVALUATION REPORT")
    print("="*80)
    print(f"Average Stability: {avg_stability:.4f} (±{std_stability:.4f})")
    print(f"Min Stability: {min_stability:.4f}")
    print(f"Max Stability: {max_stability:.4f}")
    print(f"\nInterpretation:")
    if avg_stability > 0.9:
        print("  ✓ HIGH STABILITY - Masks converge to stable patterns")
        print("  → Attack Feasibility: HIGH")
    elif avg_stability > 0.7:
        print("  ○ MODERATE STABILITY - Masks show some variation")
        print("  → Attack Feasibility: MODERATE")
    else:
        print("  ✗ LOW STABILITY - Masks are unstable")
        print("  → Attack Feasibility: LOW")
    print("="*80 + "\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = "mask_stability_results"
    os.makedirs(result_dir, exist_ok=True)
    
    filename = f"{result_dir}/stability_{config.method}_{config.dataset}_{config.distribution_type}_{timestamp}.csv"
    result_df = pd.DataFrame(results)
    result_df.to_csv(filename, index=False)
    print(f"Results saved to: {filename}")
    
    return results, avg_stability, std_stability

def run_comprehensive_experiments(config):
    methods = ['fedmask', 'fedselect']
    datasets = ['mnist', 'emnist', 'svhn', 'qmnist']
    distributions = ['random', 'dirichlet']
    
    all_results = []
    
    total_experiments = len(methods) * len(datasets) * len(distributions)
    current_exp = 0
    
    print("\n" + "="*80)
    print(f"RUNNING {total_experiments} COMPREHENSIVE EXPERIMENTS")
    print("="*80 + "\n")
    
    for method in methods:
        for dataset in datasets:
            for distribution in distributions:
                current_exp += 1
                print(f"\n[{current_exp}/{total_experiments}] Running: {method.upper()} + {dataset.upper()} + {distribution.upper()}")

                config.method = method
                config.dataset = dataset
                config.distribution_type = distribution

                try:
                    results, avg_stability, std_stability = run_mask_stability_evaluation(config)
                    
                    summary = {
                        'method': method,
                        'dataset': dataset,
                        'distribution': distribution,
                        'avg_stability': avg_stability,
                        'std_stability': std_stability
                    }
                    all_results.append(summary)
                    
                except Exception as e:
                    print(f"ERROR in experiment: {e}")
                    continue

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"mask_stability_results/summary_{timestamp}.csv"
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(summary_file, index=False)
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*80)
    print(f"Summary saved to: {summary_file}")
    print("\nSummary:")
    print(summary_df.to_string(index=False))
    
    return all_results

if __name__ == '__main__':
    config = Config()

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    print(f"Device: {config.device}")

    run_mode = 'single'  # 'single' or 'comprehensive'
    if run_mode == 'single':
        config.method = 'fedmask'  # 'fedmask' or 'fedselect'
        config.dataset = 'mnist'  # 'mnist', 'emnist', 'svhn', 'qmnist', 'cifar10'
        config.distribution_type = 'random'  # 'random' or 'dirichlet'
        config.dirichlet_alpha = 0.1  # only used if distribution_type is 'dirichlet'
        
        run_mask_stability_evaluation(config)
        
    elif run_mode == 'comprehensive':
        run_comprehensive_experiments(config)
