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
from scipy.stats import spearmanr
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors
import os
import json
import argparse
import pandas as pd

class Config:
    def __init__(self):
        parser = argparse.ArgumentParser(description="FedMask Attack Evaluation")

        # Basic Settings
        parser.add_argument('--domain', type=str, default='digit', choices=['digit', 'fashion', 'cifar10'], help='Dataset domain')
        parser.add_argument('--num_clients', type=int, default=10, help='Number of clients')
        parser.add_argument('--local_epochs', type=int, default=5, help='Local epochs')
        parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
        parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
        parser.add_argument('--personalization_rate', type=float, default=5.0, help='Personalization rate for FedMask')
        parser.add_argument('--seed', type=int, default=42, help='Random seed')

        # Experiment Settings
        parser.add_argument('--num_evaluation_trials', type=int, default=100, help='Number of evaluation trials')
        parser.add_argument('--template_runs_per_class', type=int, default=20, help='Number of template runs per class')

        # Distribution Settings
        parser.add_argument('--distribution_type', type=str, default='random', choices=['random', 'dirichlet'], help='Data distribution type')

        # Random Distribution Settings
        parser.add_argument('--total_samples_per_client', type=int, default=200, help='Total samples per client (random)')
        parser.add_argument('--min_classes_per_client', type=int, default=3, help='Min classes per client (random)')
        parser.add_argument('--max_classes_per_client', type=int, default=5, help='Max classes per client (random)')
        parser.add_argument('--dominant_class_min_ratio', type=float, default=0.7, help='Dominant class min ratio (random)')

        # Dirichlet Distribution Settings
        parser.add_argument('--dirichlet_alpha', type=float, default=0.5, help='Dirichlet alpha')
        parser.add_argument('--dirichlet_min_classes', type=int, default=3, help='Dirichlet min classes')
        parser.add_argument('--dirichlet_total_samples', type=int, default=200, help='Dirichlet total samples')

        args = parser.parse_args()

        self.domain = args.domain
        self.num_clients = args.num_clients
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.personalization_rate = args.personalization_rate
        self.seed = args.seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_evaluation_trials = args.num_evaluation_trials
        self.template_runs_per_class = args.template_runs_per_class

        self.distribution_type = args.distribution_type
        
        # Random distribution parameters (for 'random' type)
        self.total_samples_per_client = args.total_samples_per_client
        self.min_classes_per_client = args.min_classes_per_client
        self.max_classes_per_client = args.max_classes_per_client
        self.dominant_class_min_ratio = args.dominant_class_min_ratio
        
        # Dirichlet distribution parameters (for 'dirichlet' type)
        self.dirichlet_alpha = args.dirichlet_alpha
        self.dirichlet_min_classes = args.dirichlet_min_classes
        self.dirichlet_total_samples = args.dirichlet_total_samples

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(SimpleCNN, self).__init__(); self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2); self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128); self.fc2 = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x))); x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7); x = torch.relu(self.fc1(x)); x = self.fc2(x)
        return x

class Client_FedMask:
    def __init__(self, model_architecture, train_loader, config):
        self.model = model_architecture().to(config.device)
        self.train_loader = train_loader
        self.config = config

    def generate_mask(self, initial_state_dict):
        self.model.load_state_dict(copy.deepcopy(initial_state_dict))
        optimizer = optim.SGD(self.model.parameters(), lr=self.config.lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for _ in range(self.config.local_epochs):
            if not self.train_loader or len(self.train_loader.dataset) == 0:
                continue
            for data, target in self.train_loader:
                data, target = data.to(self.config.device), target.to(self.config.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        final_state_dict = self.model.state_dict()

        magnitudes = OrderedDict()
        mask = OrderedDict()
        for name in final_state_dict.keys():
            mag = final_state_dict[name].cpu().abs()
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

def generate_template_masks(config, initial_state_dict):
    print("Generating high-quality template masks from public {} dataset...".format(config.domain))
    
    if config.domain == 'cifar10':
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    else:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        if config.domain == 'digit':
            dataset = datasets.EMNIST(root='./data', split='digits', train=True, download=True, transform=transform)
        elif config.domain == 'fashion':
            dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        else:
             raise ValueError(f"Unknown domain: {config.domain}")

    template_masks = []
    for i in tqdm(range(config.num_clients), desc="Generating Templates"):
        indices = np.where(np.array(dataset.targets) == i)[0]
        loader = DataLoader(Subset(dataset, indices), batch_size=config.batch_size, shuffle=True)
        class_masks = [torch.cat([v.flatten() for v in Client_FedMask(SimpleCNN, loader, config).generate_mask(initial_state_dict).values()]) for _ in range(config.template_runs_per_class)]
        template_masks.append(torch.stack(class_masks).mean(dim=0))
    return template_masks

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

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def run_comprehensive_attack_evaluation(config, initial_state_dict, template_masks):
    print(f"\n--- Running {config.num_evaluation_trials} Attack Trials on Skewed Distributions ---")
    print(f"Distribution Type: {config.distribution_type.upper()}")
    if config.distribution_type == 'dirichlet':
        print(f"Dirichlet Alpha: {config.dirichlet_alpha} (smaller = more heterogeneous)")

    attack_results = []

    if config.domain == 'cifar10':
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        mnist_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    else:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        if config.domain == 'digit':
            mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        elif config.domain == 'fashion':
            mnist_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

    all_metrics = []
    true_sim_acc = 0
    top3_recall_sum = 0.0
    top3_hit_count = 0
    top3_order_match_count = 0
    top3_similarities_sum = 0.0
    top3_order_similarities_sum = 0.0

    for i in tqdm(range(config.num_evaluation_trials), desc="Attack Trials"):
        if config.distribution_type == 'dirichlet':
            true_distribution = generate_dirichlet_distribution(config)
        else:
            true_distribution = generate_random_distribution(config)
        
        indices = []
        for cls, num_samples in true_distribution.items():
            available_indices = np.where(np.array(mnist_train.targets) == cls)[0]
            chosen_indices = np.random.choice(available_indices, num_samples, replace=False)
            indices.append(chosen_indices)
        
        final_indices = np.concatenate(indices)
        target_loader = DataLoader(Subset(mnist_train, final_indices), batch_size=config.batch_size, shuffle=True)
        client = Client_FedMask(SimpleCNN, target_loader, config)
        target_mask_dict = client.generate_mask(initial_state_dict)
        target_mask_flat = torch.cat([v.flatten() for v in target_mask_dict.values()])
    
        similarities = np.array([F.cosine_similarity(target_mask_flat.float(), tm.float(), dim=0).item() for tm in template_masks])

        max_label = np.argmax(similarities)
        max_similarity = similarities[max_label]

        true_top3_classes = sorted(true_distribution.items(), key=lambda x: x[1], reverse=True)[:3]
        true_top3_set = set([cls for cls, count in true_top3_classes])

        predicted_top3_indices = np.argsort(similarities)[-3:][::-1]
        predicted_top3_set = set(predicted_top3_indices)

        intersection_count = len(true_top3_set.intersection(predicted_top3_set))
        current_top3_recall = intersection_count / 3.0
        top3_recall_sum += current_top3_recall

        is_top3_hit = int(true_top3_set == predicted_top3_set)
        if is_top3_hit:
            top3_hit_count += 1
        
        true_top3_list = [cls for cls, count in true_top3_classes]
        predicted_top3_list = predicted_top3_indices.tolist()
        is_order_match = int(true_top3_list == predicted_top3_list)
        if is_order_match:
            top3_order_match_count += 1
            top3_sims_ordered = [similarities[cls] for cls in true_top3_set]
            top3_order_similarities_sum += np.mean(top3_sims_ordered)

        result_entry = {
            "round": i,
            "true_distribution": json.dumps({int(k): int(v) for k, v in true_distribution.items()}),
            "similarities": ",".join(map(str, similarities)),
            "max_similarity": float(max_similarity),
            "max_label": int(max_label),
            "top3_recall": float(current_top3_recall),
            "top3_hit": is_top3_hit,
            "top3_order_match": is_order_match,
            "true_top3": json.dumps([int(x) for x in sorted(list(true_top3_set))]),
            "predicted_top3": json.dumps([int(x) for x in predicted_top3_indices.tolist()])
        }
        
        attack_results.append(result_entry)

        true_classes_set = set(true_distribution.keys()); true_dominant_class = max(true_distribution, key=true_distribution.get)
        predicted_dominant_class = np.argmax(similarities); predicted_top_k_classes = set(np.argsort(similarities)[-len(true_classes_set):])
        
        top1_acc = 1.0 if predicted_dominant_class == true_dominant_class else 0.0
        jaccard_sim = len(true_classes_set.intersection(predicted_top_k_classes)) / len(true_classes_set.union(predicted_top_k_classes))
        true_proportions_vector = np.array([true_distribution.get(c, 0) for c in range(config.num_clients)])
        rank_corr, _ = spearmanr(similarities, true_proportions_vector)

        if is_top3_hit:
            top3_sims = [similarities[cls] for cls in true_top3_set]
            top3_similarities_sum += np.mean(top3_sims)

        all_metrics.append({'top1_acc': top1_acc, 'jaccard': jaccard_sim, 'rank_corr': rank_corr if not np.isnan(rank_corr) else 0.0})

        true_sim_acc += similarities[np.argmax(similarities)]
        
        print('{}_th Attack Target Distribution: {}; Similarity: {}; Max Similarity: {}'.format(i, true_distribution, similarities, similarities[np.argmax(similarities)]))

    ensure_directory_exists("att_result")
    result_df = pd.DataFrame(attack_results)
    result_df.to_csv("att_result/attack_results.csv", index=False)
    print(f"\nAll attack results saved to: att_result/attack_results.csv")

    avg_top1_acc = np.mean([m['top1_acc'] for m in all_metrics])
    avg_jaccard = np.mean([m['jaccard'] for m in all_metrics])
    avg_rank_corr = np.mean([m['rank_corr'] for m in all_metrics])
    true_sim_acc = true_sim_acc / config.num_evaluation_trials

    print("\n--- Comprehensive Attack Evaluation Report (FedMask) ---"); print(f"Total Trials: {config.num_evaluation_trials}")
    print(f"Distribution Type: {config.distribution_type.upper()}")
    if config.distribution_type == 'random':
        print(f"Distribution Constraints: Max {config.max_classes_per_client} classes, Dominant class >= {config.dominant_class_min_ratio:.0%}")
    else:
        print(f"Dirichlet Configuration: alpha={config.dirichlet_alpha}, min_classes={config.dirichlet_min_classes} (random)")
    print(f"\n[1] Top-1 Dominant Class Accuracy: {avg_top1_acc:.4f}")
    print(f"[2] Average Top-K Jaccard Similarity: {avg_jaccard:.4f}")
    print(f"[3] Average Rank Correlation: {avg_rank_corr:.4f}")
    print(f"[4] Average Top-1 similarity true value: {true_sim_acc:.4f}")
    
    avg_top3_recall = top3_recall_sum / config.num_evaluation_trials
    top3_hit_rate = top3_hit_count / config.num_evaluation_trials
    
    print(f"[5] Average Top-3 Recall (Overlap Rate): {avg_top3_recall:.4f} (Avg correct {avg_top3_recall*3:.2f}/3 classes)")
    print(f"[6] Top-3 Hit Rate (Perfect Hit Rate): {top3_hit_rate:.4f} ({top3_hit_count}/{config.num_evaluation_trials} trials all 3 correct)")
    
    if top3_hit_count > 0:
        top3_match_accuracy = top3_order_match_count / top3_hit_count
        print(f"[7] Top-3 Match Accuracy (Order Accuracy): {top3_match_accuracy:.4f} (In {top3_hit_count} hits, {top3_order_match_count} order correct)")
    else:
        print(f"[7] Top-3 Match Accuracy (Order Accuracy): N/A (No perfect hit trials)")
    
    if top3_hit_count > 0:
        avg_top3_similarity = top3_similarities_sum / top3_hit_count
        print(f"[8] Average Top-3 Similarity (when hit): {avg_top3_similarity:.4f}")
    else:
        print(f"[8] Average Top-3 Similarity (when hit): N/A")

    if top3_order_match_count > 0:
        avg_top3_order_similarity = top3_order_similarities_sum / top3_order_match_count
        print(f"[9] Average Top-3 Similarity (when match): {avg_top3_order_similarity:.4f}")
    else:
        print(f"[9] Average Top-3 Similarity (when match): N/A")

if __name__ == '__main__':
    config = Config(); np.random.seed(config.seed); torch.manual_seed(config.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(config.seed)

    print("--- Configuration ---"); print(f"Device: {config.device}\n")
    print("--- ATTACKING FEDMASK ---")

    initial_model = SimpleCNN().to(config.device)
    initial_model_state = copy.deepcopy(initial_model.state_dict())

    template_masks = generate_template_masks(config, initial_model_state)
    run_comprehensive_attack_evaluation(config, initial_model_state, template_masks)