import os
import csv
import torch
import random
import numpy as np
import pandas as pd
from datetime import datetime
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, CIFAR100, GTSRB
from sklearn.model_selection import train_test_split
from torchvision.transforms import ToTensor
from torchvision import transforms


class NumpyDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform

        self._check_onehot_labels()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.features[idx]
        y_onehot = self.labels[idx]

        x = torch.tensor(x, dtype=torch.float32)

        y_flat = np.asarray(y_onehot).flatten()
        y_class = np.argmax(y_flat)
        y = torch.tensor(y_class, dtype=torch.long)

        if self.transform:
            x = self.transform(x)
        return x, y

    def _check_onehot_labels(self):
        sample_indices = np.random.choice(len(self.labels), min(10, len(self.labels)), replace=False)
        for idx in sample_indices:
            y = self.labels[idx]
            y_flat = np.asarray(y).flatten()

def load_dataset(dataset_name):
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2023, 0.1994, 0.2010)
    CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
    CIFAR100_STD = (0.2675, 0.2565, 0.2761)

    cifar10_train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    ])
    
    cifar10_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    ])
    
    # cifar100_train_transform = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD)
    # ])

    cifar100_train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'), 
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), 
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    ])
    
    cifar100_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD)
    ])

    gtsrb_train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
    ])
    
    gtsrb_test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
    ])

    if dataset_name == 'MNIST':
        train_data = MNIST(root='./data', train=True, transform=ToTensor(), download=True)
        test_data = MNIST(root='./data', train=False, transform=ToTensor(), download=True)
    elif dataset_name == 'FashionMNIST':
        train_data = FashionMNIST(root='./data', train=True, transform=ToTensor(), download=True)
        test_data = FashionMNIST(root='./data', train=False, transform=ToTensor(), download=True)
    elif dataset_name == 'CIFAR10':
        train_data = CIFAR10(root='./data', train=True, transform=cifar10_train_transform, download=True)
        test_data = CIFAR10(root='./data', train=False, transform=cifar10_test_transform, download=True)
    elif dataset_name == 'CIFAR100':
        train_data = CIFAR100(root='./data', train=True, transform=cifar100_train_transform, download=True)
        test_data = CIFAR100(root='./data', train=False, transform=cifar100_test_transform, download=True)
    elif dataset_name == 'GTSRB':
        train_data = GTSRB(root='./data', split='train', transform=gtsrb_train_transform, download=True)
        test_data = GTSRB(root='./data', split='test', transform=gtsrb_test_transform, download=True)
    elif dataset_name == 'texas100':
        data = np.load('./data/Purchase100_Texas100/purchase100.npz')
        features = data['features']
        labels = data['labels']
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        train_data = NumpyDataset(X_train, y_train)
        test_data = NumpyDataset(X_test, y_test)
    elif dataset_name == 'purchase100':
        data = np.load('./data/Purchase100_Texas100/purchase100.npz')
        features = data['features']
        labels = data['labels']
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        train_data = NumpyDataset(X_train, y_train)
        test_data = NumpyDataset(X_test, y_test)
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    return train_data, test_data

def pathological_non_iid_partition(train_dataset, test_dataset, num_clients, num_classes, 
                                     num_classes_per_client=2, internal_alpha=0.5):
    train_class_indices = {c: [] for c in range(num_classes)}
    for i, (_, y) in enumerate(train_dataset):
        if isinstance(y, torch.Tensor):
            y = y.item()
        if y < num_classes:
            train_class_indices[y].append(i)
    
    test_class_indices = {c: [] for c in range(num_classes)}
    for i, (_, y) in enumerate(test_dataset):
        if isinstance(y, torch.Tensor):
            y = y.item()
        if y < num_classes:
            test_class_indices[y].append(i)

    client_class_assignments = {i: [] for i in range(num_clients)}

    if num_clients > num_classes:
        num_class_sets = num_classes // num_classes_per_client
        all_classes = list(range(num_classes))
        random.shuffle(all_classes)
        class_sets = [sorted(all_classes[i*num_classes_per_client:(i+1)*num_classes_per_client]) for i in range(num_class_sets)]
        for client_id in range(num_clients):
            client_class_assignments[client_id] = class_sets[client_id % num_class_sets]
    else:

        all_classes = list(range(num_classes))
        random.shuffle(all_classes)
        base_count = num_classes // num_clients
        remainder = num_classes % num_clients
        current_class_idx = 0
        for client_id in range(num_clients):
            num_to_assign = base_count + 1 if client_id < remainder else base_count
            assigned_classes = all_classes[current_class_idx : current_class_idx + num_to_assign]
            client_class_assignments[client_id] = sorted(assigned_classes)
            current_class_idx += num_to_assign

    clients_train_data = []
    clients_test_data = []

    SAMPLES_PER_CLASS_TRAIN = 500
    SAMPLES_PER_CLASS_TEST = 100

    for client_id in range(num_clients):
        selected_classes = client_class_assignments[client_id]
        num_client_classes = len(selected_classes)

        total_train_samples = num_client_classes * SAMPLES_PER_CLASS_TRAIN
        total_test_samples = num_client_classes * SAMPLES_PER_CLASS_TEST

        if num_client_classes > 1:
            proportions_train = np.random.dirichlet([internal_alpha] * num_client_classes)
            proportions_test = np.random.dirichlet([internal_alpha] * num_client_classes)
        else:
            proportions_train = np.array([1.0])
            proportions_test = np.array([1.0])

        skewed_samples_train = (proportions_train * total_train_samples).astype(int)
        skewed_samples_test = (proportions_test * total_test_samples).astype(int)
        
        train_indices = []
        test_indices = []
        
        for i, c in enumerate(selected_classes):
            available_train_indices = train_class_indices.get(c, [])
            num_to_take_train = min(skewed_samples_train[i], len(available_train_indices))
            if num_to_take_train > 0:
                train_indices.extend(random.sample(available_train_indices, num_to_take_train))

            available_test_indices = test_class_indices.get(c, [])
            num_to_take_test = min(skewed_samples_test[i], len(available_test_indices))
            if num_to_take_test > 0:
                test_indices.extend(random.sample(available_test_indices, num_to_take_test))
            
        clients_train_data.append([train_dataset[i] for i in train_indices])
        clients_test_data.append([test_dataset[i] for i in test_indices])
        
    return clients_train_data, clients_test_data


def dirichlet_partition(train_dataset, test_dataset, num_clients, num_classes, alpha=0.5, min_samples=32):
    train_class_indices = {c: [] for c in range(num_classes)}
    for i, (_, y) in enumerate(train_dataset):
        if isinstance(y, torch.Tensor):
            y = y.item()
        if 0 <= y < num_classes:
            train_class_indices[y].append(i)
    
    test_class_indices = {c: [] for c in range(num_classes)}
    for i, (_, y) in enumerate(test_dataset):
        if isinstance(y, torch.Tensor):
            y = y.item()
        if 0 <= y < num_classes:
            test_class_indices[y].append(i)

    train_client_indices = [[] for _ in range(num_clients)]
    test_client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        proportions = np.random.dirichlet([alpha] * num_clients)

        train_idx = train_class_indices[c]
        np.random.shuffle(train_idx)
        splits = (proportions * len(train_idx)).astype(int)
        splits[-1] = len(train_idx) - np.sum(splits[:-1])
        start = 0
        for client_id, num in enumerate(splits):
            train_client_indices[client_id].extend(train_idx[start:start+num])
            start += num

        test_idx = test_class_indices[c]
        np.random.shuffle(test_idx)
        splits = (proportions * len(test_idx)).astype(int)
        splits[-1] = len(test_idx) - np.sum(splits[:-1])
        start = 0
        for client_id, num in enumerate(splits):
            test_client_indices[client_id].extend(test_idx[start:start+num])
            start += num

    all_train_indices_flat = [item for sublist in train_client_indices for item in sublist]
    all_test_indices_flat = [item for sublist in test_client_indices for item in sublist]
    
    available_train_indices = list(set(range(len(train_dataset))) - set(all_train_indices_flat))
    available_test_indices = list(set(range(len(test_dataset))) - set(all_test_indices_flat))
    
    random.shuffle(available_train_indices)
    random.shuffle(available_test_indices)

    for client_id in range(num_clients):
        if len(train_client_indices[client_id]) < min_samples:
            need = min_samples - len(train_client_indices[client_id])
            extra = available_train_indices[:need]
            train_client_indices[client_id].extend(extra)
            available_train_indices = available_train_indices[need:]

        if len(test_client_indices[client_id]) < min_samples:
            need = min_samples - len(test_client_indices[client_id])
            extra = available_test_indices[:need]
            test_client_indices[client_id].extend(extra)
            available_test_indices = available_test_indices[need:]

    clients_train_data = [[train_dataset[i] for i in idxs] for idxs in train_client_indices]
    clients_test_data = [[test_dataset[i] for i in idxs] for idxs in test_client_indices]

    return clients_train_data, clients_test_data

def load_root_dataset():
    return CIFAR10(root='./data', train=False, transform=ToTensor(), download=True)

def extract_data_distribution(dataset, num_classes):
    labels = [y for _, y in dataset]
    class_counts = np.bincount(labels, minlength=num_classes)
    return class_counts / np.sum(class_counts)

def save_evaluation_results(clients, total_clients, filename=None):
    save_folder = "eval_result"
    os.makedirs(save_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"final_evaluation_results_{timestamp}.csv"
    
    full_path = os.path.join(save_folder, filename)
    
    with open(full_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Evaluation Metric", "Client ID", "Accuracy Value"])
        writer.writerow(["Local Model Accuracy", "Client ID", "Accuracy"])
        
        acc_total = 0.0
        local_acc_dict = {}
        for c in clients:
            acc = c.compute_accuracy()
            if isinstance(acc, torch.Tensor):
                acc = acc.cpu().item()
            local_acc_dict[c.client_id] = acc
            acc_total += acc
            writer.writerow(["Local Model Accuracy", c.client_id, f"{acc:.4f}"])
        
        avg_acc = acc_total / total_clients if total_clients > 0 else 0.0
        writer.writerow(["Local Model Accuracy", "Average", f"{avg_acc:.4f}"])
        writer.writerow([])

        writer.writerow(["Evaluation Metric", "Source Client ID", "Target Client ID", "Accuracy Value"])
        writer.writerow(["Cross-Client Accuracy", "Source Client", "Target Client", "Accuracy"])
        
        cross_acc_avg_dict = {}
        for c in clients:
            acc_total = 0.0
            count = 0
            for cc in clients:
                if cc.client_id != c.client_id:
                    acc = c.compute_accuracy(local_mode=False, external_dataset=cc.test_dataset)
                    if isinstance(acc, torch.Tensor):
                        acc = acc.cpu().item()
                    writer.writerow(["Cross-Client Accuracy", c.client_id, cc.client_id, f"{acc:.4f}"])
                    acc_total += acc
                    count += 1
            
            avg_acc = acc_total / count if count > 0 else 0.0
            cross_acc_avg_dict[c.client_id] = avg_acc
            writer.writerow(["Cross-Client Accuracy", c.client_id, "Average", f"{avg_acc:.4f}"])
            writer.writerow([])

        writer.writerow(["Overall Statistics", "Metric", "Value"])
        writer.writerow(["Overall Statistics", "Average Local Accuracy", f"{avg_acc:.4f}"])
        
        cross_avg_list = list(cross_acc_avg_dict.values())
        cross_avg_list = [
            val.cpu().item() if isinstance(val, torch.Tensor) else val 
            for val in cross_avg_list
        ]
        overall_cross_avg = np.mean(cross_avg_list) if cross_avg_list else 0.0
        writer.writerow(["Overall Statistics", "Average Cross-Client Accuracy", f"{overall_cross_avg:.4f}"])
    
    print(f"Evaluation results saved to: {full_path}")

def save_loss_results(total_loss, avg_loss):
    save_folder = "eval_result"
    os.makedirs(save_folder, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"p2p_loss_{timestamp}.csv"
    full_path = os.path.join(save_folder, filename)
    
    with open(full_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Round", "Loss"])
        for i, t in enumerate(total_loss):
            writer.writerow([i + 1, f"{t:.4f}"])
        writer.writerow([])
        writer.writerow(["Average Loss", f"{avg_loss:.4f}"])
    
    print(f"Loss saved to: {full_path}")

def save_acc_results(total_acc, avg_acc, filename_prefix="total_acc"):
    save_folder = "eval_result"
    os.makedirs(save_folder, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.csv"
    full_path = os.path.join(save_folder, filename)
    
    with open(full_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Round", "acc"])
        for i, t in enumerate(total_acc):
            writer.writerow([i + 1, f"{t:.4f}"])
        writer.writerow([])
        writer.writerow(["Average acc", f"{avg_acc:.4f}"])
    
    print(f"Acc saved to: {full_path}")

def save_timing_results(timing_list, avg_time, p2p_method):
    save_folder = "eval_result"
    os.makedirs(save_folder, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"p2p_timing_{p2p_method}_{timestamp}.csv"
    full_path = os.path.join(save_folder, filename)
    
    with open(full_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Round", "Time (seconds)"])
        for i, t in enumerate(timing_list):
            writer.writerow([i + 1, f"{t:.4f}"])
        writer.writerow([])
        writer.writerow(["Average Time", f"{avg_time:.4f}"])
    
    print(f"Personalized timing saved to: {full_path}")

def save_p2p_participants_results(p2p_participants_list, p2p_method):
    save_folder = "eval_result"
    os.makedirs(save_folder, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"p2p_participants_{p2p_method}_{timestamp}.csv"
    full_path = os.path.join(save_folder, filename)
    
    with open(full_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Round", "participants list"])
        for i, t in enumerate(p2p_participants_list):
            writer.writerow([i + 1, str(t)])
        writer.writerow([])
    
    print(f"Personalized participants saved to: {full_path}")