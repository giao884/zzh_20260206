# PrivPFL

## Dataset Setup

Standard datasets like MNIST, FashionMNIST, and CIFAR10 will be automatically downloaded to the `./data` directory if they are not present.

**For Purchase100 and Texas100 datasets:**

You need to manually place the dataset files. Please create a folder named `Purchase100_Texas100` inside the `data` directory and place the `.npz` files there.

Directory structure:
```
project_root/
├── data/
│   ├── Purchase100_Texas100/
│   │   ├── purchase100.npz
│   │   └── texas100.npz
│   └── ... (other datasets)
├── main.py
└── ...
```

## Running the Experiments

You can run the main script using Python. All parameters can be configured via command-line arguments.

### Basic Usage

Run with default settings (CNN on MNIST, 20 clients, Dirichlet partition):

```bash
python main.py
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| **Model & Dataset** | | | |
| `--model_name` | str | `CNN` | Model architecture. Choices: `ResNet18`, `SimpleCNN`, `CNN`, `MLP` |
| `--dataset_name` | str | `MNIST` | Dataset. Choices: `MNIST`, `FashionMNIST`, `CIFAR10`, `CIFAR100`, `GTSRB`, `texas100`, `purchase100` |
| **FL Settings** | | | |
| `--total_clients` | int | `20` | Number of total clients participating |
| `--num_rounds` | int | `2` | Number of communication rounds |
| `--local_epochs` | int | `10` | Number of local training epochs per round |
| **Generalization** | | | |
| `--generalization_threshold` | float | `0.00005` | Threshold for parameter decoupling |
| `--p2p_participants_ratio` | float | `1.0` | Ratio of P2P participants |
| `--personalized_keep_ratio` | float | `0.0` | Ratio of personalized parameters to keep |
| **Clustering & HE** | | | |
| `--clustering_method` | str | `snf_dbscan` | Clustering algorithm to use |
| `--use_homomorphic_encryption`| flag | `True` | Enable Homomorphic Encryption (default) |
| `--no_he` | flag | `False` | Disable Homomorphic Encryption |
| **Data Partition** | | | |
| `--data_partition` | str | `dirichlet` | Partition method: `dirichlet` or `noniid` |
| `--dirichlet_alpha` | float | `0.3` | Alpha parameter for Dirichlet distribution |
| `--pathological_alpha` | float | `0.5` | Internal alpha for pathological non-IID |
| `--per_client_class` | int | `2` | Number of classes per client (for pathological non-IID) |
| **Hyperparameters** | | | |
| `--cluster_eps` | float | `0.9515` | DBSCAN epsilon |
| `--dbscan_min_samples` | int | `2` | DBSCAN min_samples |
| `--snf_K` | int | `10` | SNF K nearest neighbors |
| `--snf_sigma` | float | `0.5` | SNF sigma |

### Examples

**Run ResNet18 on CIFAR10:**
```bash
python main.py --model_name ResNet18 --dataset_name CIFAR10 --num_rounds 50 --total_clients 20
```

**Run MLP on Texas100 with Non-IID partition:**
```bash
python main.py --model_name MLP --dataset_name texas100 --data_partition noniid --pathological_alpha 0.5
```

**Run without Homomorphic Encryption (faster debugging):**
```bash
python main.py --no_he
```

