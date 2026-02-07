# PPA Attack Evaluation


## Prerequisites

Install dependencies via pip:
```bash
pip install torch torchvision numpy matplotlib scipy scikit-learn pandas tqdm
```

## Usage

Run the script from the command line:

```bash
python comprehensive_attack_evaluation_v2_fedcac.py [arguments]
```

### Arguments

The script accepts the following command-line arguments to configure the experiment:

#### Basic Settings
- `--domain`: Dataset domain to use. Choices: `digit` (EMNIST/MNIST), `fashion` (FashionMNIST), `cifar10`. Default: `fashion`.
- `--num_clients`: Number of clients in the simulation. Default: `10`.
- `--local_epochs`: Number of local training epochs. Default: `5`.
- `--batch_size`: Batch size for training. Default: `64`.
- `--lr`: Learning rate. Default: `0.01`.
- `--personalization_rate`: Personalization rate for FedCAC (percentage of parameters to mask). Default: `5.0`.
- `--num_evaluation_trials`: Number of independent attack trials to run. Default: `100`.
- `--template_runs_per_class`: Number of runs to average when generating class templates. Default: `20`.
- `--distribution_type`: Method to generate non-IID data. Choices: `random`, `dirichlet`. Default: `random`.

**Random Distribution Options (only used if `--distribution_type random`):**
- `--total_samples_per_client`: Total number of samples per client. Default: `200`.
- `--min_classes_per_client`: Minimum number of classes a client holds. Default: `3`.
- `--max_classes_per_client`: Maximum number of classes a client holds. Default: `5`.
- `--dominant_class_min_ratio`: Minimum ratio of the dominant class in the client's data. Default: `0.7` (70%).

**Dirichlet Distribution Options (only used if `--distribution_type dirichlet`):**
- `--dirichlet_alpha`: Concentration parameter for Dirichlet distribution. Smaller values mean more heterogeneity. Default: `0.5`.
- `--dirichlet_min_classes`: Minimum number of classes to include in the Dirichlet distribution. Default: `3`.
- `--dirichlet_total_samples`: Total samples per client when using Dirichlet. Default: `200`.

## Example Commands

**1. Basic run with default settings (FashionMNIST):**
```bash
python comprehensive_attack_evaluation_v2_fedcac.py
```

**2. Run on Digit dataset (MNIST/EMNIST) with Dirichlet distribution:**
```bash
python comprehensive_attack_evaluation_v2_fedcac.py --domain digit --distribution_type dirichlet --dirichlet_alpha 0.1
```

**3. Run with custom random distribution settings:**
```bash
python comprehensive_attack_evaluation_v2_fedcac.py --distribution_type random --min_classes_per_client 2 --max_classes_per_client 4 --dominant_class_min_ratio 0.8
```
