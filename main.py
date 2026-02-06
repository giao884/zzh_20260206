import LSH
import time
import torch
import random
import numpy as np
import tenseal as ts
from federated_client import FederatedClient
from aggregation_server import AggregationServer
from models import FedAvgCNN, SimpleCNN, ResNet18, ResNet18_CIFAR, texas_purchase_mlp
from data_utils import pathological_non_iid_partition, dirichlet_partition, load_dataset
from data_utils import save_evaluation_results, save_loss_results, save_acc_results, save_timing_results, save_p2p_participants_results
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Federated Learning with LSH Clustering")
    
    # Model and Dataset
    parser.add_argument('--model_name', type=str, default='CNN', choices=['ResNet18', 'SimpleCNN', 'CNN', 'MLP'], help='Model architecture')
    parser.add_argument('--dataset_name', type=str, default='MNIST', choices=['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100', 'GTSRB', 'texas100', 'purchase100'], help='Dataset name')
    
    # Federated Learning Settings
    parser.add_argument('--total_clients', type=int, default=20, help='Number of total clients')
    parser.add_argument('--num_rounds', type=int, default=2, help='Number of communication rounds')
    parser.add_argument('--local_epochs', type=int, default=10, help='Number of local epochs')
    
    # Generalization & Personalization
    parser.add_argument('--generalization_threshold', type=float, default=0.00005, help='Generalization threshold for parameter decoupling')
    parser.add_argument('--p2p_participants_ratio', type=float, default=1.0, help='Ratio of P2P participants')
    parser.add_argument('--personalized_keep_ratio', type=float, default=0.0, help='Ratio of personalized parameters to keep')
    
    # Clustering & Encryption
    parser.add_argument('--clustering_method', type=str, default='snf_dbscan', help='Clustering method')
    parser.add_argument('--use_homomorphic_encryption', action='store_true', help='Use Homomorphic Encryption')
    parser.add_argument('--no_he', action='store_false', dest='use_homomorphic_encryption', help='Disable Homomorphic Encryption')
    parser.set_defaults(use_homomorphic_encryption=True)

    # Data Partitioning
    parser.add_argument('--data_partition', type=str, default='dirichlet', choices=['dirichlet', 'noniid'], help='Data partition method')
    parser.add_argument('--dirichlet_alpha', type=float, default=0.3, help='Dirichlet distribution alpha')
    parser.add_argument('--per_client_class', type=int, default=2, help='Number of classes per client for pathological non-iid')
    parser.add_argument('--pathological_alpha', type=float, default=0.5, help='Internal alpha for pathological non-iid')
    
    # SNF & DBSCAN Hyperparameters
    parser.add_argument('--cluster_eps', type=float, default=0.9515, help='DBSCAN eps')
    parser.add_argument('--dbscan_min_samples', type=int, default=2, help='DBSCAN min_samples')
    parser.add_argument('--snf_K', type=int, default=10, help='SNF K nearest neighbors')
    parser.add_argument('--snf_sigma', type=float, default=0.5, help='SNF sigma')

    return parser.parse_args()


def main():

    args = get_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU")

    model_name = args.model_name
    dataset_name = args.dataset_name

    dataset_config = {
        'MNIST': 10, 'FashionMNIST': 10, 'CIFAR10': 10,
        'CIFAR100': 100, 'GTSRB': 43, 'texas100': 100, 'purchase100': 100
    }
    if dataset_name not in dataset_config:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    num_classes = dataset_config[dataset_name]
    total_clients = args.total_clients
    num_rounds = args.num_rounds
    local_epochs = args.local_epochs
    generalization_threshold = args.generalization_threshold
    p2p_participants_ratio = args.p2p_participants_ratio
    personalized_keep_ratio = args.personalized_keep_ratio

    clustering_method = args.clustering_method
    use_homomorphic_encryption = args.use_homomorphic_encryption

    data_partition = args.data_partition
    dirichlet_alpha = args.dirichlet_alpha
    per_client_class = args.per_client_class
    pathological_alpha = args.pathological_alpha
    cluster_eps = args.cluster_eps
    dbscan_min_samples = args.dbscan_min_samples
    snf_K = args.snf_K
    snf_sigma = args.snf_sigma

    print("\n===== Experiments =====")
    print('Model:', model_name, '| Dataset:', dataset_name)
    print('Clustering Method:', clustering_method, '| HE:', use_homomorphic_encryption)
    print('Data Partition:', data_partition)
    print('Dirichlet_alpha:', dirichlet_alpha, '| Pathological_alpha:', pathological_alpha)
    print('Cluster_eps:', cluster_eps, '| DBSCAN_min_samples:', dbscan_min_samples)
    print('Generalization_threshold:', generalization_threshold, '| p2p_participants_ratio:', p2p_participants_ratio)
    print('Personalized_keep_ratio:', personalized_keep_ratio)

    loss_per_round = []
    acc_per_round = []
    global_acc_per_round = []
    he_context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=16384,
        coeff_mod_bit_sizes=[60, 40, 40, 40, 60]
    )
    he_context.generate_galois_keys()
    he_context.global_scale = 2**40
    he_sk = he_context.secret_key()
    public_he_context = he_context.copy()
    public_he_context.make_context_public()

    pairwise_seeds = {}
    for i in range(total_clients):
        for j in range(i + 1, total_clients):
            seed = random.randint(0, 2**32 - 1)
            pairwise_seeds[tuple(sorted((i, j)))] = seed

    train_dataset, test_dataset = load_dataset(dataset_name)
    root_dataset = test_dataset
    if data_partition == 'dirichlet':
        clients_train_datasets, clients_test_data = dirichlet_partition(train_dataset, test_dataset, total_clients, alpha=dirichlet_alpha, num_classes=num_classes, min_samples=32)
    elif data_partition == 'noniid':
        clients_train_datasets, clients_test_data = pathological_non_iid_partition(train_dataset, test_dataset, total_clients, num_classes=num_classes, num_classes_per_client=per_client_class, internal_alpha=pathological_alpha)
    else:
        raise ValueError(f"Unknown data partition: {data_partition}")

    if model_name == 'ResNet18':
        if dataset_name in ['CIFAR10', 'CIFAR100',]:
            model = ResNet18_CIFAR(num_classes=num_classes)
        elif dataset_name in [ 'GTSRB']:
            model = ResNet18(num_classes=num_classes)
    elif model_name == 'SimpleCNN':
        model = SimpleCNN(num_classes=num_classes)
    elif model_name == 'CNN':
        model = FedAvgCNN(num_classes=num_classes)
    elif model_name == 'MLP':
        model = texas_purchase_mlp()

    w = 1
    param_dim = sum(np.prod(p.shape) for p in model.parameters())
    R2lsh_dis = LSH.pstable(w, num_classes, total_clients, 2, 1, 5)
    R2lsh_sof = LSH.pstable(w, 10 * num_classes, total_clients, 2, 1, 10)
    E2lsh = LSH.pstable(w, param_dim, total_clients, 2, 1, 10)

    clients = [
        FederatedClient(
            client_id=i,
            model=model.__class__(num_classes=num_classes),
            local_data=clients_train_datasets[i],
            root_dataset=root_dataset,
            test_data=clients_test_data[i],
            E2lsh = E2lsh,
            R2lsh_dis = R2lsh_dis,
            R2lsh_sof = R2lsh_sof,
            num_classes=num_classes
        ) for i in range(total_clients)
    ]

    for client in clients:
        client.set_he_key(he_sk)

    server = AggregationServer(global_model=model)
    server.set_he_context(public_he_context)
    cluster_info_per_round = []

    for round_idx in range(num_rounds):
        print(f"\n===== {round_idx + 1}/{num_rounds} =====")

        loss_round = 0
        for c in clients:
            c_loss = c.local_train(epochs=local_epochs)
            loss_round += c_loss
        loss_round = loss_round / total_clients
        loss_per_round.append(loss_round)
        print(f"loss: {loss_round:.4f}")

        for client in clients:
            client.parameter_decoupling_new(threshold=generalization_threshold, keep_ratio=personalized_keep_ratio)

        server.generalization_aggregation_mask_based(clients)
        server.clear_collected_data()
        for c in clients:
            client_hashes = c.generate_hashes_personalized()
            server.collect_client_hashes(c.client_id, client_hashes)

        clusters = server.cluster_clients_snf_dbscan(K=snf_K, sigma=snf_sigma, eps=cluster_eps, min_samples=dbscan_min_samples)

        round_cluster_info = {
            "round": round_idx + 1,
            "num_clusters": len(clusters),
            "clusters": {cid: client_ids for cid, client_ids in clusters.items()},
            "cluster_sizes": {cid: len(client_ids) for cid, client_ids in clusters.items()}
        }
        cluster_info_per_round.append(round_cluster_info)

        for cluster_id, client_ids in clusters.items():
            cluster_clients = [clients[cid] for cid in client_ids]
            server.personalized_aggregation(cluster_clients, cluster_id, use_he=use_homomorphic_encryption)
            # server.personalization_aggregation_mask_based(cluster_clients, cluster_id)

        if (round_idx + 1) % 5 == 0 or round_idx == 0:
            round_accuracy = sum(c.compute_accuracy() for c in clients) / len(clients)
            print(f"Local acc: {round_accuracy:.4f}")

            global_acc = sum(c.compute_accuracy(local_mode=False, external_dataset=test_dataset) for c in clients) / len(clients)
            print(f"Total acc: {global_acc:.4f}")
            acc_per_round.append(round_accuracy)
            global_acc_per_round.append(global_acc)

    print("\n===== Evaluation =====")
    print("\n===== Local ACC =====")
    acc_total = 0
    for c in clients:
        acc = c.compute_accuracy()
        print(f"Client {c.client_id} local accuracy: {acc:.4f}")
        acc_total += acc
    avg_acc = acc_total / total_clients
    print(f"Average accuracy: {avg_acc:.4f}")
    save_acc_results(acc_per_round, avg_acc, filename_prefix="local_acc")
    if global_acc_per_round:
        global_avg_acc = sum(global_acc_per_round) / len(global_acc_per_round)
        save_acc_results(global_acc_per_round, global_avg_acc, filename_prefix="global_acc")

    print("\n===== Cross client ACC =====")
    for c in clients:
        acc_total = 0
        for cc in clients:
            if cc.client_id != c.client_id:
                acc = c.compute_accuracy(local_mode=False, external_dataset=cc.test_dataset)
                print(f"Client {c.client_id} on client {cc.client_id} accuracy: {acc:.4f}")
                acc_total += acc
        avg_acc = acc_total / (total_clients - 1)
        print(f"Client {c.client_id} average cross client accuracy: {avg_acc:.4f}")

    if loss_per_round:
        print("\n===== Avg Loss =====")
        total_loss = sum(loss_per_round)
        avg_loss = total_loss / len(loss_per_round)
        print(f"Average loss: {avg_loss:.4f}")
        save_loss_results(loss_per_round, avg_loss)

    if cluster_info_per_round:
        save_p2p_participants_results(cluster_info_per_round, "cluster")
    
    print("\n===== Experiments =====")
    print('Model:', model_name, '| Dataset:', dataset_name)
    print('Clustering Method:', clustering_method, '| HE:', use_homomorphic_encryption)
    print('Data Partition:', data_partition)
    print('Dirichlet_alpha:', dirichlet_alpha, '| Pathological_alpha:', pathological_alpha)
    print('Cluster_eps:', cluster_eps, '| DBSCAN_min_samples:', dbscan_min_samples)
    print('Generalization_threshold:', generalization_threshold, '| p2p_participants_ratio:', p2p_participants_ratio)
    print('Personalized_keep_ratio:', personalized_keep_ratio)

    save_file_name = f"eval_{time.time()}_" + str(generalization_threshold) + "_" + str(p2p_participants_ratio) + ".csv"
    save_evaluation_results(clients, total_clients, filename=save_file_name)

if __name__ == "__main__":
    main()