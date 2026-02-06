import numpy as np
import torch
import tenseal as ts
from sklearn.cluster import DBSCAN, SpectralClustering
from torch.utils.data import DataLoader
from sklearn.metrics import pairwise_distances
import snf
import time

class AggregationServer:
    def __init__(self, global_model):
        self.global_model = global_model
        self.clusters = {}
        self.client_hashes = {}
        self.client_params = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.he_context = None

    def set_he_context(self, context):
        self.he_context = context

    def clear_collected_data(self):
        self.client_hashes.clear()
        self.client_params.clear()

    def collect_client_hashes(self, client_id, hash_dict):
        self.client_hashes[client_id] = hash_dict

    def collect_client_parameters(self, client_id, param_vector):
        self.client_params[client_id] = param_vector

    def cluster_clients_snf_dbscan(self, K=15, sigma=0.5, eps=0.5, min_samples=2):

        if not self.client_hashes:
            raise ValueError("No client hashes collected! Cannot perform clustering.")

        client_ids = list(self.client_hashes.keys())
        if len(client_ids) < min_samples:
            print(f"The number of clients ({len(client_ids)}) are less than ({min_samples})")
            self.clusters = {0: client_ids}
            return self.clusters

        data_dist_hashes = np.array([self.client_hashes[cid]["data_distribution"] for cid in client_ids])
        model_param_hashes = np.array([self.client_hashes[cid]["model_param"] for cid in client_ids])
        soft_label_hashes = np.array([self.client_hashes[cid]["soft_label"] for cid in client_ids])
        dist_data = pairwise_distances(data_dist_hashes, metric='hamming')
        affinity_data = np.exp(-dist_data**2 / (2 * sigma**2))
        dist_model = pairwise_distances(model_param_hashes, metric='euclidean')
        affinity_model = np.exp(-dist_model**2 / (2 * sigma**2))
        dist_label = pairwise_distances(soft_label_hashes, metric='hamming')
        affinity_label = np.exp(-dist_label**2 / (2 * sigma**2))

        affinity_networks = [affinity_data, affinity_model, affinity_label]
        fused_graph = snf.snf(affinity_networks, K=K)
        fused_graph = np.clip(fused_graph, 0, 1)

        distance_matrix = 1 - fused_graph
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        labels = dbscan.fit_predict(distance_matrix)

        self.clusters = {}
        noise_points_count = 0
        for i, client_id in enumerate(client_ids):
            cluster_id = labels[i]
            if cluster_id == -1:
                cluster_id = f"noise_{noise_points_count}"
                noise_points_count += 1

            if cluster_id not in self.clusters:
                self.clusters[cluster_id] = []
            self.clusters[cluster_id].append(client_id)
            
        return self.clusters

    def personalized_aggregation_he(self, clients, cluster_id):
        if self.he_context is None:
            raise ValueError("HE context has not been set in the server.")
        if cluster_id not in self.clusters:
            raise ValueError(f"ID {cluster_id} is not in clusters.")

        cluster_client_ids = self.clusters[cluster_id]
        if not cluster_client_ids or len(cluster_client_ids) < 1: return

        cluster_clients = [c for c in clients if c.client_id in cluster_client_ids]
        num_clients_in_cluster = len(cluster_clients)
        
        if not cluster_clients: return

        print(f" {cluster_id} clustering start, including {num_clients_in_cluster} clients...")
        
        encrypted_params = [c.get_encrypted_parameters(self.he_context) for c in cluster_clients]
        
        aggregated_enc_param = sum(encrypted_params)

        if num_clients_in_cluster > 0:
            inverse_len = 1.0 / num_clients_in_cluster
            avg_enc_param = aggregated_enc_param * inverse_len
        else:
            avg_enc_param = aggregated_enc_param

        for client in cluster_clients:
            client.update_model_from_encrypted_vector(avg_enc_param)
            
        print(f"Cluster {cluster_id} finished.")

    def personalized_aggregation_plaintext(self, clients, cluster_id):
        if cluster_id not in self.clusters:
            raise ValueError(f"Cluster {cluster_id} not found in server clusters.")

        cluster_client_ids = self.clusters[cluster_id]
        cluster_clients = [c for c in clients if c.client_id in cluster_client_ids]
        
        if len(cluster_clients) < 1:
            return None

        param_vectors = [c.get_model_parameters() for c in cluster_clients]

        avg_param = np.mean(param_vectors, axis=0)
        
        for client in cluster_clients:
            client.update_model_from_vector(avg_param)
            
        print(f"Cluster {cluster_id} finished, including {len(cluster_clients)} clients.")
        return avg_param

    def personalized_aggregation(self, clients, cluster_id, use_he=False):
        if use_he:
            # Call the secure HE version
            self.personalized_aggregation_he(clients, cluster_id)
        else:
            # Call the insecure plaintext version
            self.personalized_aggregation_plaintext(clients, cluster_id)

    def generalization_aggregation_mask_based(self, clients):

        masks = []
        param_vectors = []
        for client in clients:
            if client.mask is None:
                raise ValueError(f"Client {client.client_id} have not performed parameter decoupling.")
            masks.append(client.mask)
            param_vectors.append(client.get_model_parameters())
        
        masks = np.array(masks)
        param_vectors = np.array(param_vectors)

        masked_params = param_vectors * masks
        sum_masked_params = np.sum(masked_params, axis=0)

        total_mask = np.sum(masks, axis=0)

        aggregated_params = np.zeros_like(sum_masked_params)
        non_zero_mask = total_mask > 0
        aggregated_params[non_zero_mask] = sum_masked_params[non_zero_mask] / total_mask[non_zero_mask]

        for client in clients:
            current_params = client.get_model_parameters()
            new_params = current_params * (1 - client.mask) + aggregated_params * client.mask
            client.update_model_from_vector(new_params)
        
        print(f"Generalized parameter aggregation finished, including {len(clients)} clients.")

    def personalization_aggregation_mask_based(self, clients, cluster_id):

        masks = []
        param_vectors = []
        for client in clients:
            if client.mask is None:
                raise ValueError(f"clients {client.client_id} have not performed parameter decoupling.")

            if hasattr(client, 'personalized_agg_mask') and client.personalized_agg_mask is not None:
                personalized_mask = client.personalized_agg_mask
            else:
                personalized_mask = 1 - client.mask
                
            masks.append(personalized_mask)
            param_vectors.append(client.get_model_parameters())
        
        masks = np.array(masks)
        param_vectors = np.array(param_vectors)

        masked_params = param_vectors * masks
        sum_masked_params = np.sum(masked_params, axis=0)
        
        total_mask = np.sum(masks, axis=0)
        
        aggregated_params = np.zeros_like(sum_masked_params)
        non_zero_mask = total_mask > 0
        aggregated_params[non_zero_mask] = sum_masked_params[non_zero_mask] / total_mask[non_zero_mask]
        
        for client in clients:
            current_params = client.get_model_parameters()

            if hasattr(client, 'personalized_agg_mask') and client.personalized_agg_mask is not None:
                personalized_mask = client.personalized_agg_mask
            else:
                personalized_mask = 1 - client.mask
            
            new_params = current_params * (1 - personalized_mask) + aggregated_params * personalized_mask
            client.update_model_from_vector(new_params)
        
        print(f"Pesonalized parameter aggregation finished, including {len(clients)} clients in cluster {cluster_id}.")


    def p2p_generalization_simplified(self, clients, participants, threshold=0.00005):
        participant_clients = [c for c in clients if c.client_id in participants]
        for client in participant_clients:
            if client.mask is None:
                client.parameter_decoupling_new(threshold=threshold)

        masks = [client.mask for client in participant_clients]
        param_vectors = [client.get_model_parameters() for client in participant_clients]
        
        param_length = len(param_vectors[0])
        total_mask = np.sum(masks, axis=0)
        
        gsp_params = np.array([p * m for p, m in zip(param_vectors, masks)])
        sum_param = np.sum(gsp_params, axis=0)

        avg_param = np.zeros_like(sum_param)
        non_zero_mask = total_mask != 0
        avg_param[non_zero_mask] = sum_param[non_zero_mask] / total_mask[non_zero_mask]

        for client in participant_clients:
            current_param = client.get_model_parameters()
            new_param = current_param * (1 - client.mask) + avg_param * client.mask
            client.update_model_from_vector(new_param)


    def generalization_centralized(self, clients, participants, threshold=0.00005):
        participant_clients = [c for c in clients if c.client_id in participants]
        for client in participant_clients:
            if client.mask is None:
                client.parameter_decoupling_new(threshold=threshold)

        masks = [client.mask.astype(np.float64) for client in participant_clients]

        param_vectors = [client.get_model_parameters() for client in participant_clients]
        gsp_params = [p * m for p, m in zip(param_vectors, masks)]

        num_participants = len(participant_clients)
        if num_participants < 2:
            print("The number of participants is less than 2, skip generalization aggregation.")
            return 0
        param_length = len(param_vectors[0])

        for i in range(num_participants):
            for j in range(i + 1, num_participants):
                
                cover_mask = np.random.uniform(-10, 10, param_length) 
                cover_param = np.random.uniform(-0.1, 0.1, param_length)

                masks[i] += cover_mask
                gsp_params[i] += cover_param

                masks[j] -= cover_mask
                gsp_params[j] -= cover_param

        total_mask = np.sum(masks, axis=0)
        sum_param = np.sum(gsp_params, axis=0)
        avg_param = np.zeros_like(sum_param)
        non_zero_idx = np.abs(total_mask) > 1e-6 
        avg_param[non_zero_idx] = sum_param[non_zero_idx] / total_mask[non_zero_idx]

        for client in participant_clients:
            current_param = client.get_model_parameters()
            new_param = current_param * (1 - client.mask) + avg_param * client.mask
            client.update_model_from_vector(new_param)