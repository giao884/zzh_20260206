import numpy as np

class LSHHandler:
    def __init__(self, psi_dim, param_dim, label_dim, w=1.0, hash_length=10):
        self.w = w
        self.hash_length = hash_length
        
        self.cosine_a = self._generate_cosine_lsh_param(psi_dim)
        
        self.euclid_a_param = self._generate_e2lsh_param(param_dim)
        self.euclid_a_label = self._generate_e2lsh_param(label_dim)

    def _generate_cosine_lsh_param(self, dim):
        a_matrix = np.random.randn(self.hash_length, dim)
        for i in range(self.hash_length):
            a_matrix[i] = a_matrix[i] / np.linalg.norm(a_matrix[i])
        return a_matrix

    def _generate_e2lsh_param(self, dim):
        return np.random.randn(self.hash_length, dim)

    def compute_psi_hash(self, psi, b_i):
        dot_products = np.dot(self.cosine_a, psi)
        return np.floor((dot_products + b_i) / self.w).astype(int)

    def compute_param_hash(self, param_vec, b_i):
        dot_products = np.dot(self.euclid_a_param, param_vec)
        return np.floor((dot_products + b_i) / self.w).astype(int)

    def compute_label_hash(self, soft_label, b_i):
        dot_products = np.dot(self.euclid_a_label, soft_label)
        return np.floor((dot_products + b_i) / self.w).astype(int)