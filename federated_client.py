import torch
import numpy as np
import tenseal as ts
from torch.utils.data import DataLoader
import torchmetrics


class FederatedClient:
    def __init__(self, client_id, model, local_data, root_dataset, test_data, R2lsh_dis, R2lsh_sof, E2lsh, num_classes):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.nn.DataParallel(model).to(self.device) if torch.cuda.is_available() else model.to(self.device)
        self.client_id = client_id
        self.local_data = local_data
        self.root_dataset = root_dataset
        self.test_dataset = test_data
        self.R2lsh_dis = R2lsh_dis
        self.R2lsh_sof = R2lsh_sof
        self.E2lsh = E2lsh
        # self.optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4, betas=(0.9, 0.999))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-5)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.mask = None
        self.personalized_agg_mask = None
        self.param_sensitivity = None
        self.num_classes = num_classes

        self.torch_acc_cal = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)

        self.he_sk = None

    def set_he_key(self, sk):
        self.he_sk = sk

    def get_encrypted_parameters(self, he_context):
        params_vec = self.get_model_parameters()
        return ts.ckks_vector(he_context, params_vec)

    def update_model_from_encrypted_vector(self, enc_vec):
        decrypted_vec = np.array(enc_vec.decrypt(self.he_sk))
        param_len = len(self.get_model_parameters())

        if len(decrypted_vec) > param_len:
            decrypted_vec = decrypted_vec[:param_len]
        elif len(decrypted_vec) < param_len:
            print(f"The decryption length({len(decrypted_vec)}) is less than ({param_len}) (Client {self.client_id})")
            padded_vec = np.zeros(param_len)
            padded_vec[:len(decrypted_vec)] = decrypted_vec
            decrypted_vec = padded_vec

        self.update_model_from_vector(decrypted_vec)

    def compute_data_distribution(self):
        labels = [y for _, y in self.local_data]
        class_counts = np.bincount(labels, minlength=self.num_classes)

        return class_counts / np.sum(class_counts)

    def get_model_parameters(self):
        params = []
        for p in self.model.parameters():
            params.append(p.data.cpu().numpy().flatten())
        return np.concatenate(params)

    def compute_soft_labels(self):
        self.model.eval()
        soft_labels = []
        dataloader = DataLoader(self.root_dataset, batch_size=32, shuffle=False)
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.cuda() if torch.cuda.is_available() else x
                outputs = torch.softmax(self.model(x), dim=1)
                soft_labels.append(outputs.mean(dim=0).cpu().numpy())
        soft_labels = np.concatenate((soft_labels[:10]))

        return np.array(soft_labels)

    def generate_hashes_new(self):
        psi = list(self.compute_data_distribution())
        omega = list(self.get_model_parameters())
        L = list(self.compute_soft_labels())

        return {
            "data_distribution": self.R2lsh_dis.R2lsh(psi),
            "model_param": self.E2lsh.E2lsh_new(omega, self.client_id),
            "soft_label": self.R2lsh_sof.R2lsh(L)
        }
    
    def generate_hashes_personalized(self):
        if self.mask is None:
            raise ValueError(f"Client {self.client_id}has not performed parameter decoupling yet.")

        omega = self.get_model_parameters()

        personalized_params = omega * (1 - self.mask)

        psi = list(self.compute_data_distribution())
        L = list(self.compute_soft_labels())
        
        return {
            "data_distribution": self.R2lsh_dis.R2lsh(psi),
            "model_param": self.E2lsh.E2lsh_new(list(personalized_params), self.client_id),
            "soft_label": self.R2lsh_sof.R2lsh(L)
        }

    def local_train(self, epochs=5):
        total_loss = 0
        self.model.train()
        dataloader = DataLoader(self.local_data, batch_size=32, shuffle=True)
        for epoch in range(epochs):
            running_loss = 0.0
            for x, y in dataloader:
                if x.size(0) == 1:
                    continue
                x = x.cuda() if torch.cuda.is_available() else x
                y = y.cuda() if torch.cuda.is_available() else y
                
                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item() * x.size(0)
            epoch_loss = running_loss / len(dataloader.dataset)
            total_loss += epoch_loss

        self.scheduler.step()
        
        total_loss = total_loss / epochs
        return total_loss

    def compute_accuracy(self, batch_size=32, local_mode=True, external_dataset=None):

        self.model.eval()
        if local_mode:
            dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True)
        else:
            dataloader = DataLoader(external_dataset, batch_size=batch_size, shuffle=True)
        cunt = 0
        accuracy = 0
        
        with torch.no_grad():
            for x, y in dataloader:
                cunt = cunt + 1
                if x.size(0) == 1:
                    continue
                x = x.to(self.device)
                y = y.to(self.device)
                
                predicts = self.model(x)
                self.torch_acc_cal.to(self.device)
                self.torch_acc_cal.reset()
                self.torch_acc_cal.reset()
                acc = self.torch_acc_cal(predicts, y)
                accuracy = acc + accuracy
                
            accuracy = accuracy / cunt
        
        return accuracy

    def update_model_from_vector(self, param_vec):
        ptr = 0
        for p in self.model.parameters():
            shape = p.data.shape
            size = np.prod(shape)
            p.data = torch.tensor(
                param_vec[ptr:ptr+size].reshape(shape),
                dtype=torch.float32
            ).to(self.device)
            ptr += size

    def parameter_decoupling_new(self, train_epochs=1, batch_size=32, threshold = 0.002, keep_ratio=0.0):
        original_state = {
            'params': {name: param.data.clone() for name, param in self.model.named_parameters()},
            'buffers': {name: buf.clone() for name, buf in self.model.named_buffers()}
        }
       
        temp_optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.0, weight_decay=0.0)
        
        self.model.train()
        test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(train_epochs):
            running_loss = 0.0
            for x, y in test_dataloader:
                if x.size(0) == 1:
                    continue
                x = x.to(self.device)
                y = y.to(self.device)
                
                temp_optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                temp_optimizer.step()
                
                running_loss += loss.item() * x.size(0)

        original_flat = []
        for param in original_state['params'].values():
            original_flat.append(param.cpu().numpy().flatten())
        original_flat = np.concatenate(original_flat)

        new_flat = self.get_model_parameters()
        param_changes = np.abs(new_flat - original_flat)
        self.mask = (param_changes <= threshold).astype(int)
        self.personalized_agg_mask = np.zeros_like(self.mask)
        pers_indices = np.where(self.mask == 0)[0]
        
        if len(pers_indices) > 0 and keep_ratio > 0:
            pers_changes = param_changes[pers_indices]
            k = int(len(pers_indices) * keep_ratio)
            
            if k > 0:
                sorted_valid_indices = np.argsort(pers_changes) 
                indices_to_aggregate_relative = sorted_valid_indices[:-k]
                indices_to_aggregate = pers_indices[indices_to_aggregate_relative]
                
                self.personalized_agg_mask[indices_to_aggregate] = 1
            else:
                self.personalized_agg_mask[pers_indices] = 1
        else:
             self.personalized_agg_mask[pers_indices] = 1

        for name, param in self.model.named_parameters():
            if name in original_state['params']:
                param.data = original_state['params'][name]
        for name, buf in self.model.named_buffers():
            if name in original_state['buffers']:
                buf.data = original_state['buffers'][name]

        del temp_optimizer
        del original_state
        
        return self.mask