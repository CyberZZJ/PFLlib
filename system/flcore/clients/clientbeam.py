import copy
import torch
import time
import numpy as np
from flcore.clients.clientavg import clientAVG
from utils.compression import compress_model_for_communication

class clientBeam(clientAVG):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.num_models = 1 # Default
        self.uploaded_models_list = []
        self.received_models = []
        self.compressed_model = None

    def set_parameters(self, model, topk_ratios):
        self.topk_ratios = topk_ratios
        if isinstance(model, list):
            self.received_models = model
            # For testing/init purposes, just load the first one into self.model
            # though we will iterate through them in train()
            if len(model) > 0:
                # Call super().set_parameters with just the model
                super().set_parameters(model[0])
        else:
            self.received_models = [model]
            super().set_parameters(model)

    def compress_model(self, model, ratio):
        try:
            compressed_data = compress_model_for_communication(
                model,
                compression_ratio=ratio
            )
            # Manually add metadata since compress_model_for_communication returns dict
            compressed_data['_is_compressed'] = True
            compressed_data['_client_id'] = self.id
            compressed_data['_compression_ratio'] = ratio # Tag it
            return compressed_data
        except Exception as e:
            print(f"Client {self.id} compression failed: {e}")
            return copy.deepcopy(model)

    def train(self):
        trainloader = self.load_train_data()
        self.model.train()
        start_time = time.time()
        
        # Determine number of models to train based on received models
        # But respect self.num_models (strong/weak logic)
        # If strong client (num_models > 1), we expect multiple received models
        # If weak client (num_models == 1), we expect one received model
        
        models_to_train = self.received_models
        self.trained_models = []

        for k, initial_model in enumerate(models_to_train):
            # Load the specific global model state
            self.model.load_state_dict(initial_model.state_dict())
            
            # Perform local training (standard FedAvg logic)
            max_local_epochs = self.local_epochs
            if self.train_slow:
                max_local_epochs = np.random.randint(1, max_local_epochs // 2)

            for epoch in range(max_local_epochs):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    if self.train_slow:
                        time.sleep(0.1 * np.abs(np.random.rand()))
                    output = self.model(x)
                    loss = self.loss(output, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            if self.learning_rate_decay:
                self.learning_rate_scheduler.step()

            # Store the trained model
            self.trained_models.append(copy.deepcopy(self.model))
        
        # Update time cost
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def get_upload_model(self):
        # Generate upload models based on trained models and topk_ratios
        # Strong: [M0_k00, M0_k01, M1_k10, M1_k11]
        # Weak: [M0_k00, M0_k01]
        
        upload_list = []
        
        if self.num_models > 1: # Strong
            # Expecting 2 trained models: M0, M1
            # And 4 ratios: topk_ratios[0..3]
            if len(self.trained_models) >= 2 and len(self.topk_ratios) >= 4:
                m0 = self.trained_models[0]
                m1 = self.trained_models[1]
                
                # M0 -> k00, k01
                upload_list.append(self.compress_model(m0, self.topk_ratios[0]))
                upload_list.append(self.compress_model(m0, self.topk_ratios[1]))
                
                # M1 -> k10, k11
                upload_list.append(self.compress_model(m1, self.topk_ratios[2]))
                upload_list.append(self.compress_model(m1, self.topk_ratios[3]))
            else:
                # Fallback if something is wrong
                print(f"Warning: Client {self.id} (Strong) missing models or ratios")
                return self.trained_models
        else: # Weak
            # Expecting 1 trained model: M0
            # And 4 ratios, but we use first 2? Or specific ones?
            # Prompt says: "If Weak: Return [M0_k00, M0_k01]"
            # Assuming using first 2 ratios for M0 as well, or maybe split?
            # Let's assume indices 0 and 1 for consistency with M0
            if len(self.trained_models) >= 1 and len(self.topk_ratios) >= 2:
                m0 = self.trained_models[0]
                upload_list.append(self.compress_model(m0, self.topk_ratios[0]))
                upload_list.append(self.compress_model(m0, self.topk_ratios[1]))
            else:
                 print(f"Warning: Client {self.id} (Weak) missing models or ratios")
                 return self.trained_models[0] if self.trained_models else self.model

        if self.num_models == 1:
             # Weak clients usually return a single item or list?
             # Server expects:
             # Strong: list of 2 (but we are returning 4 now?)
             # Wait, server receive_and_aggregate_parameters_beam expects:
             # Strong: list of 2?
             # Let's check server code.
             pass
             
        return upload_list
