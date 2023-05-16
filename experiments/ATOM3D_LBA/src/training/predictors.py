'''
Create model with sklearn interface (output numpy arrays only)
    - fit() --> whole dataset creation and training loop
    - predict_proba() --> output with softmax; classification models only
    - predict() --> regression models only
'''

import os, sys
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torch import Tensor



class SimplestDataset(Dataset):
    def __init__(self, x: Tensor, y: Tensor):
        assert x.shape[0] == y.shape[0]
        self.x = x
        self.y = y
    
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class MLPRegressor(nn.Module):
    def __init__(self, input_dim, h_dims, nonlinearity='leaky_relu', batch_size=32, lr=0.01, n_epochs=250, verbose=True, seed=100000):
        super().__init__()
        self.input_dim = input_dim
        self.h_dims = h_dims
        self.nonlinearity = nonlinearity
        self.batch_size = batch_size
        self.lr = lr
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.rng = torch.Generator().manual_seed(seed)

        from protein_holography_pytorch.nn.activations import NONLIN_TO_ACTIVATION_MODULES

        layers = []
        h_dim_prev = input_dim
        for h_dim in h_dims:
            layers.append(nn.Linear(h_dim_prev, h_dim))
            layers.append(eval(NONLIN_TO_ACTIVATION_MODULES[self.nonlinearity]))
            h_dim_prev = h_dim
        layers.append(nn.Linear(h_dim_prev, 1))

        self.mlp = torch.nn.Sequential(*layers)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self = self.to(self.device).float()
        
    
    def forward(self, x):
        return self.mlp(x)
    
    def fit(self, x_NF, y_N, x_valid_MF=None, y_valid_M=None):
        assert x_NF.shape[0] == y_N.shape[0]
        if self.batch_size == 'N':
            self.batch_size = x_NF.shape[0]

        train_dataset = SimplestDataset(x_NF, y_N)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, generator=self.rng, shuffle=True, drop_last=False)
        
        if x_valid_MF is not None and y_valid_M is not None:
            valid_dataset = SimplestDataset(x_valid_MF, y_valid_M)
            valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, generator=self.rng, shuffle=True, drop_last=False)
        else:
            valid_dataset = None
            valid_dataloader = None

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        mse = torch.nn.MSELoss()

        loss_trace_per_epoch = []
        validation_metric_per_epoch = []
        loss_trace = []
        for epoch in range(self.n_epochs):
            for x, y_true in train_dataloader:
                x = x.to(self.device).float()
                y_true = y_true.to(self.device).float().view(-1, 1)

                optimizer.zero_grad()
                self.train()

                y_pred = self.forward(x)
                loss = mse(y_pred, y_true)
                loss_trace.append(loss.item())

                loss.backward()
                optimizer.step()
            
            if valid_dataloader is not None:
                validation_metric = self.validation_step(valid_dataloader, mse)
            else:
                # use training loss as validation metric
                validation_metric = np.mean(loss_trace)
            
            scheduler.step(validation_metric)

            if self.verbose:
                print('Epoch %d/%d: loss = %.5f' % (epoch+1, self.n_epochs, np.mean(loss_trace)), end=' - ')
                print('valid metric: %.5f' % (validation_metric))
            loss_trace_per_epoch.append(np.mean(loss_trace))
            validation_metric_per_epoch.append(validation_metric)
            loss_trace = []
        
        self.loss_trace_per_epoch = loss_trace_per_epoch
        self.validation_metric_per_epoch = validation_metric_per_epoch
        
        return self
    
    def validation_step(self, valid_dataloader, metric_fn):
        metric = []
        for x, y_true in valid_dataloader:
            x = x.to(self.device).float()
            y_true = y_true.to(self.device).view(-1, 1)

            self.eval()

            y_pred = self.forward(x)
            metric.append(metric_fn(y_pred, y_true).item())
        
        return np.mean(metric)

    def predict(self, x_MF):
        self.eval()
        x_MF = torch.Tensor(x_MF).to(self.device)
        return self.forward(x_MF).detach().cpu().numpy()[:, 0]


class LinearRegressor(nn.Module):
    def __init__(self, input_dim, batch_size=32, lr=0.01, n_epochs=250, verbose=True, seed=100000):
        super().__init__()
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.lr = lr
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.rng = torch.Generator().manual_seed(seed)

        self.linear = nn.Linear(input_dim, 1)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self = self.to(self.device).float()
        
    
    def forward(self, x):
        return self.linear(x)
    
    def fit(self, x_NF, y_N, x_valid_MF=None, y_valid_M=None):
        assert x_NF.shape[0] == y_N.shape[0]
        if self.batch_size == 'N':
            self.batch_size = x_NF.shape[0]

        train_dataset = SimplestDataset(x_NF, y_N)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, generator=self.rng, shuffle=True, drop_last=False)
        
        if x_valid_MF is not None and y_valid_M is not None:
            valid_dataset = SimplestDataset(x_valid_MF, y_valid_M)
            valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, generator=self.rng, shuffle=True, drop_last=False)
        else:
            valid_dataset = None
            valid_dataloader = None

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        mse = torch.nn.MSELoss()

        loss_trace_per_epoch = []
        validation_metric_per_epoch = []
        loss_trace = []
        for epoch in range(self.n_epochs):
            for x, y_true in train_dataloader:
                x = x.to(self.device).float()
                y_true = y_true.to(self.device).float().view(-1, 1)

                optimizer.zero_grad()
                self.train()

                y_pred = self.forward(x)
                loss = mse(y_pred, y_true)
                loss_trace.append(loss.item())

                loss.backward()
                optimizer.step()
            
            if valid_dataloader is not None:
                validation_metric = self.validation_step(valid_dataloader, mse)
            else:
                # use training loss as validation metric
                validation_metric = np.mean(loss_trace)
            
            scheduler.step(validation_metric)

            if self.verbose:
                print('Epoch %d/%d: loss = %.5f' % (epoch+1, self.n_epochs, np.mean(loss_trace)), end=' - ')
                print('valid metric: %.5f' % (validation_metric))
            loss_trace_per_epoch.append(np.mean(loss_trace))
            validation_metric_per_epoch.append(validation_metric)
            loss_trace = []
        
        self.loss_trace_per_epoch = loss_trace_per_epoch
        self.validation_metric_per_epoch = validation_metric_per_epoch
        
        return self
    
    def validation_step(self, valid_dataloader, metric_fn):
        metric = []
        for x, y_true in valid_dataloader:
            x = x.to(self.device).float()
            y_true = y_true.to(self.device).view(-1, 1)

            self.eval()

            y_pred = self.forward(x)
            metric.append(metric_fn(y_pred, y_true).item())
        
        return np.mean(metric)

    def predict(self, x_MF):
        self.eval()
        x_MF = torch.Tensor(x_MF).to(self.device)
        return self.forward(x_MF).detach().cpu().numpy()[:, 0]


class MultiClassLinearClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, batch_size=100, lr=0.01, n_epochs=250, verbose=True, seed=100000):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.lr = lr
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.rng = torch.Generator().manual_seed(seed)

        self.linear = nn.Linear(input_dim, output_dim)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self = self.to(self.device).float()
        
    
    def forward(self, x):
        return self.linear(x)
    
    def fit(self, x_NF, y_N, x_valid_MF=None, y_valid_M=None):
        assert x_NF.shape[0] == y_N.shape[0]
        if self.batch_size == 'N':
            self.batch_size = x_NF.shape[0]

        train_dataset = SimplestDataset(x_NF, y_N)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, generator=self.rng, shuffle=True, drop_last=False)
        
        if x_valid_MF is not None and y_valid_M is not None:
            valid_dataset = SimplestDataset(x_valid_MF, y_valid_M)
            valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, generator=self.rng, shuffle=True, drop_last=False)
        else:
            valid_dataset = None
            valid_dataloader = None

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        cross_entropy_with_logits = torch.nn.CrossEntropyLoss()

        loss_trace_per_epoch = []
        validation_metric_per_epoch = []
        loss_trace = []
        for epoch in range(self.n_epochs):
            for x, y_true in train_dataloader:
                x = x.to(self.device).float()
                y_true = y_true.to(self.device)

                optimizer.zero_grad()
                self.train()

                y_pred = self.forward(x)
                loss = cross_entropy_with_logits(y_pred, y_true)
                loss_trace.append(loss.item())

                loss.backward()
                optimizer.step()
            
            if valid_dataloader is not None:
                validation_metric = self.validation_step(valid_dataloader, cross_entropy_with_logits)
            else:
                # use training loss as validation metric
                validation_metric = np.mean(loss_trace)
            
            scheduler.step(validation_metric)

            if self.verbose:
                print('Epoch %d/%d: loss = %.5f' % (epoch+1, self.n_epochs, np.mean(loss_trace)), end=' - ')
                print('valid metric: %.5f' % (validation_metric))
            loss_trace_per_epoch.append(np.mean(loss_trace))
            validation_metric_per_epoch.append(validation_metric)
            loss_trace = []
        
        self.loss_trace_per_epoch = loss_trace_per_epoch
        self.validation_metric_per_epoch = validation_metric_per_epoch
        
        return self
    
    def validation_step(self, valid_dataloader, metric_fn):
        metric = []
        for x, y_true in valid_dataloader:
            x = x.to(self.device).float()
            y_true = y_true.to(self.device)

            self.eval()

            y_pred = self.forward(x)
            metric.append(metric_fn(y_pred, y_true).item())
        
        return np.mean(metric)

    def predict_proba(self, x_MF):
        self.eval()
        x_MF = torch.Tensor(x_MF).to(self.device)
        return torch.nn.functional.softmax(self.forward(x_MF), dim=1).detach().cpu().numpy()
    


