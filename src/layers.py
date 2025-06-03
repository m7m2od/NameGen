"""
Neural Network Layers for WaveNet Character-Level Language Model

This module contains custom implementations of neural network layers
including Linear, BatchNorm, Tanh, Embedding, and utility layers.
"""

import torch
import torch.nn.functional as F


class Linear:
    """Linear transformation layer: y = xW + b"""
    
    def __init__(self, input_size, output_size):
        self.weight = torch.randn(input_size, output_size) / (input_size ** 0.5)
        self.bias = torch.zeros(output_size)
    
    def __call__(self, x):
        return x @ self.weight + self.bias
    
    def parameters(self):
        return [self.weight, self.bias]
    
    def __repr__(self):
        return f"Linear({self.weight.shape[0]}, {self.weight.shape[1]})"


class BatchNorm:
    """Batch Normalization layer"""
    
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        
        # Parameters (learnable)
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        
        # Buffers (running statistics)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
    
    def __call__(self, x):
        if self.training:
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                dim = (0, 1)
            xmean = x.mean(dim, keepdim=True)
            xvar = x.var(dim, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var
        
        # Normalize to unit variance
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        
        # Update running statistics during training
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        
        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]
    
    def __repr__(self):
        return f"BatchNorm({self.gamma.shape[0]})"


class Tanh:
    """Hyperbolic tangent activation function"""
    
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    
    def parameters(self):
        return []
    
    def __repr__(self):
        return "Tanh()"


class Embedding:
    """Embedding layer for converting discrete tokens to dense vectors"""
    
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn(num_embeddings, embedding_dim)
    
    def __call__(self, x):
        self.out = self.weight[x]
        return self.out
    
    def parameters(self):
        return [self.weight]
    
    def __repr__(self):
        return f"Embedding({self.weight.shape[0]}, {self.weight.shape[1]})"


class FlattenConsecutive:
    """Flatten consecutive dimensions for hierarchical processing"""
    
    def __init__(self, n):
        self.n = n
    
    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T // self.n, C * self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        self.out = x
        return self.out
    
    def parameters(self):
        return []
    
    def __repr__(self):
        return f"FlattenConsecutive({self.n})"


class Sequential:
    """Sequential container for chaining layers"""
    
    def __init__(self, layers):
        self.layers = layers
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
    
    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
    
    def __repr__(self):
        layer_strs = [str(layer) for layer in self.layers]
        return f"Sequential(\n  " + "\n  ".join(layer_strs) + "\n)"