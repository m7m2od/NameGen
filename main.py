import torch
import random

words = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}

random.shuffle(words)
block_size = 3

def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

for x, y in zip(Xtr[:5], Ytr[:5]):
    try:
        context_chars = [itos[ix.item()] for ix in x]
        target_char = itos[y.item()]
        print(f"Context: {''.join(context_chars)} -> Target: {target_char}")
    except KeyError as e:
        print(f"KeyError for x={x}, y={y}: {e}")



# ----------------------
class Linear:
    def __init__(self, input_size, output_size):
        self.weights = torch.randn(input_size, output_size) / input_size**0.5
        self.bias = torch.zeros(output_size)

    def __call__(self, x):
        return x @ self.weights + self.bias

    def __repr__(self):
        return f"Linear({self.weights.shape[0]}, {self.weights.shape[1]})"

    def parameters(self):
        return [self.weights, self.bias]

# ----------------------
class BatchNorm:
    def __init__(self,dim,eps=1e-5,momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # parameters
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # buffers
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self,x):
        if self.training:
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var

        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize the input variance
        self.out = self.gamma * xhat + self.beta

        # update running stats (Buffers)
        if self.training:
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]

    def __repr__(self):
        return f"BatchNorm({self.gamma.shape[0]})"

# ------------------------------------
class Tanh:
    def __call__(self,x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []

    def __repr__(self):
        return "Tanh()"

# ------------------------------------
