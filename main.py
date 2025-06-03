# %%
from numpy import subtract
import torch
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt

words = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(stoi)

# %%
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

# %%

class Linear:
    def __init__(self, input_size, output_size):
        self.weight = torch.randn(input_size, output_size) / input_size**0.5
        self.bias = torch.zeros(output_size)

    def __call__(self, x):
        return x @ self.weight + self.bias

    def __repr__(self):
        return f"Linear({self.weight.shape[0]}, {self.weight.shape[1]})"

    def parameters(self):
        return [self.weight, self.bias]

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
            if x.ndim == 2:
                dim = 0
            elif x.ndim ==3:
                dim = (0,1)
            xmean = x.mean(dim, keepdim=True)
            xvar = x.var(dim, keepdim=True)
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


class Tanh:
    def __call__(self,x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []

    def __repr__(self):
        return "Tanh()"

class Embedding:
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

    def __init__(self,n):
        self.n = n

    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T//self.n, C*self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        self.out = x
        return self.out

    def parameters(self):
        return []

    def __repr__(self):
        return "Flatten()"

class Sequential:
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
            params += layer.parameters()
        return params

    def __repr__(self):
        return f"Sequential({', '.join(str(layer) for layer in self.layers)})"


# %%
n_embd = 10
n_hidden = 200
model = Sequential([
    Embedding(vocab_size, n_embd),
    FlattenConsecutive(block_size),
    Linear(n_embd * block_size , n_hidden), BatchNorm(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size)
])

with torch.no_grad():
    model.layers[-1].weight *= 0.1

params = model.parameters()

for param in params:
    param.requires_grad = True

print(sum(param.nelement() for param in params ))

# %%
max_steps = 100000
batch_size = 32
lossi = []

for step in range(max_steps):
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    X = Xtr[ix]
    Y = Ytr[ix]

    # forward pass
    logits = model(X)
    loss = F.cross_entropy(logits, Y)

    # backward pass
    for p in params:
        p.grad = None
    loss.backward()

    # update
    lr = 0.1 if step < 1000 else 0.01
    for p in params:
        p.data += -lr * p.grad

    # track stats
    lossi.append(loss.item())
    if step % 10000 == 0:
        print(f"step {step}: loss {loss.item():.4f}")


# %%
lossi = torch.tensor(lossi).view(-1, 1000).mean(1)
plt.plot(lossi)

# %%
for layer in model.layers:
    layer.training = False

# %%
@torch.no_grad()
def split_loss(split):
    x,y = {
        'train': (Xtr, Ytr),
        'val': (Xdev, Ydev),
        'test': (Xte, Yte)
    }[split]
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    print(f"loss for {split}: {loss.item():.4f}")

split_loss('train')
split_loss('val')
split_loss('test')

# %%
for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
        logits = model(torch.tensor([context]))
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print(''.join(itos[i] for i in out))

# %%

list(range(10))[:]
