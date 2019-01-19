import numpy as np
import torch as t
import torch.nn as nn
import torch.optim as optim

def make_xor_dataset(n_samples):
    popu = [1, 0]
    X = np.array([np.random.choice(popu, 2) for _ in np.arange(n_samples)])
    y = np.array([np.logical_xor(*pair) for pair in X]).astype(np.int)
    return X, y

def feed_batches(total_n, batch_n):
    idxs = np.arange(total_n)
    batch_idx = [[idxs[k:min(k+batch_n, total_n)]][0]
                 for k in np.arange(0, total_n, batch_n)]
    for batch in batch_idx:
        yield batch

class XORNet(t.nn.Module):
    def __init__(self, d_in, d_h, d_out):
        super(XORNet, self).__init__()
        self.lin1 = t.nn.Linear(d_in, d_h)
        self.lin2 = t.nn.Linear(d_h, d_out)

    def forward(self, x):
        return self.lin2(self.lin1(x).clamp(min=0)).sigmoid()

def return_ft(x, y):
    return t.FloatTensor(x), t.FloatTensor(y)

device = t.device("cpu")
dtype = t.float
ns = 1000 # number of samples
lr = 0.001 # learning rate
d_in = 2 # input dimension
d_out = 1 # output dimension
d_h = 2 # number of units in the hidden layer
batch_n = 100 # batch size
X, y = return_ft(*make_xor_dataset(ns))
net = XORNet(d_in, d_h, d_out)
loss_fn = t.nn.BCELoss()
opt_fn = optim.Adam(net.parameters(), lr=lr)

c = 0
for epoch in np.arange(1000):
    c += 1
    batch_gen = feed_batches(X.shape[0], batch_n)

    while True:
        try:
            idxs = next(batch_gen)
            y_pred = net(X[idxs])
            loss = loss_fn(y_pred.squeeze(), y[idxs])
            opt_fn.zero_grad()
            loss.backward()
            opt_fn.step()
        except StopIteration:
            break


# test the network on unseed data
ntest = 400
test_results = np.zeros(1000)

for idx in range(1000):
    Xtest, ytest = return_ft(*make_xor_dataset(ntest))
    ypred = net(Xtest)
    test_results[idx] = t.sum(t.round(ypred.squeeze()) == ytest).item() / ntest

print(test_results.mean())
