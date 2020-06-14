import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        outputs = self.softmax(self.linear(x))
        return outputs.view(-1, len(x))

all_data = np.load("./path_relation_type.npz", allow_pickle=True)
x = all_data['x']
y = all_data['y']

np.random.seed(666)
rand_ind = np.arange(len(x))
np.random.shuffle(rand_ind)

x = x[rand_ind]
y = y[rand_ind]

train_x = x[:int(len(x) * 0.8)]
train_y = y[:int(len(x) * 0.8)]

test_x = x[int(len(x) * 0.8):]
test_y = y[int(len(x) * 0.8):]

# train_x = x
# train_y = y
# test_x = x
# test_y = y


batch_size = 1
n_iters = 80000
epochs = n_iters / (len(x) / batch_size)
input_dim = 14 + 14 ** 2 + 14 ** 3
output_dim = 1
# lr = 1e-2

lrs = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005]


best_acc = 0.
for lr in lrs:
    model = LogisticRegression(input_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    iter = 0
    for epoch in range(int(epochs)):
        for i, curr_x in enumerate(train_x):
            curr_x = Variable(torch.from_numpy(curr_x).float())
            labels = Variable(torch.from_numpy(train_y[i : i + 1]).long())

            optimizer.zero_grad()
            outputs = model(curr_x)
            # print(outputs.shape)
            # print(labels.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            iter+=1
            if iter%500==0:
                # calculate Accuracy
                correct = 0
                total = 0
                for m, curr_x in enumerate(test_x):
                    curr_x = Variable(torch.from_numpy(curr_x).float())
                    labels = Variable(torch.from_numpy(test_y[m : m + 1]).long())
                    outputs = model(curr_x)
                    _, predicted = torch.max(outputs.data, 1)
                    total+= labels.size(0)
                    # for gpu, bring the predicted and labels back to cpu fro python operations to work
                    correct+= (predicted == labels).sum()
                accuracy = 100 * correct/total
                if accuracy > best_acc:
                    torch.save(model.state_dict(), "./lr_models/best_acc_" + str(int(accuracy)) + "_lr_" + str(lr) + ".pth")
                    best_acc = accuracy
                print("Iteration: {}. Loss: {}. Accuracy: {}.".format(iter, loss.item(), accuracy))




print(best_acc)
