import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0, 1, 2, 3, 4, 5"

class LSTM(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=8, n_layers=1):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.n_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, 1)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x, hidden, each_choice_num_path, device):
        # x shape (num_choice * len_path, 3, 5)
        outputs, _ = self.lstm(x)
        total = 0
        for ii in range(len(each_choice_num_path)):
            total += len(each_choice_num_path[ii])
        # print(total)
        # print(outputs.shape)
        if total != outputs.shape[0]:
            raise
        last_end = 0
        for ii in range(len(each_choice_num_path)):
            temp_ind = torch.from_numpy(each_choice_num_path[ii]).long().to(device)
            # print(temp_ind)
            temp_ind = temp_ind.view(-1, 1).repeat(1, self.hidden_dim)[:, None, :]
            # print(temp_ind)
            temp_ind -= 1
            temp_outputs = torch.gather(outputs[last_end:last_end + len(each_choice_num_path[ii])], dim=1, index=temp_ind).to(device)
            print(temp_outputs.shape)
            if ii == 0:
                new_outputs = torch.sum(temp_outputs, dim=0).to(device)
            else:
                new_outputs = torch.cat((new_outputs, torch.sum(temp_outputs, dim=0)), dim=0).to(device)
            last_end += len(each_choice_num_path[ii])
            # print(last_end)
            # print(outputs[:24])
            # print(temp_ind)

        # raise
        # print(outputs)
        raise
        outputs = self.softmax(self.fc(new_outputs))
        return outputs.view(1, -1)


    def init_hidden(self, bs, device):
        hidden_state = torch.zeros(self.n_layers, bs, self.hidden_dim).to(device)
        cell_state = torch.zeros(self.n_layers, bs, self.hidden_dim).to(device)
        hidden = (hidden_state, cell_state)
        return hidden



all_data = np.load("./q_info_lstm_all.npz", allow_pickle=True)
train_x = all_data['x_train']
train_y = all_data['y_train']
train_path_len = all_data['path_len_train']


test_x = all_data['x_test']
test_y = all_data['y_test']
test_path_len = all_data['path_len_test']


# print(x.shape)
# print(y.shape)
# print(path_len.shape)
# # raise

# np.random.seed(66666)
# rand_ind = np.arange(len(x))
# np.random.shuffle(rand_ind)

# x = x[rand_ind]
# y = y[rand_ind]
# path_len = path_len[rand_ind]

# train_x = x[:int(len(x) * 0.8)]
# train_y = y[:int(len(x) * 0.8)]
# train_path_len = path_len[:int(len(x) * 0.8)]

# test_x = x[int(len(x) * 0.8):]
# test_y = y[int(len(x) * 0.8):]
# test_path_len = path_len[int(len(x) * 0.8):]
print(len(train_x))
print(len(train_y))
print(len(train_path_len))

# train_x = x
# train_y = y
# test_x = x
# test_y = y


batch_size = 1
n_iters = 70000
epochs = n_iters / (len(train_x) / batch_size)
input_dim = 44
output_dim = 16
# lr = 1e-2

lrs = [0.1, 0.05, 0.01, 0.001, 0.0001, 0.00005]

c = torch.cuda.device_count()
print('Number of GPUs:', c)
if c > 0:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
torch.backends.cudnn.enabled = False

best_acc = 0.
for lr in lrs:
    model = LSTM(input_dim, output_dim)
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    iter = 0
    for epoch in range(int(epochs)):
        correct_train = 0
        total_train = 0
        all_empty = 0
        loss_all = []
        for i, curr_x in enumerate(train_x):
            curr_x = Variable(torch.from_numpy(curr_x).float()).to(device)
            if len(curr_x) == 0:
                all_empty += 1
                continue
            # print(train_y[i : i + 1])
            # curr_x = Variable(torch.from_numpy(curr_x).float())
            labels = Variable(torch.from_numpy(train_y[i : i + 1]).long()).to(device)

            optimizer.zero_grad()
            hidden = model.init_hidden(curr_x.shape[0], device)
            outputs = model(curr_x, hidden, train_path_len[i], device)
            _, predicted = torch.max(outputs.data, 1)
            total_train+= labels.size(0)
            # for gpu, bring the predicted and labels back to cpu fro python operations to work
            correct_train+= (predicted == labels).sum()
            # print(outputs)
            # print(labels.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            loss_all.append(loss.item())
            optimizer.step()

            iter+=1
        # if iter%500==0:
        model.eval()
        # calculate Accuracy
        correct = 0
        total = 0
        for m, curr_x in enumerate(test_x):
            curr_x = Variable(torch.from_numpy(curr_x).float()).to(device)
            if len(curr_x) == 0:
                all_empty += 1
                continue
            # print(train_y[i : i + 1])
            # curr_x = Variable(torch.from_numpy(curr_x).float())
            labels = Variable(torch.from_numpy(test_y[m : m + 1]).long()).to(device)

            optimizer.zero_grad()
            hidden = model.init_hidden(curr_x.shape[0], device)
            outputs = model(curr_x, hidden, test_path_len[m], device)
            _, predicted = torch.max(outputs.data, 1)
            total+= labels.size(0)
            # for gpu, bring the predicted and labels back to cpu fro python operations to work
            correct+= (predicted == labels).sum()
        accuracy = 100 * correct/total
        acc_train = 100 * correct_train / total_train
        if accuracy > best_acc:
            torch.save(model.state_dict(), "./lstm_models/best_acc_" + str(int(accuracy)) + "_lr_" + str(lr) + ".pth")
            best_acc = accuracy
        print("Iteration: {}. Loss: {}. Accuracy_Test: {}. Accuracy_Train: {}".format(iter, np.mean(loss_all), accuracy, acc_train))
        model.train()
        print(all_empty)



print(best_acc)





