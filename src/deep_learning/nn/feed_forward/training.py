"""
"Optimally Regular" (c) by Ignacio Slater M.
"Optimally Regular" is licensed under a
Creative Commons Attribution 4.0 International License.
You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by/4.0/>.
"""
import sys
from timeit import default_timer as timer

import torch
from matplotlib.pyplot import figure
from torch.utils.data import DataLoader

from deep_learning.dataset import RandomDataset
from deep_learning.functions.activation import celu, relu
from deep_learning.functions.loss import ce_loss
from deep_learning.nn.feed_forward.network import FFNN
from deep_learning.sgd import SGD


def train_feed_forward_nn(network: FFNN, dataset, optimizer, epochs=1, batch_size=1,
                          reports_every=1, device='cuda'):
    network.to(device)
    data = DataLoader(dataset, batch_size, shuffle=True)
    total = len(dataset)
    epochs_time = 0
    loss, accs = [], []
    for epoch in range(1, epochs + 1):
        epoch_start = timer()

        for nn_in, nn_out in data:
            nn_in, nn_out = nn_in.view(nn_in.size(0), -1).float().to(device), nn_out.to(device)

            prediction = network(nn_in)

            onehot_pred = torch.zeros_like(prediction)
            onehot_pred[torch.arange(nn_in.size(0)), nn_out] = 1.

            network.backward(nn_in, onehot_pred, prediction)

            optimizer.step()

        epochs_time += timer() - epoch_start

        if epoch % reports_every == 0:
            nn_in = dataset.data.view(len(dataset), -1).float().to(device)
            nn_out = dataset.targets.to(device)

            prediction = network.forward(nn_in).to(device)

            onehot_pred = torch.zeros_like(prediction)
            onehot_pred[torch.arange(nn_in.size(0)), nn_out] = 1.

            total_loss = ce_loss(prediction, onehot_pred)
            loss.append(total_loss)
            diff = nn_out - torch.argmax(prediction, 1)
            errors = torch.nonzero(diff).size()[0]

            acc = 100 * (total - errors) / total
            accs.append(acc)

            sys.stdout.write(
                '\rEpoch:{0:03d}'.format(epoch) + ' Acc:{0:.2f}%'.format(acc)
                + ' Loss:{0:.4f}'.format(total_loss)
                + ' Tiempo/epoch:{0:.3f}s'.format(epochs_time / epoch))

    return loss, accs


def plot_results(loss, acc):
    f1 = figure(1)
    ax1 = f1.add_subplot(111)
    ax1.set_title("Loss")
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    ax1.plot(loss, c='r')
    f1.show()

    f2 = figure(2)
    ax2 = f2.add_subplot(111)
    ax2.set_title("Accuracy")
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('acc')
    ax2.plot(acc, c='b')
    f2.show()


if __name__ == '__main__':
    N, F, C = 2000, 300, 10
    dataset = RandomDataset(N, F, C)

    model = FFNN(F, [300, 400], [celu, relu], C, [float(C), None])
    optimizer = SGD(model.parameters(), lr=1e-3)
    with torch.no_grad():
        loss1, acc1 = train_feed_forward_nn(model, dataset, optimizer, epochs=100, batch_size=32)

    plot_results(loss1, acc1)
