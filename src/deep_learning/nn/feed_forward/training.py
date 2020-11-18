"""
"Optimally Regular" (c) by Ignacio Slater M.
"Optimally Regular" is licensed under a
Creative Commons Attribution 4.0 International License.
You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by/4.0/>.
"""
from timeit import default_timer as timer

import torch
from torch.utils.data import DataLoader

from deep_learning.nn.feed_forward.network import FFNN


def train_feed_forward_nn(network: FFNN, dataset, optimizador, epochs=1, batch_size=1,
                          reports_every=1, device='cuda'):
    network.to(device)
    data = DataLoader(dataset, batch_size, shuffle=True)
    total = len(dataset)
    epochs_time = 0
    loss, acc = [], []
    for e in range(1, epochs + 1):
        inicio_epoch = timer()

        for x, y in data:
            x, y = x.view(x.size(0), -1).float().to(device), y.to(device)

            y_pred = network(x)

            y_onehot = torch.zeros_like(y_pred)
            y_onehot[torch.arange(x.size(0)), y] = 1.

            network.backward(x, y_onehot, y_pred)

            optimizador.step()

        epochs_time += timer() - inicio_epoch

        if e % reports_every == 0:
            X = dataset.data.view(len(dataset), -1).float().to(device)
            Y = dataset.targets.to(device)

            Y_PRED = network.forward(X).to(device)

            Y_onehot = torch.zeros_like(Y_PRED)
            Y_onehot[torch.arange(X.size(0)), Y] = 1.

            L_total = ce_loss(Y_PRED, Y_onehot)
            loss.append(L_total)
            diff = Y - torch.argmax(Y_PRED, 1)
            errores = torch.nonzero(diff).size(0)

            Acc = 100 * (total - errores) / total
            acc.append(Acc)

            sys.stdout.write(
                '\rEpoch:{0:03d}'.format(e) + ' Acc:{0:.2f}%'.format(Acc)
                + ' Loss:{0:.4f}'.format(L_total)
                + ' Tiempo/epoch:{0:.3f}s'.format(epochs_time / e))

    return loss, acc
