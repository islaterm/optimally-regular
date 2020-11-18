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
from torch.utils.data import DataLoader

from deep_learning.functions.loss import ce_loss
from deep_learning.nn.feed_forward.network import FFNN


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
