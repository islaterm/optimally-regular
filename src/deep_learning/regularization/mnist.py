import sys

import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


from deep_learning.nn.feed_forward.network import FFNN


def ce_loss(q, p, stable=True, epsilon=1e-7):
    if stable:
        q = q.clamp(epsilon, 1 - epsilon)
    return -(p * q.log()).sum() / q.size(0)


def test_FFNN(red, dataset, device='cuda'):
    red.to(device)
    total = len(dataset)
    loss, acc = [], []

    X = dataset.data.view(len(dataset), -1).float().to(device)
    Y = dataset.targets.to(device)

    Y_PRED = red.forward(X, True, device).to(device)

    Y_onehot = torch.zeros_like(Y_PRED)
    Y_onehot[torch.arange(X.size(0)), Y] = 1.

    L_total = RegLoss(Y_PRED, Y_onehot, red.Ws, l2_par=red.l2_par)
    loss.append(L_total)
    diff = Y - torch.argmax(Y_PRED, 1)
    errores = torch.nonzero(diff).size(0)

    Acc = 100 * (total - errores) / total
    acc.append(Acc)

    sys.stdout.write(
        '/r Acc:{0:.2f}%'.format(Acc)
        + ' Loss:{0:.4f}'.format(L_total))

    return loss, acc


# Descarga y almacena el conjunto de train de MNIST.
mnist_train_set = MNIST('mnist', train=True, transform=ToTensor(), download=True)

# Descarga y almacena el conjunto de test de MNIST.
mnist_test_set = MNIST('mnist', train=False, transform=ToTensor(), download=True)
print('Datos de train: {}, Datos de test: {}'.format(len(mnist_train_set), len(mnist_test_set)))

# Acá debes agregar tu código para entrenar usando el conjunto de
# train (mnist_train_set) y probar usando el conjunto de test (mnist_test_set)
# Debes entrenar sin usar regularización y con regularización
# Modificar tu código de entrenar_FFNN para que retorne los
# errores de entrenamiento y generalización

mnist_model_NoReg = FFNN(784, [256, 128, 64], [relu, sig, relu], [None, None, None], 10,
                         [1.0, 1.0, 1.0, 1.0])

mnist_model_Reg = FFNN(784, [256, 128, 64], [relu, sig, celu], [None, None, None], 10,
                       [1.0, 0.7, 0.7, 0.8])

mnist_model_Reg_2 = FFNN(784, [256, 128, 64], [relu, relu, sig], [None, None, None], 10,
                         [1.0, 0.7, 0.8, 0.7], l2_par=0.5)

mnist_optimizer_NoReg = SGD(mnist_model_NoReg.parameters(), lr=1e-2)

mnist_optimizer_Reg = SGD(mnist_model_Reg.parameters(), lr=1e-2)

mnist_optimizer_Reg_2 = SGD(mnist_model_Reg_2.parameters(), lr=1e-2)

with torch.no_grad():
    lossNR, accNR = train_network(mnist_model_NoReg, mnist_train_set, mnist_optimizer_NoReg,
                                  epochs=30, batch_size=200)

with torch.no_grad():
    lossNRTest, accNRTest = test_FFNN(mnist_model_NoReg, mnist_test_set)

with torch.no_grad():
    lossR, accR = train_network(mnist_model_Reg, mnist_train_set, mnist_optimizer_Reg, epochs=30,
                                batch_size=200)

with torch.no_grad():
    lossRTest, accRTest = test_FFNN(mnist_model_Reg, mnist_test_set)

with torch.no_grad():
    lossR2, accR2 = train_network(mnist_model_Reg_2, mnist_train_set, mnist_optimizer_Reg_2,
                                  epochs=30, batch_size=200)

with torch.no_grad():
    lossR2Test, accR2Test = test_FFNN(mnist_model_Reg_2, mnist_test_set)
