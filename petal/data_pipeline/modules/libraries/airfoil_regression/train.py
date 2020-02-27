import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def train(net, dataset, n_epochs=2, criterion=nn.MSELoss()):
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # TODO: Change me later!
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    try:
        for epoch in range(n_epochs):
            print('Epoch: ', epoch, flush=True)
            running_loss = 0.0
            iterator = iter(trainloader)
            i = 0
            while True:
                try:
                    inputs, labels = next(iterator)
                    print('    datapoint: ', i, flush=True)
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                    if i % 100 == 0:
                        print(running_loss)
                        print(' [%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                        running_loss = 0.0
                    i += 1
                except StopIteration:
                    break
    except KeyboardInterrupt:
        pass
    return net
