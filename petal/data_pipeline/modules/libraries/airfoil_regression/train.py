import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def train(model, reverse_model, dataset, n_epochs=2, criterion=nn.MSELoss()):
    lr = 0.001
    momentum = 0.9
    batch_size = 100

    optimizer         = optim.SGD(model.parameters(), lr=lr, momentum=momentum) 
    reverse_optimizer = optim.SGD(reverse_model.parameters(), lr=lr, momentum=momentum) 
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    try:
        for epoch in range(n_epochs):
            print('Epoch: ', epoch, flush=True)
            i = 0
            for inputs, labels in trainloader:
                reverse_optimizer.zero_grad()
                reverse_inputs = reverse_model(labels)
                reverse_loss = criterion(reverse_inputs, inputs)
                reverse_loss.backward()
                reverse_optimizer.step()

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if i % 5 == 0:
                    print(' [%d, %5d] loss: %.5f (reverse) | %.5f (forward)' % (epoch + 1, i + 1, reverse_loss.item(), loss.item()), flush=True)
                i += 1
    except KeyboardInterrupt:
        print('Interrupted training, saving..')
    return model, reverse_model
