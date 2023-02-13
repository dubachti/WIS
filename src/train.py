import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

def train(model, device, train_loader, optimizer):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\n {}: Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(epoch,
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def train_model(model, optimizer, trainloader, testloader, n_epochs):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

    test(model=model, device=device, test_loader=testloader, epoch=0)
    for epoch in range(1,n_epochs+1):
        train(model=model, device=device, train_loader=trainloader, optimizer=optimizer)
        test(model=model, device=device, test_loader=testloader, epoch=epoch)
        scheduler.step()
