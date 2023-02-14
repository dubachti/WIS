import torch
import torch.nn as nn

def train(model, device, train_loader, optimizer):
    criterion = nn.TripletMarginLoss()
    model.train()
    for a, p, n in train_loader:
        a, p, n = a.to(device), p.to(device), n.to(device)
        optimizer.zero_grad()
        a_out, p_out, n_out = model(a), model(p), model(n)
        loss = criterion(a_out, p_out, n_out)
        loss.backward()
        optimizer.step()

    ## insert result print here same as in test function

def test(model, device, test_loader, epoch):
    criterion = nn.CosineSimilarity()
    model.eval()
    test_loss = 0.
    correct = 0
    with torch.no_grad():
        for a, p, n in test_loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            a_out, p_out, n_out = model(a), model(p), model(n)
            c_pos, c_neg = criterion(a_out, p_out), criterion(a_out, n_out)
            pred = (c_pos-c_neg > 0).int()
            correct += pred.sum()

    test_loss = (len(test_loader.dataset) - correct)/len(test_loader.dataset)

    print('\n {}: Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(epoch,
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_loss

def train_model(model, optimizer, trainloader, testloader, n_epochs):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=5)


    test(model=model, device=device, test_loader=testloader, epoch=0)
    for epoch in range(1,n_epochs+1):
        train(model=model, device=device, train_loader=trainloader, optimizer=optimizer)
        err = test(model=model, device=device, test_loader=testloader, epoch=epoch)
        scheduler.step(err)
