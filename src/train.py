import torch
import torch.nn as nn
import time

def train(model: nn.Module, 
          device: torch.device, 
          train_loader: torch.utils.data.Dataloader, 
          optimizer: torch.optim.Optimizer, 
          epoch: int) -> None:
    start = time.time()
    criterion = nn.TripletMarginLoss()
    model.train()
    for a, p, n in train_loader:
        a, p, n = a.to(device), p.to(device), n.to(device)
        optimizer.zero_grad(set_to_none=True)
        a_out, p_out, n_out = model(a), model(p), model(n)
        loss = criterion(a_out, p_out, n_out)
        loss.backward()
        optimizer.step()
    
    end = time.time()
    print('\n {}: Train set: loss: {:.4f}, Time: {:02d}:{:02}\n'.format(epoch, loss, (int(end-start))//60, (int(end-start))%60))

def test(model: nn.Module, 
         device: torch.device, 
         test_loader: torch.utils.data.Dataloader,
         epoch: int) -> None:
    start = time.time()
    criterion = nn.TripletMarginLoss()
    model.eval()
    test_loss = 0.
    with torch.no_grad():
        for a, p, n in test_loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            a_out, p_out, n_out = model(a), model(p), model(n)
            loss = criterion(a_out, p_out, n_out)
            test_loss += loss

    test_loss = test_loss/len(test_loader.dataset)

    end = time.time()
    print('\n {}: Test set: loss: {:.4f}, Time: {:02d}:{:02d}\n'.format(epoch, test_loss, (int(end-start))//60, (int(end-start))%60))
    return test_loss

def train_model(model: nn.Module, 
                optimizer: torch.optim.Optimizer, 
                trainloader: torch.utils.data.Dataloader, 
                testloader: torch.utils.data.Dataloader,  
                n_epochs: int) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=5)

    test(model=model, device=device, test_loader=testloader, epoch=0)
    for epoch in range(1,n_epochs+1):
        train(model=model, device=device, train_loader=trainloader, optimizer=optimizer, epoch=epoch)
        err = test(model=model, device=device, test_loader=testloader, epoch=epoch)
        scheduler.step(err)

def predict(model: nn.Module, 
            predict_loader: torch.utils.data.Dataloader) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for a, p, n in predict_loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            a_out, p_out, n_out = model(a), model(p), model(n)

            pos = torch.nn.functional.pairwise_distance(a_out, p_out)
            neg = torch.nn.functional.pairwise_distance(a_out, n_out)

            correct += (pos < neg).sum()
            total += pos.shape[0]

    print(f'Predict: {correct}/{total} ({correct/total*100:.1f}%)')
