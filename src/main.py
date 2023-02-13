import torch.optim as optim
import torch
import argparse
from data_loader import data_loader
from train import train_model
from net import Net

def train_new_model(model, args):
    trainloader, testloader, _ = data_loader(path='small_data_transformed', batch_size_test=16, batch_size_train=32)
    optimizer = optim.Adam(model.parameters(), args.lr)
    train_model(model, optimizer, trainloader, testloader, args.epoch)
    torch.save(model.state_dict(), 'model_weights')
    print('training done, saved parameters to: model_weights/')

def parse():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--epoch', default=5, type=int, help='num epochs')
    parser.add_argument('--train', default=False, type=int, help='train parameters (else load existing ones)')
    return parser.parse_args()


def main():

    args = parse()
    model = Net()

    if args.train:
        train_new_model(model, args)
    else:
        model.load_state_dict(torch.load('model_weights'))

    model.eval()

    ##
    # create dataloader for unheard speakers
    # create metric for voice similarity
    
    # predict stuff
    from train import test
    trainloader, testloader, _ = data_loader(path='small_data_transformed', batch_size_test=16, batch_size_train=32)
    test(model, 'cpu', testloader, 0)


if __name__ == '__main__': main()