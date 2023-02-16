import torch.optim as optim
import torch
import argparse
from data_loader import data_loader
from train import train_model, predict
from net import Net

def train_new_model(model: torch.nn.Module, 
                    args: argparse.Namespace) -> None:
    trainloader, testloader = data_loader(path=args.data, batch_size_test=32, batch_size_train=32, num_workers=args.num_workers)
    optimizer = optim.Adam(model.parameters(), args.lr)
    train_model(model, optimizer, trainloader, testloader, args.epoch)
    torch.save(model.state_dict(), 'model_weights')
    print('training done, saved parameters to: model_weights/')

def eval_model(model: torch.nn.Module, 
               args: argparse.Namespace) -> None:
    model.load_state_dict(torch.load('weights/model_weights', map_location=torch.device('cpu')))
    _, testloader = data_loader(path='small_data_transformed', batch_size_test=32, batch_size_train=32, num_workers=args.num_workers)
    predict(model, testloader)

def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--epoch', default=5, type=int, help='num epochs')
    parser.add_argument('--train', default=False, type=bool, help='train parameters (else load existing ones)')
    parser.add_argument('--data', default='data_transformed', type=str, help='path to data')
    parser.add_argument('--num_workers', default=8, type=int, help='number of dataloader workers')
    return parser.parse_args()

def main():
    args = parse()
    model = Net()
    if args.train:
        train_new_model(model, args)
    else:
        eval_model(model, args)

if __name__ == '__main__': main()