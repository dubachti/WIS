import torch
import torchvision.transforms as transforms
import os
from dataset import Data

def data_loader(path: str,
                batch_size_train: int, 
                batch_size_test: int,
                num_workers: int = 8) -> tuple:

    print("==> Preparing dataset ...")

    # get file names of all data instances
    file_names = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.startswith('.'): continue # ignore .DS_store files
            with open(os.path.join(root, file), "r") as auto:
                file_names.append(auto.name)


    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    train_data, test_data = torch.utils.data.random_split(file_names, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
 
    train_list = [file_names[i] for i in train_data.indices]
    test_list = [file_names[i] for i in test_data.indices]

    trainset = Data(train_list, transform_train)
    testset = Data(test_list, transform_test)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size_train, num_workers=num_workers, shuffle=True)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size_test, num_workers=num_workers, shuffle=False)

    return trainloader, testloader