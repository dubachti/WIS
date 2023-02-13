import torch
import torchvision.transforms as transforms
import os
import pickle
from dataset import Data

def class_to_idx_map(file_names: list) -> dict:
    dic = {}
    max_idx = 0
    for file_name in file_names:
        with open(file_name, 'rb') as f:
            y, _ = pickle.load(f)
            if y not in dic:
                dic[y] = max_idx
                max_idx += 1
    return dic

def data_loader(path: str,
                batch_size_train: int, 
                batch_size_test: int):

    print("==> Preparing dataset ...")

    # get file names for all data instances
    file_names = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.startswith('.'): continue # ignore .DS_store files
            with open(os.path.join(root, file), "r") as auto:
                file_names.append(auto.name)


    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(128, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914), (0.2023)), ## compute real vlaues
    ])

    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914), (0.2023)), ## compute real vlaues
    ])

    ## class to idx mapping
    class_to_idx = class_to_idx_map(file_names)

    train_data, test_data = torch.utils.data.random_split(file_names, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
 
    train_list = [file_names[i] for i in train_data.indices]
    test_list = [file_names[i] for i in test_data.indices]

    trainset = Data(train_list, class_to_idx, transform_train)
    testset = Data(test_list, class_to_idx, transform_test)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size_train, num_workers=2, shuffle=True)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size_test, num_workers=2, shuffle=False)

    return trainloader, testloader, class_to_idx