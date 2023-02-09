from data_loader import data_loader

def main():

    trainloader, testloader = data_loader(path='small_data_transformed', batch_size_test=8, batch_size_train=8)
    
    pass



if __name__ == '__main__': main()