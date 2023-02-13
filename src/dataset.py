from torch.utils.data.dataset import Dataset
import pickle
import torch



class Data(Dataset):

    def __init__(self,
                 file_names: list,
                 class_to_idx: dict,
                 transform = None) -> None:
        self.file_names = file_names
        self.class_to_idx = class_to_idx
        self.transform = transform

    # note:
    # - maybe it is smarter to load all instance to mem first
    # - maybe need to enable list as index input
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        file = self.file_names[index]
        with open(file, 'rb') as f:
            y, x = pickle.load(f)

        if self.transform:
            x = self.transform(x)
        else:
            x = torch.tensor(x)

        return x, torch.tensor(self.class_to_idx[y])
 
    def __len__(self) -> int:
        return len(self.file_names)
