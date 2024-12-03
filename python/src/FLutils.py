import torch
import numpy as np
from torch.utils.data import Dataset

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


def hard_non_iid_mapping(areas: int, labels: int) -> dict[int, list[int]]:
    labels = np.arange(labels)
    split_classes_per_area = np.array_split(labels, areas)
    mapping_area_labels = {i: e.tolist() for i, e in enumerate(split_classes_per_area)}
    return mapping_area_labels