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


def hard_non_iid_mapping(areas: int, labels: int) -> np.ndarray:
    labels_set = np.arange(labels)
    split_classes_per_area = np.array_split(labels_set, areas)
    distribution = np.zeros((areas, labels))
    for i, elems in enumerate(split_classes_per_area):
        rows = [i for _ in elems]
        distribution[rows, elems] = 1
    return distribution

def partitioning(distribution: np.ndarray, dataset: Dataset) -> dict[int, list[int]]:
    targets = dataset.targets
    areas = distribution.shape[0]
    targets_cardinality = distribution.shape[1]
    class_counts = torch.bincount(targets)
    partitions = {}
    for area in range(areas):
        area_distribution = distribution[area, :]
        elements_per_class = torch.tensor(area_distribution) * class_counts
        elements_per_class = torch.floor(elements_per_class).to(torch.int)
        selected_indices = []
        for label in range(targets_cardinality):
            target_indices = torch.where(targets == label)[0]
            selected_count = min(len(target_indices), elements_per_class[label].item())
            if selected_count > 0:
                selected_indices.extend(target_indices[:selected_count].tolist())
        partitions[area] = selected_indices
    return partitions