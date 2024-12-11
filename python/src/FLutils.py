import torch
import numpy as np
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

def hard_non_iid_mapping(areas: int, labels: int) -> np.ndarray:
    labels_set = np.arange(labels)
    split_classes_per_area = np.array_split(labels_set, areas)
    distribution = np.zeros((areas, labels))
    for i, elems in enumerate(split_classes_per_area):
        rows = [i for _ in elems]
        distribution[rows, elems] = 1
    return distribution


def dirichlet_non_iid_mapping(areas: int, labels: int, beta: float) -> np.ndarray:
    alpha = np.full(labels, beta)
    sample = np.random.dirichlet(alpha, areas)
    return sample


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

def get_dataset(name: str) -> Dataset:
    transform = transforms.Compose([transforms.ToTensor()])
    if name == 'MNIST':
        dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
        return dataset

def to_subset(dataset, indexes):
    return Subset(dataset, indexes)

def training(model, training_data, validation_data, epochs, batch_size):
    pass

def average_models():
    pass

def seed_everything(seed):
    pass