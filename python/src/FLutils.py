import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader

class NNMnist(nn.Module):

    def __init__(self, h1=128):
        super().__init__()
        self.fc1 = torch.nn.Linear(28*28, h1)
        self.fc2 = torch.nn.Linear(h1, 27)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def average_weights(models, weigths):
    """ Averages the weights

    Args:
        models (list): a list of state_dict

    Returns:
        state_dict: the average state_dict
    """
    w_avg = copy.deepcopy(models[0])

    for key in w_avg.keys():
        w_avg[key] = torch.mul(w_avg[key], 0.0)
    sum_weights = sum(weigths)
    for key in w_avg.keys():
        for i in range(0, len(models)):
            w_avg[key] += models[i][key] * weigths[i]
        w_avg[key] = torch.div(w_avg[key], sum_weights)
    return w_avg


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

def get_dataset(name: str, train: bool = True) -> Dataset:
    transform = transforms.Compose([transforms.ToTensor()])
    if name == 'MNIST':
        dataset = datasets.MNIST(root='data', train=train, download=True, transform=transform)
        return dataset

def to_subset(dataset, indexes):
    return Subset(dataset, indexes)

def training(model_weights, training_data, epochs, batch_size, experiment):
    model = instantiate_model(model_weights, experiment)
    criterion = nn.NLLLoss()
    model.train()
    epoch_loss = []
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    data_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        batch_loss = []
        for batch_index, (images, labels) in enumerate(data_loader):
            model.zero_grad()
            log_probs = model(images)
            loss = criterion(log_probs, labels)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        mean_epoch_loss = sum(batch_loss) / len(batch_loss)
        epoch_loss.append(mean_epoch_loss)
    return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

def evaluate(model, validationData, batchSize):
    pass

def average_models():
    pass

def seed_everything(seed):
    pass

def instantiate_model(model_weights, experiment):
    if experiment == 'MNIST':
        model = NNMnist()
        model.load_state_dict(model_weights)
        return model
    else:
        raise Exception(f'Wrong experiment name ({experiment})! Please check :)')