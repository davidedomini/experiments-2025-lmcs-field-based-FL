import copy
import torch
import random
import numpy as np
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader

class NNMnist(nn.Module):

    def __init__(self, h1=128, output_size=10):
        super().__init__()
        self.fc1 = torch.nn.Linear(28*28, h1)
        self.fc2 = torch.nn.Linear(h1, output_size)

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
        distribution[rows, elems] = 1 / len(elems)
    return distribution

def iid_mapping(areas: int, labels: int) -> np.ndarray:
    percentage = 1 / labels
    distribution = np.zeros((areas, labels))
    distribution.fill(percentage)
    return distribution

def partitioning(distribution: np.ndarray, dataset: Dataset) -> dict[int, list[int]]:
    targets = dataset.targets
    areas = distribution.shape[0]
    targets_cardinality = distribution.shape[1]
    class_counts = torch.bincount(targets)
    class_to_indices = {}
    for index in range(len(dataset)):
        c = targets[index].item()
        if c in class_to_indices:
            class_to_indices[c].append(index)
        else:
            class_to_indices[c] = [index]
    max_examples_per_area = int(math.floor(len(indices) / areas))
    elements_per_class =  torch.floor(torch.tensor(distribution) * max_examples_per_area).to(torch.int)
    partitions = { a: [] for a in range(areas) }
    for area in range(areas):
        elements_per_class_in_area = elements_per_class[area, :].tolist()
        for c in sorted(class_to_indices.keys()):
            elements = min(elements_per_class_in_area[c], class_counts[c].item())
            selected_indices = random.sample(class_to_indices[c], elements)
            partitions[area].extend(selected_indices)
    return partitions

def dirichlet_partitioning(dataset: Dataset, areas: int, beta: float) -> dict[int, list[int]]:
    # Implemented as in: https://proceedings.mlr.press/v97/yurochkin19a.html
    min_size = 0
    # indices = data.indices
    targets = dataset.targets
    N = len(dataset)
    class_to_indices = {}
    for index in range(N):
        c = targets[index].item()
        if c in class_to_indices:
            class_to_indices[c].append(index)
        else:
            class_to_indices[c] = [index]
    partitions = {a: [] for a in range(areas)}
    while min_size < 10:
        idx_batch = [[] for _ in range(areas)]
        for k in sorted(class_to_indices.keys()):
            idx_k = class_to_indices[k]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, areas))
            ## Balance
            proportions = np.array([p * (len(idx_j) < N / areas) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
    for j in range(areas):
        np.random.shuffle(idx_batch[j])
        partitions[j] = idx_batch[j]
    return partitions

def get_dataset(name: str, train: bool = True) -> Dataset:
    transform = transforms.Compose([transforms.ToTensor()])
    if name == 'MNIST':
        dataset = datasets.MNIST(root='data', train=train, download=True, transform=transform)
    elif name == 'EMNIST':
        dataset = datasets.EMNIST(root='dataset', split = 'letters', train=train, download=True, transform=transform)
    elif name == 'FashionMNIST':
        dataset = datasets.FashionMNIST(root='dataset', train=train, download=True, transform=transform)
    else:
        raise Exception(f'Dataset {name} not supported! Please check :)')
    return dataset

def to_subset(dataset, indexes):
    return Subset(dataset, indexes)

def training(model_weights, training_data, epochs, batch_size, experiment):
    model = instantiate_model(model_weights, experiment)
    global_weights = copy.deepcopy(list(model.parameters()))
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
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        mean_epoch_loss = sum(batch_loss) / len(batch_loss)
        epoch_loss.append(mean_epoch_loss)
    return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

def evaluate(model_weights, validation_data, batch_size, experiment):
    model = instantiate_model(model_weights, experiment)
    criterion = nn.NLLLoss()
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    data_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)
    for batch_index, (images, labels) in enumerate(data_loader):
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct / total
    return loss, accuracy

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)

def instantiate_model(model_weights, experiment: str, from_weights: bool = True):
    if experiment == 'MNIST' or experiment == 'FashionMNIST':
        model = NNMnist()
    elif experiment == 'EMNIST':
        model = NNMnist(output_size=27)
    else:
        raise Exception(f'Wrong experiment name ({experiment})! Please check :)')
    if from_weights:
        model.load_state_dict(model_weights)
    return model