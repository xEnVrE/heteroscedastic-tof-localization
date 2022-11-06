from torch.utils.data import Dataset
from trajectory import split_trajectories


class DatasetOfSubsequences(Dataset):

    def __init__(self, trajectories, length):

        self.sequences = split_trajectories(trajectories, length)


    def __getitem__(self, idx):

        return self.sequences[idx]


    def __len__(self):

        return len(self.sequences)
