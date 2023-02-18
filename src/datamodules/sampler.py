from torch.utils.data import Sampler


class BalanceSampler(Sampler):
    """Samples elements from a dataset in a balanced way.

    See examples/rsna for example.

    Args:
        data_source (Dataset): A PyTorch dataset object with a `labels`
            attribute that contains the class labels for each sample.

    Example:
        >>> dataset = MyDataset()
        >>> sampler = BalanceSampler(dataset)
        >>> dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    """

    def __init__(self, data_source):
        """Initializes the balance sampler.

        Counts the number of samples in each class, and calculates the weight
        for each sample based on its class.

        Args:
            data_source (Dataset): A PyTorch dataset object with a `labels`
                attribute that contains the class labels for each sample.

        """
        ...

    def __iter__(self):
        """Returns an iterator over the indices of the samples to be
        included in the batch.

        Generates a list of indices for each class, then loops over the
        classes and adds samples to the batch until the desired balance is
        achieved.

        Returns:
            iter: An iterator over the indices of the samples to be included
            in the batch.

        """
        ...

    def __len__(self):
        """Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.

        """
        ...
