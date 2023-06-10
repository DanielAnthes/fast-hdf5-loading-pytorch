from torchdata import datapipes as dp
import torch
import h5py
import numpy as np


def batch_reading_pipe(dataset, chunk_size=128, shuffle_chunks=True, shuffle_buffer_size=10):
    '''
    Create a torchdata pipe for efficient reading from datasets where reads are expensive.
    This is useful for reading from HDF5 files, but can also be used for other file formats.
    Instead of reading one sample at a time, we read a chunk of samples to minimize the number of file reads.

    :param dataset: torch dataset
    :param chunk_size: number of consecutive samples to request in one read operation
    :param shuffle_chunks: whether to shuffle the order of the chunks
    :param shuffle_buffer_size: size of buffer to use for shuffling at the sample level
    :return: torchdata pipe
    '''
    pipe = dp.map.SequenceWrapper(range(len(dataset)))  # indices in dataset
    pipe = dp.iter.Batcher(pipe, batch_size=chunk_size)  # subsequent indices to load together. This minimizes number of file reads
    if shuffle_chunks:
        pipe = dp.iter.Shuffler(pipe, buffer_size=len(pipe))  # shuffle batches so we read from random start points in dataset
    pipe = dp.iter.ShardingFilter(pipe)  # make sure each worker gets its own batches
    pipe = dp.iter.Mapper(pipe, lambda x: dataset[x])  # load data from dataset into memory
    pipe = dp.iter.Mapper(pipe, lambda x: list(zip(*x))) # each datapoint becomes a tuple. For a traditional image, label dataset this is (image, label)
    if shuffle_buffer_size > 0:
        pipe = dp.iter.Shuffler(pipe, buffer_size=shuffle_buffer_size, unbatch_level=1)  # shuffle samples from consecutive batches
    return pipe

class Hdf5ImageLabelDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, split='train', transform=None, in_memory=False):
        file = h5py.File(h5_path, 'r')
        dset = file[split]
        self.data = dset['data']
        self.labels = dset['labels']
        self.transform = transform

        if in_memory:
            self.data = np.array(self.data)
            self.labels = np.array(self.labels)

        # print info
        print('Dataset: {}'.format(split))
        print('Data shape: {}'.format(self.data.shape))
        print('Labels shape: {}'.format(self.labels.shape))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # convert to float in [0,1]
        sample = sample.astype(np.float32) / 255.0
        label = self.labels[idx]
        label = label.astype(np.int64)
        # convert label to torch int8 and then tensor
        label = torch.tensor(label)
        if self.transform:
            # since we use this dataloader with a batchsampler we need to make sure to call transform on each element of the batch
            # allocate empty tensor
            shape = sample.shape
            sample_transformed = torch.zeros((shape[0], shape[3], shape[1], shape[2]))
            for i, s in enumerate(sample):
                sample_transformed[i] = self.transform(s)
            sample = sample_transformed

        return sample, label

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, length):
        self.length = length
        # generate fake data and labels
        self.data = torch.rand(length, 3, 256, 256)
        # labels are indices
        self.labels = torch.arange(length)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


if __name__ == '__main__':
    # dummy image label dataset class
    dataset = DummyDataset(1024)
    pipe = batch_reading_pipe(dataset, chunk_size=128, shuffle_chunks=True, shuffle_buffer_size=1024)
    dataloader = torch.utils.data.DataLoader(pipe, batch_size=128, num_workers=0)
    for batch in dataloader:
        im, label = batch
        print(label)
