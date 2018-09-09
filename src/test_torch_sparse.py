import torch
from torch.utils.data import DataLoader

print('torch version: ', torch.__version__)

i = torch.LongTensor([[0, 1, 1], [2, 0, 2]])
v = torch.FloatTensor([3, 4, 5])
sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size([2, 3]))

dataset_sp = 2 * [sparse_tensor]


def collate_fn(batch):
    return batch[0]


loader = DataLoader(dataset_sp, batch_size=1, collate_fn=collate_fn)

print('num_workers=0: ')
for i, b in enumerate(loader):
    print(i, b.size())
    loader = DataLoader(dataset_sp, batch_size=1, collate_fn=collate_fn, num_workers=1)
    print('num_workers=1: ')
    for i, b in enumerate(loader):
        print(i, b.size())

