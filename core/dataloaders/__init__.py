from torch.utils.data import DataLoader

import torch


class DataLoaderFactory:

    def __init__(self, args):
        self.config = args
        self.num_gpus = max(1, torch.cuda.device_count())

    def build(self, split='train'):

        if self.config.dset == 'D2MIDI':
            from .dataset import D2MIDIDataset
            ds = D2MIDIDataset.from_cfg(self.config, split=split)
        else:
            raise Exception

        loader = DataLoader(
            ds,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers* self.num_gpus,
            shuffle=(split == 'train'),
            drop_last=True,
        )

        return loader
