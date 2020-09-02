from math import ceil
from typing import Callable

import common_io
import torch
from torch.utils.data.dataset import IterableDataset

class OdpsChainIterator(object):
    def __init__(self, num_iteration: int, iterator_type, *args, **kwargs):
        super().__init__()
        self.num_iteration = num_iteration
        self.args = args
        self.kwargs = kwargs
        self.index = 0
        self.iterator_type = iterator_type
        self.iterator = iterator_type(*args, **kwargs)

    def __len__(self):
        return self.num_iteration

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self):
            raise StopIteration()
        try:
            result = next(self.iterator)
        except StopIteration:
            self.iterator = self.iterator_type(*self.args, **self.kwargs)
            result = next(self.iterator)
        self.index += 1
        return result


class OdpsIterDataset(IterableDataset):
    def __init__(
            self,
            dataset_total_len: int,
            num_workers: int,
            iterator_type: Callable,
            rank: int,
            world: int,
            pad_dataset: bool=True,
            *args,
            **kwargs
    ):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        self.dataset_total_len = dataset_total_len
        self.num_workers = num_workers
        self.pad_dataset = pad_dataset
        self.iterator_type = iterator_type
        self.rank = rank
        self.world = world

    def __iter__(self):
        if self.pad_dataset and self.world > 1:
            return OdpsChainIterator(
                num_iteration=ceil(
                    self.dataset_total_len / (self.num_workers * self.world)),
                iterator_type=self.iterator_type,
                *self.args,
                **self.kwargs,
                rank=self.rank,
                world=self.world
            )
        else:
            return self.iterator_type(
                *self.args,
                **self.kwargs,
                rank=self.rank,
                world=self.world
            )

    @property
    def collate_fn(self):
        return self.iterator_type.collate_fn


class OdpsIterator:
    def __init__(self, feature_table: str, selected_cols: str, num_prefetch: int, rank: int, world: int):
        super().__init__()
        worker_info = torch.utils.data.get_worker_info()
        assert worker_info is not None
        self.table_reader = common_io.table.TableReader(
            feature_table,
            selected_cols=selected_cols,
            slice_id=rank * worker_info.num_workers + worker_info.id,
            slice_count=worker_info.num_workers * world,
            capacity=num_prefetch
        )
        self.__dataset_len = None

    def __iter__(self):
        return self

    def __next__(self):
        try:
            records = self.table_reader.read(num_records=1)[0]
            return records
        except common_io.exception.OutOfRangeException:
            self.table_reader.close()
            raise StopIteration()

    @staticmethod
    def collate_fn(batches):
        raise NotImplementedError()