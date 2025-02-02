from typing import Dict, List, Optional, NoReturn
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from data.audiotext_dataset import AudioTextDataset


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset: object,
        val_dataset: object,
        batch_size: int,
        num_workers: int
    ):
        r"""Data module. To get one batch of data:

        code-block:: python

            data_module.setup()

            for batch_data_dict in data_module.train_dataloader():
                print(batch_data_dict.keys())
                break

        Args:
            train_sampler: Sampler object
            train_dataset: Dataset object
            num_workers: int
            distributed: bool
        """
        super().__init__()
        self._train_dataset = train_dataset
        self._val_dataset = val_dataset
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.collate_fn = collate_fn


    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage: Optional[str] = None) -> NoReturn:
        r"""called on every device."""

        # make assignments here (val/train/test split)
        # called on every process in DDP

        # SegmentSampler is used for selecting segments for training.
        # On multiple devices, each SegmentSampler samples a part of mini-batch
        # data.
        self.train_dataset = self._train_dataset
        self.val_dataset = self._val_dataset
        
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        r"""Get train loader."""
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False,
            shuffle=True,
            drop_last=True
        )

        return train_loader

    # 새로 정의한 코드 
    
    def val_dataloader(self) -> torch.utils.data.DataLoader:
        r"""Get validation loader."""
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False,
            shuffle=False,
            drop_last=True
        )

        return val_loader
    
    '''
    def val_dataloader(self):
        # val_split = Dataset(...)
        # return DataLoader(val_split)
        pass
    '''
    def test_dataloader(self):
        # test_split = Dataset(...)
        # return DataLoader(test_split)
        pass

    def teardown(self):
        # clean up after fit or test
        # called on every process in DDP
        pass

# 멀티모달한 데이터를 배치로 처리하기 위한 추가적인 함수
# text는 리스트로, waveform인 텐서는 torch.stack을 사용해 배치로 묶는다.
# data_dict라는 딕셔너리에 audio_text라는 딕셔너리로 다시 저장하여 관리
def collate_fn(list_data_dict):
    r"""Collate mini-batch data to inputs and targets for training.

    Args:
        list_data_dict: e.g., [
            {
                'text': 'a sound of dog',
                'waveform': (1, samples),
                'modality': 'audio_text'
            }
            ...
            ]
    Returns:
        data_dict: e.g. 
            'audio_text': {
                'text': ['a sound of dog', ...]
                'waveform': (batch_size, 1, samples)
        }
    """
    
    at_list_data_dict = [data_dict for data_dict in list_data_dict if data_dict['modality']=='audio_text']

    at_data_dict = {}
    
    if len(at_list_data_dict) > 0:
        for key in at_list_data_dict[0].keys():
            at_data_dict[key] = [at_data_dict[key] for at_data_dict in at_list_data_dict]
            if key == 'waveform':
                at_data_dict[key] = torch.stack(at_data_dict[key])
            elif key == 'text':
                at_data_dict[key] = [text for text in at_data_dict[key]]

    
    data_dict = {
        'audio_text': at_data_dict
    }
    
    return data_dict