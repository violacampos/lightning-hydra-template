from typing import Any, Dict, Optional, Tuple

import operator
from functools import reduce

import torch
import tqdm
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms

from transformers import AutoTokenizer
from datasets import load_dataset






class TokenizedDACSDataset(Dataset):
    def __init__(self, 
                 tokenizer, 
                 max_len_inp=32,
                 max_len_out=256):

        self.data = load_dataset("src/data/scripts/dacs_dataset_script.py")

        self.max_len_input = max_len_inp
        self.max_len_output = max_len_out
        self.tokenizer = tokenizer

        self.tokenized_data = self.data['train'].map(self.build_tokenized_samples, batched=True, remove_columns=self.data['train'].column_names)


    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, index):
        return self.tokenized_data[index]

        

    def build_tokenized_samples(self, examples):

        inputs = [(f"Number: {number}  DACS Chain: ", f"Number: {number}  DACS Sequence: ") for number in examples['number']] 
        inputs = reduce(operator.concat, inputs)
        targets = [(chain, sequence) for chain, sequence in zip(examples['chain'], examples['sequence'])]
        targets = reduce(operator.concat, targets)

        tokenized_inputs = self.tokenizer.batch_encode_plus(
                inputs, max_length=self.max_len_input,
                truncation = True,
                padding='max_length', return_tensors="pt"
            )

        tokenized_targets = self.tokenizer.batch_encode_plus(
                targets, max_length=self.max_len_output,
                truncation = True,
                padding='max_length',return_tensors="pt"
            )

        labels = tokenized_targets["input_ids"].clone()
        labels[labels==50256] = -100
        tokenized_inputs["labels"] = labels

        return tokenized_inputs
            

def collate_tensor_fn(batch):
    input_ids = torch.stack([torch.tensor(x['input_ids']) for x in batch])
    attention_mask = torch.stack([torch.tensor(x['attention_mask']) for x in batch])
    labels = torch.stack([torch.tensor(x['labels']) for x in batch])
    return {'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels}



class DACSDataModule(LightningDataModule):
    """LightningDataModule for DACS (differential addition chains) dataset.

    A DataModule implements 6 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        checkpoint: str = "Salesforce/codet5p-2b",
        train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        max_len_input: int = 32,
        max_len_output: int = 256
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None


    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            # format: {"number": datasets.Value("string"),
            #        "chain": datasets.Value("string"),
            #        "sequence": datasets.Value("string"}
            
            dataset = TokenizedDACSDataset(tokenizer = self.tokenizer) 
            
            
            
            
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset ,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )



    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=collate_tensor_fn, drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_tensor_fn,
            drop_last=True
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_tensor_fn,
            drop_last=True
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    _ = DACSDataModule()
