from lightning.pytorch import LightningDataModule
from torch.utils.data import Dataset, DataLoader, random_split
from torch import Tensor


class dataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __len__(self) -> int:
        return None

    def __getitem__(self, idx: int) -> Tensor:
        return None


class DataModule(LightningDataModule):
    def __init__(
        self, batch_size: int, num_workers: int, pin_memory: bool, train_pt: float
    ) -> None:
        super().__init__()
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.pin_memory: bool = pin_memory
        self.train_pt: float = train_pt

    def setup(self, stage: str) -> None:
        data = dataset()
        train_len = round(self.train_pt, (len(data)))
        val_len = len(data) - train_len
        self.train_ds, self.val_ds = random_split(data, [train_len, val_len])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )
