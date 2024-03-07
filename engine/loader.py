import torch

from .config import Config


class DataLoader:
    def __init__(
        self,
        tokenizer,
        text_path=None,
        block_size=None,
        batch_size=None,
        train_test_split=0.2,
        config=None,
    ):
        if config is not None and isinstance(config, Config):
            text_path = config.TRAIN.DATA_PATH
            block_size = config.MODEL.BLOCK_SIZE
            batch_size = config.TRAIN.BATCH_SIZE
        else:
            if None in [text_path, block_size, batch_size]:
                raise ValueError("Lack of defined parameters")

        if text_path is None:
            raise ValueError("Empty path")
        with open(text_path, "r") as file:
            text = file.read()

        data = tokenizer.encode(text)
        num_test = int(train_test_split * len(data))

        self.block_size = block_size
        self.batch_size = batch_size
        self.train_data = torch.tensor(data[:-num_test], dtype=torch.long)
        self.test_data = torch.tensor(data[-num_test:], dtype=torch.long)

    def get_batch(self, split):
        if split == "train":
            sample = self.train_data
        else:
            sample = self.test_data

        start_idx = torch.randint(len(sample) - self.block_size, (self.batch_size,))
        x = torch.stack([sample[i : i + self.block_size] for i in start_idx])
        y = torch.stack([sample[i + 1 : i + self.block_size + 1] for i in start_idx])
        return x, y
