import yaml

import torch


class Config:
    def __init__(self, config: dict):
        for k, v in config.items():
            if isinstance(v, dict):
                v = Config(v)
            if isinstance(v, str):
                try:
                    v = float(v)
                    if v.is_integer():
                        v = int(v)
                except:
                    pass
            config[k] = v

        self.parse_config(config)

    def parse_config(self, config: dict):
        self.__dict__.update(config)


def parse_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return Config(config)


class DataLoader:
    def __init__(self, text, tokenizer, block_size, batch_size, train_test_split=0.2):
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


def get_loader(text, tokenizer, block_size, batch_size, train_test_split=0.2):
    return DataLoader(text, tokenizer, block_size, batch_size, train_test_split)
