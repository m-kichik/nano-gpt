import os
from tqdm import tqdm

import torch
import wandb

from engine.model import NanoGPT
from engine.tokenizer import CharTokenizer
from engine.loader import DataLoader
from engine.config import parse_config
from engine.trainer import Trainer
from engine.evaluator import Evaluator

torch.manual_seed(42)


def main():
    config = parse_config("configs/basic.yaml")

    tokenizer = CharTokenizer(config.TRAIN.DATA_PATH)
    loader = DataLoader(tokenizer=tokenizer, config=config)

    if config.MODEL.DEVICE != "cpu":
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device("cpu")

    model = NanoGPT(
        vocabulary_size=tokenizer.vocab_size,
        config=config,
        device=device,
    )

    trainer = Trainer(loader=loader, tokenizer=tokenizer, config=config, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAIN.LR)

    if config.TRAIN.WANDB:
        wandb_config = {
            "num_heads": config.MODEL.NUM_HEADS,
            "num_blocks": config.MODEL.NUM_BLOCKS,
            "num_embeddings": config.MODEL.NUM_EMBEDDINGS,
            "block_size": config.MODEL.BLOCK_SIZE,
            "lr": config.TRAIN.LR,
            "batch_size": config.TRAIN.BATCH_SIZE,
        }

        wandb.init(project="NanoGPT", name=config.TRAIN.EXPERIMENT_NAME, config=wandb_config)

    trainer.train(model=model, optimizer=optimizer)

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(tokenizer.decode(model.generate(context, max_new_tokens=500)[0].tolist()))


if __name__ == "__main__":
    main()
