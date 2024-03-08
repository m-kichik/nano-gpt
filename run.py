from argparse import ArgumentParser
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


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, help="path to config file")
    parser.add_argument(
        "--mode",
        type=str,
        default="demo",
        help="train, eval or demo for appropriate task",
    )
    return parser.parse_args()


def main():
    args = get_args()
    config = parse_config(args.config)

    tokenizer = CharTokenizer(config.TRAIN.DATA_PATH)
    loader = DataLoader(tokenizer=tokenizer, config=config)

    if config.MODEL.DEVICE != "cpu":
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
    else:
        device = torch.device("cpu")

    model = NanoGPT(
        vocabulary_size=tokenizer.vocab_size,
        config=config,
        device=device,
    )

    if config.MODEL.CHECKPOINT is not None:
        model.load_state_dict(torch.load(config.MODEL.CHECKPOINT))

    if args.mode == "train":
        trainer = Trainer(
            loader=loader, tokenizer=tokenizer, config=config, device=device
        )
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

            wandb.init(
                project="NanoGPT",
                name=config.TRAIN.EXPERIMENT_NAME,
                config=wandb_config,
            )

        trainer.train(model=model, optimizer=optimizer)

    if args.mode == "eval":
        evaluator = Evaluator(loader, config.EVAL.ITERS, device)
        loss = evaluator.evaluate(model)
        print("loss = ", loss["val"].item())

    if args.mode == "demo":
        inputs = input("Input prompt in russian:\n")
        inputs = torch.tensor(tokenizer.encode(inputs), dtype=torch.long, device=device)
        inputs = inputs.unsqueeze(0)
        result = model.generate(inputs, config.DEMO.MAX_TOKENS)
        result = tokenizer.decode(result[0].tolist())

        if not result.endswith("."):
            idx = result.rfind(".")
            result = result[: idx + 1]

        print(result)


if __name__ == "__main__":
    main()
