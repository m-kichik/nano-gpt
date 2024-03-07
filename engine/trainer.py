import os

import torch
from tqdm import tqdm
import wandb

from .config import Config
from .evaluator import Evaluator


class Trainer:
    def __init__(
        self,
        loader,
        tokenizer,
        train_iters=None,
        eval_interval=None,
        eval_iters=None,
        out_dir=None,
        device=torch.device("cpu"),
        config=None,
    ):
        if config is not None and isinstance(config, Config):
            train_iters = config.TRAIN.ITERS
            eval_interval = config.TRAIN.EVAL_INTERVAL
            eval_iters = config.EVAL.ITERS
            out_dir = config.TRAIN.OUT_DIR
        else:
            if None in [
                train_iters,
                eval_interval,
                eval_iters,
                out_dir,
            ]:
                raise ValueError("Lack of defined parameters")

        self.loader = loader
        self.tokenizer = tokenizer
        self.train_iters = train_iters
        self.eval_interval = eval_interval
        self.out_dir = out_dir
        self.device = device

        self.evaluator = Evaluator(loader, eval_iters, device)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    def train(self, model, optimizer):
        val_loss = 1e9
        for i in tqdm(range(self.train_iters)):
            if i % self.eval_interval == 0 or i == self.train_iters - 1:
                losses = self.evaluator.evaluate(model)

                if losses["val"] < val_loss:
                    torch.save(model.state_dict(), self.out_dir + "/best.pt")
                    val_loss = losses["val"]

                if wandb.run is not None:
                    wandb.log(
                        {
                            "step": i,
                            "train_loss": losses["train"],
                            "val_loss": losses["val"],
                        }
                    )
            else:
                if wandb.run is not None:
                    wandb.log({"step": i})

            model.train()
            X, Y = self.loader.get_batch("train")
            X, Y = X.to(self.device), Y.to(self.device)
            _, loss = model(X, Y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), self.out_dir + "/last.pt")
