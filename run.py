from tqdm import tqdm

import torch
import wandb

from nano_gpt import NanoGPT
from tokenizer import CharTokenizer
from utils import parse_config, get_loader

torch.manual_seed(42)


@torch.no_grad()
def eval(model, dataloader, eval_iters, device):
    model.eval()

    split_losses = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = dataloader.get_batch(split)
            X, Y = X.to(device), Y.to(device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        split_losses[split] = losses.mean()

    return split_losses


def train(
    model, optimizer, dataloader, max_iters, eval_interval, eval_iters, device, log=None
):
    for i in tqdm(range(max_iters)):
        if log is not None and i % eval_interval == 0 or i == max_iters - 1:
            losses = eval(model, dataloader, eval_iters, device)

            if log == "terminal":
                print(
                    f"Iter {i}: train loss {losses['train']}, val loss: {losses['val']}"
                )
            if log == "wandb":
                wandb.log({"train_loss": losses["train"], "val_loss": losses["val"]})

        model.train()
        X, Y = dataloader.get_batch("train")
        X, Y = X.to(device), Y.to(device)
        _, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def main():
    with open("./data/Strugackie_prepared.txt", "r") as file:
        text = file.read()
    tokenizer = CharTokenizer(text)

    config = parse_config("./configuration.yaml")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = NanoGPT(
        tokenizer.vocab_size,
        config.MODEL.NUM_HEADS,
        config.MODEL.NUM_BLOCKS,
        config.MODEL.NUM_EMBEDDINGS,
        config.MODEL.BLOCK_SIZE,
        device=device,
    )

    print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAIN.LR)

    loader = get_loader(
        text, tokenizer, config.MODEL.BLOCK_SIZE, config.TRAIN.BATCH_SIZE
    )

    wandb_config = {
        "num_heads": config.MODEL.NUM_HEADS,
        "num_blocks": config.MODEL.NUM_BLOCKS,
        "num_embeddings": config.MODEL.NUM_EMBEDDINGS,
        "block_size": config.MODEL.BLOCK_SIZE,
        "lr": config.TRAIN.LR,
        "batch_size": config.TRAIN.BATCH_SIZE,
    }

    wandb.init(project="NanoGPT", name="initial", config=wandb_config)

    train(
        model,
        optimizer,
        loader,
        config.TRAIN.ITERS,
        config.TRAIN.EVAL_INTERVAL,
        config.EVAL.ITERS,
        device,
        log="wandb",
    )

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(tokenizer.decode(model.generate(context, max_new_tokens=500)[0].tolist()))
    torch.save(model.state_dict(), "checkpoint.pt")


if __name__ == "__main__":
    main()
