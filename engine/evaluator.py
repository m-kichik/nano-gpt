import torch


class Evaluator:
    def __init__(self, loader, eval_iters, device=torch.device("cpu")):
        self.loader = loader
        self.eval_iters = eval_iters
        self.device = device

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()

        split_losses = {}
        for split in ["train", "val"]:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                X, Y = self.loader.get_batch(split)
                X, Y = X.to(self.device), Y.to(self.device)
                _, loss = model(X, Y)
                losses[k] = loss.item()
            split_losses[split] = losses.mean()

        return split_losses
