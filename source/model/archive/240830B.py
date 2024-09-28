import torch
import wandb
import torch.amp as amp
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Dict
from attrs import define
from pathlib import Path
from source import console
from rich.progress import Progress
from source.model import workspace
from source.dataset.msMarco import MsMarcoDataset
from source.embedding.bgeBase import BgeBaseEmbedding as Embedding


class Model(nn.Module):

    def __init__(self, features: int, expandBy: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(features, features * expandBy)
        self.decoder = nn.Linear(features * expandBy, features)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        xbar = x - self.decoder.bias
        f = F.relu(self.encoder.forward(xbar))
        xhat = self.decoder.forward(f)
        return f, xhat


@define
class HyperParams:
    features: int
    expandBy: int
    sparsity: float


@define
class TrainParams:
    batchSize: int
    numEpochs: int
    learnRate: float


class Trainer:

    hyperParams = HyperParams(features=768, expandBy=384, sparsity=0.98)
    trainParams = TrainParams(batchSize=4096, numEpochs=256, learnRate=0.1)

    def __init__(self) -> None:
        self.name = Path(__file__).stem
        self.workDir = Path(workspace, self.name, "snapshot")
        self.workDir.mkdir(mode=0o777, parents=True, exist_ok=True)
        self.dataset = MsMarcoDataset()
        expandBy = self.hyperParams.expandBy
        self.model = nn.DataParallel(Model(Embedding.size, expandBy).cuda())
        learnRate = self.trainParams.learnRate
        self.optimizer = optim.SGD(self.model.parameters(), lr=learnRate)
        self.scaler = amp.GradScaler()
        wandb.init(project="private", entity="esae", name=self.name)
        wandb.save("source/**/*.py", policy="now")

    def trainLoss(self, x: Tensor, f: Tensor, xhat: Tensor) -> Dict[str, Tensor]:
        loss = dict()
        loss["MSE"] = F.mse_loss(x, xhat)
        PSL = torch.sum(f * (1 - f)) / (f.numel())
        sparsity = self.hyperParams.sparsity
        base = torch.clamp(torch.mean(f, dim=0) - (1.0 - sparsity), min=0.0)
        ASL = torch.sum(torch.pow(base, 2)) / f.size(1)
        loss["SPINE"] = PSL + ASL
        return loss

    def trainStep(self, x: Tensor) -> Dict[str, Tensor]:
        self.optimizer.zero_grad()
        x = x.to(self.model.device_ids[0])
        with amp.autocast("cuda"):
            f, xhat = self.model.forward(x)
            loss = self.trainLoss(x, f, xhat)
        self.scaler.scale(sum(loss.values())).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss

    def trainIter(self) -> Dict[str, float]:
        iterLoss = dict()
        batchSize = self.trainParams.batchSize
        with Progress(console=console) as progress:
            T = progress.add_task("Training", total=self.dataset.getMixLen("Train"))
            for x in self.dataset.mixEmbIter(Embedding, "Train", batchSize, 8, True):
                loss = self.trainStep(x)
                progress.advance(T, x.size(0))
                for key, val in loss.items():
                    iterLoss[key] = iterLoss.get(key, 0.0) + val.item()
            progress.stop_task(T)
        numBatches = self.dataset.getMixLen("Train") // batchSize
        for key in iterLoss.keys():
            iterLoss[key] /= numBatches
        return iterLoss

    def run(self):
        minLoss = float("inf")
        numEpochs = self.trainParams.numEpochs
        for i in range(1, numEpochs + 1):
            console.rule(f"{i:>3}/{numEpochs}")
            trainLoss = self.trainIter()
            if sum(trainLoss.values()) <= minLoss:
                minLoss, state = sum(trainLoss.values()), dict()
                state["model"] = self.model.module.state_dict()
                state["optimizer"] = self.optimizer.state_dict()
                state["scaler"] = self.scaler.state_dict()
                state["epoch"], state["minLoss"] = i, minLoss
                torch.save(state, Path(self.workDir, f"{i:03}.pth"))
                globs = sorted(self.workDir.glob("*.pth"), reverse=True)
                while len(globs) > 3:
                    globs.pop().unlink()
            health = dict()
            health.update(trainLoss)
            health["LR"] = self.optimizer.param_groups[0]["lr"]
            wandb.log(health)
            for key, val in health.items():
                console.log(f"{key:>5}={val:.8f}")
        wandb.finish()


if __name__ == "__main__":
    T = Trainer()
    T.run()
