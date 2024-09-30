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

    def forward(self, x: Tensor, activate: int) -> Tuple[Tensor, Tensor]:
        xbar = x - self.decoder.bias
        a = self.encoder.forward(xbar)
        pack = torch.topk(a, activate)
        f = torch.zeros_like(a)
        f.scatter_(1, pack.indices, F.relu(pack.values))
        xhat = self.decoder.forward(f)
        return f, xhat


@define
class HyperParams:
    features: int
    expandBy: int
    activate: int


@define
class TrainParams:
    batchSize: int
    numEpochs: int
    learnRate: float


class Trainer:

    hyperParams = HyperParams(features=768, expandBy=256, activate=32)
    trainParams = TrainParams(batchSize=16384, numEpochs=512, learnRate=1e-3)

    def __init__(self) -> None:
        self.name = Path(__file__).stem
        self.workDir = Path(workspace, self.name, "snapshot")
        self.workDir.mkdir(mode=0o777, parents=True, exist_ok=True)
        self.dataset = MsMarcoDataset()
        expandBy = self.hyperParams.expandBy
        self.model = nn.DataParallel(Model(Embedding.size, expandBy).cuda())
        learnRate = self.trainParams.learnRate
        self.optimizer = optim.Adam(self.model.parameters(), lr=learnRate)
        self.scaler = amp.GradScaler()
        wandb.init(project="private", entity="esae", name=self.name)
        wandb.save("source/**/*.py", policy="now")

    def trainLoss(self, x: Tensor, xhat: Tensor) -> Dict[str, Tensor]:
        loss = dict()
        loss["MSE"] = F.mse_loss(x, xhat)
        return loss

    def trainStep(self, x: Tensor) -> Dict[str, Tensor]:
        self.optimizer.zero_grad()
        x = x.to(self.model.device_ids[0])
        activate = self.hyperParams.activate
        with amp.autocast("cuda"):
            _, xhat = self.model.forward(x, activate)
            loss = self.trainLoss(x, xhat)
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
                console.log(f"{key:>3}={val:.8f}")
        wandb.finish()


if __name__ == "__main__":
    T = Trainer()
    T.run()
