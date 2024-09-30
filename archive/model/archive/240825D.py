import torch
import wandb
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from typing import Tuple
from pathlib import Path
from attrs import define, asdict
from source import progress, console
from source.interface import SAE
from source.dataset.msMarco import MsMarcoDataset
from source.embedding.bgeBase import BgeBaseEmbedding
from source.model import workspace


class Model(nn.Module, SAE):

    def __init__(self, features: int, expandBy: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(features, features * expandBy)
        self.decoder = nn.Linear(features * expandBy, features)

    def forward(self, x: Tensor, K: int) -> Tuple[Tensor, Tensor]:
        xbar = x - self.decoder.bias
        a = self.encoder.forward(xbar)
        pack = torch.topk(a, K)
        f = torch.zeros_like(a)
        f.scatter_(1, pack.indices, F.relu(pack.values))
        xhat = self.decoder.forward(f)
        return xhat, f


@define
class HyperParams:
    expandBy: int
    activate: int


@define
class TrainParams:
    batchSize: int
    numEpochs: int
    learnRate: float


class Trainer:

    hyperParams = HyperParams(expandBy=256, activate=256)
    trainParams = TrainParams(batchSize=4096, numEpochs=100, learnRate=1e-4)

    def __init__(self) -> None:
        self.name = Path(__file__).stem
        self.workDir = Path(workspace, self.name, "state")
        self.workDir.mkdir(mode=0o770, parents=True, exist_ok=True)
        self.dataset = MsMarcoDataset()
        dimension = BgeBaseEmbedding.size
        expandBy = self.hyperParams.expandBy
        self.model = nn.DataParallel(Model(dimension, expandBy).cuda())
        learnRate = self.trainParams.learnRate
        self.optimizer = optim.Adam(self.model.parameters(), lr=learnRate)
        wandb.init(name=self.name, project="esae", config=asdict(self.hyperParams))
        wandb.save("source/**/*.py", policy="now")

    def trainStep(self) -> float:
        stepLoss = 0.0
        self.model.train()
        activate = self.hyperParams.activate
        batchSize = self.trainParams.batchSize
        T = progress.add_task("Training", total=self.dataset.getDocLen())
        for x in self.dataset.docEmbIter(BgeBaseEmbedding, batchSize, 4, True):
            self.optimizer.zero_grad()
            x = x.to(self.model.device_ids[0])
            xhat, f = self.model.forward(x, activate)
            loss = F.mse_loss(xhat, x)
            loss.backward()
            self.optimizer.step()
            stepLoss += loss.item()
            progress.update(T, advance=batchSize)
        progress.stop_task(T)
        numBatches = self.dataset.getDocLen() // batchSize
        stepLoss /= numBatches
        progress.remove_task(T)
        return stepLoss

    def dispatch(self):
        minLoss = float("inf")
        numEpochs = self.trainParams.numEpochs
        for i in range(1, numEpochs + 1):
            console.rule(f"{i:>3}/{numEpochs}")
            trainLoss = self.trainStep()
            if trainLoss <= minLoss:
                minLoss, state = trainLoss, dict()
                state["model"] = self.model.module.state_dict()
                state["optimizer"] = self.optimizer.state_dict()
                torch.save(state, Path(self.workDir, f"{i:03}.pth"))
                globs = sorted(self.workDir.glob("*.pth"), reverse=True)
                while len(globs) > 3:
                    globs.pop().unlink()
            health = dict()
            health["MSE"] = trainLoss
            health["LR"] = self.optimizer.param_groups[0]["lr"]
            for key, val in health.items():
                console.log(f"{key:>3}={val:.8f}")
            wandb.log(health)
        wandb.finish()


if __name__ == "__main__":
    T = Trainer()
    T.dispatch()
