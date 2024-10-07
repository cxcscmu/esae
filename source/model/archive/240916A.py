import torch
import wandb
import torch.amp as amp
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Dict
from attrs import define
from pathlib import Path
from source import console
from rich.progress import Progress
from source.model import workspace
from source.interface import SAE
from source.dataset.msMarco import MsMarcoDataset
from source.embedding.bgeBase import BgeBaseEmbedding as Embedding


class Model(nn.Module, SAE):

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
    relevant: int


@define
class TrainParams:
    batchSize: int
    numEpochs: int
    learnRate: float


class Trainer:

    hyperParams = HyperParams(features=768, expandBy=256, activate=384, relevant=8)
    trainParams = TrainParams(batchSize=512, numEpochs=128, learnRate=1e-3)

    def __init__(self) -> None:
        self.name = Path(__file__).stem
        self.workDir = Path(workspace, self.name, "snapshot")
        self.workDir.parent.mkdir(mode=0o770, exist_ok=True)
        self.workDir.mkdir(mode=0o770, exist_ok=True)
        self.dataset = MsMarcoDataset()
        expandBy = self.hyperParams.expandBy
        self.model = nn.DataParallel(Model(Embedding.size, expandBy).cuda())
        learnRate = self.trainParams.learnRate
        self.optimizer = optim.Adam(self.model.parameters(), lr=learnRate)
        self.scaler = amp.GradScaler()
        numEpochs = self.trainParams.numEpochs
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=numEpochs)
        activate = self.hyperParams.activate
        relevant = self.hyperParams.relevant
        name = f"{self.name}: expand={expandBy},activate={activate},relevant={relevant}"
        wandb.init(project="esae", entity="haok", name=name)
        wandb.save("source/**/*.py", policy="now")

    def trainLoss(
        self, qry: Tensor, docs: Tensor, qryHat: Tensor, docsHat: Tensor
    ) -> Dict[str, Tensor]:
        loss = dict()
        loss["MSE"] = torch.tensor(0.0, requires_grad=True)
        loss["MSE"] = loss["MSE"] + F.mse_loss(qryHat, qry)
        loss["MSE"] = loss["MSE"] + F.mse_loss(docsHat, docs)
        return loss

    def trainStep(self, qry: Tensor, docs: Tensor) -> Dict[str, Tensor]:
        self.optimizer.zero_grad()
        qry = qry.to(self.model.device_ids[0])
        docs = docs.to(self.model.device_ids[0])
        activate = self.hyperParams.activate
        with amp.autocast("cuda"):
            _, qryHat = self.model.forward(qry, activate)
            _, docsHat = self.model.forward(docs.view(-1, docs.size(-1)), activate)
            assert isinstance(qryHat, Tensor) and isinstance(docsHat, Tensor)
            loss = self.trainLoss(qry, docs, qryHat, docsHat.view(docs.size()))
        self.scaler.scale(sum(loss.values())).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss

    def trainIter(self, i: int) -> Dict[str, float]:
        iterLoss = dict()
        relevant = self.hyperParams.relevant
        batchSize = self.trainParams.batchSize
        numEpochs = self.trainParams.numEpochs
        with Progress(console=console) as progress:
            T = progress.add_task(
                f"[{i:>3}/{numEpochs}]", total=self.dataset.getMixLen("Train")
            )
            for qry, docs in self.dataset.mixEmbIter(
                Embedding, "Train", relevant, batchSize, 8, True
            ):
                loss = self.trainStep(qry, docs)
                progress.advance(T, qry.size(0))
                for key, val in loss.items():
                    iterLoss[key] = iterLoss.get(key, 0.0) + val.item()
            progress.stop_task(T)
        numBatches = self.dataset.getMixLen("Train") // batchSize
        for key in iterLoss.keys():
            iterLoss[key] /= numBatches
        self.scheduler.step()
        return iterLoss

    def run(self):
        minLoss = float("inf")
        numEpochs = self.trainParams.numEpochs
        for i in range(1, numEpochs + 1):
            trainLoss = self.trainIter(i)
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
