import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from typing import Type
from pathlib import Path
from rich.progress import Progress
from importlib import import_module
from source.model import workspace
from source.dataset import MsMarcoDataset, BeirDataset
from source.embedding import BgeBaseEmbedding
from source.interface import Embedding, Dataset, SAE


@torch.inference_mode()
def main(embedding: Type[Embedding], dataset: Dataset, version: str):
    # load model into memory
    readBase = Path(workspace, version, "snapshot")
    module = import_module(f"source.model.{version}")
    Model, Trainer = getattr(module, "Model"), getattr(module, "Trainer")
    assert issubclass(Model, SAE)
    features, expandBy = Trainer.hyperParams.features, Trainer.hyperParams.expandBy
    assert isinstance(features, int) and isinstance(expandBy, int)
    assert features == embedding.size
    model = Model(features, expandBy)
    assert isinstance(model, nn.Module)
    activate = Trainer.hyperParams.activate
    assert isinstance(activate, int)
    state = max(readBase.glob("*.pth"))
    model.load_state_dict(torch.load(state, map_location="cpu")["model"])
    model.eval().cuda()

    # compute mse scores for validation set
    mse = torch.tensor(0.0).cuda()
    relevant = Trainer.hyperParams.relevant
    assert isinstance(relevant, int)
    batchSize = Trainer.trainParams.batchSize
    assert isinstance(batchSize, int)
    numBatches = dataset.getMixLen("Validate") // batchSize
    with Progress() as p:
        t = p.add_task("Computing...", total=numBatches)
        for qry, docs in dataset.mixEmbIter(
            embedding, "Validate", relevant, batchSize, 8, True
        ):
            qry, docs = qry.cuda(), docs.cuda()
            _, qryHat = model.forward(qry, activate)
            _, docsHat = model.forward(docs.view(-1, docs.size(-1)), activate)
            mse += F.mse_loss(qryHat, qry)
            mse += F.mse_loss(docsHat.view(docs.size()), docs)
            p.advance(t)
        mse /= numBatches
    p.log("MSE:", mse.item())


if __name__ == "__main__":
    # specify command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("embedding", type=str, choices=["BgeBase"])
    parser.add_argument("dataset", type=str, choices=["MsMarco", "Beir"])
    parser.add_argument("version", type=str)
    args = parser.parse_args()

    # parse arguments into concrete instances
    match args.dataset:
        case "MsMarco":
            dataset = MsMarcoDataset()
        case "Beir":
            dataset = BeirDataset()
        case _:
            raise NotImplementedError()
    match args.embedding:
        case "BgeBase":
            embedding = BgeBaseEmbedding
        case _:
            raise NotImplementedError()

    # run the workflow
    main(embedding, dataset, args.version)
