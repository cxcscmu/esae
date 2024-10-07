import torch
import numpy as np
import torch.nn as nn
from typing import Type
from pathlib import Path
from rich.progress import Progress
from importlib import import_module
from source import console
from source.model import workspace
from source.interface import Embedding, Dataset, SAE


@torch.inference_mode()
def saveComputed(embedding: Type[Embedding], dataset: Dataset, version: str):
    # define where to read weights and save results
    readBase = Path(workspace, version, "snapshot")
    saveBase = Path(workspace, version, "computed", dataset.name)
    saveBase.mkdir(mode=0o770, parents=True, exist_ok=True)

    # load model into memory
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

    # create memmap for saving
    docLatentIndex = np.memmap(
        Path(saveBase, "docLatentIndex.bin"),
        dtype=np.int32,
        mode="w+",
        shape=(dataset.getDocLen(), activate),
    )
    docLatentValue = np.memmap(
        Path(saveBase, "docLatentValue.bin"),
        dtype=np.float32,
        mode="w+",
        shape=(dataset.getDocLen(), activate),
    )
    docDecode = np.memmap(
        Path(saveBase, "docDecode.bin"),
        dtype=np.float32,
        mode="w+",
        shape=(dataset.getDocLen(), features),
    )
    qryLatentIndex = np.memmap(
        Path(saveBase, "qryLatentIndex.bin"),
        dtype=np.int32,
        mode="w+",
        shape=(dataset.getQryLen("Validate"), activate),
    )
    qryLatentValue = np.memmap(
        Path(saveBase, "qryLatentValue.bin"),
        dtype=np.float32,
        mode="w+",
        shape=(dataset.getQryLen("Validate"), activate),
    )
    qryDecode = np.memmap(
        Path(saveBase, "qryDecode.bin"),
        dtype=np.float32,
        mode="w+",
        shape=(dataset.getQryLen("Validate"), features),
    )

    # compute document and query features
    with Progress(console=console) as p:
        t = p.add_task("Fowarding Docs...", total=dataset.getDocLen())
        iterator = dataset.docEmbIter(embedding, 8192, 8, False)
        for i, batch in enumerate(iterator):
            latent, decode = model.forward(batch.cuda(), activate)
            pack = torch.topk(latent, activate)
            index = pack.indices.cpu().numpy()
            value = pack.values.cpu().numpy()
            docLatentIndex[i * 8192 : (i + 1) * 8192] = index
            docLatentValue[i * 8192 : (i + 1) * 8192] = value
            docDecode[i * 8192 : (i + 1) * 8192] = decode.cpu().numpy()
            p.advance(t, batch.size(0))
        p.remove_task(t)
        t = p.add_task("Fowarding Qrys...", total=dataset.getQryLen("Validate"))
        iterator = dataset.qryEmbIter(embedding, "Validate", 8192, 4, False)
        for i, batch in enumerate(iterator):
            latent, decode = model.forward(batch.cuda(), activate)
            pack = torch.topk(latent, activate)
            index = pack.indices.cpu().numpy()
            value = pack.values.cpu().numpy()
            qryLatentIndex[i * 8192 : (i + 1) * 8192] = index
            qryLatentValue[i * 8192 : (i + 1) * 8192] = value
            qryDecode[i * 8192 : (i + 1) * 8192] = decode.cpu().numpy()
            p.advance(t, batch.size(0))
        p.remove_task(t)
