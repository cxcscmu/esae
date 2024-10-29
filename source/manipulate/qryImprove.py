import os
import faiss
import argparse
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from io import TextIOWrapper
from pathlib import Path
from rich.progress import Progress
from numpy.typing import NDArray
from typing import Tuple, Dict, List
from torch import Tensor
from source import workspace, console
from source.interface import SAE
from source.dataset.msMarco import MsMarcoDataset


class Model(nn.Module, SAE):

    def __init__(self, features: int, expanded: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(features, features * expanded)
        self.decoder = nn.Linear(features * expanded, features)

    def forwardEncoder(self, x: Tensor, activate: int) -> Tensor:
        xbar = x - self.decoder.bias
        a = self.encoder.forward(xbar)
        pack = torch.topk(a, activate)
        f = torch.zeros_like(a)
        f.scatter_(1, pack.indices, F.relu(pack.values))
        return f

    def forwardDecoder(self, f: Tensor) -> Tensor:
        xhat = self.decoder.forward(f)
        return xhat

    def forward(self, x: Tensor, activate: int) -> Tuple[Tensor, Tensor]:
        f = self.forwardEncoder(x, activate)
        xhat = self.forwardDecoder(f)
        return f, xhat


dataset = MsMarcoDataset()
features, expanded, activate = 768, 256, 128
snapshotBase = Path(workspace, "model/kld_x256_k128/snapshot")
computedBase = Path(workspace, f"model/kld_x256_k128/computed/{dataset.name}")
model = Model(features, expanded)
state = torch.load(max(snapshotBase.glob("*.pth")), map_location="cpu")
model.load_state_dict(state["model"])
model.eval().cuda()

parser = argparse.ArgumentParser()
parser.add_argument("limit", type=int)
parser.add_argument("delta", type=float)
args = parser.parse_args()
assert isinstance(args.limit, int)
assert 0 <= args.limit <= activate
limit: int = args.limit
assert isinstance(args.delta, float)
delta: float = args.delta


def newIndex(vectors: NDArray[np.float32]) -> faiss.GpuIndexFlatIP:
    """
    Create a new GPU index from the given vectors.
    """
    assert vectors.ndim == 2
    assert vectors.dtype == np.float32

    cpuIndex = faiss.IndexFlatIP(vectors.shape[1])
    cpuIndex.add(vectors)
    gpuIndex = faiss.index_cpu_to_all_gpus(cpuIndex)
    return gpuIndex


def newRecon(
    qryIndex: NDArray[np.int32],
    qryValue: NDArray[np.float32],
    docIndexBulk: List[NDArray[np.int32]],
    docValueBulk: List[NDArray[np.float32]],
) -> NDArray[np.float32]:
    """
    Manipulate the reconstructed query embedding by improving the relevant latent
    dimensions using the document features.
    """
    assert qryIndex.ndim == 1
    assert qryValue.ndim == 1
    for docIndex, docValue in zip(docIndexBulk, docValueBulk):
        assert docIndex.ndim == 1
        assert docValue.ndim == 1

    qryLatent = torch.zeros(features * expanded, dtype=torch.float32).cuda()
    qryIndex = torch.from_numpy(qryIndex).to(torch.int64).cuda()
    qryValue = torch.from_numpy(qryValue).to(torch.float32).cuda()
    qryLatent.scatter_(0, qryIndex, qryValue)

    for docIndex, docValue in zip(docIndexBulk, docValueBulk):
        mask = docValue != 0
        docIndex = docIndex[mask]
        qryLatent[docIndex[:limit]] += delta

    with torch.no_grad():
        qryRecon = model.forwardDecoder(qryLatent.unsqueeze(0))
        qryRecon = qryRecon.squeeze(0).cpu().numpy()
    return qryRecon


def retrieve(
    gpuIndex: faiss.GpuIndexFlatIP,
    qids: NDArray[np.int32],
    qrys: NDArray[np.float32],
    didTable: NDArray[np.int32],
    qresFile: TextIOWrapper,
):
    """
    Perform dense retrieval on the manipulated document embedding. This index
    shall be different for each query, whose relevant documents are bumped.
    """
    D, I = gpuIndex.search(qrys, 100)
    for qid, sims, dnos in zip(qids, D, I):
        for s, d in zip(sims, dnos):
            did = didTable[d]
            qresFile.write(f"{qid}\tQ0\t{did}\t0\t{s:.6f}\tR\n")
    qresFile.flush()
    os.fsync(qresFile.fileno())


def evaluate(
    qrelPath: Path,
    qresPath: Path,
    evalFile: TextIOWrapper,
):
    """
    Evaluate the retrieval performance using the TREC evaluation.
    """
    args = []
    args.append("trec_eval")
    args.append(["-m", "all_trec"])
    args.append(qrelPath)
    args.append(qresPath)
    with console.status("Evaluating"):
        subprocess.run(args, stdout=evalFile, check=True)


docLen, qryLen = dataset.getDocLen(), dataset.getQryLen("Validate")
docLatentIndex = np.memmap(
    Path(computedBase, "docLatentIndex.bin"),
    dtype=np.int32,
    shape=(docLen, activate),
    mode="r",
)
docLatentValue = np.memmap(
    Path(computedBase, "docLatentValue.bin"),
    dtype=np.float32,
    shape=(docLen, activate),
    mode="r",
)
qryLatentIndex = np.memmap(
    Path(computedBase, "qryLatentIndex.bin"),
    dtype=np.int32,
    shape=(qryLen, activate),
    mode="r",
)
qryLatentValue = np.memmap(
    Path(computedBase, "qryLatentValue.bin"),
    dtype=np.float32,
    shape=(qryLen, activate),
    mode="r",
)
docDecode = np.memmap(
    Path(computedBase, "docDecode.bin"),
    dtype=np.float32,
    shape=(docLen, features),
    mode="r",
)
qryDecode = np.memmap(
    Path(computedBase, "qryDecode.bin"),
    dtype=np.float32,
    shape=(qryLen, features),
    mode="r",
)

basePath = Path(workspace, "manipulate/improve")
basePath.mkdir(parents=True, exist_ok=True)

qidTable, qidLookup, i = np.zeros(qryLen, dtype=np.int32), {}, 0
for batch in dataset.qidIter("Validate", 4096):
    for d in batch:
        qidTable[i] = d
        qidLookup[d] = i
        i += 1
didTable, didLookup, i = np.zeros(docLen, dtype=np.int32), {}, 0
for batch in dataset.didIter(4096):
    for d in batch:
        didTable[i] = d
        didLookup[d] = i
        i += 1
qrelDict: Dict[int, List[int]] = {}
qrelPath = dataset.getQryRel("Validate")
with qrelPath.open("r") as qrelFile:
    for line in qrelFile:
        qid, _, did, _ = line.split()
        qof, dof = qidLookup[int(qid)], didLookup[int(did)]
        qrelDict.setdefault(qof, []).append(dof)

gpuIndex = newIndex(docDecode)
qresPath = Path(basePath, f"qry_{limit:03d}_{delta:.3f}.qres")
evalPath = Path(basePath, f"qry_{limit:03d}_{delta:.3f}.eval")

with Progress(console=console) as p:
    m = 1000  # take a subset
    t = p.add_task("Improving...", total=m)

    with qresPath.open("w") as qresFile:
        for qof, rel in qrelDict.items():
            if (m := m - 1) < 0:
                break
            qryIndex, qryValue = qryLatentIndex[qof], qryLatentValue[qof]
            docIndexBulk, docValueBulk = [], []
            for dof in rel:
                docIndexBulk.append(docLatentIndex[dof])
                docValueBulk.append(docLatentValue[dof])
            qryRecon = newRecon(qryIndex, qryValue, docIndexBulk, docValueBulk)
            qids = np.array([qidTable[qof]], dtype=np.int32)
            qrys = qryRecon.reshape(1, features).astype(np.float32)
            retrieve(gpuIndex, qids, qrys, didTable, qresFile)
            p.update(t, advance=1)
