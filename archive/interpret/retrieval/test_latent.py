import io
import torch
from source.interpret.retrieval.latent import searchCore
from source.interpret.retrieval.latent import search


def test_searchCore():
    """
    Test `searchCore` function.
    """

    # define queries
    qrys = torch.tensor(
        [
            [0, 0, 1],
            [0, 1, 0],
        ],
        dtype=torch.float32,
    )
    qrys = qrys.to_sparse_coo()

    # define document ids
    dids = torch.tensor([0, 1, 2, 3])

    # define documents
    docs = torch.tensor(
        [
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, 0],
        ],
        dtype=torch.float32,
        device="cuda:0",
    )
    docs = docs.T.to_sparse_coo()

    # execute search
    K = 2
    results = searchCore(qrys, dids, docs, K)

    # validate results
    assert len(results) == qrys.size(0)
    assert all(len(result) <= K for result in results)
    assert results[0][0] == (1.0, 2)
    assert results[1][0] == (1.0, 0)


def test_search():
    """
    Test `search` function.
    """

    # define query ids
    qidBatch = []
    qidBatch0 = torch.tensor([0])
    qidBatch.append(qidBatch0)
    qidBatch1 = torch.tensor([1])
    qidBatch.append(qidBatch1)

    # define queries
    qryBatch = []
    qryBatch0 = torch.tensor([[0, 0, 1]], dtype=torch.float32)
    qryBatch0 = qryBatch0.to_sparse_coo()
    qryBatch.append(qryBatch0)
    qryBatch1 = torch.tensor([[0, 1, 0]], dtype=torch.float32)
    qryBatch1 = qryBatch1.to_sparse_coo()
    qryBatch.append(qryBatch1)

    # define document ids
    didShard = []
    didShard0 = torch.tensor([0, 1])
    didShard.append(didShard0)
    didShard1 = torch.tensor([2, 3])
    didShard.append(didShard1)

    # define documents
    docShard = []
    docShard0 = torch.tensor(
        [
            [0, 1, 0],
            [1, 0, 0],
        ],
        dtype=torch.float32,
        device="cuda:0",
    )
    docShard0 = docShard0.T.to_sparse_coo()
    docShard.append(docShard0)
    docShard1 = torch.tensor(
        [
            [0, 0, 1],
            [0, 0, 0],
        ],
        dtype=torch.float32,
        device="cuda:0",
    )
    docShard1 = docShard1.T.to_sparse_coo()
    docShard.append(docShard1)

    # execute search
    K = 3
    qresFile = io.StringIO()
    search(qidBatch, qryBatch, didShard, docShard, qresFile, K)

    # validate results
    qresFile.seek(0)
    results = qresFile.read().split("\n")
    assert results[0] == "0\tQ0\t2\t0\t1.0\tLatent"
    assert results[1] == "0\tQ0\t1\t0\t0.0\tLatent"
    assert results[2] == "0\tQ0\t3\t0\t0.0\tLatent"
    assert results[3] == "1\tQ0\t0\t0\t1.0\tLatent"
    assert results[4] == "1\tQ0\t2\t0\t0.0\tLatent"
    assert results[5] == "1\tQ0\t3\t0\t0.0\tLatent"


if __name__ == "__main__":
    test_searchCore()
    test_search()
