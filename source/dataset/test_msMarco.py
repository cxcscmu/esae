from source.dataset.msMarco import MsMarcoDataset
from source.embedding.bgeBase import BgeBaseEmbedding


def test_didIter():
    """
    Test didIter method.
    """
    dataset = MsMarcoDataset()
    ids = next(dataset.didIter(8))
    assert isinstance(ids, list) and len(ids) == 8
    assert all(isinstance(i, int) for i in ids)


def test_docIter():
    """
    Test docIter method.
    """
    dataset = MsMarcoDataset()
    docs = next(dataset.docIter(8))
    assert isinstance(docs, list) and len(docs) == 8
    assert all(isinstance(d, str) for d in docs)


def test_docEmbIter():
    """
    Test docEmbIter method.
    """
    dataset = MsMarcoDataset()
    embeddings = next(dataset.docEmbIter(BgeBaseEmbedding, 8, 0, False))
    assert embeddings.shape == (8, BgeBaseEmbedding.size)


def test_getDocLen():
    """
    Test getDocLen method.
    """
    dataset = MsMarcoDataset()
    docLen = dataset.getDocLen()
    assert isinstance(docLen, int)
    assert docLen == 8841823


def test_qidIter():
    """
    Test qidIter method.
    """
    dataset = MsMarcoDataset()
    qids = next(dataset.qidIter("Train", 8))
    assert isinstance(qids, list) and len(qids) == 8
    assert all(isinstance(q, int) for q in qids)
    qids = next(dataset.qidIter("Validate", 8))
    assert isinstance(qids, list) and len(qids) == 8
    assert all(isinstance(q, int) for q in qids)


def test_qryIter():
    """
    Test qryIter method.
    """
    dataset = MsMarcoDataset()
    qrys = next(dataset.qryIter("Train", 8))
    assert isinstance(qrys, list) and len(qrys) == 8
    assert all(isinstance(q, str) for q in qrys)
    qrys = next(dataset.qryIter("Validate", 8))
    assert isinstance(qrys, list) and len(qrys) == 8
    assert all(isinstance(q, str) for q in qrys)


def test_qryEmbIter():
    """
    Test qryEmbIter method.
    """
    dataset = MsMarcoDataset()
    embeddings = next(dataset.qryEmbIter(BgeBaseEmbedding, "Train", 8, 0, False))
    assert embeddings.shape == (8, BgeBaseEmbedding.size)
    embeddings = next(dataset.qryEmbIter(BgeBaseEmbedding, "Validate", 8, 0, False))
    assert embeddings.shape == (8, BgeBaseEmbedding.size)


def test_getQryLen():
    """
    Test getQryLen method.
    """
    dataset = MsMarcoDataset()
    qryLen = dataset.getQryLen("Train")
    assert isinstance(qryLen, int)
    assert qryLen == 808731
    qryLen = dataset.getQryLen("Validate")
    assert isinstance(qryLen, int)
    assert qryLen == 101093


def test_mixEmbIter():
    """
    Test mixEmbIter method.
    """
    dataset = MsMarcoDataset()
    qry, docs = next(dataset.mixEmbIter(BgeBaseEmbedding, "Train", 32, 8, 0, False))
    assert qry.shape == (8, BgeBaseEmbedding.size)
    assert docs.shape == (8, 32, BgeBaseEmbedding.size)
    qry, docs = next(dataset.mixEmbIter(BgeBaseEmbedding, "Validate", 32, 8, 0, False))
    assert qry.shape == (8, BgeBaseEmbedding.size)
    assert docs.shape == (8, 32, BgeBaseEmbedding.size)


def test_getMixLen():
    """
    Test getMixLen method.
    """
    dataset = MsMarcoDataset()
    mixLen = dataset.getMixLen("Train")
    assert isinstance(mixLen, int)
    assert mixLen == 808731
    mixLen = dataset.getMixLen("Validate")
    assert isinstance(mixLen, int)
    assert mixLen == 101093
