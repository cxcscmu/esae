from source.dataset.beir import BeirDataset
from source.embedding.bgeBase import BgeBaseEmbedding


def test_didIter():
    """
    Test didIter method.
    """
    dataset = BeirDataset()
    ids = next(dataset.didIter(8))
    assert isinstance(ids, list) and len(ids) == 8
    assert all(isinstance(i, str) for i in ids)


def test_docIter():
    """
    Test docIter method.
    """
    dataset = BeirDataset()
    docs = next(dataset.docIter(8))
    assert isinstance(docs, list) and len(docs) == 8
    assert all(isinstance(d, str) for d in docs)


def test_docEmbIter():
    """
    Test docEmbIter method.
    """
    dataset = BeirDataset()
    embeddings = next(dataset.docEmbIter(BgeBaseEmbedding, 8, 0, False))
    assert embeddings.shape == (8, BgeBaseEmbedding.size)


def test_getDocLen():
    """
    Test getDocLen method.
    """
    dataset = BeirDataset()
    docLen = dataset.getDocLen()
    assert isinstance(docLen, int)
    assert docLen == 7488722


def test_qidIter():
    """
    Test qidIter method.
    """
    dataset = BeirDataset()
    qids = next(dataset.qidIter("Validate", 8))
    assert isinstance(qids, list) and len(qids) == 8
    assert all(isinstance(q, str) for q in qids)


def test_qryIter():
    """
    Test qryIter method.
    """
    dataset = BeirDataset()
    qrys = next(dataset.qryIter("Validate", 8))
    assert isinstance(qrys, list) and len(qrys) == 8
    assert all(isinstance(q, str) for q in qrys)


def test_qryEmbIter():
    """
    Test qryEmbIter method.
    """
    dataset = BeirDataset()
    embeddings = next(dataset.qryEmbIter(BgeBaseEmbedding, "Validate", 8, 0, False))
    assert embeddings.shape == (8, BgeBaseEmbedding.size)


def test_getQryLen():
    """
    Test getQryLen method.
    """
    dataset = BeirDataset()
    qryLen = dataset.getQryLen("Validate")
    assert isinstance(qryLen, int)
    assert qryLen == 3902


def test_mixEmbIter():
    """
    Test mixEmbIter method.
    """
    dataset = BeirDataset()
    qry, docs = next(dataset.mixEmbIter(BgeBaseEmbedding, "Validate", 32, 8, 0, False))
    assert qry.shape == (8, BgeBaseEmbedding.size)
    assert docs.shape == (8, 32, BgeBaseEmbedding.size)


def test_getMixLen():
    """
    Test getMixLen method.
    """
    dataset = BeirDataset()
    mixLen = dataset.getMixLen("Validate")
    assert isinstance(mixLen, int)
    assert mixLen == 3902
