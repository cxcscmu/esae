from source.embedding.bgeBase import BgeBaseEmbedding


def test_forward():
    """
    Test forward method.
    """
    embedding = BgeBaseEmbedding()
    passages = ["Hello, world!", "Goodbye, world!"]
    results = embedding.forward(passages)
    assert results.shape == (len(passages), BgeBaseEmbedding.size)
