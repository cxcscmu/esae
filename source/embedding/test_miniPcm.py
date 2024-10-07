from source.embedding.miniPcm import MiniPcmEmbedding


def test_forward():
    """
    Test forward method.
    """
    embedding = MiniPcmEmbedding()
    passages = ["Hello, world!", "Goodbye, world!"]
    results = embedding.forward(passages)
    assert results.shape == (len(passages), MiniPcmEmbedding.size)
