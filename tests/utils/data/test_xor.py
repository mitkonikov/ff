from fflib.utils.data.xor import FFXOR


def test_ffxor() -> None:
    processor = FFXOR(64, 128)
    loader = processor.get_train_loader()
    assert len(loader) == 2  # two batches
    batch = next(iter(loader))
    x, y = batch
    for i in range(32):
        a = int(x[i][0].item())
        b = int(x[i][1].item())
        c = int(y[i].item())
        assert a ^ b == c
