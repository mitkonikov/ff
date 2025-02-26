import pytest
import torch

from fflib.probes.one_hot import TryAllClasses


# Mock callback function
def mock_callback(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Simulates a simple deterministic FF neural network goodness function.

    Args:
        x (torch.Tensor):
            Input Features.
            Tensor with shape (batch_size, feature_dim)
        y (torch.Tensor):
            One Hot Encoding of the output we want to test.
            Tensor with shape (batch_size, output_classes)

    Returns:
        torch.Tensor: Tensor with shape (batch_size, 1) annotating the goodness of this guess.
    """

    assert x.shape[0] == y.shape[0]

    # Compute a mock result
    result = torch.mul(torch.sum(y, 1), torch.sum(y, 1))
    result = result.unsqueeze(1)
    assert result.shape == (x.shape[0], 1)
    return result


@pytest.mark.parametrize(
    "batch_size, output_classes",
    [
        (5, 3),  # Small batch, 3 classes
        (10, 4),  # Medium batch, 4 classes
        (1, 2),  # Edge case: batch size of 1
        (20, 5),  # Larger batch, 5 classes
    ],
)
def test_predict(batch_size: int, output_classes: int) -> None:
    torch.manual_seed(42)

    # Generate random input tensor (batch_size, feature_dim)
    feature_dim = 10
    x = torch.randn(batch_size, feature_dim)

    probe = TryAllClasses(callback=mock_callback, output_classes=output_classes)

    # Get predictions
    predictions = probe.predict(x)

    # Assertions
    assert predictions.shape == (batch_size,), "Output should have shape (batch_size,)"
    assert torch.all(
        (predictions >= 0) & (predictions < output_classes)
    ), "Predicted labels must be within range [0, output_classes-1]"


def test_callback_is_called() -> None:
    """
    Ensures the callback function is properly called with correct input shapes.
    """
    callback_called = False

    def tracking_callback(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        nonlocal callback_called
        callback_called = True
        assert x.shape[0] == y.shape[0], "Batch sizes should match"
        assert y.shape[1] > 1, "y should have one-hot encoded classes"
        return torch.randn(x.shape[0], 1)

    probe = TryAllClasses(callback=tracking_callback, output_classes=3)
    x = torch.randn(5, 10)

    _ = probe.predict(x)
    assert callback_called, "Callback function should be called during prediction"


def test_output_consistency() -> None:
    """
    Checks if the function produces consistent output given a fixed seed.
    """
    torch.manual_seed(123)
    x = torch.randn(8, 10)

    probe = TryAllClasses(callback=mock_callback, output_classes=4)
    pred1 = probe.predict(x)

    torch.manual_seed(123)  # Reset seed
    x = torch.randn(8, 10)
    pred2 = probe.predict(x)

    assert torch.equal(pred1, pred2), "Predictions should be consistent with the same input"


@pytest.mark.parametrize(
    "batch_size, output_classes",
    [
        (1, 3),
        (50, 4),
    ],
)
def test_predict_correct_label(batch_size: int, output_classes: int) -> None:
    feature_dim = 10
    x = torch.randn(batch_size, feature_dim)

    truth = torch.randint(0, output_classes, (batch_size, 1))

    def mock_callback2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Mock callback that returns a high score for the correct label (1.0) and 0.0 for others.
        It simulates a model where the correct class gets the highest score.
        """
        scores = torch.zeros((x.shape[0], 1))

        # Compare the predicted label (y) with the truth (truth) and assign score 1 if they match
        # Assumes y contains the predicted labels as a tensor of shape (batch_size, 1)
        # We compare the predicted label with the truth, and assign 1.0 if they match, else 0.0
        scores[:, 0] = (
            truth.squeeze(1) == y.argmax(1)
        ).float()  # convert boolean to float (True -> 1.0, False -> 0.0)
        return scores

    probe = TryAllClasses(callback=mock_callback2, output_classes=output_classes)
    predictions = probe.predict(x)

    print("Truth: ", truth[:20,])
    print("Predictions: ", predictions[:20,])

    # Ensure the predictions are within the valid range of [0, output_classes-1]
    assert torch.all(
        (predictions >= 0) & (predictions < output_classes)
    ), "Predicted labels must be within range [0, output_classes-1]"

    # Check that the predicted label is the correct one
    assert torch.equal(predictions, truth.squeeze(1)), "Predictions should match the correct labels"
