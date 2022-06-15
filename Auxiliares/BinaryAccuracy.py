import typing
import torch


class BinaryAccuracy:
    def __init__(
        self,
        logits: bool = True,
        reduction: typing.Callable[
            [
                torch.Tensor,
            ],
            torch.Tensor,
        ] = torch.mean,
    ):
        self.logits = logits
        if logits:
            self.threshold = 0
        else:
            self.threshold = 0.5

        self.reduction = reduction

    def __call__(self, y_pred, y_true):
        return self.reduction(((y_pred > self.threshold) == y_true.bool()).float())