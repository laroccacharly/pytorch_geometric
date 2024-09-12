import torch.nn.functional as F

class TargetsTrait:
    def __init__(self):
        self.num_classes = None

    def get_targets_from_batch_as_count(self, batch):
        return batch.y.view(-1, self.num_classes).sum(dim=1).long() # shape (batch_size, 1)

    def get_targets_from_batch_as_one_hot(self, batch):
        counts = self.get_targets_from_batch_as_count(batch)
        one_hot = F.one_hot(counts, num_classes=self.num_classes)
        return one_hot # shape (batch_size, num_classes)