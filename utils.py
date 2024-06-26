from collections import defaultdict
import torch
from enum import Enum

class ClassificationType(Enum):
    SINGLELABEL = 0
    SECTION_MULTICLASS = 1
    MULTILABEL = 2
    MULTILABEL_TECHNICAL = 3
    MULTILABEL_PATHLOGICAL = 4
    MULTICLASS_TECHNICAL = 5

# https://albumentations.ai/docs/examples/pytorch_classification/
class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"], float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )