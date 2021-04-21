from torchmetrics import Metric
import math
import torch
class CollapseLevel(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.w=0.9
        self.add_state("avg_output_std", default=torch.tensor(0), dist_reduce_fx="sum")
        self.out_dim=768
        # self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, y0: torch.Tensor):
        # calculate the per-dimension standard deviation of the outputs
        # we can use this later to check whether the embeddings are collapsing
        output = y0
        output = output.detach()
        output = torch.nn.functional.normalize(output, dim=1)

        output_std = torch.std(output, 0)
        output_std = output_std.mean()
        # use moving averages to track the loss and standard deviation
        self.avg_output_std = self.w * self.avg_output_std + (1 - self.w) * output_std.item()


    def compute(self):
        # the level of collapse is large if the standard deviation of the l2
        # normalized output is much smaller than 1 / sqrt(dim)
        collapse_level = max(0., 1 - math.sqrt(self.out_dim) * self.avg_output_std)
        
        
        return collapse_level.float() 