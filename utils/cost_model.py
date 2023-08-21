from typing import List

from utils.ir import Graph, Node
from .ios_runtime import graph_latency, stage_latency


class CostModel:
    """
    Cost model is used to measure the latency of a stage, block, and computation graph. Cost model is used to guide the
    optimization of IOS.
    """
    def get_stage_latency(self, stage: List[List[Node]], batch_size, warmup, number, repeat, profile_stage=False, ngpu = 2) -> List[float]:
        raise NotImplementedError


    def get_graph_latency(self, graph: Graph, batch_size, warmup, number, repeat, profile_stage=False, ngpu = 2) -> List[float]:
        raise NotImplementedError




class IOSCostModel(CostModel):
    """
    IOS runtime cost model, which measure the latency by directly executing the stage on the hardware.
    """
    def get_graph_latency(self, graph: Graph, batch_size, warmup, number, repeat, profile_stage=False, ngpu =2):
        return graph_latency(graph, batch_size, warmup, number, repeat, profile_stage, ngpu)



    def get_stage_latency(self, stage: List[List[Node]], batch_size, warmup, number, repeat, profile_stage=False, ngpu =2):
        return stage_latency(stage, batch_size, warmup, number, repeat, profile_stage, ngpu)



