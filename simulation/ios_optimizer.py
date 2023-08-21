from typing import List, Optional, Dict, Tuple, Set, FrozenSet, List
import functools
import operator
import numpy as np
import logging
import itertools
import random
#from tqdm import tqdm
#from ios.ir import Graph, Block, Conv, Value, Pool, Placeholder, Node, Identity, Sequential
#from ios.cost_model import CostModel, IOSCostModel, RandomCostModel
#from ios.utils import iter_subset
import networkx as nx

logging.disable(logging.WARNING)

# The representation of computation graph by index
IGraph = Dict[int, List[int]]



def optimize(modelgraph,
             batch_size=1,
             warmup=2, number=6, repeat=6,
             max_num_groups=2, max_part_size=10, max_group_size=3):


    all_nodes = list(modelgraph.nodes)


    nid: Dict[int, int] = {node: i for i, node in enumerate(all_nodes)}
    idn: Dict[int, int] = {i: node for i, node in enumerate(all_nodes)}


    node_parts = []
    for idx_part in range((len(all_nodes) + max_part_size - 1) // max_part_size):
        begin = idx_part * max_part_size
        end = min((idx_part + 1) * max_part_size, len(all_nodes))
        node_parts.append([all_nodes[i] for i in range(begin, end)])


    stage_list = []

    for part_index, npart in enumerate(node_parts):
        ipart = [nid[nd] for nd in npart]

        dp: Dict[int, float] = {}
        ep: Dict[int, Tuple[List[int], str]] = {}
        #merge_latency: Dict[int, float] = {}
        parallel_latency: Dict[int, float] = {}
        part_graph = build_graph(modelgraph, npart, nid)
        chains = graph_chain_decomposition(part_graph)

        max_num_endings = functools.reduce(operator.mul, [len(chain) + 1 for chain in chains])




        ustate = sum(1 << i for i in ipart)
        dop(modelgraph, ustate, chains, idn, nid, dp, ep, max_group_size,
            max_num_groups, parallel_latency, batch_size, warmup, number, repeat)
        stage_list.extend(get_stage_list(ep, ustate))


    graph_latency = get_ios_graph_latency(modelgraph, idn, stage_list)
    return graph_latency


def get_ios_graph_latency(modelgraph, idn, stage_list):
    graph_latency = 0
    for stage in stage_list:
        graph_latency += ios_get_latency_of_stage(modelgraph, idn, stage)

    return graph_latency








def count_bits(s):
    """
    Count the number of bit 1 in the binary representation of non-negative number s
    """
    cnt = 0
    while s > 0:
        s -= s & (-s)  # (s & (-s)) = 2^k, where k is the index of least significant bit 1 of s.
        cnt += 1
    return cnt


def state2iset(s):
    """
    Return a set that contains the index of each 1 in the binary representation of non-negative number s
    """
    iset = []
    i = 0
    while s > 0:
        if (s & 1) != 0:
            iset.append(i)
        s = s >> 1
        i += 1
    return iset



def state2nset(s, idn):
    return [idn[i] for i in state2iset(s)]





def check_parallel(ss, successor_dict, max_num_streams):
    """
    Check whether a set of operators is valid to parallelly execute
    """
    iset = state2iset(ss)
    if len(iset) > max_num_streams:  # stream number requirement
        return False
    suc_list = [successor_dict[u] for u in iset]
    num = len(iset)
    for i in range(num):
        for j in range(i + 1, num):
            if not suc_list[i].isdisjoint(suc_list[j]):  # successors keep disjoint
                return False
    return True





def build_graph(modelgraph, all_nodes: List[int], nid):
    """
    Build a graph of given operators. The global index in nid is used to represent the operator in the graph.
    """
    g: Dict[int, List[int]] = {}
    for nu in all_nodes:
        iu = nid[nu]
        g[iu] = []
        for successor in modelgraph.successors(nu):
            #nv = use[0]
            if successor in all_nodes:
                iv = nid[successor]
                g[iu].append(iv)
        g[iu] = list(set(g[iu]))  # dump duplicate targets
    return g


def topological_order(graph: IGraph) -> List[int]:
    """
    Generate a topological order for given graph
    """
    in_degree = {u: 0 for u in graph.keys()}
    for u in graph.keys():
        for v in graph[u]:
            in_degree[v] += 1
    qu = [u for u in graph.keys() if in_degree[u] == 0]
    order = []
    while len(qu) > 0:
        u = qu.pop()
        order.append(u)
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                qu.append(v)
    assert len(order) == len(graph)  # no loop
    return order


def graph_transitive_closure(graph: IGraph, include_self=False) -> Dict[int, Set[int]]:
    """
    Generate the transitive closure of a computation graph.
    """
    tc: Dict[int, Set[int]] = {u: set() for u in graph.keys()}
    for u in reversed(topological_order(graph)):
        if include_self:
            tc[u].update([u])
        for v in graph[u]:
            tc[u].update(tc[v])
            tc[u].update([v])
    return tc


def transitive_closure_to_graph(tc):
    """
    Convert a transitive closure to IGraph format.
    """
    graph: IGraph = {}
    for u in tc:
        graph[u] = list(tc[u])
    return graph


def longest_chain(graph: IGraph) -> List[int]:
    """
    Return the longest chain in the directed acyclic graph (DAG).
    """
    depth: Dict[int, int] = {u: 1 for u in graph}
    comes: Dict[int, int] = {u: None for u in graph}
    for u in reversed(topological_order(graph)):
        for v in graph[u]:
            if depth[u] < depth[v] + 1:
                depth[u] = depth[v] + 1
                comes[u] = v
    u = max(depth.keys(), key=lambda u: depth[u])
    chain = []
    while u is not None:
        chain.append(u)
        u = comes[u]
    return chain


def sub_graph(graph: IGraph, uset) -> IGraph:
    """
    Generate the sub-graph derived from a subset of operators in the graph.
    """
    new_graph = {}
    for u in graph:
        if u in uset:
            new_graph[u] = [v for v in graph[u] if v in uset]
    return new_graph


def graph_chain_decomposition(graph: IGraph) -> List[List[int]]:
    """
    Conduct a graph chain decomposition. At each time, split out a longest chain. Repeat this progress until no
    operators are left.
    """
    chains = []
    graph = transitive_closure_to_graph(graph_transitive_closure(graph))

    while len(graph) > 0:
        chain = longest_chain(graph)
        chains.append(chain)
        graph = sub_graph(graph, set(graph.keys()) - set(chain))
    return chains


def ending_iterator(modelgraph,
        s: int,
        chains: List[List[int]],
        nid: Dict[int, int],
        idn: Dict[int, int],
        max_group_size: int,
        max_num_groups: int):
    """
    Enumerate endings of a set of operators. An ending of operator set S is defined as a subset S' of S, such that all
    edges between S-S' and S' are from S-S' to S'.
    """
    iset = state2iset(s)
    begins = []
    ends = []
    lengths = []

    # get the range for each chain
    for ichain, chain in enumerate(chains):
        end = 0
        for iu, u in enumerate(chain):
            if u in iset:
                end = iu + 1
            else:
                break
        begin = max(0, end - max_group_size)
        begins.append(begin)
        ends.append(end)
        lengths.append(end - begin)
    bases = [length + 1 for length in lengths]
    strides = list(itertools.accumulate(bases, operator.mul))
    total = strides[-1]

    # build sub graph and transitive clousure
    tc = graph_transitive_closure(build_graph(modelgraph, state2nset(s, idn), nid), include_self=True)

    # enuermate ending
    for w in range(total):
        end_list = []
        for i, chain in enumerate(chains):
            div = strides[i - 1] if i >= 1 else 1
            idx = (w % strides[i]) // div
            if idx == lengths[i]:  # empty
                continue
            end_list.append(chain[begins[i] + idx])
        if len(end_list) == 0:
            continue
        if len(end_list) > max_num_groups:
            continue
        isdisjoint = True
        for i in range(len(end_list)):
            for j in range(i + 1, len(end_list)):
                if not tc[end_list[i]].isdisjoint(tc[end_list[j]]):
                    isdisjoint = False
                    break
            if not isdisjoint:
                break
        if isdisjoint:
            groups = [sorted(tc[u]) for u in end_list]
            if any(len(group) > max_group_size for group in groups):
                continue
            yield groups


def dop(modelgraph, s: int,
        chains, idn, nid, dp, ep, max_group_size, max_num_groups,
        parallel_latency, batch_size, warmup, number, repeat) -> float:
    """
    The main dynamic programming progress.
    """
    if s == 0:
        return 0.0
    if s in dp:
        return dp[s]


    iset = state2iset(s)
    successor_dict: Dict[int, Set] = {u: set() for u in iset}
    for u in reversed(iset):
        successors = successor_dict[u]
        successors.add(u)
        for successor in modelgraph.successors(idn[u]):
            if successor in nid and nid[successor] in iset:
                successors.update(successor_dict[nid[successor]])
        """
        for use in idn[u].uses:
            if use[0] in nid and nid[use[0]] in iset:
                successors.update(successor_dict[nid[use[0]]])
        """

    dpv = 1e19
    s1 = sum(1 << u for u in iset if len(successor_dict[u]) == 1)

    # # the follow method is used previously, which is inefficient and is replaced by the second implementation.
    # s2 = sum(1 << u for u in iset if len(successor_dict[u]) <= max_group_size)
    # if "parallel" in opt_type:
    #     for ss in iter_subset(s2):
    #         if check_parallel(ss, successor_dict, max_num_groups):
    #             stage = [list(sorted(list(successor_dict[u]))) for u in state2iset(ss)], 'parallel'
    #             consumed = sum(1 << u for u in itertools.chain(*stage[0]))
    #             val1 = dop(s - consumed, block, chains, on_debug, debug_dp_info, idn, nid, dp, ep, opt_type,
    #                        max_group_size, max_num_groups, merge_latency, parallel_latency, cost_model, batch_size,
    #                        warmup, number, repeat, bar_state)
    #             val2 = latency(stage, block, merge_latency, parallel_latency, cost_model, idn, nid, batch_size,
    #                            warmup, number, repeat)
    #             val = val1 + val2
    #             if on_debug:
    #                 debug_dp_info['#transitions'][-1] += 1
    #                 debug_dp_info['meta'][-1][s] += debug_dp_info['meta'][-1][s - consumed]
    #                 debug_dp_info['width'][-1] = max(debug_dp_info['width'][-1], len(stage[0]))
    #             if val < dpv:
    #                 dpv = val
    #                 ep[s] = stage

    for groups in ending_iterator(modelgraph, s, chains, nid, idn, max_group_size, max_num_groups):
        stage = groups
        consumed = sum(1 << u for u in itertools.chain(*stage))
        val1 = dop(modelgraph, s - consumed, chains, idn, nid, dp, ep, max_group_size,
                   max_num_groups, parallel_latency, batch_size, warmup, number, repeat)
        val2 = ios_get_latency_of_stage(modelgraph, idn, stage)
        val = val1 + val2

        if val < dpv:
            dpv = val
            ep[s] = stage

    dp[s] = dpv

    return dpv


def ios_get_latency_of_group(modelgraph, idn, group):
    group_latency = 0
    group_utilization = 0
    for item in group:
        group_latency += modelgraph.nodes[idn[item]]['weight']
        group_utilization += modelgraph.nodes[idn[item]]['weight'] * modelgraph.nodes[idn[item]]['utilization']

    group_utilization = group_utilization/group_latency
    return group_latency, group_utilization


def ios_get_latency_of_stage(modelgraph, idn, stage):
    max_bound = 0
    utl_bound = 0
    alpha = .5
    groups_latency = []
    group_latency = {}
    group_utilization = {}
    group_id = 0
    for group in stage:
        group_latency[group_id], group_utilization[group_id] = ios_get_latency_of_group(modelgraph, idn, group)
        group_id += 1
    for id in range(group_id):
        max_bound += group_latency[id]
        utl_bound += group_latency[id] * group_utilization[id]
        groups_latency.append(group_latency[id])

    max_groups_latency = max(groups_latency)
    min_bound = max(utl_bound, max_groups_latency)
    stage_latency = alpha*max_bound + (1-alpha)*min_bound
    return stage_latency


def get_stage_list(ep, s):
    """
    Get the list of stages according to the choices of each state stored in ep.
    """
    stage_list = []
    while s != 0:
        stage = ep[s]
        stage_list.append(stage)
        s = s - sum(1 << u for seq in ep[s] for u in seq)
    stage_list = list(reversed(stage_list))
    return stage_list

