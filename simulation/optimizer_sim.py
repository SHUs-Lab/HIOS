import networkx as nx
import copy
import crpath_sim
from typing import List, Optional, Dict, Tuple, Set, FrozenSet, List
import functools
import random
#from .crpath_sim import *






def construct_graph(modelgraph, gpugraph,longest_path):
    pairs = [longest_path[i: i + 2] for i in range(len(longest_path) - 1)]
    for pair in pairs:
        if gpugraph.has_node(pair[0]) == False:
            gpugraph.add_node(pair[0], weight=modelgraph.nodes[pair[0]]['weight'])
        if gpugraph.has_node(pair[1]) == False:
            gpugraph.add_node(pair[1], weight=modelgraph.nodes[pair[1]]['weight'])

        if gpugraph.has_edge(pair[0], pair[1]) == False:
            gpugraph.add_edge(pair[0], pair[1], weight=modelgraph.edges[pair[0], pair[1]]['weight'])

    remove_path(modelgraph, pairs)


def graph_latency(modelgraph, src_node, dst_node, ngpu):
    latency_graph = 0

    ests, efts, lsts, lfts, slack = [], [], [], [], []
    nx.set_node_attributes(modelgraph, ests, 'est')
    nx.set_node_attributes(modelgraph, efts, 'eft')
    nx.set_node_attributes(modelgraph, ests, 'lst')
    nx.set_node_attributes(modelgraph, ests, 'lft')
    nx.set_node_attributes(modelgraph, ests, 'slack')

    for node in modelgraph.nodes:
        modelgraph.nodes[node]['est'] = 0
        modelgraph.nodes[node]['eft'] = 0
        modelgraph.nodes[node]['lst'] = 0
        modelgraph.nodes[node]['lft'] = 0
        modelgraph.nodes[node]['slack'] = 0

    topological_order = list(nx.topological_sort(modelgraph))
    print(topological_order[-1])
    for vertex in topological_order:
        for neighbor in modelgraph.neighbors(vertex):
            modelgraph.nodes[neighbor]['est'] = max(modelgraph.nodes[neighbor]['est'],
                                                    modelgraph.nodes[vertex]['eft'] +
                                                    modelgraph.edges[vertex, neighbor]['weight'])

            modelgraph.nodes[neighbor]['eft'] = modelgraph.nodes[neighbor]['est'] + modelgraph.nodes[neighbor]['weight']


    tpsort_bylst = sorted(modelgraph.nodes(), key=lambda n: modelgraph.nodes[n]['lst'])
    gpu_latency = [0]*ngpu

    for node in tpsort_bylst:
        predecessors = modelgraph.predecessors(node)

        node_gpu = modelgraph.nodes[node]['gpu']
        max_gpu_latency = gpu_latency[node_gpu]
        for predecessor in predecessors:
            pred_gpu = modelgraph.nodes[predecessor]['gpu']
            pred_gpu_latency = gpu_latency[pred_gpu]
            if pred_gpu != node_gpu:
                tmp_gpu_latency = pred_gpu_latency + modelgraph.edges[predecessor, node]['weight']
            else:
                tmp_gpu_latency = pred_gpu_latency
            if max_gpu_latency < tmp_gpu_latency:
                max_gpu_latency = tmp_gpu_latency


        gpu_latency[node_gpu] = max_gpu_latency + modelgraph.nodes[node]['weight']

    latency_graph = max(gpu_latency)
    return latency_graph



def intra_op_graph_latency(modelgraph,execution_order, src_node, dst_node, ngpu):
    latency_graph = 0


    gpu_latency = [0]*ngpu

    for node in execution_order:
        predecessors = modelgraph.predecessors(node)

        node_gpu = modelgraph.nodes[node]['gpu']
        max_gpu_latency = gpu_latency[node_gpu]
        for predecessor in predecessors:
            pred_gpu = modelgraph.nodes[predecessor]['gpu']
            pred_gpu_latency = gpu_latency[pred_gpu]
            if pred_gpu != node_gpu:
                tmp_gpu_latency = pred_gpu_latency + modelgraph.edges[predecessor, node]['weight']
            else:
                tmp_gpu_latency = pred_gpu_latency
            if max_gpu_latency < tmp_gpu_latency:
                max_gpu_latency = tmp_gpu_latency


        gpu_latency[node_gpu] = max_gpu_latency + modelgraph.nodes[node]['weight']

    latency_graph = max(gpu_latency)
    return latency_graph






def remove_path(modelgraph, longest_path):
    pairs = [longest_path[i: i + 2] for i in range(len(longest_path) - 1)]
    for pair in pairs:
        modelgraph.remove_edge(pair[0], pair[1])

def hios_lp( modelgraph, source, destination, ngpu):
    gpu_graph = {}
    cpgraph_prime = copy.deepcopy(modelgraph)

    while cpgraph_prime.number_of_edges() > 0:
        print(cpgraph_prime.number_of_edges())
        crpath = get_optcrpath(cpgraph_prime)

        for rank in range(ngpu):
            gpu_graph[rank] = copy.deepcopy(modelgraph)
            crpath_sim.setrank_path(gpu_graph[rank], crpath, rank)
            crpath_sim.update_combinedgraph(gpu_graph[rank])
            gpu_graph[rank].graph['latency'] = graph_latency(gpu_graph[rank], source, destination, ngpu)
        opt_rank, opt_latency = -1, 1000000
        for rank in range(ngpu):
            if opt_latency >= gpu_graph[rank].graph['latency']:
                opt_rank = rank
                opt_latency = gpu_graph[rank].graph['latency']
            #elif opt_latency == gpu_graph[rank].graph['latency']:
            #    opt_rank = 0
        crpath_sim.setrank_path(modelgraph, crpath, opt_rank)
        crpath_sim.exclude_pathupdated(cpgraph_prime, crpath)
        for rank in range(ngpu):
            del gpu_graph[rank]



"""
def path_latency(modelgraph, path):
    node_latency = 0
    edge_latency = 0
    for node in path[1:-1]:
        node_latency += modelgraph.nodes[node]['weight']
    pairs = [path[i: i + 2] for i in range(len(path) - 1)]
    for pair in pairs:
        edge_latency += modelgraph.get_edge_data(pair[0], pair[1])['weight']

    return node_latency + edge_latency
"""

def check_validity(modelgraph, path, src_list, dst_list):
    nodes = path.node_list[1:-1]
    if len(nodes) == 0:
        return True
    if (path.node_list[0] not in src_list) or (path.node_list[-1] not in dst_list):
        print("hi")

    refined_src_list = src_list.remove(path.node_list[0])
    refined_dst_list = dst_list.remove(path.node_list[-1])
    for node in nodes:
        successors = modelgraph.successors(node)
        for successor in successors:
            if refined_dst_list != None and successor in refined_dst_list and modelgraph.nodes[successor]['lst'] <= modelgraph.nodes[path[-1]]['lst']:
                return False
        predecessors = modelgraph.predecessors(node)
        for predecessor in predecessors:
            if refined_src_list != None and predecessor in refined_src_list and modelgraph.nodes[predecessor]['lst'] >= modelgraph.nodes[path[0]]['lst']:
                return False
    return True


"""
def get_optcrpath(cpgraph:CPGraph):
    src_list: list[str] = [key for key in cpgraph.nametocpnode.keys() if
                           cpgraph.nametocpnode[key].rem_indeg == 0 and cpgraph.nametocpnode[key].rem_outdeg >= 1]

    if len(src_list) > 1:
        high_lft:float = 0
        high_crpath:CRPath = CRPath()
        for src in src_list:
            edges = get_rechabledges(cpgraph, src)
            src_graph = copy.deepcopy(cpgraph)
            src_graph.keep_edges(edges)
            dst_list: list[str] = [key for key in src_graph.nametocpnode.keys() if
                                   src_graph.nametocpnode[key].rem_outdeg == 0 and src_graph.nametocpnode[key].rem_indeg >= 1]
            if len(dst_list) > 1:
                edges = src_graph.nodelist_multiDest()
                src_graph = copy.deepcopy(cpgraph)
                src_graph.keep_edges(edges)
                src_crpath = src_graph.generate_crpath(src)
            else:
                src_crpath = src_graph.generate_crpath(src)
            lft = src_graph.nametocpnode[src_crpath.node_list[-1]].lft
            if high_lft < lft:
                high_lft = lft
                high_crpath = src_crpath
            del src_graph
        return high_crpath
    else:
        return cpgraph.generate_crpath(src_list[0])
"""
def get_rechabledges(modelgraph, src):
    queue = []
    edgelist = []
    queue.append(src)
    all_edges = set (modelgraph.edges())
    while queue:
        vertex = queue.pop(0)
        neighbors = list(modelgraph.successors(vertex))
        for neighbor in neighbors:
            if (vertex, neighbor) in all_edges:
                edgelist.append((vertex, neighbor))
                all_edges.remove((vertex, neighbor))
                queue.append(neighbor)


    return edgelist

def keep_edges(modelgraph, edges):
    current_edges = set(modelgraph.edges()) #.difference(set(edges))
    edges_to_keep = set(edges)
    edges_to_remove = current_edges.difference(edges_to_keep)
    for edge in edges_to_remove:
        frm, to = edge[0], edge[1]
        modelgraph.remove_edge(frm, to)

    nodes = list(modelgraph.nodes)
    for vertex in nodes:
        if modelgraph.in_degree(vertex) == 0 and modelgraph.out_degree(vertex) == 0:
            modelgraph.remove_node(vertex)




def get_optcrpath(modelgraph):
    src_list: list[str] = [node for node in modelgraph.nodes if
                           modelgraph.in_degree(node) == 0 and modelgraph.out_degree(node) >= 1]
    dst_list: list[str] = [node for node in modelgraph.nodes if
                           modelgraph.out_degree(node) == 0 and modelgraph.in_degree(node) >= 1]
    paths = []
    if len(src_list) == 1 and len(dst_list) == 1:
        crpath = crpath_sim.generate_crpath(modelgraph, src_list[0], dst_list[0])
        return crpath
    elif len(src_list) == 1 and len(dst_list) > 1:
        for dst in dst_list:
            if dst in modelgraph.nodes:
                edges = crpath_sim.nodelist_multiDest(modelgraph, src_list[0], dst, dst_list)
                src_dst_graph = copy.deepcopy(modelgraph)
                keep_edges(src_dst_graph, edges)
                crpath = crpath_sim.generate_crpath(modelgraph, src_list[0], dst)
                # result = list(nx.all_simple_paths(modelgraph, source=src, target=dst))
                paths.append(crpath)
    else:
        for src in src_list:
            edges = get_rechabledges(modelgraph, src)
            src_graph = copy.deepcopy(modelgraph)
            keep_edges(src_graph, edges)

            for dst in dst_list:
                if dst in src_graph.nodes:
                    edges = crpath_sim.nodelist_multiDest(src_graph, src, dst, dst_list)
                    src_dst_graph = copy.deepcopy(src_graph)
                    keep_edges(src_dst_graph,edges)
                    crpath  = crpath_sim.generate_crpath(src_dst_graph, src, dst)
                    #result = list(nx.all_simple_paths(modelgraph, source=src, target=dst))
                    paths.append(crpath)


    latency = []
    for path in paths:
        latency.append(path.latency)
    sorted_paths = [x for _, x in sorted(zip(latency, paths), key=lambda pair: pair[0])]


    for path in reversed(sorted_paths):
        if check_validity(modelgraph,path, src_list, dst_list) == True:
            valid_path = path
            break
    #crpath = crpath_sim.CRPath()
    #crpath.node_list = valid_path
    #valid_path.generate_edge(modelgraph)
    if len(valid_path.node_list) == 0:
        print("stop")
    return valid_path

def get_base_cost(path, cost_list, pred_node, each_gpu, last_node):
    base_cost = 0
    pred_gpu = 0

    for each_gpu_cell in path:
        if pred_node in each_gpu_cell:
            pred_gpu = each_gpu_cell[pred_node]
            cost_pred_gpu = cost_list[pred_gpu]
            cost_pred_node = cost_pred_gpu[pred_node]
            base_cost = cost_pred_node
            return base_cost, pred_gpu
    return base_cost, pred_gpu

def get_gpu_cost(path, cost_list,base_gpu):
    cost_node = 0
    for each_gpu_cell in reversed(path):
        val, = each_gpu_cell.values()
        if val == base_gpu:
            key, = each_gpu_cell.keys()
            cost_base_gpu = cost_list[base_gpu]
            cost_node = cost_base_gpu[key]
            return cost_node
    return cost_node


def min_costgpu (modelgraph, gpu_node_path, gpu_node_cost, pred_nodes, node, last_node, cr_gpu, ngpu, tpsort_bylst):
    mincost, mingpu = 100000, 100000

    cost = 100000
    gpu = -1

    multiple_pred_modes: Dict[str:(int, int)] = {}

    for pred_node in pred_nodes:
        for each_gpu in range(0, ngpu):
            each_gpu_path = gpu_node_path[each_gpu]
            each_gpu_cell_path = each_gpu_path[last_node]

            gpu = each_gpu
            base_cost, base_gpu = get_base_cost(each_gpu_cell_path, gpu_node_cost, pred_node, each_gpu, last_node)

            if (base_gpu == cr_gpu):
                gpu_cost = gpu_node_cost[each_gpu]
                if cr_gpu == each_gpu:
                    base_cost = gpu_cost[last_node]
                else:
                    base_cost = get_gpu_cost(each_gpu_cell_path, gpu_node_cost,base_gpu)
                cost = base_cost + modelgraph.nodes[node]['weight']
            else:
                last_node_cost = get_gpu_cost(each_gpu_cell_path, gpu_node_cost,cr_gpu)
                if (base_cost + modelgraph.edges[pred_node, node]['weight'] > last_node_cost):
                    cost = base_cost + modelgraph.edges[pred_node, node]['weight'] + \
                       modelgraph.nodes[node]['weight']
                else:
                    cost = last_node_cost + modelgraph.nodes[node]['weight']

            if cost < mincost:
                mincost = cost
                mingpu = gpu

        multiple_pred_modes[pred_node] = (mincost, mingpu)

    ret_cost = -1
    ret_gpu = -1

    for pred_node in pred_nodes:
        (mincost, mingpu) = multiple_pred_modes[pred_node]
        if ret_cost < mincost:
            ret_cost = mincost
            ret_gpu = mingpu

    return (ret_cost, ret_gpu)



def hios_pdp( modelgraph, source, destination, ngpu):
    gpu_node_path: Dict[int:Dict[str:List[Dict[str:int]]]] = {}
    gpu_node_cost: Dict[int:Dict[str:float]] = {}

    crpath_sim.generate_crpath(modelgraph, source)
    tpsort_bylst = sorted(modelgraph.nodes(), key=lambda n: modelgraph.nodes[n]['lst'])


    # for node in tpsort_bylst:
    node = tpsort_bylst[0]

    node_cost: Dict[str:float] = {}
    node_path: Dict[str:List[Dict[str:int]]] = {}
    node_cost[node] = 0

    cost = 0

    for gpu in range(0, ngpu):
        gpu_node_path[gpu] = {}
        gpu_node_cost[gpu] = {}

        gpu_row_path = gpu_node_path[gpu]

        path: List[Dict[str:int]] = []
        path_item: Dict[str:int] = {}
        if gpu == 0:
            path_item[node] = 0
            path.append(path_item)

        gpu_row_path[node] = copy.deepcopy(path)
        gpu_node_path[gpu] = gpu_row_path

        gpu_row_cost = gpu_node_cost[gpu]
        gpu_row_cost[node] = copy.deepcopy(cost)
        gpu_node_cost[gpu] = gpu_row_cost

    for index, node in enumerate(tpsort_bylst):
        if index == 0:
            continue
        else:
            last_node = tpsort_bylst[index - 1]

        pred_nodes = list(modelgraph.predecessors(node))
        for gpu in range(0, ngpu):
            cost = 0
            path: List[Dict[str:int]] = []
            path_item: Dict[str:int] = {}
            cr_gpu = gpu
            cost, min_gpu = min_costgpu(modelgraph, gpu_node_path, gpu_node_cost, pred_nodes, node, last_node, cr_gpu,
                                        ngpu, tpsort_bylst)


            node_path = gpu_node_path[min_gpu]
            last_node_path = node_path[last_node]
            path = copy.deepcopy(last_node_path)

            path_item[node] = cr_gpu
            path.append(path_item)
            updated_node_path = gpu_node_path[cr_gpu]
            updated_node_path[node] = path

            updated_node_cost = copy.deepcopy(gpu_node_cost[cr_gpu])
            updated_node_cost[node] = cost
            gpu_node_cost[cr_gpu] = updated_node_cost


    min_cost_gpu_path = {}
    min_cost = 1000000
    last_node = tpsort_bylst[-1]
    for gpu in range(0, ngpu):
        gpu_row_cost = gpu_node_cost[gpu]
        gpu_row_path = gpu_node_path[gpu]
        if gpu_row_cost[last_node] < min_cost:
            min_cost = gpu_row_cost[last_node]
            min_cost_gpu_path = gpu_row_path[last_node]

    crpath_sim.setrank_block(modelgraph, min_cost_gpu_path)

def get_execution_order(tpsort_bylst,id_pos_map):
    execution_order = []
    for node in tpsort_bylst:
        if id_pos_map[node] not in execution_order:
            execution_order.append(id_pos_map[node])
    return execution_order

def intra_op_parallelism(modelgraph, successors_list,window_size, max_num_streams, ngpu,src_node, dst_node):
    brank_stage: Dict[int:List[List[int]]] = {rank: [] for rank in range(ngpu)}
    rank_stage: Dict[int:List[set(int)]] = {rank: [] for rank in range(ngpu)}
    map_pos_to_id = {}
    set_position(modelgraph, rank_stage, map_pos_to_id,ngpu)

    cr_exec_time = graph_latency(modelgraph, src_node, dst_node, ngpu)
    bexec_time = 0
    #find_parallelism(modelgraph, batch_size=1, warmup=2, number=5, repeat=5, ngpu=2):
    #bexec_time = graph_latency(modelgraph, src_node, dst_node, ngpu)
    rank_stage_dup = copy.deepcopy(rank_stage)
    brank_stage = copy.deepcopy(rank_stage)
    tpsort_bylst = sorted(modelgraph.nodes(), key=lambda n: modelgraph.nodes[n]['lst'])

    for node in tpsort_bylst:
        rank = modelgraph.nodes[node]['gpu']

        brank_stage_tmp = copy.deepcopy(brank_stage)

        if {node} in rank_stage_dup[rank]:
            start = 0

            for step in range(window_size, 1, -1):
                window = rank_stage_dup[rank][start: start + step]
                combs = comb_list(successors_list, window, max_num_streams)
                bcomb = []
                bexec_time = cr_exec_time

                for comb in combs:
                    rank_stage_c = copy.deepcopy(brank_stage)

                    #flatten_comb = [item for sublist in comb for subsublist in sublist for item in subsublist]
                    #stage = [[self.nametonode[item]] for item in flatten_comb]
                    #self.latency(stage, nametocpnode_t, parallel_latency, nid, idn, batch_size, warmup, number,repeat)

                    positions = [rank_stage_c[rank].index(item) for item in comb]
                    min_position = min(positions)

                    merged_comb = set([item for sublist in comb for item in sublist])
                    rank_stage_c[rank].insert(min_position, merged_comb)
                    for item in comb:
                        rank_stage_c[rank].remove(item)

                    positional_graph, src_node, dst_node, id_pos_map = check_cycle(modelgraph, rank_stage_c, ngpu)
                    if positional_graph == None:
                        continue

                    """
                    position = 0                    
                    for items in rank_stage_c[rank]:
                        for item in items:
                            nametocpnode_t[item].position = position
                        position = position + 1
                    new_exec = 0
                    new_exec = self.generate_exectime(nametocpnode_t, rank_stage_c, ngpu)
                    """
                    execution_order = get_execution_order(tpsort_bylst, id_pos_map)
                    new_exec = intra_op_graph_latency(positional_graph, execution_order, src_node, dst_node, ngpu)
                    if bexec_time >= new_exec:
                        bexec_time = new_exec
                        bcomb = comb
                        brank_stage_tmp = rank_stage_c



            if  cr_exec_time >= bexec_time:
                brank_stage = brank_stage_tmp
                cr_exec_time = bexec_time
            if len(bcomb) > 1:
                for item in bcomb:
                    rank_stage_dup[rank].remove(item)
            else:
                if {node} in rank_stage_dup[rank]:
                    rank_stage_dup[rank].remove({node})
                else:
                    print('hi')
    return cr_exec_time




def set_successors(modelgraph, successors_list):

    for u in reversed(list(nx.topological_sort(modelgraph))):
        successors = list(modelgraph.successors(u))
        successors.append(u)
        for neighbour in modelgraph.successors(u):
            if neighbour in successors_list:
                successors = successors + successors_list[neighbour]

        successors_list[u] = [*set(successors)]


def check_parallel(successors_list, comb):
    """
    Check whether a set of operators is valid to parallelly execute
    """
    comb = [item for items in comb for item in items]
    num = len(comb)
    for i in range(0, num):
        for j in range(i + 1, num):

            if comb[j] in successors_list[comb[i]]:
                return False

            if comb[i] in successors_list[comb[j]]:
                return False
    return True

def comb_list(successors_list, window, max_num_streams):
    combs = []
    comb_len = max_num_streams
    while comb_len > 1:
        slice = window[0:comb_len]
        comb_len = comb_len - 1
        if len(slice) > 1:
            if check_parallel(successors_list, slice) and len(slice) > 1:
                combs.append(slice)
    return combs


def set_position(modelgraph, rank_stage, map_pos_to_id, ngpu):
    tpsort_bylst = sorted(modelgraph.nodes(), key=lambda n: modelgraph.nodes[n]['lst'])
    for node in tpsort_bylst:
        node_gpu = modelgraph.nodes[node]['gpu']
        rank_stage[node_gpu].append({node})

def get_latency_of_stage(modelgraph,stage):
    max_bound = 0
    utl_bound = 0
    alpha = .5
    operators_latency = []
    for item in stage:
        max_bound += modelgraph.nodes[item]['weight']
        utl_bound += modelgraph.nodes[item]['weight'] * modelgraph.nodes[item]['utilization']
        operators_latency.append(modelgraph.nodes[item]['weight'])

    max_operators_latency = max(operators_latency)
    min_bound = max(utl_bound, max_operators_latency)
    stage_latency = alpha*max_bound + (1-alpha)*min_bound
    return stage_latency

def map_pos_id(modelgraph,positionalgraph, brank_stage, ngpu):
    id_pos_map = {}
    pos_stage_map = {}
    position = 0
    for gpu in range(ngpu):
        stages = brank_stage[gpu]
        for stage in stages:
            pos_stage_map[position] = stage
            if len(stage) == 1:

                positioanl_node_weight = modelgraph.nodes[list(stage)[0]]['weight']
                positionalgraph.add_node(position, weight=positioanl_node_weight)
                positionalgraph.nodes[position]['gpu'] = gpu
                id_pos_map[list(stage)[0]] = position
            else:
                positioanl_node_weight = get_latency_of_stage(modelgraph, stage)
                positionalgraph.add_node(position, weight=positioanl_node_weight)
                positionalgraph.nodes[position]['gpu'] = gpu
                for item in stage:
                    id_pos_map[item] = position
            position += 1
    return id_pos_map, pos_stage_map





def build_graph_by_position(modelgraph, positionalgraph, id_pos_map, pos_stage_map):


    inner_edge = 0
    normal_edge = 0
    for edge in modelgraph.edges:
        pos_src = id_pos_map[edge[0]]
        pos_dst = id_pos_map[edge[1]]


        weight = modelgraph.get_edge_data(*edge)['weight']
        if positionalgraph.has_edge(pos_src, pos_dst):
            weight += positionalgraph.get_edge_data(pos_src, pos_dst)['weight']
            positionalgraph.edges[pos_src, pos_dst]['weight'] = weight
            inner_edge += 1
        else:
          positionalgraph.add_edge(pos_src, pos_dst, weight=weight)


    assert(positionalgraph.number_of_edges() + inner_edge  == modelgraph.number_of_edges())




def check_cycle(modelgraph, brank_stage, ngpu):
    positionalgraph = nx.DiGraph(latency=0)

    id_pos_map,pos_stage_map = map_pos_id(modelgraph,positionalgraph, brank_stage, ngpu)
    build_graph_by_position(modelgraph, positionalgraph, id_pos_map, pos_stage_map)

    try:
        cycles = list(nx.find_cycle(positionalgraph, orientation="original"))
        return None, -1, -1, id_pos_map
    except nx.exception.NetworkXNoCycle as e:
        return positionalgraph, 0, len(id_pos_map), id_pos_map

