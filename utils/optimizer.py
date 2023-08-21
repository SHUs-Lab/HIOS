from .cost_model import IOSCostModel
import json
from .crpath import *

def get_rechabledges(cpgraph:CPGraph, src):
    queue = []
    edgelist:List[(str,str)] = []
    queue.append(src)
    nametoedge = copy.deepcopy(cpgraph.nametoedge)
    while queue:
        vertex = queue.pop(0)
        last_vertex = vertex
        neighbors = set(cpgraph.cpgraph[vertex])
        for neighbor in neighbors:
            if nametoedge.get((vertex, neighbor)):
                edgelist.append((vertex, neighbor))
                queue.append(neighbor)
                del nametoedge[(vertex, neighbor)]

    return edgelist

def check_validity( cpgraph, path, src_list, dst_list):
    nodes = path.node_list[1:-1]
    if len(nodes) == 0:
        return True

    refined_src_list = src_list.remove(path.node_list[0])
    refined_dst_list = dst_list.remove(path.node_list[-1])
    for node in nodes:
        successors = cpgraph.nametocpnode[node].im_successors
        for successor in successors:
            if refined_dst_list != None and successor in refined_dst_list: # and modelgraph.nodes[successor]['lst'] <= modelgraph.nodes[path[-1]]['lst']:
                return False
        predecessors = cpgraph.nametocpnode[node].im_predessors
        for predecessor in predecessors:
            if refined_src_list != None and predecessor in refined_src_list:# and modelgraph.nodes[predecessor]['lst'] >= modelgraph.nodes[path[0]]['lst']:
                return False
    return True



def get_optcrpath(cpgraph:CPGraph):
    src_list: list[str] = [key for key in cpgraph.nametocpnode.keys() if
                           cpgraph.nametocpnode[key].rem_indeg == 0 and cpgraph.nametocpnode[key].rem_outdeg >= 1]
    dst_list: list[str] = [key for key in cpgraph.nametocpnode.keys() if
                           cpgraph.nametocpnode[key].rem_outdeg == 0 and cpgraph.nametocpnode[key].rem_indeg >= 1]

    paths = []
    if len(src_list) == 1 and len(dst_list) == 1:
        crpath = cpgraph.generate_crpath(src_list[0], dst_list[0])
        return crpath
    elif len(src_list) == 1 and len(dst_list) > 1:
        for dst in dst_list:
            edges = cpgraph.nodelist_multiDest(dst, dst_list)
            src_dst_graph = copy.deepcopy(cpgraph)
            src_dst_graph.keep_edges(edges)
            crpath = src_dst_graph.generate_crpath(src_list[0], dst)
            paths.append(crpath)
    else:
        for src in src_list:
            edges = get_rechabledges(cpgraph, src)
            src_graph = copy.deepcopy(cpgraph)
            src_graph.keep_edges(edges)

            for dst in dst_list:
                edges = src_graph.nodelist_multiDest( dst, dst_list)
                src_dst_graph = copy.deepcopy(src_graph)
                src_dst_graph.keep_edges(edges)
                crpath  = src_dst_graph.generate_crpath(src, dst)

                paths.append(crpath)


    latency = []
    for path in paths:
        latency.append(path.latency)
    sorted_paths = [x for _, x in sorted(zip(latency, paths), key=lambda pair: pair[0])]


    for path in reversed(sorted_paths):
        if check_validity(cpgraph, path, src_list, dst_list) == True:
            valid_path = path
            break


    return valid_path




def hios_lp( cpgraph, cost_model, batch_size, warmup, number, repeat, ngpu):
    gpu_graph: Dict[int, CPGraph] = {}
    cpgraph_prime = copy.deepcopy(cpgraph)
    while cpgraph_prime.isempty() is False:
        crpath: CRPath = get_optcrpath(cpgraph_prime)

        for rank in range(ngpu):
            gpu_graph[rank] = copy.deepcopy(cpgraph)
            gpu_graph[rank].setrank_path(crpath, rank)
            gpu_graph[rank].update_combinedgraph()

            gpu_graph[rank].measure_graphlatency( batch_size, warmup, number, repeat)
        opt_rank, opt_latency = -1, 1000000
        for rank in range(ngpu):
            if opt_latency > gpu_graph[rank].cpgraphlatency:
                opt_rank = rank
                opt_latency = gpu_graph[rank].cpgraphlatency
            elif opt_latency == gpu_graph[rank].cpgraphlatency:
                opt_rank = 0
        cpgraph.setrank_path(crpath, opt_rank)
        cpgraph_prime.exclude_pathupdated(crpath)
        for rank in range(ngpu):
            del gpu_graph[rank]


def optimize(height, width,graph: Graph, opt_type,
             batch_size,
             warmup, number, repeat, ngpu, device) -> Graph:


    cost_model = IOSCostModel()
    graph_enter = Placeholder(graph.input.name, graph.input.hint_name, graph.input.output_shape)
    graph_enter.output_shape = graph.enter_node.output_shape
    blocks = []



    for bindex, block in enumerate(graph.blocks):
        cpgraph:CPGraph = CPGraph(block, cost_model, ngpu)
        cpgraph.generate_cpgraph(batch_size, warmup, number, repeat, ngpu)
        cpgraph.set_imsuccessors()
        cpgraph.set_impredessors()

        if opt_type == 'hios_mr':
            hios_pdp(cpgraph, ngpu)
        else:
            hios_lp(cpgraph, cost_model, batch_size, warmup, number, repeat, ngpu)


        new_block = cpgraph.construct_execgraph()
#------------------------------------------------------------------------------------------------------------------
        all_nodes = [block.enter_node] + block.inner_nodes + [block.exit_node]


        nid: Dict[str, int] = {node.name: i for i, node in enumerate(all_nodes)}
        idn: Dict[int, str] = {i: node.name for i, node in enumerate(all_nodes)}
        cpgraph.set_successors()
        window_size = 2
        max_num_streams = 2
        new_block.pr_stages = cpgraph.intra_op_parallelism(nid, idn, window_size, max_num_streams,batch_size, warmup, number, repeat, ngpu)
# ------------------------------------------------------------------------------------------------------------------
        if(len(blocks) > 0):
            if new_block.exit_node.rank < 0 and blocks[-1].exit_node.rank < 0:
                blocks[-1].exit_node.rank = -2

            elif new_block.exit_node.rank < 0 and blocks[-1].exit_node.rank >= 0:
                new_block.enter_node.rank = -1
                blocks[-1].exit_node.rank = -1
                stages = blocks[-1].stages[0].stages
                pr_stages = blocks[-1].pr_stages[0].stages
                stage = stages[-1]
                pr_stage = pr_stages[-1]
                for rank in range(1, ngpu):
                    blocks[-1].stages[rank].stages.append(stage)
                    blocks[-1].pr_stages[rank].stages.append(pr_stage)
                    
            new_block.enter_node = blocks[-1].exit_node

        blocks.append(new_block)
    
    graph_enter.rank = blocks[0].enter_node.rank
    new_graph = Graph(graph.name, graph_enter, blocks)


    name_nonpar = device + "_" + opt_type + "_"+ new_graph.name +'_nonpar_' + str(batch_size) +'_'+ str(height) + '_'+ str(width)

    with open(f'{name_nonpar}', 'w') as f:
        f.write(json.dumps(new_graph.export_config()))

    for bindex, block in enumerate(blocks):
        blocks[bindex].stages = blocks[bindex].pr_stages
    name_par = device + "_" + opt_type + "_"+ new_graph.name +'_par_' + str(batch_size) +'_'+ str(height) + '_'+ str(width)


    with open(f'{name_par}', 'w') as f:
        f.write(json.dumps(new_graph.export_config()))


    return new_graph



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

def min_costgpu (cpgraph, gpu_node_path, gpu_node_cost, pred_nodes, node, last_node, cr_gpu, ngpu, tpsort_bylst):
    mincost, mingpu = 100000, 100000

    cost = 100000
    gpu = -1

    multiple_pred_modes: Dict[str:(int, int)] = {}

    for pred_node in pred_nodes:
        for each_gpu in range(0, ngpu):
            each_gpu_path = gpu_node_path[each_gpu]
            each_gpu_cell_path = each_gpu_path[last_node.name]

            gpu = each_gpu
            base_cost, base_gpu = get_base_cost(each_gpu_cell_path, gpu_node_cost, pred_node, each_gpu, last_node)

            if (base_gpu == cr_gpu):
                gpu_cost = gpu_node_cost[each_gpu]
                if cr_gpu == each_gpu:
                    base_cost = gpu_cost[last_node.name]
                else:
                    base_cost = get_gpu_cost(each_gpu_cell_path, gpu_node_cost,base_gpu)
                cost = base_cost + cpgraph.nametocpnode[node.name].latency
            else:
                last_node_cost = get_gpu_cost(each_gpu_cell_path, gpu_node_cost,cr_gpu)
                if (base_cost + cpgraph.nametoedge[pred_node, node.name].latency > last_node_cost):
                    cost = base_cost + cpgraph.nametoedge[pred_node, node.name].latency + \
                       cpgraph.nametocpnode[node.name].latency
                else:
                    cost = last_node_cost + cpgraph.nametocpnode[node.name].latency

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


def hios_pdp(cpgraph, ngpu):
    gpu_node_path: Dict[int:Dict[str:List[Dict[str:int]]]] = {}
    gpu_node_cost: Dict[int:Dict[str:float]] = {}
    src_list: list[str] = [key for key in cpgraph.nametocpnode.keys() if
                           cpgraph.nametocpnode[key].rem_indeg == 0 and cpgraph.nametocpnode[key].rem_outdeg >= 1]
    cpgraph.generate_crpath(src_list[0])
    cpgraph.set_impredessors()
    tpsort_bylst = sorted(cpgraph.nametocpnode.values(), key=operator.attrgetter('lst'))


    node = tpsort_bylst[0]

    node_cost: Dict[str:float] = {}
    node_path: Dict[str:List[Dict[str:int]]] = {}
    node_cost[node.name] = 0

    cost = 0

    for gpu in range(0, ngpu):
        gpu_node_path[gpu] = {}
        gpu_node_cost[gpu] = {}

        gpu_row_path = gpu_node_path[gpu]

        path: List[Dict[str:int]] = []
        path_item: Dict[str:int] = {}
        if gpu == 0:
            path_item[node.name] = 0
            path.append(path_item)

        gpu_row_path[node.name] = copy.deepcopy(path)
        gpu_node_path[gpu] = gpu_row_path

        gpu_row_cost = gpu_node_cost[gpu]
        gpu_row_cost[node.name] = copy.deepcopy(cost)
        gpu_node_cost[gpu] = gpu_row_cost

    for index, node in enumerate(tpsort_bylst):
        if index == 0:
            continue
        else:
            last_node = tpsort_bylst[index - 1]

        pred_nodes = node.im_predessors
        for gpu in range(0, ngpu):
            cost = 0
            path: List[Dict[str:int]] = []
            path_item: Dict[str:int] = {}
            cr_gpu = gpu
            cost, min_gpu = min_costgpu(cpgraph, gpu_node_path, gpu_node_cost, pred_nodes, node, last_node, cr_gpu,
                                        ngpu, tpsort_bylst)


            node_path = gpu_node_path[min_gpu]
            last_node_path = node_path[last_node.name]
            path = copy.deepcopy(last_node_path)

            path_item[node.name] = cr_gpu
            path.append(path_item)
            updated_node_path = gpu_node_path[cr_gpu]
            updated_node_path[node.name] = path

            updated_node_cost = copy.deepcopy(gpu_node_cost[cr_gpu])
            updated_node_cost[node.name] = cost
            gpu_node_cost[cr_gpu] = updated_node_cost


    min_cost_gpu_path = {}
    min_cost = 1000000
    last_node = tpsort_bylst[-1]
    for gpu in range(0, ngpu):
        gpu_row_cost = gpu_node_cost[gpu]
        gpu_row_path = gpu_node_path[gpu]
        if gpu_row_cost[last_node.name] < min_cost:
            min_cost = gpu_row_cost[last_node.name]
            min_cost_gpu_path = gpu_row_path[last_node.name]

    cpgraph.setrank_block(min_cost_gpu_path)















