import copy
import operator
import math
from .models.common import *
from .ir import *
from utils.cost_model import CostModel
import itertools
import networkx as nx

class CRPath:
    def __init__(self):
        self.node_list = []
        self.edge_list = []
        self.latency = 0

    def verify_edge(self, nametoedge):
        self.edge_list = []
        if len(self.node_list)== 1:
            print('cr path size 1')
            return (-5, -5)
        else:
            rerun = 0
            node_list = copy.deepcopy(self.node_list)
            for idx, x in enumerate(node_list):
                if (idx + 1  < len(node_list)):
                    edge = (x, node_list[idx + 1])
                    if nametoedge.get(edge) == None:
                        print('slack not exact zero')
                        rerun = 1
                        self.node_list.remove(node_list[idx + 1])
                        break

            if rerun == 1:
                self.verify_edge(nametoedge)
        return (-1, -1)



    def generate_edge(self, nametoedge):
        self.edge_list = []

        if len(self.node_list)== 1:
            print('cr path size 1')
        else:
            for idx, x in enumerate(self.node_list):
                if (idx + 1  < len(self.node_list)):
                    edge = (x, self.node_list[idx + 1])
                    if nametoedge.get(edge) != None:
                        self.edge_list.append(edge)

    def set_rank(self, rank):
        self.rank = rank

class CPNode:
    def __init__(self, name, indeg, outdeg, latency):
        self.name = name
        self.indeg, self.outdeg = indeg, outdeg
        self.rem_indeg, self.rem_outdeg = indeg, outdeg

        self.latency = latency
        self.est, self.lst, self.eft, self.lft, self.slack, self.rank, self.hop = 0, 0, 0, 0, 0, -1, 0
        self.successors = []
        self.position = -1
        self.im_predessors = []
        self.im_successors = []


    def measure_latency(self, cost_model: CostModel, nametonode, batch_size, warmup, number, repeat, ngpu =2):
        stage_seqs_nodes = [[nametonode[self.name]]]

        self.latency = float(
            np.mean(cost_model.get_stage_latency(stage_seqs_nodes, batch_size, warmup, number, repeat, ngpu)))
        return self.latency



class CPEdge:
    def __init__(self, frm, to, latency):
        self.frm = frm
        self.to = to
        self.latency = latency

    def measure_latency(self, cost_model: CostModel, nametonode, batch_size, warmup, number, repeat,ngpu):

        from_node = nametonode[self.frm]
        to_node = nametonode[self.to]
        reset_name()
        v = placeholder(output_shape=from_node.output_shape)

        block = Block(enter_node=v.node)
        v1 = mpi_sr(block, inputs=[[v]], com_type=1, rank=0)
        v1.node.outputs = [[1, v1.tag]]
        v2 = mpi_sr(block, inputs=[[v1]], com_type=0, rank=1, is_exit=True)

        stage0 = Stage((0, [[[1]]]))
        stage1 = Stage((1, [[[2]]]))
        stage_list = [stage0, stage1]
        block.stages = stage_list
        graph = Graph(name="demo", input=v.node, blocks=[block])

        self.latency = float(
            np.mean(cost_model.get_graph_latency(graph, batch_size, warmup, number, repeat, profile_stage = 0)))
        return self.latency


class CPGraph:
    def __init__(self, block:Block, cost_model: CostModel, ngpu):

        self.cpgraph: Dict[str, List[str]] = {}
        self.nametocpnode: Dict[str, CPNode] = {}
        self.nametoedge: Dict[(str, str), CPEdge] = {}

        self.block:Block = block
        self.sn = block.enter_node.name
        self.en = block.exit_node.name
        self.cost_model = cost_model
        self.crpath:CRPath = CRPath()
        self.nametonode: Dict[str, Node] = {}
        self.cpgraphlatency = 0
        self.result_tpsort:List[str] = []
        self.rank_stage: Dict[int:List[List[str]]] = {rank: [] for rank in range(ngpu)}


    def generate_cpgraph(self, batch_size, warmup = 2, number =5 , repeat = 5, ngpu =2):
        """
        Build a graph of given operators. The global index in nid is used to represent the operator in the graph.
        """

        blocknodes = [self.block.enter_node] + self.block.inner_nodes + [self.block.exit_node]

        self.cpgraph: Dict[str, List[str]] = {}
        self.nametonode = {node.name: node for i, node in enumerate(blocknodes)}
        for nu in blocknodes:
            self.cpgraph[nu.name] = []
            for use in nu.uses:
                nv = use[0]
                if nv in blocknodes:
                    self.cpgraph[nu.name].append(nv.name)
                    cpedge:CPEdge = CPEdge(nu.name, nv.name ,-1)
                    cpedge.measure_latency(self.cost_model, self.nametonode, batch_size, warmup, number, repeat, ngpu)
                    self.nametoedge[(nu.name,nv.name)] = cpedge

            self.cpgraph[nu.name] = list(self.cpgraph[nu.name])  # dump duplicate targets list(set(self.cpgraph[nu.name]))

        for node in blocknodes:
            cpnode:CPNode = CPNode(node.name, node.indeg, node.outdeg, 0)
            if self.nametonode[cpnode.name].type == 'placeholder':
                cpnode.latency = 0
            if cpnode.name != self.sn:
                cpnode.latency = cpnode.measure_latency(self.cost_model, self.nametonode, batch_size, warmup, number, repeat, ngpu)
            if cpnode.name == self.sn:
                cpnode.rem_indeg = 0
            if cpnode.name == self.en:
                cpnode.rem_outdeg = 0
            self.nametocpnode[cpnode.name] = cpnode

        self.result_tpsort = self.topological_sort()



    def topological_sort(self):
        in_degree = {node: self.nametocpnode[node].rem_indeg for node in self.nametocpnode.keys()}

        cpgraph = copy.deepcopy(self.cpgraph)
        queue = [key for key in in_degree.keys()  if in_degree[key] == 0]
        self.result_tpsort = []
        while queue:
            vertex = queue.pop(0)
            self.result_tpsort.append(vertex)
            neighbors = copy.deepcopy(cpgraph[vertex])
            for neighbor in neighbors:
                cpgraph[vertex].remove(neighbor)
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        return self.result_tpsort



    def update_combinedgraph(self):
        ndlist_toremove = [node for node in self.nametocpnode.keys() if self.nametocpnode[node].rank == -1] # and node != self.en and node != self.sn]
        edgelist_toremove = []
        for edge in self.nametoedge.keys(): 
            frm, to = edge[0], edge[1]
            for vertex in ndlist_toremove:
                if frm == vertex or to == vertex:
                    edgelist_toremove.append(edge)
        
        ndlist_toremove = [*set(ndlist_toremove)]
        edgelist_toremove = [*set(edgelist_toremove)]
        for edge in edgelist_toremove:
            del self.nametoedge[edge]
            frm, to = edge[0], edge[1]
            count = self.cpgraph[frm].count(to)

            self.nametocpnode[frm].outdeg -= count
            self.nametocpnode[to].indeg -= count

            for _ in range(count):
                self.cpgraph[frm].remove(to)

        for vertex in ndlist_toremove:
                del self.nametocpnode[vertex]
                del self.cpgraph[vertex]



    def nodelist_multiDest(self, dst, dst_list):

        topological_order = self.topological_sort()

        node_list = []
        edge_list = []
        node_list.append(dst)

        for vertex in dst_list:
            if vertex in topological_order:
                topological_order.remove(vertex)

        for vertex in reversed(topological_order):
            for neighbor in self.cpgraph[vertex]:
                if neighbor in node_list:
                    node_list.insert(0, vertex)
                    edge_list.append((vertex, neighbor))

        return edge_list


    def generate_crpath(self, src, dst = None):
        self.crpath = CRPath()
        if dst not in self.nametocpnode:
            self.crpath.latency = 0
            return self.crpath


        for node in self.nametocpnode:
            self.nametocpnode[node].est = 0
            self.nametocpnode[node].eft = 0
            self.nametocpnode[node].lst = 0
            self.nametocpnode[node].lft = 0
            self.nametocpnode[node].slack = 0

        topological_order = self.topological_sort()
        max_eft = 0
        max_dst = ''

        for vertex in topological_order:
            for neighbor in self.cpgraph[vertex]:
                edge = self.nametoedge[(vertex, neighbor)]
                self.nametocpnode[neighbor].est = max(self.nametocpnode[neighbor].est,
                                                      self.nametocpnode[vertex].eft + edge.latency)
                self.nametocpnode[neighbor].eft = self.nametocpnode[neighbor].est + self.nametocpnode[neighbor].latency
                if self.nametocpnode[neighbor].eft > max_eft:
                    max_eft = self.nametocpnode[neighbor].eft
                    max_dst = neighbor

        if dst == None:
            dst = max_dst
        self.nametocpnode[dst].eft = self.nametocpnode[dst].est + self.nametocpnode[dst].latency
        self.nametocpnode[dst].lft = self.nametocpnode[dst].eft
        self.nametocpnode[dst].lst = self.nametocpnode[dst].est

        for node in self.nametocpnode:
            self.nametocpnode[node].lft = self.nametocpnode[dst].lft

        for vertex in reversed(topological_order):
            for neighbor in self.cpgraph[vertex]:
                edge = self.nametoedge[(vertex, neighbor)]
                self.nametocpnode[vertex].lft = min(self.nametocpnode[vertex].lft,
                                                    self.nametocpnode[neighbor].lst - edge.latency)
                self.nametocpnode[vertex].lst = self.nametocpnode[vertex].lft - self.nametocpnode[vertex].latency

        for vertex in self.nametocpnode:
            self.nametocpnode[vertex].slack = round(self.nametocpnode[vertex].eft, 8) - round(
                self.nametocpnode[vertex].lft, 8)

        for vertex in topological_order:
            if math.isclose(self.nametocpnode[vertex].slack, 0, abs_tol=1e-08):
                self.crpath.node_list.append(vertex)


        abs_tol = 1e-08
        while (True):
            frm, to = self.crpath.verify_edge(self.nametoedge)
            if (frm == -1 and to == -1):
                self.crpath.generate_edge(self.nametoedge)
                break
            else:
                abs_tol = abs_tol * 10
                self.crpath.node_list.clear()
                for idx, vertex in enumerate(topological_order):

                    if math.isclose(self.nametocpnode[vertex].slack, 0, abs_tol=abs_tol):
                        self.crpath.node_list.append(vertex)

        self.crpath.latency = self.nametocpnode[dst].eft
        return self.crpath


        
    def isempty(self):
        total_remindeg = sum(self.nametocpnode[vertex].rem_indeg for vertex in self.nametocpnode.keys())
        total_remoutdeg = sum(self.nametocpnode[vertex].rem_outdeg for vertex in self.nametocpnode.keys())

        print("total_remindeg " + str(total_remindeg) + " total_remoutdeg " + str(total_remoutdeg))

        for vertex in self.nametocpnode.keys():
            if self.nametocpnode[vertex].rem_indeg > 0 or self.nametocpnode[vertex].rem_outdeg > 0:
                return False
            
        return True


    def exclude_pathupdated(self, crpath: CRPath):
        for edge in crpath.edge_list:
            del self.nametoedge[edge]
            frm, to = edge[0], edge[1]
            count = self.cpgraph[frm].count(to)
            self.nametocpnode[frm].rem_outdeg -= count
            self.nametocpnode[to].rem_indeg -= count
            for _ in range(count):
                self.cpgraph[frm].remove(to)

            
        for vertex in crpath.node_list:
            if vertex in self.nametocpnode:
                if self.nametocpnode[vertex].rem_outdeg == 0 and self.nametocpnode[vertex].rem_indeg == 0:
                    del self.nametocpnode[vertex]



    def keep_edges(self, edges):
        node_list = []
        edges_to_remove = set(self.nametoedge.keys()).difference(set(edges))
        for edge in edges_to_remove:
            del self.nametoedge[edge]
            frm, to = edge[0], edge[1]
            node_list.append(frm)
            node_list.append(to)

            count = self.cpgraph[frm].count(to)
            if count > 1:
                for _ in range(count):
                    self.nametocpnode[frm].rem_outdeg -= 1
                    self.nametocpnode[to].rem_indeg -= 1
                    self.cpgraph[frm].remove(to)
            else:
                self.nametocpnode[frm].rem_outdeg -= 1
                self.nametocpnode[to].rem_indeg -= 1
                self.cpgraph[frm].remove(to)
        node_list = [*set(node_list)] 
        for vertex in node_list:
            if self.nametocpnode[vertex].rem_outdeg == 0 and self.nametocpnode[vertex].rem_indeg == 0:
                del self.nametocpnode[vertex]


    def setrank_path(self, crpath:CRPath, rank):   
        for vertex in crpath.node_list:
            if self.nametocpnode[vertex].rank == -1:
                self.nametocpnode[vertex].rank = rank

        for edge in crpath.edge_list:
            frm, to = edge[0], edge[1]
            if self.nametocpnode[frm].rank == self.nametocpnode[to].rank:
                self.nametoedge[edge].latency = 0

    def setrank_block(self, min_cost_gpu_path):
        for vertex in min_cost_gpu_path:
            for vertex_key in vertex.keys():
                self.nametocpnode[vertex_key].rank = vertex[vertex_key]

    def measure_graphlatency(self,  batch_size, warmup, number, repeat, stage_list = None, ngpu = 2):
        blocks = []


        block = self.construct_execgraph()
        if stage_list != None:
            block.stages = stage_list

        blocks.append(block)
        graph_enter = Placeholder(blocks[0].enter_node.name, blocks[0].enter_node.hint_name, blocks[0].enter_node.output_shape)


        graph_enter.rank = blocks[0].enter_node.rank
        new_graph = Graph("intermediate" + "_" + "graph", graph_enter, blocks)

        if len(block.stages[0].stages) <= 1:
            self.cpgraphlatency = 1000000
        else:
            self.cpgraphlatency = float(
                np.mean(self.cost_model.get_graph_latency(new_graph, batch_size, warmup, number, repeat, profile_stage = 0, ngpu=2)))

        return self.cpgraphlatency 


    def construct_execgraph(self, ngpu = 2):
        """
        Construct the optimized computation graph.
        """
        inner_nodes = []
        for rank in range(ngpu):
            self.rank_stage[rank].clear()

        def get_new_terms(terms, new_node, max_rank, do_sort=True):
            nterms = []
            for ti, term in enumerate(terms):
                nterm = []
                for vi, value in enumerate(term):
                    if out_dict.get(value.node) != None:
                        nv = out_dict[value.node]
                        tag = random.randint(1, 30000)
                        tag = random.randint(1+tag, 30000+tag)
                        if new_node.name != self.en and max_rank >= 1:
                            out_dict[value.node][0].outputs.append([new_node.rank, tag])

                        nterm.append(Value(nv[0], nv[1] + value.begin, nv[0].output_shape[0], nv[0].output_shape[1],
                                               nv[0].output_shape[2], nv[0].rank, tag))

                nterms.append(nterm)
            if do_sort:
                nterms = sorted(nterms,
                                key=lambda nterm: (len(nterm), nterm[0].node.name))  # git rid of duplicates terms
            for ti, term in enumerate(nterms):
                for vi, value in enumerate(term):
                    value.node.uses.append((new_node, ti, vi))

            return nterms

        snodes = []

        for node in self.result_tpsort:
            if self.nametocpnode.get(node) != None:
                if node != self.sn and self.nametocpnode[node].rank != -1:
                    snodes.append(self.nametonode[node])


        max_rank = max(self.nametocpnode[snode.name].rank for snode in snodes)


        v = placeholder(output_shape=self.block.enter_node.output_shape, name=self.block.enter_node.name)
        v.node.indeg = 0
        v.node.outdeg = self.nametocpnode[self.sn].outdeg


        block = Block(enter_node=v.node)
        out_dict = {self.block.enter_node: (v.node, 0, v.node.output_shape[0])}


        new_nodes = []

        for snode in snodes:
            snode_config = snode.export_config()
            if isinstance(snode, Sequential):
                snode_config["nodes"][0]["inputs"] = []
                new_node = Node.from_config(snode_config, {})
                new_node.indeg = self.nametocpnode[snode.name].indeg
                new_node.outdeg = self.nametocpnode[snode.name].outdeg
                new_node.rank = self.nametocpnode[snode.name].rank

                new_node.nodes[0].inputs = get_new_terms(snode.nodes[0].inputs, new_node, max_rank, do_sort=False)
                new_node.inputs = new_node.nodes[0].inputs

                for index, nd in enumerate(new_node.nodes):
                    nd.infer_shape()
                    if index >= 1:
                        for ti, terms in enumerate(nd.inputs):
                            for vi, value in enumerate(terms):
                                value.end = new_node.nodes[index - 1].output_shape[0]
                                value.height = new_node.nodes[index - 1].output_shape[1]
                                value.width = new_node.nodes[index - 1].output_shape[2]

                for nd in new_node.nodes:
                    nd.rank = self.nametocpnode[snode.name].rank
                    nd.tag = new_node.tag

                new_node.infer_shape()
                out_dict[snode] = (new_node, 0, new_node.output_shape[0])
            else:
                snode_config["inputs"] = []
                new_node = Node.from_config(snode_config, {})
                new_node.indeg = self.nametocpnode[snode.name].indeg
                new_node.outdeg = self.nametocpnode[snode.name].outdeg
                new_node.rank = self.nametocpnode[snode.name].rank
                new_node.inputs = get_new_terms(snode.inputs, new_node, max_rank, do_sort=False)
                new_node.infer_shape()
                out_dict[snode] = (new_node, 0, new_node.output_shape[0])

        block.enter_node.rank = -1 if max_rank > 0 else 0

        for snode in snodes:
            nv = out_dict[snode]
            new_nodes.append(nv[0])
        inner_nodes.extend(new_nodes)

        block.exit_node = inner_nodes.pop()
        block.inner_nodes = inner_nodes



        for node in (sorted(self.nametocpnode.values(), key=operator.attrgetter('lst'))):
            if node.name == self.sn or node.name == self.en:
                continue
            else:
                self.rank_stage[node.rank].append([[node.name]])

        block.exit_node.rank = -3 if max_rank > 0 else 0
        if(max_rank > 0):
            for rank in range(ngpu):
                self.rank_stage[rank].append([[self.en]])
        else:
            self.rank_stage[0].append([[self.en]])


        self.nametocpnode[block.enter_node.name].rank = block.enter_node.rank
        self.nametocpnode[block.exit_node.name].rank = block.exit_node.rank

        stage_list = [Stage((rank, self.rank_stage[rank])) for rank in range(ngpu)]

        block.stages = stage_list
        return block

    def set_impredessors(self):
        for u in self.result_tpsort:
            for use in self.nametonode[u].uses:
                if self.nametocpnode.get(use[0].name):
                    self.nametocpnode[use[0].name].im_predessors.append(u)

    def set_successors(self):
        for u in reversed(self.result_tpsort):
            successors = self.nametocpnode[u].successors
            successors.append(u)
            for use in self.nametonode[u].uses:
                if self.nametocpnode.get(use[0].name):
                    successors = successors + self.nametocpnode[use[0].name].successors
            self.nametocpnode[u].successors = [*set(successors)]

    def check_parallel(self, comb):
        """
        Check whether a set of operators is valid to parallelly execute
        """
        comb = [item for items in comb for item in items]
        num = len(comb)
        for i in range(0, num):
            for j in range(i + 1, num):

                if comb[j] in self.nametocpnode[comb[i]].successors:
                    return False

                if comb[i] in self.nametocpnode[comb[j]].successors:
                    return False
        return True



    def comb_list(self, window, max_num_streams):
        combs = []
        comb_len = max_num_streams
        while comb_len > 1:
            slice = window[0:comb_len]
            comb_len = comb_len - 1
            if len(slice) > 1:
                if self.check_parallel(slice) and len(slice) > 1:
                    combs.append(slice)
        return combs



    def latency(self, stage: List[List[Node]], nametocpnode_t, parallel_latency, nid, idn, batch_size, warmup,
                number, repeat):
        """
        Measure the latency of a stage.
        """

        ssl = sorted([nid[u.name] for u in itertools.chain(*stage)])
        ss = ""
        for item in ssl:
            ss = ss + idn[item]

        if ss not in parallel_latency:
            parallel_latency[ss] = float(
                np.mean(self.cost_model.get_stage_latency(stage, batch_size, warmup, number, repeat)))

        for u in itertools.chain(*stage):
            nametocpnode_t[u.name].latency = parallel_latency[ss]



    def map_pos_id(self, positionalgraph, brank_stage, ngpu):
        id_pos_map = {}
        pos_stage_map = {}

        position = -1
        pos_stage_map[position] = {self.block.enter_node.name}
        id_pos_map[self.block.enter_node.name] = position
        position += 1


        for gpu in range(ngpu):
            stages = brank_stage[gpu]
            for stage in stages:
                pos_stage_map[position] = stage
                if len(stage) == 1:

                    positionalgraph.add_node(position)
                    positionalgraph.nodes[position]['gpu'] = gpu
                    id_pos_map[list(stage)[0]] = position
                else:
                    positionalgraph.add_node(position)
                    positionalgraph.nodes[position]['gpu'] = gpu
                    for item in stage:
                        id_pos_map[item] = position
                position += 1
        return id_pos_map, pos_stage_map

    def build_graph_by_position(self, positionalgraph, id_pos_map, pos_stage_map):

        inner_edge = 0

        for edge in self.nametoedge:
            pos_src = id_pos_map[edge[0]]
            pos_dst = id_pos_map[edge[1]]

            weight = self.nametoedge[edge].latency
            if positionalgraph.has_edge(pos_src, pos_dst):
                weight += positionalgraph.get_edge_data(pos_src, pos_dst)['weight']
                positionalgraph.edges[pos_src, pos_dst]['weight'] = weight
                inner_edge += 1
            else:
                positionalgraph.add_edge(pos_src, pos_dst, weight=weight)

        assert (positionalgraph.number_of_edges() + inner_edge == len(self.nametoedge))

    def check_cycle(self, brank_stage, ngpu):
        positionalgraph = nx.DiGraph(latency=0)

        id_pos_map, pos_stage_map = self.map_pos_id( positionalgraph, brank_stage, ngpu)
        self.build_graph_by_position( positionalgraph, id_pos_map, pos_stage_map)

        try:
            cycles = list(nx.find_cycle(positionalgraph, orientation="original"))
            return None, id_pos_map
        except nx.exception.NetworkXNoCycle as e:
            # print("Found the no cycle exception")
            return positionalgraph, id_pos_map, pos_stage_map

    def get_execution_order(self, tpsort_bylst, id_pos_map, pos_stage_map, ngpu = 2):
        rank_stage: Dict[int:List[List[str]]] = {rank: [] for rank in range(ngpu)}
        rank_pos: Dict[int:List[int]] = {rank: [] for rank in range(ngpu)}

        for node in tpsort_bylst:
            rank = node.rank
            if id_pos_map[node.name] < 0:
                continue
            if rank < 0:
                for gpu in range(ngpu):
                    rank_pos[gpu].append(id_pos_map[node.name])
            else:
                rank_pos[rank].append(id_pos_map[node.name])

        for rank in range(ngpu):
            prev = -1
            for pos in rank_pos[rank]:
                if pos == prev:
                    continue
                else:
                    prev = pos
                parallel_ids = pos_stage_map[pos]
                exec_order = []
                for id in parallel_ids:
                    exec_order.append([id])
                rank_stage[rank].append(exec_order)

        stage_list = [Stage((rank, rank_stage[rank])) for rank in range(ngpu)]
        return stage_list

    def intra_op_parallelism(self, nid, idn, window_size, max_num_streams, batch_size, warmup=2, number=5, repeat=5, ngpu=2):


        brank_stage_formatted: Dict[int:List[List[str]]] = {rank: [] for rank in range(ngpu)}
        rank_stage: Dict[int:List[set(str)]] = {rank: [] for rank in range(ngpu)}

        for rank in range(ngpu):
            position = 0
            for sublist in self.rank_stage[rank]:
                for subsublist in sublist:
                    for element in subsublist:
                        rank_stage[rank].append({element})
                        if self.nametocpnode[element].position < position:
                            self.nametocpnode[element].position = position
                position = position + 1

        cr_exec_time = self.measure_graphlatency(batch_size, warmup, number, repeat)          #self.generate_exectime(nametocpnode, rank_stage, ngpu)


        bexec_time = 0
        rank_stage_dup = copy.deepcopy(rank_stage)
        brank_stage = copy.deepcopy(rank_stage)
        tpsort_bylst = sorted(self.nametocpnode.values(), key=operator.attrgetter('lst'))

        for node in tpsort_bylst:
            rank = node.rank
            if rank < 0:
                continue

            brank_stage_tmp = copy.deepcopy(brank_stage)

            if {node.name} in rank_stage_dup[rank]:
                start = 0

                for step in range(window_size, 1, -1):
                    window = rank_stage_dup[rank][start: start + step]
                    combs = self.comb_list(window, max_num_streams)
                    bcomb = []
                    bexec_time = cr_exec_time

                    for comb in combs:
                        rank_stage_c = copy.deepcopy(brank_stage)

                        positions = [rank_stage_c[rank].index(item) for item in comb]
                        min_position = min(positions)

                        merged_comb = set([item for sublist in comb for item in sublist])
                        rank_stage_c[rank].insert(min_position, merged_comb)
                        for item in comb:
                            rank_stage_c[rank].remove(item)

                        positional_graph, id_pos_map, pos_stage_map = self.check_cycle(rank_stage_c, ngpu)
                        if positional_graph == None:
                            continue

                        execution_order = self.get_execution_order(tpsort_bylst, id_pos_map, pos_stage_map, ngpu)
                        new_exec = self.measure_graphlatency(batch_size, warmup, number, repeat, execution_order,ngpu)

                        if bexec_time >= new_exec:
                            bexec_time = new_exec
                            bcomb = comb
                            brank_stage_tmp = rank_stage_c

                if cr_exec_time >= bexec_time:
                    brank_stage = brank_stage_tmp
                    cr_exec_time = bexec_time
                if len(bcomb) > 1:
                    for item in bcomb:
                        rank_stage_dup[rank].remove(item)
                else:
                    if {node} in rank_stage_dup[rank]:
                        rank_stage_dup[rank].remove({node})
                    


        for rank in range(ngpu):
            stage = brank_stage[rank]
            for ids in stage:
                exec_ids = []
                for id in ids:
                    exec_ids.append([id])
                brank_stage_formatted[rank].append(exec_ids)

        stage_list = [Stage((rank, brank_stage_formatted[rank])) for rank in range(ngpu)]
        return stage_list

    def set_imsuccessors(self):
        for u in reversed(self.result_tpsort):
            successors = self.nametocpnode[u].im_successors
            for use in self.nametonode[u].uses:
                if self.nametocpnode.get(use[0].name):
                    successors.append(use[0].name)
            self.nametocpnode[u].im_successors = [*set(successors)]

    def set_successors(self):
        for u in reversed(self.result_tpsort):
            successors = self.nametocpnode[u].successors
            successors.append(u)
            for use in self.nametonode[u].uses:
                if self.nametocpnode.get(use[0].name):
                    successors = successors + self.nametocpnode[use[0].name].successors
            self.nametocpnode[u].successors = [*set(successors)]





