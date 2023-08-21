import random
import networkx as nx
import math

class CRPath:
    def __init__(self):
        self.node_list = []
        self.edge_list = []
        self.latency = 0

    def verify_edge(self,modelgraph):
        self.edge_list = []
        if len(self.node_list) == 1:
            print('cr path size 1')
            return (-5, -5)
        else:
            rerun = 0
            pairs = [self.node_list[i: i + 2] for i in range(len(self.node_list) - 1)]
            for pair in pairs:
                if modelgraph.has_edge(pair[0], pair[1]) == False:
                    rerun = 1
                    self.node_list.remove(pair[1])
                    break

            if rerun == 1:
                self.verify_edge(modelgraph)
        return (-1, -1)
    def generate_edge(self, modelgraph):
        self.edge_list = []

        if len(self.node_list) == 1:
            print('cr path size 1')
        else:
            pairs = [self.node_list[i: i + 2] for i in range(len(self.node_list) - 1)]
            for pair in pairs:
                if modelgraph.has_edge(pair[0], pair[1]) == True:
                    self.edge_list.append((pair[0], pair[1]))



def setrank_path(modelgraph, crpath, rank):
    for vertex in crpath.node_list:
        if modelgraph.nodes[vertex]['gpu'] == -1:
            modelgraph.nodes[vertex]['gpu']= rank
    """
    for edge in crpath.edge_list:
        frm, to = edge[0], edge[1]
        if modelgraph.nodes[frm]['gpu'] == modelgraph.nodes[to]['gpu']:
            modelgraph.edges[frm, to]['weight'] = 0
    """

def setrank_block(modelgraph, min_cost_gpu_path):
    for vertex in min_cost_gpu_path:
        for vertex_key in vertex.keys():
            modelgraph.nodes[vertex_key]['gpu']  = vertex[vertex_key]
            #print(modelgraph.nodes[vertex_key]['gpu'])


def exclude_pathupdated(modelgraph, crpath):
    for edge in crpath.edge_list:
        if edge not in modelgraph.edges:
            print("hi")
        else:
            modelgraph.remove_edge(*edge)

    for vertex in crpath.node_list:
        if vertex in modelgraph.nodes:
            if modelgraph.out_degree(vertex) == 0 and modelgraph.in_degree(vertex) == 0:
                modelgraph.remove_node(vertex)

def update_combinedgraph(modelgraph):
    ndlist_toremove = [node for node in modelgraph.nodes if
                       modelgraph.nodes[node]['gpu'] == -1]

    edgelist_toremove = []
    for edge in modelgraph.edges:
        frm, to = edge[0], edge[1]
        for vertex in ndlist_toremove:
            if frm == vertex or to == vertex:
                edgelist_toremove.append(edge)

    ndlist_toremove = [*set(ndlist_toremove)]
    edgelist_toremove = [*set(edgelist_toremove)]
    for edge in edgelist_toremove:
        modelgraph.remove_edge(*edge)

    for vertex in ndlist_toremove:
        modelgraph.remove_node(vertex)


def nodelist_multiDest(modelgraph, src, dst, dst_list):

    topological_order = list(nx.topological_sort(modelgraph))

    node_list = []
    edge_list = []
    node_list.append(dst)

    for vertex in dst_list:
        if vertex in topological_order:
            topological_order.remove(vertex)

    for vertex in reversed(topological_order):
        for neighbor in modelgraph.successors(vertex):
            if neighbor in node_list:
                node_list.insert(0, vertex)
                edge_list.append((vertex, neighbor))

    return edge_list


def keep_edges(modelgraph, edges):
    node_list = []
    edges_to_remove = set(modelgraph.edges).difference(set(edges))
    for edge in edges_to_remove:
        modelgraph.remove_edge(*edge)
        frm, to = edge[0], edge[1]
        node_list.append(frm)
        node_list.append(to)


    node_list = [*set(node_list)]
    for vertex in node_list:
        if modelgraph.out_degree(vertex) == 0 and modelgraph.in_degree(vertex) == 0:
            modelgraph.remove_node(vertex)




def generate_crpath(modelgraph, src, dst = None):
    crpath = CRPath()

    """
    if dst != None:
        if nx.has_path(modelgraph, src, dst) == False :
            return crpath
    """

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
    """
    if dst != None:
        src_index = topological_order.index(src)
        dst_index = topological_order.index(dst)
        topological_order = topological_order[src_index: dst_index + 1]
    """

    max_eft = 0
    max_dst = ''
    for vertex in topological_order:
        for neighbor in modelgraph.neighbors(vertex):
            modelgraph.nodes[neighbor]['est'] = max(modelgraph.nodes[neighbor]['est'],
                                                  modelgraph.nodes[vertex]['eft'] + modelgraph.edges[vertex, neighbor]['weight'])

            modelgraph.nodes[neighbor]['eft'] = modelgraph.nodes[neighbor]['est'] + modelgraph.nodes[neighbor]['weight']
            if modelgraph.nodes[neighbor]['eft'] > max_eft:
                max_eft = modelgraph.nodes[neighbor]['eft']
                max_dst = neighbor

    if dst == None:
        dst = max_dst

    modelgraph.nodes[dst]['eft'] = modelgraph.nodes[dst]['est'] + modelgraph.nodes[dst]['weight']
    modelgraph.nodes[dst]['lft'] = modelgraph.nodes[dst]['eft']
    modelgraph.nodes[dst]['lst'] = modelgraph.nodes[dst]['est']

    for vertex in modelgraph.nodes:
        modelgraph.nodes[vertex]['lft']  = modelgraph.nodes[dst]['lft']

    for vertex in reversed(topological_order):
        for neighbor in modelgraph.successors(vertex):
            modelgraph.nodes[vertex]['lft'] = min(modelgraph.nodes[vertex]['lft'],
                                                modelgraph.nodes[neighbor]['lst'] - modelgraph.edges[vertex, neighbor]['weight'])
            modelgraph.nodes[vertex]['lst'] = modelgraph.nodes[vertex]['lft'] - modelgraph.nodes[vertex]['weight']

    for vertex in reversed(topological_order):
        modelgraph.nodes[vertex]['slack'] = round(modelgraph.nodes[vertex]['eft']) - round(modelgraph.nodes[vertex]['lft'])


    for vertex in topological_order:
        if math.isclose(modelgraph.nodes[vertex]['slack'], 0, abs_tol=1e-08):
            crpath.node_list.append(vertex)

    abs_tol = 1e-08
    crpath.generate_edge(modelgraph)
    counter = 0
    while (True):
        frm, to = crpath.verify_edge(modelgraph)
        if (frm == -1 and to == -1):
            crpath.generate_edge(modelgraph)
            break
        else:
            if counter == 20:
                nocrpath = CRPath()
                return nocrpath

            abs_tol = abs_tol * 10
            crpath.node_list.clear()
            counter += 1

            for vertex in topological_order:
                if math.isclose(modelgraph.nodes[vertex]['slack'], 0, abs_tol=1e-08):
                    crpath.node_list.append(vertex)

    if src not in crpath.node_list or dst not in crpath.node_list:
        nocrpath = CRPath()
        return nocrpath

    crpath.latency = modelgraph.nodes[dst]['eft']
    return crpath

