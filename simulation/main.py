# This is a sample Python script.
import copy
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import random
import networkx as nx
import crpath_sim
import optimizer_sim
import ios_optimizer
import os
import math
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Set, FrozenSet, List
import gc
import time
import sys

import copy

graphcount = 1
experiment = "gpu"
#experiment = "struct"
#experiment = "node"
#experiment = "edge"
#experiment = "comm"
node_weight = {}
#edge_weight = {}

#file_name = "result"
window_size = max_num_streams = 2
node_size_list = [200]
#avg_nodes_per_layer_list = [ 15, 20]
#node_edge_ratio_list = [400]
edge_size_list = [400]
#layer_size_list = [16]
layer_size_list = [8]
#layer_size_list = [14]
#gpu_num_list = [2, 4, 6, 8, 10, 12]
gpu_num_list = [4]
import time
#comm_to_comp_ratio_list = [ .44, .6, .8, .10, .12,.14]
comm_to_comp_ratio_list = [.8]
sys.setrecursionlimit(20000)

def generate_nodes(modelgraph, num_layer, num_nodes, max_nodes_per_layer, min_nodes_per_layer):
    list_of_layers = [[] for _ in range(num_layer)]
    node_id = 0
    list_of_layers[0].append(node_id)
    node_id += 1
    while node_id < num_nodes-1:
        for layer_id in range(1, num_layer - 1):
            rand = random.randint(max(min_nodes_per_layer - len(list_of_layers[layer_id]), 0 ), min(min_nodes_per_layer, max_nodes_per_layer - len(list_of_layers[layer_id])))
            rand = min(rand, num_nodes-1 - node_id)
            for _ in range(rand):
                list_of_layers[layer_id].append(node_id)
                node_id += 1
            if node_id == num_nodes-1:
                break
    list_of_layers[num_layer-1].append(node_id)


    for id in range(0, num_nodes):
        node_weight[id] = random.randint(1000,40000)/10000

        modelgraph.add_node(id, weight=node_weight[id])
        utilization = random.randint(6000, 10000) / 10000
        modelgraph.add_node(id, utilization = utilization)
        #print(str(id) + str(" ") + str(node_weight[id]) + str("\n"))

    """
    for layer_id in range(num_layer):
        print(list_of_layers[layer_id])
    """
    return list_of_layers



def generate_edges(modelgraph, list_of_layers, num_layer, edge_count, num_nodes):
    edge_list = []
    for layer_id in range(0, num_layer-1):
        source_nodes = list_of_layers[layer_id]
        dest_nodes = list_of_layers[layer_id + 1]

        for each_node in source_nodes:
            dst_node = dest_nodes[random.randint(0, len(dest_nodes)-1)]
            if (each_node, dst_node) not in edge_list:
                edge_list.append((each_node, dst_node))
                edge_count -= 1
                if edge_count < 0:
                    return []

        for each_node in dest_nodes:
            src_node = source_nodes[random.randint(0, len(source_nodes)-1)]
            if (src_node, each_node) not in edge_list:
                edge_list.append((src_node, each_node))
                edge_count -= 1
                if edge_count < 0:
                    return []
    distant_edge = 0
    while edge_count > 0:
        src_layer = random.randint(1, num_layer-2)
        dst_layer = random.randint(src_layer+1, num_layer - 1)

        if dst_layer <= src_layer:
            continue


        source_nodes = list_of_layers[src_layer]
        dest_nodes = list_of_layers[dst_layer]

        src_node = source_nodes[random.randint(0, len(source_nodes)-1)]
        dst_node = dest_nodes[random.randint(0, len(dest_nodes)-1)]
        if (src_node, dst_node) in edge_list:
            continue
        else:
            edge_list.append((src_node, dst_node))
            edge_count -= 1
            if dst_layer - src_layer > 1:
                distant_edge += 1

    for edge in edge_list:
        src_node, dst_node = edge
        """
        if dst_node == num_nodes - 1:
            edge_weight[edge] = node_weight[dst_node] * comp_to_comm_ratio
        else:
            edge_weight[edge] = node_weight[src_node] * comp_to_comm_ratio
        """
        modelgraph.add_edge(src_node, dst_node, weight = 0)


    return edge_list


def read_graph(fname, modelgraph, comm_to_comp_ratio):
    with open(fname, 'r') as f:
        node_count = int(f.readline())
        for _ in range(node_count):
            id, weight, utilization = f.readline().split()
            weight = float(weight)
            utilization = float(utilization)
            id = int(id)
            modelgraph.add_node(id, weight=weight)
            modelgraph.add_node(id, utilization=utilization)

        edge_count = int(f.readline())
        for i in range(edge_count):
            u, v, weight = f.readline().split()
            u, v = int(u), int(v)
            weight = float(weight)
            weight = (weight * comm_to_comp_ratio)/0.8
            weight = max(.1, weight)
            modelgraph.add_edge(u, v, weight = weight)



def write_graph(fname, modelgraph):
    with open(fname, 'w') as f:
        node_count = modelgraph.number_of_nodes()
        f.write(str(node_count) + "\n")
        for node in range(node_count):
            f.write(str(node) + " "  +"{:.6f}".format(modelgraph.nodes[node]['weight']) +" " +"{:.6f}".format(modelgraph.nodes[node]['utilization']) + "\n")

        edge_count = modelgraph.number_of_edges()
        f.write(str(edge_count) + "\n")

        for edge in modelgraph.edges:
            u, v = edge
            f.write(str(u) + " " + str(v) + " " + "{:.6f}".format(modelgraph.edges[u, v]['weight']) + str() + "\n")

    f.close()

def load_graph(graph_id, num_nodes, edge_count, num_layer, comm_to_comp_ratio):
    modelgraph = nx.DiGraph(latency=0)
    if experiment == "comm":
        file_name = str(experiment) + "_" + str(graph_id) + "_" + str(num_nodes) + "_" + str(
            edge_count) + "_" + str(num_layer) + ".txt"
    else:
        file_name = str(experiment) + "_" + str(graph_id) + "_" + str(num_nodes) + "_" + str(edge_count) + "_" + str(
            num_layer) + "_" + str(comm_to_comp_ratio) + ".txt"
    if os.path.isfile(file_name) == False:
        print("file not Exist!!")
        return

    read_graph(file_name, modelgraph, comm_to_comp_ratio)

    print(file_name)

    """
    seq_latency = 0
    for node in modelgraph.nodes:
        seq_latency += modelgraph.nodes[node]['weight']

    print("Seq latency")
    print(seq_latency)
    """

    return modelgraph, file_name

def generate_graph( num_nodes, edge_count, num_layer, max_nodes_per_layer, min_nodes_per_layer, comm_to_comp_ratio):
    for graph_id in range(graphcount): #graphcount
        time.sleep(1)
        modelgraph = nx.DiGraph(latency=0)
        file_name = str(experiment)+"_"+ str(graph_id) +"_"+ str(num_nodes) +"_"+ str(edge_count) +"_"+str(num_layer) +"_"+str(comm_to_comp_ratio)+".txt"
        if os.path.isfile(file_name):
            print("file Exist!!")
            continue
        list_of_layers = generate_nodes(modelgraph, num_layer, num_nodes, max_nodes_per_layer, min_nodes_per_layer)
        edge_list = generate_edges(modelgraph, list_of_layers, num_layer, edge_count, num_nodes)
        set_edge_weight(modelgraph, comm_to_comp_ratio, num_nodes)
        if edge_list == []:
            continue
        write_graph(file_name, modelgraph)

        print(file_name)
        seq_latency = 0
        for node in modelgraph.nodes:
            seq_latency += modelgraph.nodes[node]['weight']

        print("Seq latency")
        print(seq_latency)


        del modelgraph

# Press the green button in the gutter to run the script.

def set_edge_weight(modelgraph, comm_to_comp_ratio, num_nodes):
    for edge in modelgraph.edges:
        src_node, dst_node = edge
        edge_weight = 0
        if dst_node == num_nodes - 1:
            #edge_weight[edge] = node_weight[dst_node] * comp_to_comm_ratio
            edge_weight = node_weight[dst_node] * comm_to_comp_ratio
            modelgraph.edges[src_node, dst_node]['weight'] = 0
        else:
            #edge_weight[edge] = node_weight[src_node] * comp_to_comm_ratio
            edge_weight = node_weight[dst_node] * comm_to_comp_ratio
            modelgraph.edges[src_node, dst_node]['weight'] = 0


        modelgraph.edges[src_node, dst_node]['weight'] = edge_weight




def assign_gpu(gpugraph, subs_nodes, gpu):
    for node in subs_nodes:
        gpugraph.nodes[node]['gpu'] = gpu


def create_graph_files():
    for node_size in node_size_list:
        for edge_size in edge_size_list:
            for layer_size in layer_size_list:

                min_nodes_per_layer = math.ceil(layer_size * .5)
                max_nodes_per_layer = math.ceil(layer_size *10)  # int((1/min_max_ratio_to_avg) * avg_nodes_per_layer) + 1

                if min_nodes_per_layer * layer_size >= node_size:
                    continue

                for comm_to_comp_ratio in comm_to_comp_ratio_list:

                    generate_graph(node_size, edge_size, layer_size, max_nodes_per_layer,
                                            min_nodes_per_layer, comm_to_comp_ratio)



def generate_result():
    for node_size in node_size_list:
        for edge_size in edge_size_list:
            for layer_size in layer_size_list:
                num_nodes = node_size
                src_node = 0
                dst_node = node_size - 1
                edge_count = edge_size
                num_layer = layer_size  # math.ceil((node_size - 2) / avg_nodes_per_layer) + 2
                #comm_to_comp_ratio = 0.8
                for comm_to_comp_ratio in comm_to_comp_ratio_list:
                    for gid in range(graphcount):
                        for graph_id in range(gid, gid + 1):  #graphcount13
                            gc.collect()
                            if experiment == "comm":
                                file_name = str(experiment) + "_" + str(graph_id) + "_" + str(num_nodes) + "_" + str(
                                    edge_count) + "_" + str(num_layer) + ".txt"
                            else:
                                file_name = str(experiment) + "_" + str(graph_id) + "_" + str(num_nodes) + "_" + str(
                                    edge_count) + "_" + str(num_layer) + "_" + str(comm_to_comp_ratio) + ".txt"
                            if os.path.isfile(file_name) == False:
                                print("file not Exist!!")
                                continue

                            lpgraph, file_name = load_graph(graph_id, num_nodes, edge_count, num_layer,
                                                               comm_to_comp_ratio)
                            pdpgraph, file_name = load_graph(graph_id, num_nodes, edge_count, num_layer,
                                                            comm_to_comp_ratio)
                            iosgraph, file_name = load_graph(graph_id, num_nodes, edge_count, num_layer,
                                                             comm_to_comp_ratio)
                            if experiment == "gpu":
                                file_name = str(experiment) + "_" + str(graph_id) + "_" + str(num_nodes) + "_" + str(
                                    edge_count) + "_" + str(num_layer) + "_" + str(comm_to_comp_ratio) + ".txt"
                                file_name = "result_" + file_name
                            if experiment == "comm":
                                file_name = str(experiment) + "_" + str(graph_id) + "_" + str(num_nodes) + "_" + str(
                                    edge_count) + "_" + str(num_layer) + ".txt"
                                file_name = "result_" + file_name
                            if experiment == "struct":
                                file_name = str(experiment) + "_" + str(num_nodes) + "_" + str(
                                    edge_count) + "_" + str(num_layer) + "_" + str(comm_to_comp_ratio) + ".txt"
                                file_name = "result_" + file_name
                            elif experiment == "node":
                                file_name = str(experiment) + "_" + str(num_nodes) + "_" + str(
                                    edge_count) + "_" + str(num_layer) + "_" + str(comm_to_comp_ratio) + ".txt"
                                file_name = "result_" + file_name
                            elif experiment == "edge":
                                file_name = str(experiment) + "_" + str(num_nodes) + "_" + str(
                                    edge_count) + "_" + str(num_layer) + "_" + str(comm_to_comp_ratio) + ".txt"
                                file_name = "result_" + file_name

                            if experiment == "gpu" :
                                if os.path.isfile(file_name) == True:
                                    print("file Exist!!")
                                    continue

                            f = open(file_name, "a")
                            for ngpu in gpu_num_list:
                                gpus = []
                                nx.set_node_attributes(lpgraph, gpus, 'gpu')
                                nx.set_node_attributes(pdpgraph, gpus, 'gpu')

                                for id in range(0, num_nodes):
                                    lpgraph.nodes[id]['gpu'] = -1
                                    pdpgraph.nodes[id]['gpu'] = -1

                                seq_latency = 0
                                for node in lpgraph.nodes:
                                    seq_latency += lpgraph.nodes[node]['weight']

                                print("Seq latency")
                                print(seq_latency)

                                ios_latency = ios_optimizer.optimize(iosgraph)
                                print("IOS latency")
                                print(ios_latency)

                                #lpgraph = copy.deepcopy(modelgraph)

                                t0 = time.time()
                                optimizer_sim.hios_lp(lpgraph, src_node, dst_node, ngpu)
                                t1 = time.time()
                                hios_lp_time = t1 - t0
                                successors_list = {}
                                optimizer_sim.set_successors(lpgraph, successors_list)
                                lplatencygraph = optimizer_sim.graph_latency(lpgraph, src_node, dst_node, ngpu)

                                print("LP latency")
                                print(lplatencygraph)

                                intra_op_lplatency = optimizer_sim.intra_op_parallelism(lpgraph, successors_list,
                                                                                        window_size,
                                                                                        max_num_streams, ngpu, src_node,
                                                                                        dst_node)

                                print("Intra op LP latency")
                                print(intra_op_lplatency)

                                #pdpgraph = copy.deepcopy(modelgraph)
                                t0 = time.time()
                                optimizer_sim.hios_pdp(pdpgraph, src_node, dst_node, ngpu)

                                t1 = time.time()

                                hios_pdp_time = t1 - t0

                                pdplatencygraph = optimizer_sim.graph_latency(pdpgraph, src_node, dst_node, ngpu)

                                print("PDP latency")
                                print(pdplatencygraph)

                                intra_op_pdplatency = optimizer_sim.intra_op_parallelism(pdpgraph, successors_list,
                                                                                         window_size,
                                                                                         max_num_streams, ngpu, src_node,
                                                                                         dst_node)

                                print("Intra op PDP latency")
                                print(intra_op_pdplatency)

                                if experiment == "gpu":
                                    output = str(ngpu) + " " + str(seq_latency) + " " + str(ios_latency) + " "+ str(lplatencygraph) + " " + str(
                                        intra_op_lplatency) + " " + str(pdplatencygraph) + " " + str(intra_op_pdplatency)
                                if experiment == "comm":
                                    output = str(comm_to_comp_ratio) + " " + str(seq_latency) + " " + str(ios_latency)+ " " + str(lplatencygraph) + " " + str(
                                        intra_op_lplatency) + " " + str(pdplatencygraph) + " " + str(intra_op_pdplatency)
                                if experiment == "struct":
                                    output =  str(graph_id) + " " + str(ngpu) + " " + str(layer_size) + " " + str(seq_latency) + " " + str(ios_latency) + " " + str(lplatencygraph) + " " + str(
                                        intra_op_lplatency) + " " + str(pdplatencygraph) + " " + str(intra_op_pdplatency)
                                elif experiment == "node":
                                    output = str(graph_id) + " " + str(ngpu) + " " + str(node_size) + " " + str(edge_size) + " " + str(seq_latency) + " " + str(ios_latency) + " " + str(lplatencygraph) + " " + str(
                                        intra_op_lplatency) + " " + str(pdplatencygraph) + " " + str(intra_op_pdplatency)
                                elif experiment == "edge":
                                    output = str(graph_id) + " " + str(ngpu) +  " " + str(node_size) + " " + str(edge_size) + " " + str(seq_latency) + " " + str(ios_latency)+ " " + str(lplatencygraph) + " " + str(
                                        intra_op_lplatency) + " " + str(pdplatencygraph) + " " + str(intra_op_pdplatency)
                                f.write(str(output) + str("\n"))

                                f.flush()
                            f.close()

if __name__ == '__main__':
    create_graph_files()
    generate_result()


