from utils.ir import Graph
import json
import os
from .ctypes_utils import *
import subprocess as sp




def graph_latency(graph: Graph, batch_size, warmup, number, repeat, profile_stage, ngpu):

    graph_json = json.dumps(graph.export_config())
    if os.path.exists("conv2alg.txt"):
        os.remove("conv2alg.txt")
    
    with open("graph_json", "w") as outfile:
        outfile.write(graph_json)
    try:
        exe = sp.Popen(['mpirun', '-n', str(ngpu), './output', str("graph_json"), str(batch_size), str(warmup), str(number), str(repeat), str(profile_stage)], stdin=sp.PIPE, stdout=sp.PIPE)
        data = exe.communicate()

        data = data[0].split(b'\n')
        data = [float(v) for v in data[:-1]]
    except:
        print("inside except")
        with open("problem_json", "w") as outfile:
            outfile.write(graph_json)
        data = [float(1000) for _ in range(repeat)]
    return data


def stage_latency(stage: List[List[Node]], batch_size, warmup, number, repeat, profile_stage, ngpu):


    if os.path.exists("conv2alg.txt"):
        os.remove("conv2alg.txt")
    stage_seqs = []
    input_nodes = []
    for seq in stage:
        seq_nodes = []
        for node in seq:
            seq_nodes.append(node.export_config())
        stage_seqs.append(seq_nodes)
        input_nodes.extend(value.node for nd in seq for term in nd.inputs for value in term)
    stage_node_names = [nd.name for seq in stage for nd in seq]
    input_nodes: List[Node] = list(ind for ind in dict.fromkeys(input_nodes) if ind.name not in stage_node_names)

    str_stage_seqs = json.dumps(stage_seqs)
    str_input_seqs = json.dumps({nd.name: nd.output_shape for nd in input_nodes})

    exe = sp.Popen(['mpirun', '-n', str(ngpu), './output', str_stage_seqs, str_input_seqs, str(batch_size), str(warmup), str(number), str(repeat), str(profile_stage)], stdin=sp.PIPE, stdout=sp.PIPE)
    data = exe.communicate()

    data = data[0].split(b'\n')
    data = [float(v) for v in data[:-1]]

    return data



