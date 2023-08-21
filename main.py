
from utils.optimizer import optimize

from utils.models.inception_v3 import inception_v3
from utils.models.nasnet import *
from utils.models.randwire import *
from utils.models.squeezenet import *

from utils.models.common import Graph, Block, placeholder
from utils.models.common import conv2d, identity
import subprocess as sp

import logging
import argparse
import sys
import time
import shutil
logging.disable(logging.WARNING)

argparser = argparse.ArgumentParser()

argparser.add_argument('--device', type=str, required=False, default='A5500',
                       choices=['A5500', 'A40', 'v100', 'titanxp', '2080ti', 'cpu', 'k80'])
argparser.add_argument('--model', type=str, required=True,
                       choices=['inception_v3', 'randwire', 'nasnet', 'squeezenet'])
argparser.add_argument('--bs', type=int, required=False, default=1)

argparser.add_argument('--opt_type', type=str, required=True, choices=['hios_lp', 'hios_mr'])
argparser.add_argument('--r', type=int, default=3)
argparser.add_argument('--s', type=int, default=8)
argparser.add_argument('--warmup', type=int, required=False, default=2)
argparser.add_argument('--number', type=int, required=False, default=6)
argparser.add_argument('--repeat', type=int, required=False, default=6)
argparser.add_argument('--height', type=int, required=False, default=6)
argparser.add_argument('--width', type=int, required=False, default=6)
argparser.add_argument('--index', type=int, required=False, default=6)
argparser.add_argument('--ngpu', type=int, required=False, default=6)

args = argparser.parse_args()

name2model = {
    'inception_v3': inception_v3,
    'randwire': randwire_large,
    'nasnet': nasnet_large,
    'squeezenet': squeezenet
}
os.makedirs("./outputs", exist_ok=True)


def sample_network():
    v = placeholder(output_shape=(1, 5, 5))
    block = Block(enter_node=v.node)
    v1 = conv2d(block, inputs=[[v]], out_channels=1, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act='relu')
    v2 = conv2d(block, inputs=[[v]], out_channels=1, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act='relu')
    v3 = conv2d(block, inputs=[[v]], out_channels=1, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act='relu')
    v1 = conv2d(block, inputs=[[v1]], out_channels=1, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act='relu')
    out = identity(block, inputs=[[v1, v2],[v3]], is_exit=True)  # reduce v1, v2, and concat v3
    block1 = Block(enter_node=out.node)
    v11 = conv2d(block1, inputs=[[out]], out_channels=1, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act='relu')
    v21 = conv2d(block1, inputs=[[out]], out_channels=1, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act='relu')
    v31 = conv2d(block1, inputs=[[out]], out_channels=1, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act='relu')
    v11 = conv2d(block1, inputs=[[v11]], out_channels=1, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act='relu')
    out1 = identity(block1, inputs=[[v11, v21, v31]], is_exit=True)  # reduce v1, v2, and v3
    graph = Graph(name="demo", input=v.node, blocks=[block, block1])
    graph.init_weights()
    return graph


def main():
    height = args.height
    width = args.height

    print(height)
    batch_size = args.bs
    device = args.device
    ngpu = args.ngpu
    opt_type = args.opt_type
    t1 = time.time()
    sys.setrecursionlimit(7000)

    graph = name2model[args.model](args.height, args.width)

    optimize(height, width, graph, opt_type, batch_size=batch_size, warmup=2, number=6, repeat=6, ngpu =ngpu, device = device)
    t2 = time.time()
    optimization_cost = t2 - t1
    print("Optimization cost::")
    print(optimization_cost)

    dump_results(graph.name, height, width, optimization_cost, opt_type, batch_size=1, warmup=2, number=2, repeat=6, ngpu=ngpu, device = device)
    


    
def dump_results( name,  height, width, opt_cost, opt_type, batch_size, warmup, number, repeat, ngpu , device):
    if os.path.exists("conv2alg.txt"):
        os.remove("conv2alg.txt")

    profile_stage = 0

    name_nonpar = device + "_"+ opt_type + "_"+ name + "_nonpar_" + str(batch_size) + "_" + str(height) + "_" + str(width)
    name_par =device + "_"+ opt_type + "_"+ name + "_par_" + str(batch_size) + "_" + str(height) + "_" + str(width)
    print(name_nonpar)
    exe = sp.Popen(['mpirun', '-n', str(ngpu), './output', name_nonpar, str(batch_size), str(warmup), str(number), str(repeat),
                    str(profile_stage)], stdin=sp.PIPE, stdout=sp.PIPE)
    data = exe.communicate()
    print(data)
    data = data[0].split(b'\n')
    data = [float(v) for v in data[:-1]]

    print("latency:")
    print(np.mean(data))
    print(name_par)

    statement = "Only Inter Operator latency:\n"
    statement = statement + str(np.mean(data)) + "\n"

    if os.path.exists("conv2alg.txt"):
        os.remove("conv2alg.txt")

    exe = sp.Popen(['mpirun', '-n', str(ngpu), './output', name_par, str(batch_size), str(warmup), str(number), str(repeat),
                    str(profile_stage)], stdin=sp.PIPE, stdout=sp.PIPE)
    data = exe.communicate()
    print(data)
    data = data[0].split(b'\n')
    data = [float(v) for v in data[:-1]]


    print("With Intra Operator latency:")
    print(np.mean(data))
    statement = statement + "With Intra  Operator latency:\n"
    statement = statement + str(np.mean(data)) + "\n"
    statement = statement + "Optimization cost:" + "\n"
    results = device + "_" + opt_type + "_" + name+"_"+ "results_"+str(batch_size) + "_" + str(height) + "_" + str(width)
    statement = statement + str(opt_cost) + "\n"
    with open(f'{results}', 'w') as f:
        f.write(statement)

    if os.path.exists("conv2alg.txt"):
        os.remove("conv2alg.txt")
    shutil.move(name_nonpar, f'./outputs/{name_nonpar}')
    shutil.move(name_par, f'./outputs/{name_par}')
    shutil.move(results, f'./outputs/{results}')
    

if __name__ == '__main__':
    main()
