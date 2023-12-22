#  Hierarchical Inter-Operator Scheduler (HIOS)
To accelerate CNN inference on large image input, a common strategy is to reduce image size, which results in lower accuracy. However, many applications such as remote sensing, and biomedical analysis where high-resolution images are quite common, need high accuracy as well as low latency. 

To lower the inference latency of the directed acyclic graph (DAG) structured DNN model, Inter-operator parallelism is an effective method when the input image size is small. However, since the computational workload of a CNN operator on a large input image is very high, it is hard to parallelize operators in a Single GPU. During our experiment, we found that, for large input images, parallelizing two identical convolution operators even increases the latency instead of lowering latency than the sequential execution.

In our work HIOS, we study the scope of inter-operator parallelism beyond a single GPU to accelerate DAG structured model for high resolution input images. The main bottleneck of inter-operator parallelism on multiple GPUs is communication latency. However, with the new technological advancements of high-speed interconnect (NVLink), the communication latency is not as high as before. This opens up new opportunities for inter-operator parallelism on multiple GPUs connected by high-speed interconnect (NVLink) and we design the Hierarchical Inter-Operator Scheduler (HIOS) to automatically schedule the execution of multiple operators in parallel in such platform.

## Scheduling Challenges:
<img align="right" width="400" height="350" src="https://github.com/SHUs-Lab/HIOS/assets/18241223/05296027-39d4-491f-97e1-b2705ee19b7e">

*  Spatially, mapping operators onto GPUs ​
    *  Allocate independent operators onto different GPUs ​
        *  To improve the **degree of parallelism**
    *  Assign dependent operators onto the same GPU ​
        *  To minimize **data transfer time** ​

*  Temporally, assigning operators in streams​
    *  avoid **hardware under-utilization** and **resource contention** in a single GPU​
    *  Maintain **dependencies** between operators on different GPUs. ​

​The interaction between spatial and temporal optimization makes the joint two-dimensional optimization problem very intractable.​




## Methodology:
HIOS works on two steps


1.  **Inter-GPU** inter-operator parallelization​
      *  Longest-path-based operator scheduling
      *  In this step, operators are mapped to GPUs
2.  **Intra-GPU** inter-operator parallelization​
      *  Strats when operators are already mapped onto a GPU in the previous step
      *  In this step, operators are mapped to Streams within a GPU
      *  On a sliding window,it finds operators to parallelize into Streams within a GPU
        
![image](https://github.com/SHUs-Lab/HIOS/assets/18241223/02316260-ea89-4969-b41f-3a3724b8ea96)

## Implementation

Please follow this section to run the HIOS on Multiple GPUs connected via NVLink. We conduct our experiment on two GPUs connected via NVLink in a server. This code base is not tested for more than two GPUs due to unavailability of more GPUs connected via NVLinK.

### Environment Prerequisites

- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) 10.0 or higher
- [cuDNN](https://developer.nvidia.com/cudnn) 7.6.5 or higher
- [CUPTI](https://developer.nvidia.com/cupti)
- [Cuda-aware MPI](https://www.open-mpi.org/faq/?category=runcuda)

### HIOS system Components
There are two major components of HIOS system
1.   Python based scheduler
      *   Takes input computation graph defined in python
      *   Parse the Graph into nodes and edges
      *   Profile each nodes, edges and sub graph by using execution engine (another component of HIOS)
      *   Using Profiling information, HIOS algorithm generate schedule
      *   The schedules generated by HIOS are in JSON format
2.   Execution engine,a Cuda-aware MPI application
      *   It takes input schedules of Nodes, edges and graph in JSON format
      *   It executes the schedules and return latency
      *   Before measuring latency, it does warm-up execution of the schedules multiple times
   
### Instruction to Run:
Please follow the following instruction to profile  Inception_v3, NASNet and Randwire computation graph 

1.   Download souce code from github
2.   Go to executor folder and set cuda and CUPTI library path in buildfile.sh
3.   Run sh buildfile.sh in terminal. It will generate the executanle file of execution engine( a Cuda-aware MPI application)
4.   For measuring latency for Inception_v3, NASNet and Randwire run sh run_expr_batchsize.sh on parent folder
5.   Input image size and batch size are configurable in run_expr_batchsize.sh file
6.   The schedule and optimization cost will be generated in output folder

For custom computation Graph

1.   Define your computation graph in main.py file in parent folder like following example
2.   Build the execution engine
3.   Run python main.py in parent folder
4.   The schedule and optimization cost will be generated in output folder

```
def sample_network():
    v = placeholder(output_shape=(1, 500, 500))
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
  
    return graph


def main():

    graph = sample_network()
    opt_type = "hios_lp"
    batch_size = 1
    height = 500
    width = 500
    ngpu = 2
    device = "v100"
    t1 = time.time()
    sys.setrecursionlimit(7000)
    optimize(height, width, graph, opt_type, batch_size=batch_size, warmup=2, number=6, repeat=6, ngpu =ngpu, device = device)
    t2 = time.time()
    optimization_cost = t2 - t1
    print("Optimization cost::")
    print(optimization_cost)

    dump_results(graph.name, height, width, optimization_cost, opt_type, batch_size=1, warmup=2, number=2, repeat=6, ngpu=ngpu, device = device)
```
## Results
The following figure shows inference latency for Inceptiont v3 and NASNet network for different image sizes
<br> <br>


![image](https://github.com/SHUs-Lab/HIOS/assets/18241223/d21e9588-9f8b-484a-b545-4f87ccf94c70)



<br><br>
The following figure shows performance gain for small and Large image input
<br><br>

![image](https://github.com/SHUs-Lab/HIOS/assets/18241223/42b63de3-63bd-48fb-a3ee-3990421052b3)

<br><br>
The following figure shows the optimization cost for Inception v3 and NASNet network
<br><br>

![image](https://github.com/SHUs-Lab/HIOS/assets/18241223/8bc52c52-5f0b-4522-9f56-e9be8ef9a3bf)

