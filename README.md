#  Hierarchical Inter-Operator Scheduler (HIOS)
To accelerate CNN inference on large image input, a common strategy is to reduce image size, which results in lower accuracy. However, many applications such as remote sensing, and biomedical analysis where high-resolution images are quite common, need high accuracy as well as low latency. 

To lower the inference latency of the directed acyclic graph (DAG) structured DNN model, Inter-operator parallelism is an effective method when the input image size is small. However, since the computational workload of a CNN operator on a large input image is very high, it is hard to parallelize operators in a Single GPU. During our experiment, we found that, for large input images, parallelizing two identical convolution operators even increases the latency instead of lowering latency than the sequential execution.

In our work HIOS, we study the scope of inter-operator parallelism beyond a single GPU. The main bottleneck of inter-operator parallelism on multiple GPUs is communication latency. However, with the new technological advancements of high-speed interconnect (NVLink), the communication latency is not as high as before. This opens up new opportunities for inter-operator parallelism on multiple GPUs connected by high-speed interconnect (NVLink) and we design the Hierarchical Inter-Operator Scheduler (HIOS) to automatically schedule the execution of multiple operators in parallel in such platform.

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
      *  On a sliding window, finds operators to parallelize into Streams within a GPU
        
![image](https://github.com/SHUs-Lab/HIOS/assets/18241223/02316260-ea89-4969-b41f-3a3724b8ea96)

## System Environment

Please follow this section to run the HIOS on Multiple GPUs connected via NVLink

### Prerequisites

- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) 10.0 or higher
- [cuDNN](https://developer.nvidia.com/cudnn) 7.6.5 or higher


### Instruction to Run:
1. Go to executor folder
2. set path in buildfile.sh
3. run sh buildfile.sh
4. Go to parent folder
5. run sh run_expr_batchsize.sh
