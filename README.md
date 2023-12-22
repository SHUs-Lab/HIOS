#  Hierarchical Inter-Operator Scheduler
To accelerate CNN inference on large image input, a common strategy is to reduce image size, which results in lower accuracy. However, many applications such as remote sensing, and biomedical analysis where high-resolution images are quite common, need high accuracy as well as low latency. 

To lower the inference latency of the directed acyclic graph (DAG) structured DNN model, Inter-operator parallelism is an effective method when the input image size is small. However, since the computational workload of a CNN operator on a large input image is very high, it is hard to parallelize operators in a Single GPU. During our experiment, we found that, for large input images, parallelizing two identical convolution operators even increases the latency instead of lowering latency than the sequential execution.

In our work HIOS, we study the scope of inter-operator parallelism beyond a single GPU. The main bottleneck of inter-operator parallelism on multiple GPUs is communication latency. However, with the new technological advancements of high-speed interconnect (NVLink), the communication latency is not as high as before. This opens up new opportunities for inter-operator parallelism on multiple GPUs connected by high-speed interconnect (NVLink) and we design the Hierarchical Inter-Operator Scheduler (HIOS) to automatically schedule the execution of multiple operators in parallel in such platform.

## Instruction to Run:
1. Go to executor folder
2. set path in buildfile.sh
3. run sh buildfile.sh
4. Go to parent folder
5. run sh run_expr_batchsize.sh
