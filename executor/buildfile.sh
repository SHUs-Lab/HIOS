export PATH=/usr/local/cuda/extras/CUPTI/lib64:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/include:$LD_LIBRARY_PATH
gcc -c -o utils.o utils.cc
g++  -c  jsoncpp.cpp json.h

nvcc ops.cu -c -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -L/usr/local/cuda/extras/CUPTI/lib64  -I/usr/local/cuda/extras/CUPTI/include   -lcuda -lcupti  -lcublasLt -lcudart -lcublas -lcudnn -lstdc++ -lpthread -lcuda


mpicc -c ios_runtime.cpp  -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -L/usr/local/cuda/extras/CUPTI/lib64  -I/usr/local/cuda/extras/CUPTI/include   -lcuda -lcupti  -lcublasLt -lcudart -lcublas -lcudnn -lstdc++ -lpthread -lcuda
mpicc fresh.cpp -c -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -L/usr/local/cuda/extras/CUPTI/lib64  -I/usr/local/cuda/extras/CUPTI/include   -lcuda -lcupti  -lcublasLt -lcudart -lcublas -lcudnn -lstdc++ -lpthread -lcuda
mpicc  *.o  -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -L/usr/local/cuda/extras/CUPTI/lib64  -I/usr/local/cuda/extras/CUPTI/include   -lcuda -lcupti  -lcublasLt -lcudart -lcublas -lcudnn -lstdc++ -lpthread -lcuda -o output

mv output ../
