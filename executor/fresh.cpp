#include <iostream>
#include "json.h"
#include <fstream>
#include <sstream>
#include "json.h"
#include <iostream>
#include <mpi.h>
#include <fstream>

using namespace std;
extern void graph_latency(const char *graph_json, int batch_size, int warmup, int number, int repeat, int profile_stage_latency, float *results, float *stage_results, int myrank);
extern void stage_latency(const char *stage_json, const char *input_json, int batch_size, int warmup, int number, int repeat, int profile_stage_latency, float *results, float *stage_results);


int main(int argc, char *argv[]) 
{

	float results[2000] ={0};
	float stage_results[6000] ={0};
	int myrank = 0;	
	MPI_Status status;	
	std::string stage_json;
	std::string input_json;
	int num_stages;
	int repeat;
	int batch_size;
	int warmup;
	int number;
	std::string strJson= "graph_json";
	std::stringstream buffer;	


	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	
	double ans;
    	if(myrank == 0 and argc == 8){    			 
        	stage_latency(argv[1], argv[2], std::stoi(argv[3]), std::stoi(argv[4]), std::stoi(argv[5]), std::stoi(argv[6]), 1, results, stage_results);
        	for(int i = 0 ; i< std::stoi(argv[6]); i++){
		printf("%f\n",results[i]);
		}
    	}
    	
    	if(argc == 7){	
        	graph_latency(argv[1], std::stoi(argv[2]), std::stoi(argv[3]), std::stoi(argv[4]), std::stoi(argv[5]), 1, results, stage_results, myrank);
        	if (myrank == 0){
			for(int i = 0 ; i< std::stoi(argv[5]); i++){
			printf("%f\n",results[i]);
			}
		}
    	}

	MPI_Finalize();

	return 0;
}

