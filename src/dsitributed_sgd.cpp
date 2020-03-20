#include<bits/stdc++.h>
#include <sys/time.h>
#include <mpi.h>
using namespace std;
using namespace std::chrono; 
double W[505];
double X[20005][505];
double allX[20005][505];
int Y[20005];
int allY[20005];
double finalW[505];
int main()
{
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    int num_per_procs;
    if(world_rank == 0)
    {
    	double t1, t2; 
    	ifstream  trainfile ("train.txt");
		ifstream labelfile ("labels.txt");
		ifstream num_samples ("samples.txt")
		ifstream num_features ("features.txt")
		int n_samples;
		int n_features;
		num_features >> n_features;
		num_samples >> n_samples;
		num_per_procs=n_samples/world_size;
		for(int i=0;i<n_samples;i++)
		{
			for(int j=0;j<n_features;j++)
				trainfile>>allX[i][j];
		}
		for(int i=0;i<n_samples;i++)
		{
			labelfile>>allY[i];
			if(allY[i]==0)
			{
				allY[i]=-1;
			}
		}
    }
    MPI_Bcast(num_per_procs,1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Scatter(allX,505*num_per_procs,MPI_DOUBLE,X,505*num_per_procs,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Scatter(allX,505*num_per_procs,MPI_DOUBLE,X,505*num_per_procs,MPI_DOUBLE,0,MPI_COMM_WORLD);
    
	if(world_rank==0)
	{
		t1=MPI_Wtime(); 
	}
	
	double lambda=1.0;
	for(int iters=0;iters<num_per_procs;iters++)
	{
		double lr=1.0/(lambda*(iters+1));

		int rand_choice=iters;
		//cout<<rand_choice<<endl;
		double pred_output=0;

		{
			for(int i=0;i<n_features;i++)
			{
				pred_output+=W[i]*X[rand_choice][i];
			}
		}
		if( Y[rand_choice]*pred_output >= 1.0)
		{
			for(int i=0;i<n_features;i++)
			{
				W[i]=(1.0 - lr*lambda)*W[i];
			}
		}
		else
		{
			for(int i=0;i<n_features;i++)
			{
				W[i]=(1.0 - lr*lambda)*W[i] + (lr*Y[rand_choice])*X[rand_choice][i];
			}
		}
	}
	if(world_rank==0)
	{
		t2 = MPI_Wtime(); 
	}
	MPI_Gather(W,505,MPI_DOUBLE,finalW,505,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Reduce(W, finalW, 1, MPI_DOUBLE, MPI_SUM, 0,MPI_COMM_WORLD);

	
	//Inference
	if(world_rank==0)
	{
		for(int i=0;i<n_features;i++)
		{
			finalW[i]/=world_size;
		}
		double correct=0.0;
		for(int i=0;i<n_samples;i++)
		{
			double val=0.0;
			for(int j=0;j<n_features;j++)
			{
				val+=finalW[j]*allX[i][j];
			}
			
			if(val*Y[i]>=0)
				correct+=1;
		}
		cout << correct << " " << n_samples << endl;;
		cout<<correct/n_samples<<endl;
		cout<<"Time Elapsed "<<t2-t1<<endl;
	}
	MPI_Finalize();
	return 0;
}