#include<bits/stdc++.h>
using namespace std;
using namespace std::chrono; 
long double W[505];
long double X[20005][505];
int Y[20005];

int main()
{
	srand ( time(NULL) );
	ifstream  trainfile ("train.txt");
	ifstream labelfile ("labels.txt");
	int n_samples=20000;
	int n_features=500;
	for(int i=0;i<n_samples;i++)
	{
		for(int j=0;j<n_features;j++)
			trainfile>>X[i][j];
	}
	for (int i=0;i<n_samples;i++)	X[i][500] = 0;
	n_features = 501;
	for(int i=0;i<n_samples;i++)
	{
		labelfile>>Y[i];
		if(Y[i]==0)
		{
			Y[i]=-1;
		}
	}
	auto start = high_resolution_clock::now(); 
	//Train
	int num_iters=500000;
	long double lambda=1.0;
	for(int iters=1;iters<=num_iters;iters++)
	{
		long double lr=1.0/(lambda*iters);
		// cout << "lr " << lr << endl;
		int rand_choice=rand()%n_samples;
		// cout << rand_choice << endl;
		long double pred_output=0;
		for(int i=0;i<n_features;i++)
		{
			pred_output+=W[i]*X[rand_choice][i];
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
	auto stop = high_resolution_clock::now(); 
	auto duration = duration_cast<microseconds>(stop - start); 
	cout<<"Train Time "<< ((double) duration.count()) / 1e6 << endl; 
	//Inference
	long double correct=0.0;
	for(int i=0;i<n_samples;i++)
	{
		long double val=0.0;
		for(int j=0;j<n_features;j++)
		{
			val+=W[j]*X[i][j];
		}
		
		if(val*Y[i]>=0)
			correct+=1;
	}
	cout<<correct/n_samples<<endl;
	return 0;
}