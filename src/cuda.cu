#include<bits/stdc++.h>
using namespace std;
#define BLOCK_SIZE 256

__global__ void type1(int n, double lr, double lambda, double * W) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i=index;i<n;i+=stride) {
        W[i] = (1.0 - lr* lambda) * W[i];
    }
}

__global__ void type2(int n, double lr, double lambda, double * W, int rand_choice, double * X, double * Y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i=index;i<n;i+=stride) {
        W[i] = (1.0 - lr* lambda) * W[i] + (lr * Y[rand_choice])*X[rand_choice * n + i];
    }
}

__device__ double atomicAddDouble(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
	    old = atomicCAS(address_as_ull, assumed,
        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ void dot(int n, double * W, double *X, int rand_choice, double * res) {
   __shared__ double temp[BLOCK_SIZE];
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   if (index < n)   temp[threadIdx.x] = W[index] * X[rand_choice * n + index];
   else             temp[threadIdx.x] = 0;
   __syncthreads();
   if (threadIdx.x == 0){
       double sum = 0;
       for (int i=0;i<BLOCK_SIZE;i++) sum += temp[i];
       atomicAddDouble(res, sum);
   }
}


int main() {
    srand(time(NULL));
    ifstream  trainfile ("train.txt");
    ifstream labelfile ("labels.txt");
    int n_samples=200;
    int n_features=50000;

    double *W, *X, *Y, *res;
    double *d_W, *d_X, *d_Y, *d_res;
    cudaEvent_t start, stop;
    float elapsedTime;
    
    W = (double *) malloc(n_features * sizeof(double));
    X = (double *) malloc(n_samples * n_features * sizeof(double));
    Y = (double *) malloc(n_samples * sizeof(double));
    res = (double *) malloc(sizeof(double));

    cudaMalloc(&d_W, n_features * sizeof(double));
    cudaMalloc(&d_X, n_samples * n_features * sizeof(double));
    cudaMalloc(&d_Y, n_samples * sizeof(double));
    cudaMalloc(&d_res, sizeof(double));
    
    for (int i=0;i<n_samples;i++) {
		for (int j=0;j<n_features;j++)
			trainfile >> X[i*n_features + j];
    }
	for (int i=0;i<n_samples;i++) {
		labelfile >> Y[i];
		if (Y[i] == 0) {
			Y[i] = -1;
		}
	}
    for (int i=0;i<n_features;i++)  W[i] = 0;

    cudaMemcpy(d_X, X, n_samples * n_features * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, W, n_features * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, Y, n_samples * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, W, n_features * sizeof(double), cudaMemcpyHostToDevice);

    int num_iters = 100;
    double lambda = 1.0;
    cudaEventCreate(&start);
    cudaEventRecord(start,0);
    for (int iters=1;iters<=num_iters;iters++) {
    	int numBlocks = (n_features + BLOCK_SIZE - 1) / BLOCK_SIZE;
        double lr = 1.0 / (lambda * iters);
        int rand_choice = rand() % n_samples;
        cout << rand_choice << endl;
 	   	*res = 0;
 	    cudaMemcpy(d_res, res, sizeof(double), cudaMemcpyHostToDevice);
 	    dot<<<numBlocks, BLOCK_SIZE>>>(n_features, d_W, d_X, rand_choice, d_res);
 	    cudaMemcpy(res, d_res, sizeof(double), cudaMemcpyDeviceToHost);
	    if (Y[rand_choice] * res[0] >= 1.0)
	        type1<<<numBlocks, BLOCK_SIZE>>>(n_features, lr, lambda, d_W);
	    else
	        type2<<<numBlocks, BLOCK_SIZE>>>(n_features, lr, lambda, d_W, rand_choice, d_X, d_Y);
        cudaMemcpy(W, d_W, n_features * sizeof(double), cudaMemcpyDeviceToHost);
    }
    cudaMemcpy(W, d_W, n_features * sizeof(double), cudaMemcpyDeviceToHost);
    
    cudaEventCreate(&stop);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start,stop);

    cout << "Train time " << elapsedTime << endl;
    double correct = 0.0;
    for (int i=0;i<n_samples;i++) {
        double val = 0.0;
        for (int j=0;j<n_features;j++)
            val += W[j] * X[i * n_features + j];
        if (val * Y[i] >= 0)
            correct += 1;
    }
    cout << "Correct " << correct << endl;
    printf("Accuracy %lf\n", correct / n_samples);
    return 0;
}
