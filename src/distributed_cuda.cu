#include<bits/stdc++.h>
using namespace std;
#define BLOCK_SIZE 256

__global__ void pegasos_per_thread(int num_samples, int num_features, double * W, double * X, double * Y, double lambda, int num_iters, double * random_arr,  int k) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int n_samples_per_thread = num_samples / k;
    for(int iters=1; iters<=num_iters; iters++)
    {
        double lr = 1.0 / (lambda * iters);
        int rand_choice = random_arr[iters];
        double pred_output = 0;
        for (int i=0; i<num_features; i++)
            pred_output += W[index * num_features + i] * X[(n_samples_per_thread * index + rand_choice) * num_features + i];
        if (Y[rand_choice] * pred_output >= 1.0) {
            for (int i=0; i<num_features; i++)
                W[index * num_features + i] = (1.0 - lr * lambda) * W[index * num_features + i];
        } else {
            for (int i=0; i<num_features; i++)
                W[index * num_features + i] = (1.0 - lr * lambda) * W[index * num_features + i] + (lr * Y[rand_choice]) * X[(n_samples_per_thread * index + rand_choice) * num_features + i];
        }
    }
}


int main() {
    srand(time(NULL));
    ifstream  trainfile ("train.txt");
    ifstream labelfile ("labels.txt");
    int n_samples=20000;
    int n_features=500;
    int k = 1000;
    int numBlocks = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int num_iters = 100;
    double lambda = 1.0;
    int n_samples_per_thread = n_samples / k;

    double *W, *X, *Y, *final_W, *random_arr;
    double *d_W, *d_X, *d_Y, *d_random_arr;

    W = (double *) malloc(k * n_features * sizeof(double));
    final_W = (double *) malloc(n_features * sizeof(double));
    X = (double *) malloc(n_samples * n_features * sizeof(double));
    Y = (double *) malloc(n_samples * sizeof(double));
    random_arr = (double *) malloc(num_iters * sizeof(double));

    cudaMalloc(&d_W, k * n_features * sizeof(double));
    cudaMalloc(&d_X, n_samples * n_features * sizeof(double));
    cudaMalloc(&d_Y, n_samples * sizeof(double));
    cudaMalloc(&d_random_arr, num_iters * sizeof(double));
    
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
    for (int i=0;i<k;i++)  {
        for (int j=0;j<n_features;j++)   W[i * n_features + j] = 0;
    }
    for (int i=0;i<n_features;i++)  final_W[i] = 0;
    for (int i=0;i<num_iters;i++)   random_arr[i] = rand() % n_samples_per_thread;   

    cudaMemcpy(d_X, X, n_samples * n_features * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, W, k * n_features * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, Y, n_samples * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_random_arr, random_arr, num_iters * sizeof(double), cudaMemcpyHostToDevice);
    pegasos_per_thread<<<numBlocks, BLOCK_SIZE>>>(n_samples, n_features, d_W, d_X, d_Y, lambda, num_iters, d_random_arr, k);
    cudaMemcpy(W, d_W, k * n_features * sizeof(double), cudaMemcpyDeviceToHost);
    
    // for (int i=0;i<k;i++)
        // for (int j=0;j<n_features;j++)
            // cout << W[i * n_features + j]  << " ";
    for (int i=0;i<k;i++) {
        for (int j=0;j<n_features;j++)  final_W[j] += (W[i * n_features + j]);
    }
    // cout << "\nFinalW\n";
    // for (int i=0;i<n_features;i++)  {final_W[i] /= k;   cout << final_W[i] << " ";}

    double correct = 0.0;
    for (int i=0;i<n_samples;i++) {
        double val = 0.0;
        for (int j=0;j<n_features;j++)
            val += final_W[j] * X[i * n_features + j];
        if (val * Y[i] >= 0)
            correct += 1;
    }
    cout << "Correct " << correct << endl;
    printf("Accuracy %lf\n", correct / n_samples);
    return 0;
}
