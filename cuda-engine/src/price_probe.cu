#include "runtime.cuh"
#include <iostream>
#include <vector>
using namespace kagcuda;

struct Query { int item; int inv; };
__global__ void probe_kernel(DeviceTables tab, const Query* q, int n, int* out) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) out[i]=market_price(&tab,q[i].item,q[i].inv);
}
int main(){
    configure_device(); TablesOwner tables; std::vector<Query> q; int item,inv;
    while(std::cin>>item>>inv){if(item<0||item>=N_PRODUCTS)return 2;q.push_back({item,inv});}
    if(q.empty())return 0; Query* dq=nullptr; int* dout=nullptr;
    KAGCUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dq),q.size()*sizeof(Query)));
    KAGCUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dout),q.size()*sizeof(int)));
    KAGCUDA_CHECK(cudaMemcpy(dq,q.data(),q.size()*sizeof(Query),cudaMemcpyHostToDevice));
    probe_kernel<<<(q.size()+255)/256,256>>>(tables.dev,dq,static_cast<int>(q.size()),dout);
    KAGCUDA_CHECK(cudaGetLastError()); KAGCUDA_CHECK(cudaDeviceSynchronize());
    std::vector<int> out(q.size()); KAGCUDA_CHECK(cudaMemcpy(out.data(),dout,out.size()*sizeof(int),cudaMemcpyDeviceToHost));
    for(int v:out)std::cout<<v<<'\n'; cudaFree(dq);cudaFree(dout);return 0;
}
