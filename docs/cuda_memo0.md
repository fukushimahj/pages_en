cuda コーディングを学習する際のメモです。

# Hello, world!

まずは、プログラム学習の定番である、"Hello, Wrold !"の出力を通して、GPUでの実行を練習します。

```cpp
#include <cuda_runtime.h>
#include <stdio.h>


__global__ void print_hello(){
  printf("Hello, World!\n");
}


int main(void){

  print_hello<<<1,1,0>>>();

  cudaDeviceSynchronize();
  cudaDeviceReset();

}
```

上のコードは, Hello, world!を出力する関数をGPU上で実行するというものです。
ターミナル上でのコンパイルには、`nvcc hello.cu`とすると(cudaのコードは`.cu`の拡張子を使います)、実行ファイルが生成されます。
`print_hello`という関数の前の`__global__`という接頭文字は、GPU上で実行する関数につけるものであり、その関数の戻り値はvoidである必要があります。
また、このようにGPU上で実行される関数をカーネル(kernel)と読んだりもします。

今後、CPU側をホスト, GPU側をデバイスと呼ぶことにします。
ホストからデバイス上で, `print_hello`という関数を実行するには、C言語系統の通常の関数呼び出しと比べて、関数名と引数の間に`<<<1,1,0>>>`を挟むことで呼び出すことできます(数字の意味は下で)。
先頭のcuda特有の関数を使用するには`#include <cuda_runtime.h>`を導入する必要があります。

`cudaDeviceSynchronize()`はデバイスとの同期をとる関数で、通常はデイバス関数の終了をまってからホスト側の操作が実行されますが、
デバイス関数がコードの末尾にある場合は、その終了を待つ必要があり、上の場合では呼び出す必要があります。

# グリッド, ブロック, スレッド

cudaでは、デバイス関数を多数並列で実行する機構が備えられていて、ユーザーが自在に設定することが可能です。
Kernelはスレッドごとに実行される仕様になっています。
このスレッドの集合をブロック、さらにブロックの集まりをグリッドと呼びます([ここ](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)の図見ると想像しやすいです)。

ブロック内のスレッド数の設定は、以下のように行います。

```cpp
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void get_thread_id(){
  int i = threadIdx.x;
  printf("thread Id:%d \n", i);
}


int main(void){

  dim3 block(32);
  get_thread_id<<<1,block,0>>>();

  cudaDeviceSynchronize();
  cudaDeviceReset();

}

```

上のコードはスレッドを32つ持つブロックを１つもつグリッドを使用して、カーネルを実行するコードです。
同様に、グリッドも複数作成して計算することも可能です。

```cpp
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void get_thread_id(){
  int id_thread = threadIdx.x;
  int id_brock  = blockIdx.x;
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  printf("thread id:%d block id:%d i:%d \n", id_thread, id_brock, i);
}


int main(void){

  dim3 block(32);
  dim3 grid(8);
  get_thread_id<<<grid,block,0>>>();

  cudaDeviceSynchronize();
  cudaDeviceReset();

}

```

ここでは、32のスレッドを持つブロックを8つ生成するコードとなっています。
kernel内部では、`threadIdx.x`はスレッドがブロック内部でのID, `blockIdx.x`はグリッド内部でのIDを表します。
上の計算では、計256個のスレッドが立ち上がっているので、各々のIDは各ブロックが持っているスレッドの総数を示す`blockDim.x`を用いて、
コードにあるように`blockIdx.x*blockDim.x + threadIdx.x`。
似た方法で、ブロック、グリッドともに２次元・３次元構造として設定することも可能です。

# デバイスでのメモリ確保: cudaMalloc, cudaFree

次にcudaで、デバイス上のメモリを用いて配列を定義する方法についてみていきます。
メモリの確保は`cudaMalloc`、その開放は`cudaFree`という関数を用います。
それでは、具体的にその使用方法を見ていきましょう。

```
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void get_thread_id(int *d_array){
  int nthread = blockIdx.x*blockDim.x + threadIdx.x;
  d_array[nthread] = 2*nthread;
}

__global__ void test_output(int *d_array){
  int nthread = blockIdx.x*blockDim.x + threadIdx.x;
  printf("nthread:%d array=%d \n", nthread, d_array[nthread]);
}


int main(void){

  int n_block = 32;
  int n_grid  = 8;
  dim3 block(n_block);
  dim3 grid(n_grid);

  int *d_array;

  // メモリの確保
  cudaMalloc(&d_array, sizeof(int)*n_block*n_grid);

  get_thread_id<<<grid,block,0>>>(d_array);
  test_output<<<grid,block,0>>>(d_array);

  cudaFree(d_array);

  cudaDeviceSynchronize();
  cudaDeviceReset();

}
```

上のコードは`cudaMalloc`を使用してデバイス上に配列を確保しています。この際、配列のポインタは`d_array`というポインタ変数に格納されます。
ここで重要なことは、`d_arrray`という変数はホスト上で定義されていますが、そのポインタが示すメモリはデバイス上にあるという点です。
ホスト上からは通常、デバイス上のメモリの値を見ることはできないため（その逆も）、ホスト上で`d_array[0]`などの値を確認しようとするとエラーとなります。
また、確保した領域は自動では削除されないため、かならず`cudaFree`を実行する必要があります。

上では、整数型の配列を`cudaMalloc`によりメモリを確保しましたが、他の実数型なども`cudaMalloc(&d_array, sizeof(double))`などで確保することが可能です。

# デバイスメモリのコピー: cudaMemcpy

上記に述べたように、`cudaMalloc`で確保した配列の値は基本的にはデバイス上でしか確認することはできません。
このため、ホストとデバイス間で値を通信する関数`cudaMemcpy`が用意されています。

```
#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void get_thread_id(int *d_array){
  int nthread = blockIdx.x*blockDim.x + threadIdx.x;
  d_array[nthread] += 2;
}

__global__ void test_output(int *d_array){
  int nthread = blockIdx.x*blockDim.x + threadIdx.x;
  printf("nthread:%d array=%d \n", nthread, d_array[nthread]);
}


int main(void){

  int n_block = 32;
  int n_grid  = 8;
  dim3 block(n_block);
  dim3 grid(n_grid);

  int *d_array;
  int *h_array;

  // メモリの確保
  int n_array = n_block*n_grid;
  cudaMalloc(&d_array, sizeof(int)*n_array);
  h_array = new int[n_array];

  // 初期値代入
  for(int i=0; i<n_array; i++){
    h_array[i] = i;
  }

  // host からdeviceへ転送
  cudaMemcpy(d_array, h_array, sizeof(int)*n_array, cudaMemcpyHostToDevice);

  test_output<<<grid,block,0>>>(d_array);
  get_thread_id<<<grid,block,0>>>(d_array);

  // deviceからhostへ転送
  cudaMemcpy(h_array, d_array, sizeof(int)*n_array, cudaMemcpyDeviceToHost);

  for(int i=0; i<n_array; i++){
    std::cout << i << " " << h_array[i] << std::endl;
  }

  // メモリの削除
  cudaFree(d_array);
  delete h_array;

  cudaDeviceSynchronize();
  cudaDeviceReset();

}
```

`cudaMemcpy`では、受信側のポインタ、送信側のポインタの順で設定します。
また、ホストからデバイスに送るときには`cudaMemcpyHostToDevice`, デバイスからホストに送るときには`cudaMemcpyDeviceToHost`を指定する必要があります。

# ストリーム

cudaにはカーネル実行に関して、その実行順番を制御するStreamという機能が存在します。
実はこれまでの

## ストリームとイベントの作成

## ストリームの同期: cudaStreamSynchronize

## ストリームの待機: cudaStreamWaitEvent

## ストリームを使ったメモリ確保、コピー: cudaMemcpyAsync
