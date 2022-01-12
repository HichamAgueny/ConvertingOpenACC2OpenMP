
# GPU-programming: Porting OpenACC to OpenMP  

# Summary

This tutorial is designed for beginners in GPU-programming and who want to get familiar 
with available directives programming models, in particular, OpenACC and OpenMP offloading
models. The tutorial does not require any basic knowledge in GPU-programming. It aims at
providing an overview of these two models, and at guiding users towards their optimal
use. This tutorial ultimately aims at initiating user interest in GPU-programming. 
Specifically, it will familiarise the users with the most needed
constructs and clauses via a practical tool based on solving numerically the Laplace
equation, and carrying out experiments on their performance.

By the end of this tutorial, the user will be able to: 

*	Recognise the necessity of GPU-programming.
*	Recognise the GPU-architecture and its functionality.
*	Use appropriate constructs and clauses on either programming model to offload compute regions to the GPU device.
*	Select and map regions of a code into a target device.
*	Identify and assess the differences and similarities between the OpenACC and OpenMP offload features.
*	Get some highlights of the OpenACC-to-OpenMP translation using the `Clacc` compiler platform.


#### Table of Contents

- [Introduction](#Introduction)
- [Computational model](#Computational)
- [Experiment on OpenACC offloading](#ExperimentACC)
- [Experiment on OpenMP offloading](#ExperimentOMP)
- [Mapping OpenACC to OpenMP](#MappingACC2OMP)
- [Discussion on porting OpenACC to OpenMP](#DiscussionACC2OMP)


# Introduction

OpenACC and OpenMP are the most widely used programming models for heterogeneous computing on moderm HPC architectures. OpenACC was developed a decade ago and was designed for parallel programming of heterogenous CPU & GPU systems. Whereas OpenMP is historically known to be directed to shared-memory multi-core programming, and only recently has provided support for heterogenous systems. OpenACC and OpenMP are directive-based programming models for offloading compute regions 
from CPU host to GPU devices. These models are referred to as Application Programming 
Interfaces (APIs), which here enable to communicate between two heterogenous systems 
(i.e. CPU host and GPU device in our case) and specifically enable offloading to target
 devices. The offloading process is controlled by a set of compiler directives, library 
 runtime routines as well as environment variables. These three components will be 
 addressed in the following for both models. Furthermore, differences and similarities 
 will be assessed in the aim of converting OpenACC to OpenMP.

*Motivation:* NVIDIA-based Programming models are bounded by some barriers related to the GPU-architecture. 
Specifically, the models do not enable support on different devices such as AMD and [Intel](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-cpp-fortran-compiler-openmp/top.html) accelerators nor by 
the corresponding compilers, i.e. *Clang/Flang* and *Icx/Ifx*. Removing such barriers is one of the bottelneck 
in GPU-programming, which is the case for instance with OpenACC. The latter is one of 
the most popular programming model that requires a special focus in terms of support on all avaialble architectures.  

Although the GCC compiler supports the OpenACC offload feature, the compilation process, in general, 
suffers from some weaknesses, such as low performance, issues with the diagnostics ..etc. 
(see e.g.). This calls for an alternative that goes beyond the GCC compiler, and which ensures higher performance. On the other hand, the OpenMP offloading is supported on multiple devices by a set of compilers such as *Clang/Flang* and *Cray*, and *Icx/Ifx* which are well-known to
 provide the highest performance with respect to GCC. Therefore, converting OpenACC to OpenMP becomes a necessity to overcome the limitations of the OpenACC model set by the NVIDIA vendor. This has been the subject of a project, in which this documentation is inspired by....
 
 This tutorial is organised as follows. In sec. II, we provide a computational model, which is based on solving the Laplace equation. Section III is devoted to 
 the analysis of experiments performed using the OpenACC and OpenMP offload features. Section IV is directed to a discussion about converting OpenACC to OpenMP in the Clang/Flang compiler. In Sec. V we provide a short description on the compilation process. Finally, conclusions are given in Sec. VI.
 

# Computational model

We give a brief description of the numerical model used to slove the Laplace equation &Delta;f=0. For the sake of simplicity, we solve the eqution in a two-dimentional (2D) uniform grid according to

```math
$$\Delta f(x,y)=\frac{\partial^{2} f(x,y)}{\partial x^{2}} + \frac{\partial^{2} f(x,y)}{\partial y^{2}}=0$$.
```
Here we use the finite-difference method to approximate the partial derivative of the form `$\frac{\partial^{2} f(x)}{\partial x^{2}}$`. The spatial discretisation in the second-order scheme can be written as 

```math
$$\frac{\partial^{2} f(x,y)}{\partial^{2} x}=$\frac{f(x_{i+1},y) - 2f(x_{i},y) + f(x_{i-1},y)}{\Delta x}$
```

The Eq.(x) can be further simplified and takes the final form

```math
$$f(x_i,y_j)=\frac{f(x_{i+1},y) + f(x_{i-1},y) + f(x,y_{i+1}) + f(x,y_{i-1})}{4}$$
```

The Eq. (xx) can be solved iteratively by defining some initial conditions that reflect the geometry of the problem at-hand. The iteration process can be done  either using ...or Jacobi algorithm. In this tutorial, we apt for the Jacobi algorithm due to its simplicity. The laplace equation is solved in a 2D-grid having 8192 points in both `x` and `y` directions. The compute code is written in *Fortran 90* and a *C*-based code can be found [here](https://documentation.sigma2.no/code_development/guides/openacc.html?highlight=openacc). The serial code can be written as

```bash
       do while (max_err.gt.error.and.iter.le.max_iter)
         do j=2,ny-1
            do i=2,nx-1
               d2fx = f(i+1,j) + f(i-1,j)
               d2fy = f(i,j+1) + f(i,j-1)
               f_k(i,j) = 0.25*(d2fx + d2fy)
             enddo
          enddo

          max_err=0.

          do j=2,ny-1
            do i=2,nx-1
               max_err = max(dabs(f_k(i,j) - f(i,j)),max_err)
               f(i,j) = f_k(i,j)
            enddo
          enddo

          iter = iter +1
        enddo
```

# Comparative study: OpenACC versus OpenMP

In the following we cover both the implementation of the OpenACC model to accelerate the Jacobi algorithm and the OpenMP offloading model in the aim of conducting a comparative experiment. The experiments are systematically performed with a fixed number of grid points as well as the number of iterations that ensures the convergence of the algorithm.

## Experiment on OpenACC offloading

We begin first by illustarting the functionality of the OpenACC model in terms of parallelism, which is implemented via the directives **kernels** or **parallel loop**. The concept of parallelism functions via the generic directives: **gang**, **worker** and **vector** as schematically represented in [Fig. 1](#fig) (left-hand side). Here, the compiler initiates the parallelism by generating parallel gangs, in which each gang consists of a set of workers represented by a matrix of threads. This group of threads within a gang execute the same instruction (SIMT, Single Instruction Multiple Threads) via the vectorization process. In this scenario, a block of loops is assigned to each gang, which gets vectorized and executed redundantly by a group of threads.  

<img src="https://user-images.githubusercontent.com/95568317/149146826-e54d09ef-b428-466e-9f05-1bd2b95c3461.jpg" width="1000" height="300">

***Fig. 1.** GPU-architecture. Left-hand-side: software concept; right-hand-side: hardware aspect.*

In the hardware picture, a GPU-device consists of a block of Compute Units (CUs) (CU is a general term for a Streaming Multiprocessor, SM) each of which is organized as a matrix of Processing Elements (PEs) (PE is a general term for a CUDA core), as shown in Fig. 1 (right-hand side). As an example, the [NVIDIA P100 GPU-accelerators](https://images.nvidia.com/content/tesla/pdf/nvidia-tesla-p100-PCIe-datasheet.pdf) [see also [here](http://web.engr.oregonstate.edu/~mjb/cs575/Handouts/gpu101.2pp.pdf)] have 56 CUs (or 56 SMs) and each CU has 64 PEs (or 64 CUDA cores) with a total of 3584 PEs (i.e. 3584 FP32 cores/GPU or 1792 FP64 cores/GPU), while the [NVIDIA V100](https://images.nvidia.com/content/technologies/volta/pdf/volta-v100-datasheet-update-us-1165301-r5.pdf) has 80 CUs and each CU has 64 PEs with a total of 5120 PEs (5120 FP32/GPU or 2560 FP64/GPU), where FP32 and FP64 correspond to the single-precision Floating Point (FP) (i.e. 32 bit) and double precision (64 bit), respectively. 

The execution of the parallelism is mapped on the GPU-device in the following: each compute unit is associated to one gang of threads generated via the directive **gang**, in which a block of loops is assigned to. In addition, each block of loops is run on the processing element via the directive **vector**. In short, the role of these directives for processing the parallelism is summarized [here](https://www.openacc.org/sites/default/files/inline-files/OpenACC_Programming_Guide_0_0.pdf): 

- The **gang** clause has a role of partitioning the loop across gangs, and each gang has num_gang. 
- The **worker** clause enables the partition across workers, and each worker has num_worker.
- The **vector** clause enables the vectorization of the loop, and ....

num_gangs: Controls how many parallel gangs are created.
num_workers: Controls how many workers are created in each gang.
vector_length: Controls the vector length on each worker.
the vector length indicates how many data elements can be operated on

different gangs operate independently. 



We move now to discuss our OpenACC experiment, in which we evaluate the performance of different compute constructs and clauses and interprete their role. The GPU-based code is shown below. 


In Fig. 2 we show the performance of the three main compute constructs: **kernels** and **parallel**. These directives determine a looped compute region to be executed on the GPU-device. More specifically, they tell the compiler to transfer the control of the looped region to the GPU-device and excute the region in a sequence of operations. These two constructs differ in terms of mapping the parallelism into the device. Here, when specifying the **kernels** construct, the compiler perofmes the partition of the parallelism explicitly by choosing the optimal numbers of gangs, workers and the length of the vectors. Whereas, the use of the **parallel** construct offers some additional functionality: it allows the programmer to control the execution in the device by specifying the **gang**, **worker** and **vector**. This specification is optional, and thus it does not affect the functionality of the OpenACC code, except if the number associated to these clauses is specified. 

Here the compiler makes the optimal choice of the numbers of gang, worker and vector that can be used to performe the parallelization. 

One can specify the number of **gang**, **worker** and the length of the **vector** clause to parallelise a looped region. Accoriding to the OpenACC [specification](https://www.openacc.org/sites/default/files/inline-files/OpenACC_Programming_Guide_0_0.pdf) the order of these clauses should be enforced: the **gang** clause must determine the outermost loop and the **vector** clause should define the innermost parallel loop, while the **worker** clause should be in between these two clauses.

Under the utilization of these constructs, no parallelism is performed yet. This explains the low performance observed in Fig. 2 compared to the CPU-serial code.       

When using these constructs, the compiler will generate arrays that will be copied back and forth between the host and the device if they are not already present in the device. 

To be specific, the construct **parallel** indicates that the compiler will generate a number of parallel gangs to execute the looped region redundantly.
The clause **loop** tells the compiler to perform the parallelism for the specified looped region. These two directives can be combined in one directive.



In the scenario shown in Fig. 3 (left-hand side), only the directive **parallel loop** is introduced. Here the construct **parallel** indicates that the compiler will generate a number of parallel gangs to execute the looped region redundantly. When it is combined with the clause **loop**, the compiler will perform the parallelism over all the generated gangs for the specified looped region. In this case the compiler copies the data first to the device in the begining of the loop and then copies it back to the host at the end of the loop. This process repeats itself at each iteration, which makes it time consumming, thus rending the GPU-acceleration inefficient. To overcome this issue, one need to copy the data to the device only in the begining of the iteration and copy it back to the host at the end of the iteration, once the result converges. This can be done by introducing the data locality concepts via the directives **data**, **copyin** and **copyout**, as shown in Fig3 (right-hand side). Here, the clause **copyin** transfers the data to the GPU-device, while the clause **copyout** copies the data back to the host. Implementing this approach shows a vast improvement of the performance: the computing time get reduced by almost a factor of 53: it decreases from 111.2 s to 2.12 s. One can further tun the process by adding additional control, for instance, by introducing the **collapse** clause. Collapsing two or more loops into a single loop is beneficial for the compiler, as it allows to enhance the parallelism when mapping the looped region into the device. In addition, one can specify the clause **reduction**, which allows to compute the maximum of two elements in a parallel way. These additional clauses affect slightly the computing time: it goes from 2.12 s to 1.95 s.


Once can also introduce explicit control of the parallelism. This can be achieved by incorporating the clauses: `gang`, `worker` and `vector`. 

`todo: how to add colors in the bash code`

```bash
          **OpenACC without data locality**            |              **OpenACC with data locality**
                                                       |  !$acc data copyin(f) copyout(f_k)
   do while (max_err.gt.error.and.iter.le.max_iter)    |     do while (max_err.gt.error.and.iter.le.max_iter)
!$acc parallel loop gang worker vector                 |  !$acc parallel loop gang worker vector collapse(2)  
      do j=2,ny-1                                      |        do j=2,ny-1 
        do i=2,nx-1                                    |          do i=2,nx-1 
           d2fx = f(i+1,j) + f(i-1,j)                  |             d2fx = f(i+1,j) + f(i-1,j)
           d2fy = f(i,j+1) + f(i,j-1)                  |             d2fy = f(i,j+1) + f(i,j-1) 
           f_k(i,j) = 0.25*(d2fx + d2fy)               |             f_k(i,j) = 0.25*(d2fx + d2fy)
        enddo                                          |           enddo
      enddo                                            |         enddo
!$acc end parallel                                     |  !$acc end parallel
                                                       |
       max_err=0.                                      |          max_err=0.
                                                       |
!$acc parallel loop                                    |  !$acc parallel loop collapse(2) reduction(max:max_err)  
      do j=2,ny-1                                      |        do j=2,ny-1
        do i=2,nx-1                                    |          do i=2,nx-1
           max_err = max(dabs(f_k(i,j)-f(i,j)),max_err)|             max_err = max(dabs(f_k(i,j)-f(i,j)),max_err)
           f(i,j) = f_k(i,j)                           |             f(i,j) = f_k(i,j)
        enddo                                          |          enddo 
       enddo                                           |        enddo
!$acc end parallel                                     |  !$acc end parallel 
                                                       |
       iter = iter + 1                                 |        iter = iter + 1 
    enddo                                              |     enddo
                                                       |  !$acc end data
```

![Fig2](https://user-images.githubusercontent.com/95568317/148846246-39e4610e-1878-4812-8850-551b12c5e0b4.jpeg)

***Fig. 2.** Performance of different OpenACC directives.*


### Compiling and running OpenACC-program

We run our OpenACC-program on the `NVIDIA-GPU P100`. Compiling the code requires first loading the NVHPC module. This can be done on the [saga](https://documentation.sigma2.no/hpc_machines/saga.html) platform via
```bash
$ module load NVHPC/21.2
```
The syntax of the compilation process is
```bash
$ nvfortran -fast -acc -Minfo=accel -o laplace_acc.exe laplace_acc.f90
or
$ nvfortran ta=tesla:cc60 -Minfo=accel -o laplace_acc.exe laplace_acc.f90
```
where the flags `-acc` and `-⁠ta=[target]` enables OpenACC directives. The option `[target]` reflects the name of the GPU device. The latter is set at [tesla:cc60] for the device name Tesla P100 and [tesla:cc70] for the device tesla V100. This information can be viewed by running the command `pgaccelinfo`. Last, the flag option `-Minfo` enables the compiler to print out the feedback messages on optimizations and transformations.

The generated binary (i.e. `laplace_acc.exe`) can be lauched with the use of a Slurm script
```bash
#!/bin/bash
#SBATCH --account=nn1234k 
#SBATCH --job-name=laplace_acc
#SBATCH --partition=accel --gpus=1
#SBATCH --qos=devel
#SBATCH --time=00:01:00
#SBATCH --mem-per-cpu=2G
#SBATCH -o laplace_acc.out

#loading modules
module purge
module load NVHPC/21.2
 
$ srun ./laplace_acc.exe
```
In the script above, the option `partition--accell` enables the access to the GPU, as already shown [here](https://documentation.sigma2.no/code_development/guides/openacc.html?highlight=openacc). One can also use the command `sinfo` to get information about which nodes are connected to the GPUs. 

## Experiment on OpenMP offloading

In this section, we carry out an experiment on [OpenMP](https://www.openmp.org/wp-content/uploads/OpenMP-API-Specification-5-1.pdf) offloading by adopting the same scenario as in the previous section but with the use of a different GPU-architecture: AMD Mi100 accelerator. The functionality of OpenMP is somewhat similar to the one of OpenACC as schematically presented above. In the OpenMP concept, a block of loops is offloaded to a device via the construct **target**. A set of threads is then created on each compute unit (CU) by means of the directive **teams** to execute the offloaded region. Here, the offloaded region (i.e. block of loops) gets assigned to teams via the clause **distribute**, and get executed on the processing elements by means of the directive **parallel do simd**.  

These compute constructs define the parallelism concept in OpenMP as show in the application below. This implementation of the OpenMP application is presented for two cases: (i) OpenMP without introducing the data directive and (ii) OpenMP with the data directive. This Comparison allows us to identify the role of data management during the data-transfer between the host and a device, which in turn permits to get some insights into the performance of the OpenMP features. This performance is summarised in Fig.. The figure shows how the performance gets improved when introducing the data directive in the begining of the iteration. This indeed, allows to keep the data in the device during the iteration process and copy the data back only at the end of the iteration. By doing so, the performance is improved by almost a factor of 20: it goes from .. in the absence of the data directive to 2.1 s their presence. 


The performance is presented in Fig. 

copy arrays from the host to the device and back to the host at the end of the loop. This process will repeat itself at each iteration

The description of the constructs and clauses implemented in our application is provided in the table below.

An additonal application on OpenMP offloading can be found [here](https://documentation.sigma2.no/code_development/guides/ompoffload.html)

As in the previous section, we begin by briefly describing the AMD architecture. This is schematically illustrated in Fig. 1. Here one can see that each block 

..... 

```bash
          **OpenMP without data directive**            |                 **OpenMP with data directive**
                                                       |  !$omp target data map(to:f) map(from:f_k)
   do while (max_err.gt.error.and.iter.le.max_iter)    |     do while (max_err.gt.error.and.iter.le.max_iter)
!$omp target teams distribute parallel do simd         |  !$omp target teams distribute parallel do simd collapse(2) 
      map(to:f) map(from:f_k)                          |        schedule(static,1) 
      do j=2,ny-1                                      |        do j=2,ny-1 
        do i=2,nx-1                                    |          do i=2,nx-1 
           d2fx = f(i+1,j) + f(i-1,j)                  |             d2fx = f(i+1,j) + f(i-1,j)
           d2fy = f(i,j+1) + f(i,j-1)                  |             d2fy = f(i,j+1) + f(i,j-1) 
           f_k(i,j) = 0.25*(d2fx + d2fy)               |             f_k(i,j) = 0.25*(d2fx + d2fy)
        enddo                                          |           enddo
      enddo                                            |         enddo
!$omp end target teams distribute parallel do simd     |  !$omp end target teams distribute parallel do simd
                                                       |
       max_err=0.                                      |          max_err=0.
                                                       |
!$omp target teams distribute parallel do simd         |  !$omp target teams distribute parallel do simd collapse(2) 
      reduction(max:max_err)                           |         schedule(static,1) reduction(max:max_err) 
      do j=2,ny-1                                      |        do j=2,ny-1
        do i=2,nx-1                                    |          do i=2,nx-1
           max_err = max(dabs(f_k(i,j)-f(i,j)),max_err)|             max_err = max(dabs(f_k(i,j)-f(i,j)),max_err)
           f(i,j) = f_k(i,j)                           |             f(i,j) = f_k(i,j)
        enddo                                          |          enddo 
       enddo                                           |        enddo
!$omp end target teams distribute parallel do simd     |  !$omp end target teams distribute parallel do simd
                                                       |
       iter = iter + 1                                 |        iter = iter + 1 
    enddo                                              |     enddo
                                                       |  !$omp end target data
```

![Fig3](https://user-images.githubusercontent.com/95568317/148846520-6b1f8540-abf1-4953-9677-f72c347cc5cc.jpg)

***Fig. 3.** Performance of different OpenMP directives.*


### Compiling and running OpenMP-program

Our OpenMP benchmark test runs on AMD Mi100 accelerator. The syntax of the compilation process can be written in the following form (see [here](https://gitlab.sigma2.no/teams/gpu/planning/-/issues/41) for more details):

```bash
flang -fopenmp=libomp -fopenmp-targets=<target> -Xopenmp-target=<target> -march=<arch> laplace_omp.f90
```

The flag -fopenmp activates the OpenMP directives (i.e. !$omp [construct] in Fortran). The option -fopenmp-targets=<target> is used to enable the target offloading to GPU-accelerators and tells the Flang compiler to use <target>=amdgcn-amd-amdhsa as the AMD target. The -Xopenmp-target flag enables options to be passed to the target offloading toolchain. In addition, we need to specify the architecture of the GPU to be used. This is done via the flag -march=<arch>, where <arch> specifies the name of the GPU-architecture. This characteristic feature can be extracted from the machine via the command `rocminfo` for the AMD device. These commands provide a general view of the GPU-accelerator and additional related information. For instance, the AMD Mi100 accelerator architecture is specified by the flag -march=gfx908 amd-arch.   

> :memo: **Note:** The compilation process requires loading a module of AOMP, i.e. `AOMP/13.0-2-GCCcore-10.2.0` or a newer version.

 
## Mapping OpenACC to OpenMP

We compare our OpenACC application agains the OpenMP one. This is summarised below:

We present a direct comparison between the OpenACC and OpenMP offload features. This comparison is shown below and further illustrated in the table, in which we emphasise the meaning of some of the basic constructs and clauses underlying our application. Here, evaluating the behavior of OpenACC and OpenMP by one-to-one mapping is a key feature of the convertion procedure. A closer look at OpenACC and OpenMP codes reveals some similarities and differences in terms of constructs and clauses as summerised in the table. In particular, it shows that the syntax of both programming models is so similar, thus making the implemention of the translation procedure at the syntactic level straightforward. Therefore, such a Comparison is critical for determining the correct mappings to OpenMP.

We thus discuss this conversion procedure in the next section.
 
```bash
                    **OpenACC**                        |                    **OpenMP**
!$acc data copyin(f) copyout(f_k)                      |  !$omp target data map(to:f) map(from:f_k)
   do while (max_err.gt.error.and.iter.le.max_iter)    |     do while (max_err.gt.error.and.iter.le.max_iter)
!$acc parallel loop gang worker vector collapse(2)     |  !$omp target teams distribute parallel do simd collapse(2) 
                                                       |        schedule(static,1) 
      do j=2,ny-1                                      |        do j=2,ny-1 
        do i=2,nx-1                                    |          do i=2,nx-1 
           d2fx = f(i+1,j) + f(i-1,j)                  |             d2fx = f(i+1,j) + f(i-1,j)
           d2fy = f(i,j+1) + f(i,j-1)                  |             d2fy = f(i,j+1) + f(i,j-1) 
           f_k(i,j) = 0.25*(d2fx + d2fy)               |             f_k(i,j) = 0.25*(d2fx + d2fy)
        enddo                                          |           enddo
      enddo                                            |         enddo
!$acc end parallel                                     |  !$omp end target teams distribute parallel do simd
                                                       |
       max_err=0.                                      |          max_err=0.
                                                       |
!$acc parallel loop collapse(2) reduction(max:max_err) |  !$omp target teams distribute parallel do simd collapse(2) 
                                                       |         schedule(static,1) reduction(max:max_err) 
      do j=2,ny-1                                      |        do j=2,ny-1
        do i=2,nx-1                                    |          do i=2,nx-1
           max_err = max(dabs(f_k(i,j)-f(i,j)),max_err)|             max_err = max(dabs(f_k(i,j)-f(i,j)),max_err)
           f(i,j) = f_k(i,j)                           |             f(i,j) = f_k(i,j)
        enddo                                          |          enddo 
       enddo                                           |        enddo
!$acc end parallel                                     |  !$omp end target teams distribute parallel do simd
                                                       |
       iter = iter + 1                                 |        iter = iter + 1 
    enddo                                              |     enddo
!$acc end data                                         |  !$omp end target data
```

OpenACC | OpenMP | interpretation |
-- | -- | -- |
acc parallel | omp target teams | to execute a compute region on a device|
acc kernels  | No explicit counterpart   | - -|
acc parallel loop gang worker vector | omp target teams distribute parallel do simd | to parallelize a block of loops on a device|
acc data     | omp target data | to share data between multiple parallel regions in a device|
-- | -- | -- |
acc loop | omp teams distribute | to workshare for parallelism on a device|
acc loop gang | omp teams(num_teams) | to partition a loop accross gangs/teams|
acc loop worker | omp parallel simd | - - |
acc loop vector | omp parallel simd | - - |
num_gangs       | num_teams         | to control how many gangs/teams are created |
num_workers     | num_threads       | to control how many worker/threads are created in each gang/teams |
vector_length   | No counterpart    | to control how many data elements can be operated on |
-- | --  | -- |
acc create() | omp map(alloc:) | to allocate a memory for an array in a device|
acc copy()   | omp map(tofrom:) | to copy arrays from the host to a device and back to the host|
acc copyin() | omp map(to:) | to copy arrays to a device|
acc copyout()| omp map(from:) | to copy arrays from a device to the host|
-- | --  | -- |
acc reduction(operator:var)| omp reduction(operator:var) | to reduce the number of elements in an array to one value |
acc collapse(N)  | omp collapse(N)   | to collapse N nested loops into one loop |
No counterpart  | omp schedule(,)  | to schedule the work for each thread according to the collapsed loops|
private(var)         | private(var)          | to allocate a copy of the variable `var` on each gang/teams|
firstprivate    | firstprivate     | to allocate a copy of the variable `var` on each gang/teams and to initialise it with the value of the local thread| 

***Table 1.** Description of various directives and clauses: OpenACC vs OpenMP.*
 
Details about OpenACC and OpenMP library routines can be found, respectively, [here] and [here].

For completness, we provide 

Fig. depicts ....

# Discussion on porting OpenACC to OpenMP

For completness, we provide in this section some highlights of the available open-source tools that provide support of OpenACC in terms of compilers and hardwares. According to the work of [J. Vetter et al.](https://ieeexplore.ieee.org/document/8639349) and the [OpenACC website](https://www.openacc.org/tools), the only open-source OpenACC compiler that supports offloading to various GPU-archeterctures (i.e. NVIDIA, AMD, Intel...) is GCC 10. Recently, there has been an effort in developing an open-source compiler to complement the existing one, thus allowing to perform experiments on a broad range of architectures. The compiler is called [Clacc](https://ieeexplore.ieee.org/document/8639349) and its development is funded by Exascale Computing Project: [Clacc project](https://www.exascaleproject.org/highlight/clacc-an-open-source-openacc-compiler-and-source-code-translation-project/), which is described in the work of [J. Vetter et al.](https://ieeexplore.ieee.org/document/8639349). We thus focus here on providing some basic features of the Clacc compiler platform, without addressing the tehnical side of the compiler as it is considered to be beyond the scope of this tutorial.

Clacc is an open-source OpenACC compiler platform that has support for [Clang](https://clang.llvm.org/) and [LLVM](https://llvm.org/), and aims at facilitating GPU-programming in its broad use. The key behind the design of Clacc is based on converting OpenACC to OpenMP, taking advantage of the existing OpenMP debugging tools to be re-used for OpenACC. Clacc was designed to mimic the exact behavior of OpenMP as explicit as possible. The Clacc strategy for interpreting OpenACC is based on one-to-one mapping of [OpenACC directives to OpenMP directives](https://ieeexplore.ieee.org/document/8639349) as we have already shown in the table above.

Despite the new development of Clacc compiler platform, it suffers from some limitations, [mainly](https://ieeexplore.ieee.org/document/8639349): (i) translating OpenACC to OpenMP in Clang/Flang is currently supported only in C and Fortran but not yet in C++. (ii) Clacc has so far focused primarily on compute constructs, and thus lacks support of data-sharing between the CPU-host and a GPU-device. These limitations however are expected to be overcame in the near future. So far, Clacc has been tested and benchmarked against a series of different configuartions, and it is found to provide an acceptable GPU-performance, as stated [here](https://www.exascaleproject.org/highlight/clacc-an-open-source-openacc-compiler-and-source-code-translation-project/). Note that Clacc is publicly available [here](https://github.com/llvm-doe-org/llvm-project/wiki).

 

# Conclusion


In conclusion, we have presented an overview of the OpenACC and OpenMP offload features via an application based on solving the Laplace equation in a 2D uniform grid. This benchamrk application was used to experiment the peroformance of some of the basic diretives and clauses in order to highlight the gain behind the use of GPU-accelerators. The performance here was found to be improved by almost a factor of 20. We have also presented an evaluation of differences and similarties between OpenACC and OpenMP programming models. Furthermore, we have illustrated a one-to-one mapping of OpenACC directives to OpenMP directives in the aim of a conversion procedure between these two models. In this context, we have emphasised the recent development of the Clacc compiler platform aiming for such a convertion procedure, although the platfrom support is so far limited to C and lacks data-transfer in host-device. 

Last but not least, writing an efficient GPU-based program requires some basic knowledge of the target architectures and how regions of a program is mapped into a target device. This tutorial thus was designed to provide such basic knowledge in the aim of triggering the interest of users to GPU-programming. It thus functions as a benchmark for future advanced GPU-based parallel programming models. 





