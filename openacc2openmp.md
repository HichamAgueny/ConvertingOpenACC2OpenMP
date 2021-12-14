
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
*	Convert an OpenACC to OpenMP offloading using `clacc` compiler.


# Introduction

Heterogenous computing…
OpenACC & OpenMP are directive-based programming models for offloading compute regions 
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
$$f(x,y)=\frac{f(x_{i+1},y) + f(x_{i-1},y) + f(x,y_{i+1}) + f(x,y_{i-1})}{4}$$
```

The Eq. (xx) can be solved iteratively by defining some initial conditions that reflect the geometry of the problem at-hand. The iteration process can be done  either using ...or Jacobi algorithm. In this tutorial, we apt for the Jacobi algorithm due to its simplicity. The laplace equation is solved in a 2D-grid having 4096 points in both `x` and `y` directions. The compute code is written in *Fortran 90* and a *C*-based code can be found [here](https://documentation.sigma2.no/code_development/guides/openacc.html?highlight=openacc).

# Comparative study: OpenACC versus OpenMP

In the following we cover both the implementation of the OpenACC model to accelerate the Jacobi algorithm and the OpenMP offloading model in the aim of conducting a comparative experiment. The experiments are systematically performed with a fixed number of grid points as well as the number of iterations that ensures the convergence of the algorithm.

## Experiment on OpenACC offloading

We begin first by briefly describing the NVIDIA architecture. This is schematically illustrated in Fig. 1. Here one can see that each block 

We begin with our first OpenACC experiment, in which we evaluate the performance of different compute constructs and clauses and interprete their functionality. 


In Fig. 2 we show the performance of the three main compute constructs: **serial**, **kernels** and **parallel**. These directives determine a looped compute region to be executed on the GPU-device. More specifically, they tell the compiler to transfer the control of the looped region to the GPU-device and excute the region either in a serial way (i.e. by selecting **serial**) or as a sequence of operations (i.e. by choosing **kernels** and **parallel**). The use of **parallel** construct, however, offers some additional functionality. This manifests in the control of the execution on the device via the specification of the **gang**, **worker** and **vector**. Whereas the **kernels** construct .....

Under the utilization of these constructs, no parallelism is performed yet. This explains the observed low performance in Fig. 2 in the case of introducing the compute constructs compared to the CPU-serial code. Note that we should not confuse the construct **serial** with the CPU-serial case.         

Here the compiler makes the optimal choice of the numbers of gang, worker and vector that can be used to performe the parallelization. 

talk about the concept of different constructs and clause....interprete.

summarize them in a table.

Fig.1: speed up vs constructs, clauses.  

![Tux, the Linux mascot](/assets/images/tux.png)

For completness, we provide 


### Compiling and running OpenMP-program

We run our OpenACC-program on the `NVIDIA-GPU P100`. Compiling the code requires first loading the NVHPC module. This can be done in the [saga](https://documentation.sigma2.no/hpc_machines/saga.html) platform via
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
In the script above, the option `partition--accell` enables the access to the GPU, as already shown [here](https://documentation.sigma2.no/code_development/guides/openacc.html?highlight=openacc). One can also use the command `sinfo` to get information about which nodes are connected to the GPU. 

## Experiment on OpenMP offloading



### Compiling and running OpenMP-program


## Comparative study: OpenACC versus OpenMP


# Discussion on porting OpenACC to OpenMP


# Conclusion


Writing an efficient GPU-based program requires some basic knowledge of target architectures and how regions of a program is mapped into the target device. It thus functions as a benchmark for future advanced GPU-based parallel programming models. 


```bash
$ module load AOMP/13.0-2-gcccuda-2020a ncurses/6.2-GCCcore-9.3.0
```


```{eval-rst}
:download:`mandelbrot_serial.c <ompoffload/mandelbrot_serial.c>`

```

```{eval-rst}
.. literalinclude:: ompoffload/mandelbrot_serial.c
   :language: c++

```


```{eval-rst}
.. literalinclude:: ompoffload/mandelbrot_serial.c
   :language: c++
   :lines: 121-135
   :emphasize-lines: 8-9

```


```{note}
blabla
```

```bash
$ make omp
$ srun --account=<your project number> --time=10:00 --mem-per-cpu=1G ./omp 8k 10000
``` 

Summary of the execution times
==========================

Image Size | Iterations |OMP-Directive | CPU time in ms. | GPU time in ms.
-- | -- | -- | -- | --
1280x720 | 10,000 | -- | 10869.028 | -- 
1280x720 | 10,000 | `parallel` | 15025.200 | --
1280x720 | 10,000 | `parallel for` | 542.429 | --
1280x720 | 10,000 | `target`| -- | 147998.497
1280x720 | 10,000 | `target teams` | -- | 153735.213
1280x720 | 10,000 | `target teams parallel for` | -- | 2305.166
1280x720 | 10,000 | `target teams parallel for collapse` | -- | 2296.626
1280x720 | 10,000 | `target teams distribute parallel for collapse schedule` | -- | 143.434
8K	 | 10,000 | `parallel for` |  16722.152 | --
8k	 | 10,000 | `target teams distribute parallel for collapse schedule` | -- | 881.921

