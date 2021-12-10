```{index} GPU; Intorduction to OpenMP-Offload with Nvidia-GPU; Intorduction to OpenMP-Offload with Nvidia-GPU; Intorduction to OpenMP-Offload with Nvidia-GPU
```

Introduction
============

This tutorial is designed for beginners in GPU-programming and who want to get familiar 
with available directives programming models, in particular, OpenACC and OpenMP offloading
models. The tutorial does not require any basic knowledge in GPU-programming. It aims to
provide an overview of these programming models, and to guide users towards the optimal
use of programming models. The tutorial will familiarise the users with the most needed
constructs and clauses via a practical tool based on solving numerically the Laplace
equation…and carrying out experiment on the performance of various constructs and clauses.

This tutorial ultimately aims at initiating user interest in GPU-programming; 
it is thus considered as a first step towards advanced and efficient parallel 
programming models. 

It does function as a benchmark for future advanced GPU-based programming models.

By the end of this tutorial, the user will be able to: 

*	Recognise the necessity of GPU-programming.
•	Interpret the compiler feedback messages.
•	Select and map regions of a code into a target device.
•	Use appropriate constructs and clauses on either programming model to offload compute regions to the GPU device.
•	Identify and assess the differences and similarities between OpenACC and OpenMP.
•	Convert an OpenACC to OpenMP offloading using `clacc` compiler.


OpenACC & OpenMP
================

Heterogenous computing…
OpenACC & OpenMP are directive-based programming models for offloading compute regions 
from CPU host to GPU devices. These models are referred to as Application Programming 
Interfaces (APIs), which here enable to communicate between two heterogenous systems 
(i.e. CPU host and GPU device in our case) and specifically enable offloading to target
 devices. The offloading process is controlled by a set of compiler directives, library 
 runtime routines as well as environment variables. These three components will be 
 addressed in the following for both models. Furthermore, differences and similarities 
 will be assessed in the aim of converting OpenACC to OpenMP.

Motivation: OpenACC is supported for offloading to NVIDIA device by the NVIDIA and PGI 
compilers. However, it is not supported on the AMD device by AMD compilers such as Clang. 
This hardware-dependency has limited the use of the OpenACC model. Although the GCC 
compiler supports OpenACC offloading to AMD target devices, the compilation process 
suffers from some weaknesses, such as low performance, issues with the diagnostics 
(see e.g.). This calls for an alternative that goes beyond the GCC compiler to support 
OpenACC offloading to AMD GPU-accelerators. On the other hand, OpenMP offloading to AMD 
device is supported by a set of compilers such as Clang and Cray, which are well-known to
 provide the highest. 

[Saga](https://documentation.sigma2.no/hpc_machines/saga.html#saga)

Computational model
===================


Experiment on OpenACC offloading
================================

Experiment on OpenMP offloading
===============================

Comparative study: OpenACC versus OpenMP
=======================================

Discussion on porting OpenACC to OpenMP
======================================

Conclusion
==========

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


```bash
$ make serial
$ srun --ntasks=1 --account=<your project number> --time=00:10:00 --mem-per-cpu=125M ./serial
```

We found, on [Saga](https://documentation.sigma2.no/hpc_machines/saga.html#saga), that executing 1280x720 pixels image with 10,000 iterations takes 11 seconds. 


Let’s build up on this and start applying OpenMP directives to transform our serial code into a parallel code on CPU. As we know, the potential parallelizable region is the code with the nested ‘for’ loop, as shown in the figure. 

```{eval-rst}
.. literalinclude:: ompoffload/mandelbrot_serial.c
   :language: c++
   :lines: 121-135
   :emphasize-lines: 6-11

```


```{eval-rst}
.. literalinclude:: ompoffload/omptarget.c
   :language: c++
   :lines: 2-14
   :emphasize-lines: 1,5

```

```{note}
Thread creation is an expensive operation, thread synchronization is also a point of concern which may deteriorate the overall performance. So, don’t let the compiler do all the optimizations for you; use OpenMp wisely.
```



```{eval-rst}
.. literalinclude:: ompoffload/omptarget.c
   :language: c++
   :lines: 17-27
   :emphasize-lines: 1

```

	

```{eval-rst}
.. literalinclude:: ompoffload/omptarget.c
   :language: c++
   :lines: 79-92
   :emphasize-lines: 1

```


At this point, we conslude that GPUs are optimised for throughput whereas CPUs are optimised for latency. Therefore, in order to benefit from using GPUs, we must give enough tasks to process per unit time on the GPU. In our code example, for instance, we care more about pixels per second than the latency of any particular pixel. 

In order to highlight the benefit of using GPUs, we consider an example, in which the size of our input image is increased. As previously, we rerun the code on the GPU as well as on the CPU.

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


Resources
=========

The complete code is available in the compressed format.

```{eval-rst}
:download:`mandelbrot_gpu.tar.gz <ompoffload/mandelbrot_gpu.tar.gz>`

```

One can download the given `tarball` file on his/her computer and copy it to `Saga` using `scp` command, as shown below.

```bash
$ scp <source_directory/mandelbrot_gpu.tar.gz> username@saga.sigma2.no:/cluster/home/<target_directory>
```
`source directory` should be the absolute path of the downloaded `tarball` on your computer, and the target direcetory should be the directory where you want to keep and uncompress the `tarball`.

To uncompress the `tarball` file, execute the following command on the terminal.

```bash
$ tar -zxvf mandelbrot_gpu.tar.gz
```



Makefile
========
For our sample code, we used `Makefile` to build. `Makefile` contains all the code that is needed to automate the boring task of transforming the source code into an excutable. One could argue; why not `batch` script? The advantage of `make` over the script is that one can specify the relationships between the elements of the program to `make`, and through this relationship together with timestamps it can figure out exactly what steps need to be repeated in order to produce the desired program each time. In short, it saves time by optimizing the build process.

The complete `Makefile` is listed here.

```{eval-rst}
.. literalinclude:: ompoffload/Makefile.txt
   :language: c++
   :lines: 1-15

```

In our `Makefile` we wrote some rules to compile and link our code. Each of the rules defines target. <Few other things, in this section, will be added by Hicham>


Compilation process
===================

We briefly describe the syntax of the compilation process with the Clang compiler to implement the OpenMP offload targeting NVIDIA-GPUs on the [Saga](https://documentation.sigma2.no/hpc_machines/saga.html#saga) platform. The syntax is given below:

clang -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target=nvptx64-nvidia-cuda -march=sm_60 gpu_code.c

Here the flag -fopenmp activates the OpenMP directives (i.e. #pragma omp). The option -fopenmp-targets is used to enable target offloading to NVIDIA-GPUs and the -Xopenmp-target flag enables options to be passed to the target offloading toolchain. Last, the flag -march specifies the name of the NVIDIA GPU architecture.


<todo: spelling check>
