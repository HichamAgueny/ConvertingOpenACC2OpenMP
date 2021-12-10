
GPU-programming: Porting OpenACC to OpenMP
=========================================    

$\sum_{n=1}^{10} n^2$

```math
$$\frac{\partial^{2} f(x,y)}{\partial^{2} x} + \frac{\partial^{2} f(x,y)}{\partial^{2} y}=0$$

```

$ \sum_{\forall i}{x_i^{2}} $

Some display math:
```math
e^{i\pi} + 1 = 0
```
and some inline math, $`a^2 + b^2 = c^2`$.

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
*	Interpret the compiler feedback messages.
*	Select and map regions of a code into a target device.
*	Use appropriate constructs and clauses on either programming model to offload compute regions to the GPU device.
*	Identify and assess the differences and similarities between OpenACC and OpenMP.
*	Convert an OpenACC to OpenMP offloading using `clacc` compiler.


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

`Motivation:` NVIDIA-based Programming models are bounded by some barriers related to the GPU-architecture. 
Specifically, the models do not enable support on different devices such as AMD and Intel accelerators nor by 
the corresponding compilers, i.e. Clang/Flang and Icx/Ifx. Removing such barriers is one of the bottelneck 
in GPU-programming, which is the case for instance with OpenACC. The latter is one of 
the most popular programming model that requires a special focus in terms of support on all avaialble architectures.  

Although the GCC compiler supports the OpenACC offload feature, the compilation process, in general, 
suffers from some weaknesses, such as low performance, issues with the diagnostics ..etc. 
(see e.g.). This calls for an alternative that goes beyond the GCC compiler, and which ensures higher performance. On the other hand, the OpenMP offloading is supported on multiple devices by a set of compilers such as Clang and Cray, and Icx/Ifx which are well-known to
 provide the highest performance with respect to GCC. Therefore, converting OpenACC to OpenMP becomes a necessity to overcome the limitations of the OpenACC model set by the NVIDIA vendor. This has been the subject of a project, in which this documentation is inspired by....

 
 [Intel](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-cpp-fortran-compiler-openmp/top.html#Intel)

[Saga](https://documentation.sigma2.no/hpc_machines/saga.html#saga)

Computational model
===================
We give a brief description of the numerical model used to slove the Laplace equation nabla f=0. For the sake of simplicity, we solve the eqution in a two-dimentional (2D) uniform grid. Here we use the finite-difference method to approximate `nabla f`. The spatial discretisation in the second-order scheme can be written as 

h<sub>&theta;</sub>(x) = &theta;<sub>o</sub> x + &theta;<sub>1</sub>x

\begin{equation}
\Big[ H_0 + H_I(t) - i\frac{\partial}{\partial t}\big]|\psi(t) \rangle=0,
\end{equation}

The Eq.(x) can be further simplified and takes the final form
```
$\Big[ H_0 + H_I(t) - i\frac{\partial}{\partial t}\big]|\psi(t) \rangle=0$

```

The Eq. (xx) can be solved iteratively by defining some initial conditions that reflect the geometry of the problem at-hand. The iteration process can be done  either using or Jacobi algorithm. In this tutorial, we apt for the Jacobi algorithm due to its simplicity. In the following we implement the OpenACC model with the use of the `Fortran` language to translate the Jacobi algorithm and perform an experiment. In the second section, we adopt the same scenario but with the OpenMP offloading model in the aim of conducting a comparative experiment.

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

