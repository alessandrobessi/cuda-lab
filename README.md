# cuda-lab
Playing with CUDA and GPUs in Google Colab.

```
!git clone https://github.com/alessandrobessi/cuda-lab.git
!chmod 755 cuda-lab/INSTALL.sh
!./cuda-lab/INSTALL.sh

import os
os.environ['PATH'] += ':/usr/local/cuda/bin'

!nvcc cuda-lab/add.cu -o add -Wno-deprecated-gpu-targets
!nvprof ./add

%%writefile snippet.cu
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
...

!nvcc snippet.cu -o snippet -Wno-deprecated-gpu-targets
!nvprof ./snippet
```
