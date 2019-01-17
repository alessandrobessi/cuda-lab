# cuda-lab
Playing with CUDA and GPUs.

```
!git clone https://github.com/alessandrobessi/cuda-lab.git
!chmod 755 cuda-lab/INSTALL.sh
!./cuda-lab/INSTALL.sh

import os
os.environ['PATH'] += ':/usr/local/cuda/bin'

%%writefile example.cu
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
...

!nvcc example.cu -o example -Wno-deprecated-gpu-targets
!nvprof ./example
```
