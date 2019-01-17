# cuda-lab
Playing with CUDA and GPUs in Google Colab.

## Usage
1. Open a Colab notebook: https://colab.research.google.com/
2. Create a new Python 3 notebook
3. Change runtime type using GPU as hardware accelerator
4. Git clone this repository:
```
!git clone https://github.com/alessandrobessi/cuda-lab.git
```
5. Change permissions:
```
!chmod 755 cuda-lab/INSTALL.sh
```
6. Install cuda, nvcc, gcc, and g++:
```
!./cuda-lab/INSTALL.sh
```

7. Add `/usr/local/cuda/bin` to `PATH`:
```python
import os
os.environ['PATH'] += ':/usr/local/cuda/bin'
```

8. Compile an existing Cuda source:
```
!nvcc cuda-lab/add.cu -o add -Wno-deprecated-gpu-targets
```

9. Run the compiled Cuda source using the Nvidia profiler tool:
```
!nvprof ./add
```

You can also create a Cuda source file using the magic command `%%writefile <filename.cu>`:
``` 
%%writefile snippet.cu
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
...
```

and then compile and run it!
```
!nvcc snippet.cu -o snippet -Wno-deprecated-gpu-targets
!nvprof ./snippet
```
