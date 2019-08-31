Table of Contents
=================

   * [SLIM](#slim)
      * [Downloading SLIM](#downloading-slim)
      * [Building standalone SLIM binary and shared library](#building-standalone-slim-binary-and-shared-library)
      * [Python package installation](#python-package-installation)
      * [Getting started](#getting-started)
      * [Citing](#citing)
      * [References](#references)
      * [Credits &amp; Contact Information](#credits--contact-information)
      * [Copyright &amp; License Notice](#copyright--license-notice)



# SLIM 

Sparse LInear Method (SLIM) [1] is an item-based top-N recommendation approach that combines the advantages of neighborhood- and model-based collaborative filtering methods. It achieves state-of-the-art recommendation performance and has low computational requirements.

This package provides a C-based optimized multi-threaded implementation of SLIM that consists of a set of command-line programs and a user-callable library for estimating and applying SLIM models as well as an easy to use Python interface. 

##  Downloading SLIM

SLIM uses Git submodules to manage external dependencies. Hence, please specify the `--recursive` option while cloning the repo as follow:
```bash
git clone --recursive https://github.umn.edu/dminers/slim.git
```

## Building standalone SLIM binary and shared library

To build SLIM you can follow the instructions below:

### Dependencies

General dependencies for building slim are: gcc, cmake, build-essential.
In Ubuntu systems these can be obtained from the apt package manager (e.g., apt-get install cmake, etc) 

```bash
sudo apt-get install build-essential
sudo apt-get install cmake
```

### Building and installing SLIM  

In order to build SLIM, first build GKlib by running:

```bash
cd lib/GKlib
make config openmp=set
make
cd ../../
```


After building GKlib, you can build and install SLIM by running:

```bash
make config shared=1 cc=gcc cxx=gcc prefix=~/.local
make install
```

#### Building and installing SLIM with Intel's MKL support (optional)

In order to use SLIM's ADMM solver, you will need to install [Intel's MKL library](https://software.intel.com/en-us/mkl). 

For Ubuntu machines on which you have `sudo` privilages, we provided the `depmkl.sh` script that automates the process of obtaining and installing MKL, which can be used as follows:

```bash
bash depmkl.sh
source ~/.bashrc 
```

For machines on which you do not have `sudo` privilages, you should download the MKL tarball from [Intel's website](https://software.intel.com/en-us/mkl) and then install it locally using the `install.sh` script they provide. After installing it you should add `your-path-to-intel/intel/mkl/bin/mklvars.sh intel64`in your bashrc and run `source ~/.bashrc`.

Next you can build and install SLIM with MKL support by running:

```bash  
make config shared=1 cc=gcc cxx=gcc with_mkl=1 prefix=~/.local
make install
```

Note that SLIM's ADMM solver usually outperforms the default optimizer included in SLIM when the number of items in the dataset is relatively small compared to the number of users and the number of non-zeros in the dataset is large. 


## Python package installation

The Python package is located at `python-package/`. 
The installation of python-package requires Python `distutils` module and is often part of the core Python package or can be installed using a package manager, e.g., in Debian use

```bash
sudo apt-get install python-setuptools
```

After building the SLIM library, follow one of the following steps to install the python-package:

1. Install the python-package system-wide (this requires sudo priveleges):
```bash
cd python-package
sudo python setup.py install
```

2. Install the python-package only for the current user (without sudo priveleges):
```bash
cd python-package
python setup.py install --user
```

## Getting started

Here are some examples to quickly try out SLIM on the sample datasets that are provided with SLIM.

### Python interface

```python
import pandas as pd
from SLIM import SLIM, SLIMatrix

#read training data stored as triplets <user> <item> <rating>
traindata = pd.read_csv('../test/AutomotiveTrain.ijv', delimiter = ' ', header=None)
trainmat = SLIMatrix(traindata)

#set up parameters to learn model, e.g., use Coordinate Descent with L1 and L2
#regularization
params = {'algo':'cd', 'nthreads':2, 'l1r':1.0, 'l2r':1.0}

#learn the model using training data and desired parameters
model = SLIM()
model.train(params, trainmat)

#read test data having candidate items for users
testdata = pd.read_csv('../test/AutomotiveTest.ijv', delimiter = ' ', header=None)
#NOTE: model object is passed as an argument while generating test matrix
testmat = SLIMatrix(testdata, model)

#generate top-10 recommendations
prediction_res = model.predict(testmat, nrcmds=10, outfile = 'output.txt')

#dump the model to files on disk
model.save_model(modelfname='model.csr', # filename to save the model as a csr matrix
                 mapfname='map.csr' # filename to save the item map
                )

#load the model from from disk
model_new = SLIM()
model_new.load_model(modelfname='model.csr', # filename of the model
                 mapfname='map.csr' # filename of the item map
                )
```

The users can also refer to the python notebook [UserGuide.ipynb](./python-package/UserGuide.ipynb) located at
`./python-package/UserGuide.ipynb` for more examples on using the python api.

###  Command-line programs
SLIM can be used by running the command-line programs that are located under `./build` directory. Specifically, SLIM provides the following three command-line programs:
- `slim_learn`: for estimating a model
- `slim_predict`: for applying a previously estimated model, and
- `slim_mselect`: for exploring a set of hyper-parameters in order to select the best performing model.

Additional information about how to use these command-line programs is located in [reference manual](./doxygen/latex/refman.pdf).

###  Library interface

You can also use SLIM by direclty linking into your C/C++ program via its library interface. SLIM's API is described in [reference manual](./doxygen/latex/refman.pdf).


## Citing
If you use any part of this library in your research, please cite it using the
following BibTex entry:

```
@online{slim,
  title = {{SLIM}: Sparse LInear Model library},
  author = {Ning, Xia and Nikolakopoulos, Athanasios N. and Shui, Zeren and Sharma, Mohit and Karypis, George},
  url = {https://github.com/KarypisLab/SLIM},
  publisher = {GitHub},
  journal = {GitHub Repository},
  year = {2019},
}
```

## References
1. [Slim: Sparse linear methods for top-n recommender systems](http://glaros.dtc.umn.edu/gkhome/node/774)
## Credits & Contact Information

This implementation of SLIM was written by George Karypis with contributions by Xia Ning, Athanasios N. Nikolakopoulos, Zeren Shui and Mohit Sharma.

If you encounter any problems or have any suggestions, please contact George Karypis at <a href="mailto:karypis@umn.edu">karypis@umn.edu</a>.


## Copyright & License Notice
Copyright 2019, Regents of the University of Minnesota

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
