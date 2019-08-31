sudo apt-get update
sudo apt-get install build-essential
sudo apt-get install cmake
sudo apt-get install libopenblas-dev
sudo apt-get install liblapacke-dev

wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB

sudo sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list'

sudo apt-get update

sudo apt-get install intel-mkl-2019.4-070

sudo sh -c "echo '/opt/intel/lib/intel64'     >  /etc/ld.so.conf.d/mkl.conf"
sudo sh -c "echo '/opt/intel/mkl/lib/intel64' >> /etc/ld.so.conf.d/mkl.conf"
sudo ldconfig

rm GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB

sudo echo ". /opt/intel/mkl/bin/mklvars.sh intel64" >>~/.bashrc
