#!/bin/bash
cd lib/GKlib
make config openmp=set
make

cd ../../

if [ "$1" == "--with_mkl" ] 
then
    echo "Building SLIM with MKL!"
    make config shared=1 cc=gcc cxx=gcc with_mkl=1
else
    make config shared=1 cc=gcc cxx=gcc 
fi
make
