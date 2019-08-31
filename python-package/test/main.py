#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 00:03:05 2019

@author: shuizeren
"""

import argparse
import pandas as pd
from SLIM import SLIM, SLIMatrix


if __name__ == '__main__':
    param = argparse.ArgumentParser()
    param.add_argument('--traindata', type=str, default='../../test/AutomotiveTrain.ijv')
    param.add_argument('--valdata', type=str, default='../../test/AutomotiveTest.ijv')
    param.add_argument('--dbglvl', type=int, default=1)
    param.add_argument('--nnbrs', type=int, default=0)
    param.add_argument('--simtype', type=str, default='cos')
    param.add_argument('--algo', type=str, default='cd')
    param.add_argument('--nthreads', type=int, default=1)
    param.add_argument('--niters', type=int, default=100)
    
    param.add_argument('--l1r', type=float, default=1.)
    param.add_argument('--l2r', type=float, default=1.)
    param.add_argument('--optTol', type=float, default=1e-7)

    config = param.parse_args()
    
    traindata = pd.read_csv(config.traindata, delimiter=' ', header=None)
    valdata = pd.read_csv(config.valdata, delimiter=' ', header=None)
    
    trainmat = SLIMatrix(traindata)
    valmat = SLIMatrix(valdata, trainmat)
    
    model = SLIM()
    model.train(config, trainmat)
    model.save_model(modelfname='model.csr', mapfname='map.csv')
    model.load_model('model.csr', mapfname='map.csv')
    topn = model.predict(trainmat, nrcmds=10, outfile='output.txt')
