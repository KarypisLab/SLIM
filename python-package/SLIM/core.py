#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 00:49:28 2019

@author: dminerx007
"""

import os
import site
import time
import scipy
import numpy as np

from ctypes import *
from scipy.sparse import csr_matrix

from .config import *

# determine if pandas is installed
try:
    from pandas import DataFrame

    PANDAS_INSTALLED = True
except:
    PANDAS_INSTALLED = False

# load slimlib from libslim.so


def load_libslim():
    try:
        site_dir = site.getsitepackages()[0]
        lib_dir = site_dir + '/SLIM'
        slimlib = cdll.LoadLibrary(lib_dir + '/libslim.so')
    except:
        raise RuntimeError(
            "SLIM library %s could not be loaded. Please check if the program is installed correctly.", lib_dir + '/libslim.so')

    return slimlib


slimlib = load_libslim()


def check_obj_params(params):
    # sannity check for the parameters
    if hasattr(params, 'dbglvl'):
        if type(params.dbglvl) != int or params.dbglvl < 0:
            raise TypeError(
                "Please select dbglvl from {0, 1, 2, 4, 16, 2048}.")
    else:
        params.dbglvl = 0

    if hasattr(params, 'nnbrs'):
        if type(params.nnbrs) != int or params.nnbrs < 0:
            raise TypeError(
                "Please provide non-negative integer value for nnbrs.")
    else:
        params.nnbrs = 0

    if hasattr(params, 'simtype'):
        if params.simtype not in slim_simtype_et:
            raise TypeError(
                "Please select simtytpe from {'cos', 'jacc', 'dotp'}.")
    else:
        params.simtype = 'cos'

    if hasattr(params, 'algo'):
        if params.algo not in slim_algo_et:
            raise TypeError("Please select algo from {'admm', 'cd'}.")
    else:
        params.algo = 'cd'

    if hasattr(params, 'nthreads'):
        if type(params.nthreads) != int or params.nthreads <= 0:
            raise TypeError(
                "Please provide positive integer value for nthreads.")
    else:
        params.nthreads = 1

    if hasattr(params, 'niters'):
        if type(params.niters) != int or params.niters <= 0:
            raise TypeError(
                "Please provide positive integer value for niters.")
    else:
        params.niters = 50

    if hasattr(params, 'nrcmds'):
        if type(params.nrcmds) != int or params.nrcmds <= 0:
            raise TypeError(
                "Please provide positive integer value for nrcmds.")
    else:
        params.nrcmds = 10

    if hasattr(params, 'l1r'):
        if not isinstance(params.l1r, (int, float)) or params.l1r < 0:
            raise TypeError("Please provide non-negative value for l1r.")
    else:
        params.l1r = 1.

    if hasattr(params, 'l2r'):
        if not isinstance(params.l2r, (int, float)) or params.l2r < 0:
            raise TypeError("Please provide non-negative value for l2r.")
    else:
        params.l2r = 1.

    if hasattr(params, 'optTol'):
        if not isinstance(params.optTol, (int, float)) or params.optTol < 0:
            raise TypeError("Please provide non-negative value for optTol.")
    else:
        params.optTol = 1e-7

    # sanity check for fSLIM
    if params.nnbrs > 0 and params.algo != 'cd':
        print('A fSLIM model cannot be trained with ADMM. Changing the algorithm to coordinate descent.')
        params.algo = 'cd'

    # not in use
    params.ordered = 0


def check_dict_params(params):
    assert isinstance(params, dict)

    # sannity check for the parameters
    if 'dbglvl' in params:
        if type(params['dbglvl']) != int or params['dbglvl'] < 0:
            raise TypeError(
                "Please select dbglvl from {0, 1, 2, 4, 16, 2048}.")
    else:
        params['dbglvl'] = 0

    if 'nnbrs' in params:
        if type(params['nnbrs']) != int or params['nnbrs'] < 0:
            raise TypeError("Please provide positive integer value for nnbrs.")
    else:
        params['nnbrs'] = 0

    if 'simtype' in params:
        if params['simtype'] not in slim_simtype_et:
            raise TypeError(
                "Please select simtytpe from {'cos', 'jacc', 'dotp'}.")
    else:
        params['simtype'] = 'cos'

    if 'algo' in params:
        if params['algo'] not in slim_algo_et:
            raise TypeError("Please select algo from {'admm', 'cd'}.")
    else:
        params['algo'] = 'cd'

    if 'nthreads' in params:
        if type(params['nthreads']) != int or params['nthreads'] <= 0:
            raise TypeError(
                "Please provide positive integer value for nthreads.")
    else:
        params['nthreads'] = 1

    if 'niters' in params:
        if type(params['niters']) != int or params['niters'] < 0:
            raise TypeError(
                "Please provide positive integer value for niters.")
    else:
        params['niters'] = 50

    if 'nrcmds' in params:
        if type(params['nrcmds']) != int or params['nrcmds'] < 0:
            raise TypeError(
                "Please provide positive integer value for nrcmds.")
    else:
        params['nrcmds'] = 10

    if 'l1r' in params:
        if not isinstance(params['l1r'], (int, float)) or params['l1r'] < 0:
            raise TypeError("Please provide non-negative value for l1r.")
    else:
        params['l1r'] = 1.

    if 'l2r' in params:
        if not isinstance(params['l2r'], (int, float)) or params['l2r'] < 0:
            raise TypeError("Please provide non-negative value for l2r.")
    else:
        params['l2r'] = 1.

    if 'optTol' in params:
        if not isinstance(params['optTol'], (int, float)) or params['optTol'] < 0:
            raise TypeError("Please provide non-negative value for optTol.")
    else:
        params['optTol'] = 1e-7

    # sanity check for fSLIM
    if params['nnbrs'] > 0 and params['algo'] != 'cd':
        print('A fSLIM model cannot be trained with ADMM. Changing the algorithm to coordinate descent.')
        params['algo'] = 'cd'

    # not in use
    params['ordered'] = 0


def set_obj_params(params):
    # prepare the parameters to pass to the function
    ioptions = np.full(SLIM_NOPTIONS, -1, dtype=np.int32)
    doptions = np.full(SLIM_NOPTIONS, -1., dtype=np.float64)

    # set the parameters
    ioptions[SLIM_OPTION_DBGLVL] = params.dbglvl
    ioptions[SLIM_OPTION_NNBRS] = params.nnbrs
    ioptions[SLIM_OPTION_SIMTYPE] = slim_simtype_et[params.simtype]
    ioptions[SLIM_OPTION_ALGO] = slim_algo_et[params.algo]
    ioptions[SLIM_OPTION_NTHREADS] = params.nthreads
    ioptions[SLIM_OPTION_ORDERED] = params.ordered
    ioptions[SLIM_OPTION_MAXNITERS] = params.niters
    ioptions[SLIM_OPTION_NRCMDS] = params.nrcmds

    doptions[SLIM_OPTION_L1R] = params.l1r
    doptions[SLIM_OPTION_L2R] = params.l2r
    doptions[SLIM_OPTION_OPTTOL] = params.optTol

    return ioptions, doptions


def set_dict_params(params):
    # prepare the parameters to pass to the function
    ioptions = np.full(SLIM_NOPTIONS, -1, dtype=np.int32)
    doptions = np.full(SLIM_NOPTIONS, -1., dtype=np.float64)

    # set the parameters
    ioptions[SLIM_OPTION_DBGLVL] = params['dbglvl']
    ioptions[SLIM_OPTION_NNBRS] = params['nnbrs']
    ioptions[SLIM_OPTION_SIMTYPE] = slim_simtype_et[params['simtype']]
    ioptions[SLIM_OPTION_ALGO] = slim_algo_et[params['algo']]
    ioptions[SLIM_OPTION_NTHREADS] = params['nthreads']
    ioptions[SLIM_OPTION_ORDERED] = params['ordered']
    ioptions[SLIM_OPTION_MAXNITERS] = params['niters']
    ioptions[SLIM_OPTION_NRCMDS] = params['nrcmds']

    doptions[SLIM_OPTION_L1R] = params['l1r']
    doptions[SLIM_OPTION_L2R] = params['l2r']
    doptions[SLIM_OPTION_OPTTOL] = params['optTol']

    return ioptions, doptions


class SLIMatrix(object):
    def __init__(self, data, oldmat=None):
        ''' @brief  generate a matrix to feed into slim
            @params data: ijv triplets or csr matrix
                    oldmat: a SLIMatrix object
        '''
        self._get_gk_csr()

        # prepare the training matrix for passing to the estimate function
        if isinstance(data, scipy.sparse.csr.csr_matrix):
            self.nUsers = data.shape[0]
            self.nItems = data.shape[1]

            if oldmat != None and isinstance(oldmat, SLIMatrix):
                if self.nUsers != oldmat.nUsers or self.nItems != oldmat.nItems:
                    raise TypeError(
                        "The size of the input matrix does not match the size of oldmat.")

            if oldmat != None and isinstance(oldmat, SLIM):
                if self.nItems != oldmat.id2item.size:
                    raise TypeError(
                        "The size of the input matrix does not match the size of oldmat.")

            self.id2item = np.arange(self.nItems)
            self.item2id = self.id2item
            self.id2user = np.arange(self.nUsers)
            self.user2id = self.id2user
            R = data
            self._set_csr(R)

        elif isinstance(data, (list, np.ndarray)):
            self.data_from_np2d(data, oldmat)

        elif PANDAS_INSTALLED and isinstance(data, DataFrame):
            self.data_from_np2d(data.values, oldmat)

        else:
            raise TypeError("Input data type %s is not supported. Please provide ijv triplets in numpy.ndarray/list[List]/pandas.DataFrame \
                  or a row based sparse matrix in scipy csr_matrix." % (type(data).__name__))

    def __del__(self):
        self._csr_free(self.handle)
        del self.id2item, self.item2id, self.id2user, self.user2id

    def data_from_np2d(self, data, oldmat=None):
        if oldmat != None:
            assert isinstance(
                oldmat, (SLIMatrix, SLIM)), 'Please feed in a SLIMatrix object or a SLIM model for oldmat.'

            if isinstance(oldmat, SLIMatrix):
                self.id2item = oldmat.id2item.copy()
                self.item2id = oldmat.item2id.copy()
                self.id2user = oldmat.id2user.copy()
                self.user2id = oldmat.user2id.copy()
            else:
                self.id2item = oldmat.id2item.copy()
                self.item2id = oldmat.item2id.copy()
                self.user2id = {}
                self.id2user = []

                nUsers = 0
                for tri in data:
                    if tri[0] not in self.user2id:
                        self.user2id[tri[0]] = nUsers
                        self.id2user.append(tri[0])
                        nUsers += 1
        else:
            self.item2id = {}
            self.id2item = []
            self.user2id = {}
            self.id2user = []

            nUsers = 0
            nItems = 0
            for tri in data:
                if tri[0] not in self.user2id:
                    self.user2id[tri[0]] = nUsers
                    self.id2user.append(tri[0])
                    nUsers += 1
                if tri[1] not in self.item2id:
                    self.item2id[tri[1]] = nItems
                    self.id2item.append(tri[1])
                    nItems += 1

            self.id2item = np.array(self.id2item)
            self.id2user = np.array(self.id2user)

        val = []
        col = []
        row = []
        miss = 0
        for tri in data:
            if tri[0] in self.user2id and tri[1] in self.item2id:
                row.append(self.user2id[tri[0]])
                col.append(self.item2id[tri[1]])
                val.append(tri[2])
            else:
                miss += 1

        if miss > 0:
            print(
                "%d of the events fall out of the range of oldmat. Partial entries collected." % (miss))

        self.nUsers = len(self.id2user)
        self.nItems = len(self.id2item)
        R = csr_matrix((val, (row, col)), shape=(self.nUsers, self.nItems))
        self._set_csr(R)

    def _set_csr(self, R):
        handle = c_void_p()
        self._csr_wrapper(
            R.shape[0],  # nrows
            np.ascontiguousarray(R.indptr, dtype=np.intp),  # rowptr
            np.ascontiguousarray(R.indices, dtype=np.int32),  # rowind
            np.ascontiguousarray(R.data, dtype=np.float32),  # rowval
            byref(handle)
        )
        self.handle = handle

    def _get_gk_csr(self):
        # access Py_csr_wrapper from libslim.so
        self._csr_wrapper = wrap_function(
            slimlib,
            "Py_csr_wrapper",
            restype=c_int32,  # flag
            argtypes=[c_int32,  # nrows
                      array_1d_ssize_t,  # rowptr
                      array_1d_int32_t,  # rowind
                      array_1d_float32_t,  # rowval
                      c_void_p  # out
                      ]
        )

        # access Py_csr_free from libslim.so
        self._csr_free = wrap_function(
            slimlib,
            "Py_csr_free",
            restype=c_int32,  # flag
            argtypes=[c_void_p  # mathandle
                      ]
        )


class SLIM(object):
    def __init__(self):
        self.ismodel = 0
        self._get_slim()

    def __del__(self):
        try:
            self._slim_free(self.handle)
        except:
            pass

    def train(self, params, data):
        ''' @brief  train a slim model
            @params params: training parameters 
                    data: a SLIMatrix object
        '''
        assert type(data) == SLIMatrix, 'trndata must be a SLIMatrix object.'

        self.nItems = data.nItems

        if isinstance(params, dict):
            check_dict_params(params)
            ioptions, doptions = set_dict_params(params)
        else:
            try:
                check_obj_params(params)
                ioptions, doptions = set_obj_params(params)
            except TypeError:
                raise
            except:
                raise TypeError('Parameter type %s is not supported!' %
                                (type(params).__name__))

        handle = c_void_p()
        start = time.time()
        self.ismodel = self._slim_learn(
            data.handle,
            ioptions,
            doptions,
            byref(handle)
        )
        self.handle = handle
        end = time.time()

        self.id2item = data.id2item.copy()
        self.item2id = data.item2id.copy()

        if self.ismodel == SLIM_OK:
            print("Learning takes %.3f secs." % (end - start))
        else:
            raise RuntimeError("Something went wrong with model estimation.")

    def mselect(self, params, trndata, tstdata, arrayl1, arrayl2, nrcmds):
        ''' @brief  cross validation
            @params params: training parameters 
                    trndata: a SLIMatrix object that contains the training matrix
                    tstdata: a SLIMatrix object that contains the test matrix
                    arrayl1: a list of l1 values
                    arrayl2: a list of l2 values
                    nrcmds:  number of recommended items for each user
        '''
        assert type(trndata) == SLIMatrix, 'trndata must be a SLIMatrix object.'
        assert type(tstdata) == SLIMatrix, 'tstdata must be a SLIMatrix object.'
        assert type(arrayl1) in [
            list, np.ndarray], 'Please provide a list of l1 values.'
        assert type(arrayl2) in [
            list, np.ndarray], 'Please provide a list of l2 values.'

        # prepare the parameters to pass to the function
        ioptions = np.full(SLIM_NOPTIONS, -1, dtype=np.int32)
        doptions = np.full(SLIM_NOPTIONS, -1., dtype=np.float64)

        if isinstance(params, dict):
            check_dict_params(params)
            params['nrcmds'] = nrcmds
            ioptions, doptions = set_dict_params(params)
        else:
            try:
                check_obj_params(params)
                params.nrcmds = nrcmds
                ioptions, doptions = set_obj_params(params)
            except TypeError:
                raise
            except:
                raise TypeError('Parameter type %s is not supported!' %
                                (type(params).__name__))

        if len(arrayl1) < 1:
            raise TypeError('The l1 array must not be empty.')

        if len(arrayl2) < 1:
            raise TypeError('The l2 array must not be empty.')

        bestl1HR = c_double(0.)
        bestl2HR = c_double(0.)
        bestHRHR = c_double(0.)
        bestARHR = c_double(0.)
        bestl1AR = c_double(0.)
        bestl2AR = c_double(0.)
        bestHRAR = c_double(0.)
        bestARAR = c_double(0.)

        start = time.time()
        rstatus = self._slim_mselect(
            trndata.handle,
            tstdata.handle,
            ioptions,
            doptions,
            np.ascontiguousarray(np.sort(arrayl1), dtype=np.float64),
            np.ascontiguousarray(np.sort(arrayl2), dtype=np.float64),
            len(arrayl1),
            len(arrayl2),
            byref(bestl1HR),
            byref(bestl2HR),
            byref(bestHRHR),
            byref(bestARHR),
            byref(bestl1AR),
            byref(bestl2AR),
            byref(bestHRAR),
            byref(bestARAR)
        )
        end = time.time()
        if rstatus == SLIM_OK:
            print("Model selection takes %.3f secs." % (end - start))
            print('The best HR is achieved by, l1: %.4f, l2:%.4f, HR:%.4f, AR:%.4f.' % (
                bestl1HR.value, bestl2HR.value, bestHRHR.value, bestARHR.value))
            print('The best AR is achieved by, l1: %.4f, l2:%.4f, HR:%.4f, AR:%.4f.' % (
                bestl1AR.value, bestl2AR.value, bestHRAR.value, bestARAR.value))
        else:
            raise RuntimeError(
                'Something went wrong with model estimation or evaluation when l1=%.4f, l2=%.4f. Please check the input matrix.' % (bestl1HR, bestl2HR))

    def predict(self, data, nrcmds=10, outfile=None, negitems=None, nnegs=0, returnscores=False):
        ''' @brief  predict using the learned SLIM model
            @params data:     a SLIMatrix object to be predicted
                    nrcmds:   number of recommended items for each user
                    outfile:  a filename to dump the topn lists
                    negitems: negative items
                    nnegs:    number of negative items
            @return out:        an numpy ndarray of shape (nUsers, nrcmds) with recommended item ids
                    outscores:  an numpy ndarray of shape (nUsers, nrcmds) with recommended scores of the corresponding items
        '''
        if self.ismodel != SLIM_OK:
            raise TypeError("Model not found. Please train a model.")

        assert self.nItems == data.nItems, \
            'The shape of the input matrix should match the model.'

        # initialize the result matrix
        res = np.full(data.nUsers * nrcmds, -1, dtype=np.int32)
        scores = np.zeros(data.nUsers * nrcmds, dtype=np.float32)
        
        if negitems != None:
            assert nnegs >= nrcmds, \
            'The number of negative items must be larger than the number of items to be recommended.'
            
            if isinstance(data.user2id, dict):
                assert data.user2id.keys() == negitems.keys(), \
                'The users in the negative items should be the same with the input matrix.'
            else:
                assert np.array_equal(data.user2id, np.array(sorted(list(negitems.keys())))), \
                'The users in the negative items should be the same with the input matrix.'
            
            slim_negitems = np.full(data.nUsers * nnegs, -1, dtype=np.int32)
            nusers = 0
            newitems = 0
            for key, value in negitems.items():
                assert len(value) == nnegs, \
                'The number of negative items should match nngs.'
                for i in range(nnegs):
                    try:
                        slim_negitems[nusers * nnegs + i] = self.item2id[value[i]]
                    except:
                        newitems += 1
                nusers += 1
        
            if newitems > 0:
                print('%d negative items not in the training set.' % (newitems))
            
            rstatus = self._slim_predict_1vsk(
                nrcmds,
                nnegs,
                self.handle,
                data.handle,
                slim_negitems,
                res,
                scores)
            
        else:
            rstatus = self._slim_predict(
                nrcmds,
                self.handle,
                data.handle,
                res,
                scores)
        
        if rstatus == SLIM_OK:
            res = self.id2item[res].reshape(data.nUsers, nrcmds)
            scores = scores.reshape(data.nUsers, nrcmds)
            out = dict()
            outscores = dict()
            
            if isinstance(data.user2id, dict): 
                for key, value in data.user2id.items():
                    out[key] = res[value, :]
                    outscores[key] = scores[value, :]
            else:
                for key in data.user2id:
                    out[key] = res[key, :]
                    outscores[key] = scores[key, :]

            if outfile:
                f = open(outfile, 'w')
                for key, value in out.items():
                    f.write(str(key) + ': ' + np.array2string(value,
                                                              max_line_width=np.inf) + '\n')
                    if returnscores:
                        f.write(str(key) + ': ' + np.array2string(outscores[key],
                                                                  max_line_width=np.inf) + '\n')
        else:
            raise RuntimeError(
                'Something went wrong during prediction. Please check 1) if the model is estimated correctly; 2) if the input matrix for prediction is correct.')
        
        if returnscores:
            return out, outscores 
        else:
            return out

    def save_model(self, modelfname, mapfname):
        ''' @brief  save the model
            @params modelfname: filename to save the model
                    mapfname:   filename to save the item map
            @return None
        '''
        # save the model if there is one
        if self.ismodel == SLIM_OK:
            self._slim_save(self.handle, c_char_p(modelfname.encode('utf-8')))
            np.savetxt(mapfname, self.id2item, fmt='%s')
        else:
            raise RuntimeError("Not exist a model to save.")

    def load_model(self, modelfname, mapfname):
        ''' @brief  load a model
            @params modelfname: filename of the model
                    mapfname:   filename of the item map
            @return None
        '''
        # if there is a model, destruct the model
        if os.path.isfile(modelfname) and os.path.isfile(mapfname): 
            if self.ismodel == SLIM_OK:
                self._slim_free(self.handle)
            else:
                self.handle = c_void_p()
            self.ismodel = self._slim_load(
                byref(self.handle), c_char_p(modelfname.encode('utf-8')))
    
            try:
                self.id2item = np.genfromtxt(mapfname, dtype=np.int32)
            except:
                self.id2item = np.genfromtxt(mapfname)
            self.item2id = {}
            for i in range(len(self.id2item)):
                self.item2id[self.id2item[i]] = i
            self.nItems = len(self.id2item)
    
            if self.ismodel != SLIM_OK:
                raise RuntimeError("Fail to laod the model.")
        else:
            raise RuntimeError('File does not exist or invalid filename.')
            
    def to_csr(self, returnmap=False):
        ''' @brief  export the model as a scipy csr
            @params returnmap: return the map or not
            @return modelcsr: the model as a scipy csr
                    itemmap (optional): the item map attached with the model
        '''
        if self.ismodel == SLIM_OK:
            nnz = c_int(0)
            self._slim_stat(self.handle, byref(nnz))
            
            indptr = np.zeros(self.nItems + 1, dtype=np.int32)
            indices = np.zeros(nnz.value, dtype=np.int32)
            data = np.ones(nnz.value, dtype=np.float32)
            
            self._slim_export(self.handle, indptr, indices, data)
            
            modelcsr = csr_matrix((data, indices, indptr), shape=(self.nItems, self.nItems))
            
            if returnmap:
                itemmap = self.id2item[:]
                return modelcsr, itemmap
            else:
                return modelcsr
        else:
            raise RuntimeError("Not exist a model to export.")
        
        
        
    def _get_slim(self):
        ''' @brief  wrap up slim functions from c library for python
            @params None
            @return None
        '''

        # access Py_SLIM_Learn from libslim.so
        self._slim_learn = wrap_function(
            slimlib,
            "Py_SLIM_Learn",
            restype=c_int32,  # resmat
            argtypes=[c_void_p,  # trnhandle
                      array_1d_int32_t,  # ioptions
                      array_1d_double_t,  # doptions
                      c_void_p  # out
                      ]
        )

        # access Py_SLIM_Mselect from libslim.so
        self._slim_mselect = wrap_function(
            slimlib,
            "Py_SLIM_Mselect",
            restype=c_int32,
            argtypes=[c_void_p,  # trnhandle
                      c_void_p,  # tsthandle
                      array_1d_int32_t,  # ioptions
                      array_1d_double_t,  # doptions
                      array_1d_double_t,  # arrayl1
                      array_1d_double_t,  # arrayl2
                      c_int,  # nl1
                      c_int,  # nl2
                      c_void_p,  # bestl1HR
                      c_void_p,  # bestl2HR
                      c_void_p,  # bestHRHR
                      c_void_p,  # bestARHR
                      c_void_p,  # bestl1AR
                      c_void_p,  # bestl2AR
                      c_void_p,  # bestHRAR
                      c_void_p  # bestARAR
                      ]
        )

        # access Py_SLIM_Predict from libslim.so
        self._slim_predict = wrap_function(
            slimlib,
            "Py_SLIM_Predict",
            restype=c_int32,  # resmat
            argtypes=[c_int,  # nrcmds
                      c_void_p,  # slimhandle
                      c_void_p,  # trnhandle
                      array_1d_int32_t,  # output
                      array_1d_float32_t  # scores
                      ]
        )
        
        # access Py_SLIM_Predict_1vsk from libslim.so
        self._slim_predict_1vsk = wrap_function(
            slimlib,
            "Py_SLIM_Predict_1vsk",
            restype=c_int32,  # resmat
            argtypes=[c_int,  # nrcmds
                      c_int,  # nnegs
                      c_void_p,  # slimhandle
                      c_void_p,  # trnhandle
                      array_1d_int32_t,  # negitems
                      array_1d_int32_t,  # output
                      array_1d_float32_t  # scores
                      ]
        )
        
        # access Py_csr_save from libslim.so
        self._slim_save = wrap_function(
            slimlib,
            "Py_csr_save",
            restype=c_int32,  # flag
            argtypes=[c_void_p,  # mathandle
                      c_char_p  # fname
                      ]
        )

        # access Py_csr_load from libslim.so
        self._slim_load = wrap_function(
            slimlib,
            "Py_csr_load",
            restype=c_int32,  # flag
            argtypes=[c_void_p,  # mathandle
                      c_char_p  # fname
                      ]
        )

        # access Py_csr_free from libslim.so
        self._slim_free = wrap_function(
            slimlib,
            "Py_csr_free",
            restype=c_int32,  # flag
            argtypes=[c_void_p  # mathandle
                      ]
        )
        
        # access Py_csr_stat from libslim.so
        self._slim_stat = wrap_function(
            slimlib,
            "Py_csr_stat",
            restype=c_int32,  # flag
            argtypes=[c_void_p,  # mathandle
                      c_void_p  # nnz
                      ]
        )
        
        # access Py_csr_stat from libslim.so
        self._slim_export = wrap_function(
            slimlib,
            "Py_csr_export",
            restype=c_int32,  # flag
            argtypes=[c_void_p,  # mathandle
                      array_1d_int32_t,  # indptr
                      array_1d_int32_t,  # indices
                      array_1d_float32_t  # data
                      ]
        )