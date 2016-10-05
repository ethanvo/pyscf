#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import time
import ctypes
import tempfile
import numpy
import h5py
from pyscf import lib
from pyscf import symm
from pyscf.lib import logger
from pyscf.cc import _ccsd
from pyscf.ao2mo.outcore import balance_partition

'''
CCSD(T)
'''

# t3 as ijkabc

# JCP, 94, 442.  Error in Eq (1), should be [ia] >= [jb] >= [kc]
def kernel(mycc, eris, t1=None, t2=None, verbose=logger.NOTE):
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(mycc.stdout, verbose)
    cpu1 = cpu0 = (time.clock(), time.time())
    if t1 is None: t1 = mycc.t1
    if t2 is None: t2 = mycc.t2

    nocc, nvir = t1.shape
    nmo = nocc + nvir

    _tmpfile = tempfile.NamedTemporaryFile()
    ftmp = h5py.File(_tmpfile.name)
    eris_vvop = ftmp.create_dataset('vvop', (nvir,nvir,nocc,nmo), 'f8')
    orbsym = _sort_eri(mycc, eris, nocc, nvir, eris_vvop, log)

    ftmp['t2'] = t2  # read back late.  Cache t2T in t2 to reduce memory footprint
    mo_energy, t1T, t2T, vooo = _sort_t2_vooo(mycc, orbsym, t1, t2, numpy.asarray(eris.ovoo))

    cpu2 = [time.clock(), time.time()]
    orbsym = numpy.hstack((numpy.sort(orbsym[:nocc]),numpy.sort(orbsym[nocc:])))
    o_ir_loc = numpy.append(0, numpy.cumsum(numpy.bincount(orbsym[:nocc], minlength=8)))
    v_ir_loc = numpy.append(0, numpy.cumsum(numpy.bincount(orbsym[nocc:], minlength=8)))
    o_sym = orbsym[:nocc]
    oo_sym = (o_sym[:,None] ^ o_sym).ravel()
    oo_ir_loc = numpy.append(0, numpy.cumsum(numpy.bincount(oo_sym, minlength=8)))
    nirrep = max(oo_sym) + 1

    orbsym   = orbsym.astype(numpy.int32)
    o_ir_loc = o_ir_loc.astype(numpy.int32)
    v_ir_loc = v_ir_loc.astype(numpy.int32)
    oo_ir_loc = oo_ir_loc.astype(numpy.int32)
    def contract(a0, a1, b0, b1, cache):
        cache_row_a, cache_col_a, cache_row_b, cache_col_b = cache
        drv = _ccsd.libcc.CCsd_t_contract
        drv.restype = ctypes.c_double
        et = drv(mo_energy.ctypes.data_as(ctypes.c_void_p),
                 t1T.ctypes.data_as(ctypes.c_void_p),
                 t2T.ctypes.data_as(ctypes.c_void_p),
                 vooo.ctypes.data_as(ctypes.c_void_p),
                 ctypes.c_int(nocc), ctypes.c_int(nvir),
                 ctypes.c_int(a0), ctypes.c_int(a1),
                 ctypes.c_int(b0), ctypes.c_int(b1),
                 ctypes.c_int(nirrep),
                 o_ir_loc.ctypes.data_as(ctypes.c_void_p),
                 v_ir_loc.ctypes.data_as(ctypes.c_void_p),
                 oo_ir_loc.ctypes.data_as(ctypes.c_void_p),
                 orbsym.ctypes.data_as(ctypes.c_void_p),
                 cache_row_a.ctypes.data_as(ctypes.c_void_p),
                 cache_col_a.ctypes.data_as(ctypes.c_void_p),
                 cache_row_b.ctypes.data_as(ctypes.c_void_p),
                 cache_col_b.ctypes.data_as(ctypes.c_void_p))
        cpu2[:] = log.timer_debug1('contract %d:%d,%d:%d'%(a0,a1,b0,b1), *cpu2)
        return et

    # The rest 20% memory for cache b
    mem_now = lib.current_memory()[0]
    max_memory = max(2000, mycc.max_memory - mem_now)
    bufsize = max(1, (max_memory*1e6/8-nocc**3*100)*.8/(nocc*nmo))
    log.debug('max_memory %d MB (%d MB in use)', max_memory, mem_now)
    et = 0
    handler = None
    for a0, a1, na in reversed(tril_prange(0, nvir, bufsize)):
        if handler is not None:
            et += handler.get()
            handler = None
        # DO NOT prefetch here to reserve more memory for cache_a
        cache_row_a = numpy.asarray(eris_vvop[a0:a1,:a1])
        cache_col_a = numpy.asarray(eris_vvop[:a0,a0:a1])
        handler = lib.background_thread(contract, a0, a1, a0, a1,
                                        (cache_row_a,cache_col_a,
                                         cache_row_a,cache_col_a))

        for b0, b1, nb in tril_prange(0, a0, bufsize/10):
            cache_row_b = numpy.asarray(eris_vvop[b0:b1,:b1])
            cache_col_b = numpy.asarray(eris_vvop[:b0,b0:b1])
            if handler is not None:
                et += handler.get()
                handler = None
            handler = lib.background_thread(contract, a0, a1, b0, b1,
                                            (cache_row_a,cache_col_a,
                                             cache_row_b,cache_col_b))
            cache_row_b = cache_col_b = None
        cache_row_a = cache_col_a = None
    if handler is not None:
        et += handler.get()
        handler = None

    t2[:] = ftmp['t2']
    ftmp.close()
    _tmpfile = None
    et *= 2
    log.timer('CCSD(T)', *cpu0)
    log.info('CCSD(T) correction = %.15g', et)
    return et

def _sort_eri(mycc, eris, nocc, nvir, vvop, log):
    cpu1 = (time.clock(), time.time())
    mol = mycc.mol
    nmo = nocc + nvir

    mol = mycc.mol
    if mol.symmetry:
        orbsym = symm.addons.label_orb_symm(mol, mol.irrep_id, mol.symm_orb,
                                            mycc.mo_coeff, check=False)
        orbsym = numpy.asarray(orbsym, dtype=numpy.int32) % 10
    else:
        orbsym = numpy.zeros(nmo, dtype=numpy.int32)

    o_sorted = _irrep_argsort(orbsym[:nocc])
    v_sorted = _irrep_argsort(orbsym[nocc:])
    vrank = numpy.argsort(v_sorted)

    max_memory = max(2000, mycc.max_memory - lib.current_memory()[0])
    max_memory = min(8000, max_memory*.5)
    blksize = min(nvir, max(16, int(max_memory*1e6/8/(nvir*nocc*nmo))))
    buf = numpy.empty((blksize,nvir,nocc,nmo))
    fn = _ccsd.libcc.CCsd_t_sort_transpose
    for j0, j1 in lib.prange(0, nvir, blksize):
        vvopbuf = numpy.ndarray((j1-j0,nvir,nocc,nmo), buffer=buf)
        ovov = numpy.asarray(eris.ovov[:,j0:j1])
        ovvv = numpy.asarray(eris.ovvv[:,j0:j1])
        fn(vvopbuf.ctypes.data_as(ctypes.c_void_p),
           ovov.ctypes.data_as(ctypes.c_void_p),
           ovvv.ctypes.data_as(ctypes.c_void_p),
           orbsym.ctypes.data_as(ctypes.c_void_p),
           ctypes.c_int(nocc), ctypes.c_int(nvir), ctypes.c_int(j1-j0))
        for j in range(j0,j1):
            vvop[vrank[j]] = vvopbuf[j-j0]
        cpu1 = log.timer_debug1('transpose %d:%d'%(j0,j1), *cpu1)

    return orbsym

def _sort_t2_vooo(mycc, orbsym, t1, t2, ovoo):
    nocc, nvir = t1.shape
    if mycc.mol.symmetry:
        orbsym = numpy.asarray(orbsym, dtype=numpy.int32)
        o_sorted = _irrep_argsort(orbsym[:nocc])
        v_sorted = _irrep_argsort(orbsym[nocc:])
        mo_energy = numpy.hstack((mycc.mo_energy[:nocc][o_sorted],
                                  mycc.mo_energy[nocc:][v_sorted]))
        t1T = numpy.asarray(t1.T[v_sorted][:,o_sorted], order='C')

        t2T = lib.transpose(t2.reshape(nocc**2,-1))
        _ccsd.libcc.CCsd_t_sort_t2(t2.ctypes.data_as(ctypes.c_void_p),
                                   t2T.ctypes.data_as(ctypes.c_void_p),
                                   orbsym.ctypes.data_as(ctypes.c_void_p),
                                   ctypes.c_int(nocc), ctypes.c_int(nvir))
        t2T = t2.reshape(nvir,nvir,nocc,nocc)
        vooo = numpy.empty((nvir,nocc,nocc,nocc))
        _ccsd.libcc.CCsd_t_sort_vooo(vooo.ctypes.data_as(ctypes.c_void_p),
                                     ovoo.ctypes.data_as(ctypes.c_void_p),
                                     orbsym.ctypes.data_as(ctypes.c_void_p),
                                     ctypes.c_int(nocc), ctypes.c_int(nvir))
        ovoo = None
    else:
        t1T = t1.T.copy()
        t2T = lib.transpose(t2.reshape(nocc**2,-1))
        t2T = lib.transpose(t2T.reshape(-1,nocc,nocc), axes=(0,2,1), out=t2)
        t2T = t2T.reshape(nvir,nvir,nocc,nocc)
        vooo = ovoo.transpose(1,0,2,3).copy()
        mo_energy = mycc.mo_energy
    return mo_energy, t1T, t2T, vooo

def tril_prange(start, stop, step):
    cum_costs = numpy.arange(stop+1)**2
    tasks = balance_partition(cum_costs, step, start, stop)
    return tasks

def _irrep_argsort(orbsym):
    return numpy.hstack([numpy.where(orbsym == i)[0] for i in range(8)])


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import cc

    mol = gto.M()
    numpy.random.seed(12)
    nocc, nvir = 5, 12
    eris = lambda :None
    eris.ovvv = numpy.random.random((nocc,nvir,nvir*(nvir+1)//2)) * .1
    eris.ovoo = numpy.random.random((nocc,nvir,nocc,nocc)) * .1
    eris.ovov = numpy.random.random((nocc,nvir,nocc,nvir)) * .1
    t1 = numpy.random.random((nocc,nvir)) * .1
    t2 = numpy.random.random((nocc,nocc,nvir,nvir)) * .1
    t2 = t2 + t2.transpose(1,0,3,2)
    mf = scf.RHF(mol)
    mcc = cc.CCSD(mf)
    mcc.mo_energy = mcc._scf.mo_energy = numpy.arange(0., nocc+nvir)
    print(kernel(mcc, eris, t1, t2) + 8.4953387936460398)

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -.957 , .587)],
        [1 , (0.2,  .757 , .487)]]

    mol.basis = 'ccpvdz'
    mol.build()
    rhf = scf.RHF(mol)
    rhf.conv_tol = 1e-14
    rhf.scf()
    mcc = cc.CCSD(rhf)
    mcc.conv_tol = 1e-14
    mcc.ccsd()
    e3a = kernel(mcc, mcc.ao2mo())
    print(e3a - -0.0033300722704016289)

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -.757 , .587)],
        [1 , (0. ,  .757 , .587)]]
    mol.symmetry = True

    mol.basis = 'ccpvdz'
    mol.build()
    rhf = scf.RHF(mol)
    rhf.conv_tol = 1e-14
    rhf.scf()
    mcc = cc.CCSD(rhf)
    mcc.conv_tol = 1e-14
    mcc.ccsd()
    e3a = kernel(mcc, mcc.ao2mo())
    print(e3a - -0.003060022611584471)
