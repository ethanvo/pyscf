#!/usr/bin/env python

"""
Urestricted DFT+U.
Based on KRKSpU routine.

Refs: PRB, 1998, 57, 1505.
"""

import copy
import numpy as np
import itertools as it
import scipy.linalg as la
from functools import reduce

from pyscf import lib
from pyscf.lib import logger
from pyscf.dft import uks
from pyscf.data.nist import HARTREE2EV
from pyscf import lo
from pyscf.lo import iao
from pyscf import gto
from pyscf.dft.rkspu import set_U, make_minao_lo, mdot

def get_veff(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
    '''Coulomb + XC functional + Hubbard U terms.

    
    
    '''

    if mol is None: mol = ks.mol
    if dm is None: dm = ks.make_rdm1()

    # J + V_xc
    vxc = uks.get_veff(ks, mol=mol, dm=dm, dm_last=dm_last, vhf_last=vhf_last, 
                       hermi=hermi)

    # V_U
    C_ao_lo = ks.C_ao_lo
    ovlp = ks.get_ovlp()
    nlo = C_ao_lo.shape[-1]

    rdm1_lo  = np.zeros((2, nlo, nlo), dtype=np.complex128)
    for s in range(2):
        C_inv = np.dot(C_ao_lo[s].conj().T, ovlp)
        rdm1_lo[s] = mdot(C_inv, dm[s], C_inv.conj().T)

    E_U = 0.0
    logger.info(ks, "-" * 79) # Come back to this later
    with np.printoptions(precision=5, suppress=True, linewidth=1000):
        for idx, val, lab in zip(ks.U_idx, ks.U_val, ks.U_lab):
            lab_string = " "
            for l in lab:
                lab_string += "%9s" %(l.split()[-1])
            lab_sp = lab[0].split()
            logger.info(ks, "local rdm1 of atom %s: ",
                        " ".join(lab_sp[:2]) + " " + lab_sp[2][:2])
            U_mesh = np.ix_(idx, idx)
            for s in range(2):
                P_loc = 0.0
                S = ovlp
                C = C_ao_lo[s][:, idx]
                P = rdm1_lo[s][U_mesh]
                SC = np.dot(S, C)
                vxc[s] += mdot(SC, (np.eye(P.shape[-1]) - P * 2.0)
                               * (val * 0.5), SC.conj().T)
                E_U += (val * 0.5) * (P.trace() - np.dot(P, P).trace() * 0.5)
                P_loc += P
            P_loc = P_loc.real
            logger.info(ks, "%s\n%s", lab_string, P_loc)
            logger.info(ks, "-" * 79)

    if E_U.real < 0.0 and all(np.asarray(ks.U_val) > 0):
        logger.warn(ks, "E_U (%s) is negative...", E_U.real)
    vxc = lib.tag_array(vxc, E_U=E_U)
    return vxc

def energy_elec(mf, dm=None, h1e=None, vhf=None):
    """
    Electronic energy for RKSpU.
    """
    if dm is None: dm = mf.make_rdm1()
    if h1e is None: h1e = mf.get_hcore()
    if vhf is None or getattr(vhf, 'ecoul', None) is None:
        vhf = mf.get_veff(mf.mol, dm) # Question

    e1 = np.einsum('ij,ji->', h1e, dm[0]) + np.einsum('ij,ji->', h1e, dm[1])
    tot_e = e1 + vhf.ecoul + vhf.exc + vhf.E_U
    mf.scf_summary['e1'] = e1.real
    mf.scf_summary['coul'] = vhf.ecoul.real
    mf.scf_summary['exc'] = vhf.exc.real
    mf.scf_summary['E_U'] = vhf.E_U.real

    logger.debug(mf, 'E1 = %s  Ecoul = %s  Exc = %s  EU = %s', e1, vhf.ecoul,
                 vhf.exc, vhf.E_U)
    return tot_e.real, vhf.ecoul + vhf.exc + vhf.E_U

class UKSpU(uks.UKS):
    """
    UKSpU class.
    """
    def __init__(self, mol, xc='LDA,VWN',
                 U_idx=[], U_val=[], C_ao_lo='minao', **kwargs):
        """
        DFT+U args:
            U_idx: can be
                   list of list: each sublist is a set of LO indices to add U.
                   list of string: each string is one kind of LO orbitals,
                                   e.g. ['Ni 3d', '1 O 2pz'], in this case,
                                   LO should be aranged as ao_labels order.
                   or a combination of these two.
            U_val: a list of effective U [in eV], i.e. U-J in Dudarev's DFT+U.
                   each U corresponds to one kind of LO orbitals, should have
                   the same length as U_idx.
            C_ao_lo: LO coefficients, can be
                     np.array, shape ((spin,), nkpts, nao, nlo),
                     string, in 'minao'.

        Kwargs:
            minao_ref: reference for minao orbitals, default is 'MINAO'.
        """
        try:
            uks.UKS.__init__(self, mol, xc=xc)
        except TypeError:
            # backward compatibility
            uks.UKS.__init__(self, mol)
            self.xc = xc

        set_U(self, U_idx, U_val)

        if isinstance(C_ao_lo, str):
            if C_ao_lo == 'minao':
                minao_ref = kwargs.get("minao_ref", "MINAO")
                self.C_ao_lo = make_minao_lo(self, minao_ref)
            else:
                raise NotImplementedError
        else:
            self.C_ao_lo = np.asarray(C_ao_lo)
        if self.C_ao_lo.ndim == 3:
            self.C_ao_lo = np.asarray((self.C_ao_lo, self.C_ao_lo))
        if self.C_ao_lo.ndim == 4:
            if self.C_ao_lo.shape[0] == 1:
                self.C_ao_lo = np.asarray((self.C_ao_lo[0], self.C_ao_lo[0]))
            assert self.C_ao_lo.shape[0] == 2
        else:
            raise ValueError
        
        self._keys = self._keys.union(["U_idx", "U_val", "C_ao_lo", "U_lab"])

    get_veff = get_veff
    energy_elec = energy_elec

    def nuc_grad_method(self):
        raise NotImplementedError