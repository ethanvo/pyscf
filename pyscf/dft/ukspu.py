#!/usr/bin/env python

from pyscf.dft import uks

"""
Restricted DFT+U.
Based on KUKSpU routine.

Refs: PRB, 1998, 57, 1505.
"""
#!/usr/bin/env python

"""
Restricted DFT+U.
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
from pyscf.dft import rks
from pyscf.data.nist import HARTREE2EV
from pyscf import lo
from pyscf.lo import iao
from pyscf import gto

def get_veff(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
    '''Coulomb + XC functional + Hubbard U terms.

    
    
    '''

    if mol is None: mol = ks.mol
    if dm is None: dm = ks.make_rdm1()

    # J + V_xc
    vxc = rks.get_veff(ks, mol=mol, dm=dm, dm_last=dm_last, vhf_last=vhf_last, 
                       hermi=hermi)

    # V_U
    C_ao_lo = ks.C_ao_lo
    ovlp = ks.get_ovlp()
    nlo = C_ao_lo.shape[-1]

    rdm1_lo  = np.zeros((nlo, nlo), dtype=np.complex128)
    C_inv = np.dot(C_ao_lo.conj().T, ovlp)
    rdm1_lo = mdot(C_inv, dm, C_inv.conj().T)

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
            P_loc = 0.0
            S = ovlp
            C = C_ao_lo[:, idx]
            P = rdm1_lo[U_mesh]
            SC = np.dot(S, C)
            vxc += mdot(SC, (np.eye(P.shape[-1]) - P)
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

def energy_elec(ks, dm=None, h1e=None, vhf=None):
    """
    Electronic energy for RKSpU.
    """
    if dm is None: dm = ks.make_rdm1()
    if h1e is None: h1e = ks.get_hcore()
    if vhf is None or getattr(vhf, 'ecoul', None) is None:
        vhf = ks.get_veff(ks.mol, dm)
    e1 = np.einsum('ij,ji->', h1e, dm)
    tot_e = e1 + vhf.ecoul + vhf.exc + vhf.E_U
    ks.scf_summary['e1'] = e1.real
    ks.scf_summary['coul'] = vhf.ecoul.real
    ks.scf_summary['exc'] = vhf.exc.real
    ks.scf_summary['E_U'] = vhf.E_U.real
    logger.debug(ks, 'E1 = %s  Ecoul = %s  Exc = %s  EU = %s', e1, vhf.ecoul,
                 vhf.exc, vhf.E_U)
    return tot_e.real, vhf.ecoul + vhf.exc + vhf.E_U

def set_U(ks, U_idx, U_val):
    """
    Regularize the U_idx and U_val to each atom,
    and set ks.U_idx, ks.U_val, ks.U_lab.
    """
    assert len(U_idx) == len(U_val)
    ks.U_val = []
    ks.U_idx = []
    ks.U_lab = []

    lo_labels = np.asarray(ks.cell.ao_labels())
    for i, idx in enumerate(U_idx):
        if isinstance(idx, str):
            lab_idx = ks.cell.search_ao_label(idx)
            labs = lo_labels[lab_idx]
            labs = zip(lab_idx, labs)
            for j, idxj in it.groupby(labs, key=lambda x: x[1].split()[0]):
                ks.U_idx.append(list(list(zip(*idxj))[0]))
                ks.U_val.append(U_val[i])
        else:
            ks.U_idx.append(copy.deepcopy(idx))
            ks.U_val.append(U_val[i])
    ks.U_val = np.asarray(ks.U_val) / HARTREE2EV
    logger.info(ks, "-" * 79)
    logger.debug(ks, 'U indices and values: ')
    for idx, val in zip(ks.U_idx, ks.U_val):
        ks.U_lab.append(lo_labels[idx])
        logger.debug(ks, '%6s [%.6g eV] ==> %-100s', format_idx(idx),
                     val * HARTREE2EV, "".join(lo_labels[idx]))
    logger.info(ks, "-" * 79)

def make_minao_lo(ks, minao_ref):
    """
    Construct minao local orbitals.
    """
    mol = ks.mol
    nao = mol.nao_nr()
    ovlp = ks.get_ovlp()
    C_ao_minao, labels = proj_ref_ao(mol, minao=minao_ref, return_labels=True)
    C_ao_minao = lo.vec_lowdin(C_ao_minao, ovlp)
    labels = np.asarray(labels)

    C_ao_lo = np.zeros((nao, nao), dtype=np.complex128)
    for idx, lab in zip(ks.U_idx, ks.U_lab):
        idx_minao = [i for i, l in enumerate(labels) if l in lab]
        assert len(idx_minao) == len(idx)
        C_ao_sub = C_ao_minao[:, idx_minao]
        C_ao_lo[:, idx] = C_ao_sub
    return C_ao_lo

def proj_ref_ao(mol, minao='minao', return_labels=False):
    """
    Get a set of reference AO spanned by the calculation basis.
    Not orthogonalized.

    Args:
        return_labels: if True, return the labels as well.
    """
    nao = mol.nao_nr()
    pmol = iao.reference_mol(mol, minao)
    s1 = np.asarray(mol.intor('int1e_ovlp', hermi=1)) # hermi=1?
    s2 = np.asarray(pmol.intor('int1e_ovlp', hermi=1))
    s12 = np.asarray(gto.mole.intor_cross('int1e_ovlp', mol, pmol)) # Most unsure about this line
    s21 = np.swapaxes(s12, -1, -2).conj()
    C_ao_lo = np.zeros((s1.shape[-1], s2.shape[-1]), dtype=np.complex128)
    s1cd = la.cho_factor(s1)
    s2cd = la.cho_factor(s2)
    C_ao_lo = la.cho_solve(s1cd, s12)

    if return_labels:
        labels = pmol.ao_labels()
        return C_ao_lo, labels
    else:
        return C_ao_lo

def mdot(*args):
    return reduce(np.dot, args)

def format_idx(idx_list):
    string = ''
    for k, g in it.groupby(enumerate(idx_list), lambda ix: ix[0] - ix[1]):
        g = list(g)
        if len(g) > 1:
            string += '%d-%d, '%(g[0][1], g[-1][1])
        else:
            string += '%d, '%(g[0][1])
    return string[:-2]

class RKSpU(rks.RKS):
    """
    RKSpU class.
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
            rks.RKS.__init__(self, mol, xc=xc)
        except TypeError:
            # backward compatibility
            rks.RKS.__init__(self, mol)
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
        if self.C_ao_lo.ndim == 4:
            self.C_ao_lo = self.C_ao_lo[0]
        self._keys = self._keys.union(["U_idx", "U_val", "C_ao_lo", "U_lab"])

    get_veff = get_veff
    energy_elec = energy_elec

    def nuc_grad_method(self):
        raise NotImplementedError