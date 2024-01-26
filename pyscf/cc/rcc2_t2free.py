import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__
from pyscf.cc import ccsd
from pyscf.cc import rintermediates as imd

BLKMIN = getattr(__config__, 'cc_ccsd_blkmin', 4)
MEMORYMIN = getattr(__config__, 'cc_ccsd_memorymin', 2000)


def make_t2(cc, t1, eris):
    assert(isinstance(eris, ccsd._ChemistsERIs))
    nocc, nvir = t1.shape
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + cc.level_shift

    C = eris.mo_coeff.copy()
    X = C[:, nocc:] - lib.einsum('ui,ia->ua', C[:, :nocc], t1)
    Y = C[:, :nocc] + lib.einsum('ua,ia->ui', C[:, nocc:], t1)
    C[:, :nocc] = Y
    C[:, nocc:] = X

    t1_eris = cc.ao2mo(mo_coeff=C)
    t2 = np.asarray(t1_eris.ovov).transpose(0,2,1,3)

    eia = mo_e_o[:,None] - mo_e_v
    eijab = lib.direct_sum('ia,jb->ijab',eia,eia)
    t2 /= eijab

    return t2

def update_t1(cc, t1, eris):
    # Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004) Eqs.(35)-(36)
    assert(isinstance(eris, ccsd._ChemistsERIs))
    nocc, nvir = t1.shape
    fock = eris.fock
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + cc.level_shift

    fov = fock[:nocc,nocc:].copy()

    t2 = cc.make_t2(t1, eris)

    # T1 equation
    Foo = imd.cc_Foo(t1,t2,eris)
    Fvv = imd.cc_Fvv(t1,t2,eris)
    Fov = imd.cc_Fov(t1,t2,eris)

    # Move energy terms to the other side
    Foo[np.diag_indices(nocc)] -= mo_e_o
    Fvv[np.diag_indices(nvir)] -= mo_e_v

    t1new  =-2*np.einsum('kc,ka,ic->ia', fov, t1, t1)
    t1new +=   np.einsum('ac,ic->ia', Fvv, t1)
    t1new +=  -np.einsum('ki,ka->ia', Foo, t1)
    t1new += 2*np.einsum('kc,kica->ia', Fov, t2)
    t1new +=  -np.einsum('kc,ikca->ia', Fov, t2)
    t1new +=   np.einsum('kc,ic,ka->ia', Fov, t1, t1)
    t1new += fov.conj()
    t1new += 2*np.einsum('kcai,kc->ia', eris.ovvo, t1)
    t1new +=  -np.einsum('kiac,kc->ia', eris.oovv, t1)
    eris_ovvv = np.asarray(eris.get_ovvv())
    t1new += 2*lib.einsum('kdac,ikcd->ia', eris_ovvv, t2)
    t1new +=  -lib.einsum('kcad,ikcd->ia', eris_ovvv, t2)
    t1new += 2*lib.einsum('kdac,kd,ic->ia', eris_ovvv, t1, t1)
    t1new +=  -lib.einsum('kcad,kd,ic->ia', eris_ovvv, t1, t1)
    eris_ovoo = np.asarray(eris.ovoo, order='C')
    t1new +=-2*lib.einsum('lcki,klac->ia', eris_ovoo, t2)
    t1new +=   lib.einsum('kcli,klac->ia', eris_ovoo, t2)
    t1new +=-2*lib.einsum('lcki,lc,ka->ia', eris_ovoo, t1, t1)
    t1new +=   lib.einsum('kcli,lc,ka->ia', eris_ovoo, t1, t1)

    eia = mo_e_o[:,None] - mo_e_v
    t1new /= eia

    return t1new

# t1: ia
def kernel(mycc, eris=None, t1=None, max_cycle=50, tol=1e-8,
           tolnormt=1e-6, verbose=None):
    log = logger.new_logger(mycc, verbose)
    if eris is None:
        eris = mycc.ao2mo(mycc.mo_coeff)
    if t1 is None:
        t1 = mycc.get_init_guess(eris)[0]

    cput1 = cput0 = (logger.process_clock(), logger.perf_counter())
    eold = 0
    t2 = mycc.make_t2(t1, eris)
    eccsd = mycc.energy(t1, t2, eris)
    log.info('Init E_corr(CC2) = %.15g', eccsd)

    if isinstance(mycc.diis, lib.diis.DIIS):
        adiis = mycc.diis
    elif mycc.diis:
        adiis = lib.diis.DIIS(mycc, mycc.diis_file, incore=mycc.incore_complete)
        adiis.space = mycc.diis_space
    else:
        adiis = None

    conv = False
    for istep in range(max_cycle):
        t1new = mycc.update_t1(t1, eris)
        tmpvec = t1new.ravel() - t1.ravel()
        normt = np.linalg.norm(tmpvec)
        tmpvec = None
        if mycc.iterative_damping < 1.0:
            alpha = mycc.iterative_damping
            t1new = (1-alpha) * t1 + alpha * t1new
        t1 = t1new
        #t1new = t2new = None
        t1 = mycc.run_diis(t1, istep, normt, eccsd-eold, adiis)
        t2 = mycc.make_t2(t1, eris)
        eold, eccsd = eccsd, mycc.energy(t1, t2, eris)
        log.info('cycle = %d  E_corr(CC2) = %.15g  dE = %.9g  norm(t1) = %.6g',
                 istep+1, eccsd, eccsd - eold, normt)
        cput1 = log.timer('CC2 iter', *cput1)
        if abs(eccsd-eold) < tol and normt < tolnormt:
            conv = True
            break
    log.timer('CC2', *cput0)
    return conv, eccsd, t1

class RCC2_T2FREE(ccsd.CCSD):
    '''restricted CC2
    '''
    kernel = kernel
    update_t1 = update_t1
    make_t2 = make_t2

    def run_diis(self, t1, istep, normt, de, adiis):
        nocc, nvir = t1.shape
        if (adiis and
            istep >= self.diis_start_cycle and
            abs(de) < self.diis_start_energy_diff):
            vec = t1.ravel() 
            t1 = adiis.update(vec).reshape((nocc, nvir))
            logger.debug1(self, 'DIIS for step %d', istep)
        return t1

