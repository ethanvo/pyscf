import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__
from pyscf.cc import ccsd
from pyscf.cc import rintermediates as imd

BLKMIN = getattr(__config__, 'cc_ccsd_blkmin', 4)
MEMORYMIN = getattr(__config__, 'cc_ccsd_memorymin', 2000)

def update_t1(cc, t1, t2, eris):
    # Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004) Eqs.(35)-(36)
    assert(isinstance(eris, ccsd._ChemistsERIs))
    nocc, nvir = t1.shape
    fock = eris.fock
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + cc.level_shift

    fov = fock[:nocc,nocc:].copy()

    Foo = imd.cc_Foo(t1,t2,eris)
    Fvv = imd.cc_Fvv(t1,t2,eris)
    Fov = imd.cc_Fov(t1,t2,eris)

    # Move energy terms to the other side
    Foo[np.diag_indices(nocc)] -= mo_e_o
    Fvv[np.diag_indices(nvir)] -= mo_e_v

    # T1 equation
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

def update_t2(cc, t1, t2, eris):
    # Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004) Eqs.(35)-(36)
    assert(isinstance(eris, ccsd._ChemistsERIs))
    nocc, nvir = t1.shape
    fock = eris.fock
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + cc.level_shift

    fov = fock[:nocc,nocc:].copy()
    foo = fock[:nocc,:nocc].copy()
    fvv = fock[nocc:,nocc:].copy()

    Foo = imd.cc_Foo(t1,t2,eris)
    Fvv = imd.cc_Fvv(t1,t2,eris)

    # Move energy terms to the other side
    Foo[np.diag_indices(nocc)] -= mo_e_o
    Fvv[np.diag_indices(nvir)] -= mo_e_v

    eris_ovvv = np.asarray(eris.get_ovvv())
    eris_ovoo = np.asarray(eris.ovoo, order='C')

    # T2 equation
    tmp2  = lib.einsum('kibc,ka->abic', eris.oovv, -t1)
    tmp2 += np.asarray(eris_ovvv).conj().transpose(1,3,0,2)
    tmp = lib.einsum('abic,jc->ijab', tmp2, t1)
    t2new = tmp + tmp.transpose(1,0,3,2)
    tmp2  = lib.einsum('kcai,jc->akij', eris.ovvo, t1)
    tmp2 += eris_ovoo.transpose(1,3,0,2).conj()
    tmp = lib.einsum('akij,kb->ijab', tmp2, t1)
    t2new -= tmp + tmp.transpose(1,0,3,2)
    t2new += np.asarray(eris.ovov).conj().transpose(0,2,1,3)
    Woooo2 = np.asarray(eris.oooo).transpose(0,2,1,3).copy()
    Woooo2 += lib.einsum('lcki,jc->klij', eris_ovoo, t1)
    Woooo2 += lib.einsum('kclj,ic->klij', eris_ovoo, t1)
    Woooo2 += lib.einsum('kcld,ic,jd->klij', eris.ovov, t1, t1)
    t2new += lib.einsum('klij,ka,lb->ijab', Woooo2, t1, t1)
    Wvvvv = lib.einsum('kcbd,ka->abcd', eris_ovvv, -t1)
    Wvvvv = Wvvvv + Wvvvv.transpose(1,0,3,2)
    Wvvvv += np.asarray(imd._get_vvvv(eris)).transpose(0,2,1,3)
    t2new += lib.einsum('abcd,ic,jd->ijab', Wvvvv, t1, t1)
    Lvv2 = fvv - np.einsum('kc,ka->ac', fov, t1)
    Lvv2 -= np.diag(np.diag(fvv))
    tmp = lib.einsum('ac,ijcb->ijab', Lvv2, t2)
    t2new += (tmp + tmp.transpose(1,0,3,2))
    Loo2 = foo + np.einsum('kc,ic->ki', fov, t1)
    Loo2 -= np.diag(np.diag(foo))
    tmp = lib.einsum('ki,kjab->ijab', Loo2, t2)
    t2new -= (tmp + tmp.transpose(1,0,3,2))

    eia = mo_e_o[:,None] - mo_e_v
    eijab = lib.direct_sum('ia,jb->ijab',eia,eia)
    t2new /= eijab

    return t2new

def update_amps(cc, t1, t2, eris):
    t1new = update_t1(cc, t1, t2, eris)
    t2new = update_t2(cc, t1, t2, eris)
    return t1new, t2new

# t1: ia
# t2: ijab
def kernel(mycc, eris=None, t1=None, t2=None, max_cycle=50, tol=1e-8,
           tolnormt=1e-6, verbose=None):
    log = logger.new_logger(mycc, verbose)
    if eris is None:
        eris = mycc.ao2mo(mycc.mo_coeff)
    if t1 is None and t2 is None:
        t1, t2 = mycc.get_init_guess(eris)
    elif t2 is None:
        t2 = mycc.get_init_guess(eris)[1]

    cput1 = cput0 = (logger.process_clock(), logger.perf_counter())
    eold = 0
    eccsd = mycc.energy(t1, t2, eris)
    log.info('Init E_corr(CCSD) = %.15g', eccsd)

    if isinstance(mycc.diis, lib.diis.DIIS):
        adiis = mycc.diis
    elif mycc.diis:
        adiis = lib.diis.DIIS(mycc, mycc.diis_file, incore=mycc.incore_complete)
        adiis.space = mycc.diis_space
    else:
        adiis = None

    conv = False
    for istep in range(max_cycle):
        t1new, t2new = mycc.update_amps(t1, t2, eris)
        tmpvec = mycc.amplitudes_to_vector(t1new, t2new)
        tmpvec -= mycc.amplitudes_to_vector(t1, t2)
        normt = np.linalg.norm(tmpvec)
        tmpvec = None
        if mycc.iterative_damping < 1.0:
            alpha = mycc.iterative_damping
            t1new = (1-alpha) * t1 + alpha * t1new
            t2new *= alpha
            t2new += (1-alpha) * t2
        t1, t2 = t1new, t2new
        t1new = t2new = None
        t1, t2 = mycc.run_diis(t1, t2, istep, normt, eccsd-eold, adiis)
        eold, eccsd = eccsd, mycc.energy(t1, t2, eris)
        log.info('cycle = %d  E_corr(CCSD) = %.15g  dE = %.9g  norm(t1,t2) = %.6g',
                 istep+1, eccsd, eccsd - eold, normt)
        cput1 = log.timer('CCSD iter', *cput1)
        if abs(eccsd-eold) < tol and normt < tolnormt:
            conv = True
            break
    log.timer('CCSD', *cput0)
    return conv, eccsd, t1, t2

class RCC2(ccsd.CCSD):
    '''restricted CCSD with IP-EOM, EA-EOM, EE-EOM, and SF-EOM capabilities

    Ground-state CCSD is performed in optimized ccsd.CCSD and EOM is performed here.
    '''
    kernel = kernel
    update_amps = update_amps