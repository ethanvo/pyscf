import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__
from pyscf.cc import ccsd
from pyscf.cc import rintermediates as imd
from pyscf.ao2mo import _ao2mo

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

    Loo = eris.Loo
    Lov = eris.Lov
    Lvv = eris.Lvv

    eia = mo_e_o[:,None] - mo_e_v
    
    # T1 equation
    t1new  =-2*np.einsum('kc,ka,ic->ia', fov, t1, t1)
    t1new +=   np.einsum('ac,ic->ia', Fvv, t1)
    t1new +=  -np.einsum('ki,ka->ia', Foo, t1)
    t1new += 2*np.einsum('kc,kica->ia', Fov, t2)
    t1new +=  -np.einsum('kc,ikca->ia', Fov, t2)
    t1new +=   np.einsum('kc,ic,ka->ia', Fov, t1, t1)
    t1new += fov.conj()
    t1new += 2*np.einsum('Lkc,Lia,kc->ia', Lov, Lov, t1)
    t1new +=  -np.einsum('Lki,Lac,kc->ia', Loo, Lvv, t1)
    t1new += 2*lib.einsum('Lkd,Lac,ikcd->ia', Lov, Lvv, t2)
    t1new +=  -lib.einsum('Lkc,Lad,ikcd->ia', Lov, Lvv, t2)
    t1new += 2*lib.einsum('Lkd,Lac,kd,ic->ia', Lov, Lvv, t1, t1)
    t1new +=  -lib.einsum('Lkc,Lad,kd,ic->ia', Lov, Lvv, t1, t1)
    t1new +=-2*lib.einsum('Llc,Lki,klac->ia', Lov, Loo, t2)
    t1new +=   lib.einsum('Lkc,Lli,klac->ia', Lov, Loo, t2)
    t1new +=-2*lib.einsum('Llc,Lki,lc,ka->ia', Lov, Loo, t1, t1)
    t1new +=   lib.einsum('Lkc,Lli,lc,ka->ia', Lov, Loo, t1, t1)

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

    Loo = eris.Loo
    Lov = eris.Lov
    Lvv = eris.Lvv

    # T2 equation
    tmp2  = lib.einsum('Lki,Lbc,ka->abic', Loo, Lvv, -t1)
    tmp2 += lib.einsum('Lia,Lbc->acib', Lov, Lvv).conj()
    tmp = lib.einsum('abic,jc->ijab', tmp2, t1)
    t2new = tmp + tmp.transpose(1,0,3,2)
    tmp2  = lib.einsum('Lkc,Lia,jc->akij', Lov, Lov, t1)
    tmp2 += lib.einsum('Lia,Ljk->akij', Lov, Loo).conj()
    tmp = lib.einsum('akij,kb->ijab', tmp2, t1)
    t2new -= tmp + tmp.transpose(1,0,3,2)
    t2new += lib.einsum('Lia,Ljb->ijab', Lov, Lov).conj()
    Woooo2 = lib.einsum('Lij,Lkl->ikjl', Loo, Loo)
    Woooo2 += lib.einsum('Llc,Lki,jc->klij', Lov, Loo, t1)
    Woooo2 += lib.einsum('Lkc,Llj,ic->klij', Lov, Loo, t1)
    Woooo2 += lib.einsum('Lkc,Lld,ic,jd->klij', Lov, Lov, t1, t1)
    t2new += lib.einsum('klij,ka,lb->ijab', Woooo2, t1, t1)
    Wvvvv = lib.einsum('Lkc,Lbd,ka->abcd', Lov, Lvv, -t1)
    Wvvvv = Wvvvv + Wvvvv.transpose(1,0,3,2)
    Wvvvv += lib.einsum('Lab,Lcd->acbd', Lvv, Lvv)
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

def make_t2(cc, t1, eris):
    nocc = eris.nocc
    nmo = eris.fock.shape[0]
    nvir = nmo - nocc
    naux = cc._scf.with_df.get_naoaux()
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + cc.level_shift
    eia = mo_e_o[:,None] - mo_e_v
    eijab = lib.direct_sum('ia,jb->ijab',eia,eia)
    C = eris.mo_coeff.copy()
    X = C[:, nocc:] - lib.einsum('ui,ia->ua', C[:, :nocc], t1)
    Y = C[:, :nocc] + lib.einsum('ua,ia->ui', C[:, nocc:], t1)
    C[:, :nocc] = Y
    C[:, nocc:] = X
    Lov = np.empty((naux,nocc,nvir))
    ijslice = (0, nmo, 0, nmo)
    Lpq = None
    p1 = 0
    for eri1 in cc._scf.with_df.loop():
        Lpq = _ao2mo.nr_e2(eri1, C, ijslice, aosym='s2', out=Lpq).reshape(-1,nmo,nmo)
        p0, p1 = p1, p1 + Lpq.shape[0]
        Lov[p0:p1] = Lpq[:,:nocc,nocc:]
    t2 = lib.einsum('Lia,Ljb->ijab', Lov, Lov)
    t2 /= eijab
    return t2

def update_amps(cc, t1, t2, eris):
    t2 = make_t2(cc, t1, eris)
    t1new = update_t1(cc, t1, t2, eris)
    t2new = make_t2(cc, t1new, eris)
    # Compare t2_t1 with t2
    print("T2 COMPARE")
    #print(np.allclose(t2_t1, t2))
    #t2new = update_t2(cc, t1, t2, eris)
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

class DFRCC2(ccsd.CCSD):
    '''restricted CCSD with IP-EOM, EA-EOM, EE-EOM, and SF-EOM capabilities

    Ground-state CCSD is performed in optimized ccsd.CCSD and EOM is performed here.
    '''
    kernel = kernel
    update_amps = update_amps

    def ao2mo(self, mo_coeff=None):
        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.stdout, self.verbose)
        eris = ccsd._ChemistsERIs()
        eris._common_init_(self, mo_coeff)

        mo_coeff = np.asarray(eris.mo_coeff, order='F')
        nocc = eris.nocc
        nao, nmo = mo_coeff.shape
        nvir = nmo - nocc
        nvir_pair = nvir*(nvir+1)//2

        naux = self._scf.with_df.get_naoaux()
        Loo = np.empty((naux,nocc,nocc))
        Lov = np.empty((naux,nocc,nvir))
        Lvo = np.empty((naux,nvir,nocc))
        Lvv = np.empty((naux,nvir_pair))
        ijslice = (0, nmo, 0, nmo)
        Lpq = None
        p1 = 0
        for eri1 in self._scf.with_df.loop():
            Lpq = _ao2mo.nr_e2(eri1, mo_coeff, ijslice, aosym='s2', out=Lpq).reshape(-1,nmo,nmo)
            p0, p1 = p1, p1 + Lpq.shape[0]
            Loo[p0:p1] = Lpq[:,:nocc,:nocc]
            Lov[p0:p1] = Lpq[:,:nocc,nocc:]
            Lvo[p0:p1] = Lpq[:,nocc:,:nocc]
            Lvv[p0:p1] = lib.pack_tril(Lpq[:,nocc:,nocc:].reshape(-1,nvir,nvir))
        Loo = Loo.reshape(naux,nocc*nocc)
        Lov = Lov.reshape(naux,nocc*nvir)
        Lvo = Lvo.reshape(naux,nocc*nvir)

        eris.feri1 = lib.H5TmpFile()
        eris.oooo = eris.feri1.create_dataset('oooo', (nocc,nocc,nocc,nocc), 'f8')
        eris.oovv = eris.feri1.create_dataset('oovv', (nocc,nocc,nvir,nvir), 'f8', chunks=(nocc,nocc,1,nvir))
        eris.ovoo = eris.feri1.create_dataset('ovoo', (nocc,nvir,nocc,nocc), 'f8', chunks=(nocc,1,nocc,nocc))
        eris.ovvo = eris.feri1.create_dataset('ovvo', (nocc,nvir,nvir,nocc), 'f8', chunks=(nocc,1,nvir,nocc))
        eris.ovov = eris.feri1.create_dataset('ovov', (nocc,nvir,nocc,nvir), 'f8', chunks=(nocc,1,nocc,nvir))
        eris.ovvv = eris.feri1.create_dataset('ovvv', (nocc,nvir,nvir_pair), 'f8')
        eris.vvvv = eris.feri1.create_dataset('vvvv', (nvir_pair,nvir_pair), 'f8')
        eris.oooo[:] = lib.ddot(Loo.T, Loo).reshape(nocc,nocc,nocc,nocc)
        eris.ovoo[:] = lib.ddot(Lov.T, Loo).reshape(nocc,nvir,nocc,nocc)
        eris.oovv[:] = lib.unpack_tril(lib.ddot(Loo.T, Lvv)).reshape(nocc,nocc,nvir,nvir)
        eris.ovvo[:] = lib.ddot(Lov.T, Lvo).reshape(nocc,nvir,nvir,nocc)
        eris.ovov[:] = lib.ddot(Lov.T, Lov).reshape(nocc,nvir,nocc,nvir)
        eris.ovvv[:] = lib.ddot(Lov.T, Lvv).reshape(nocc,nvir,nvir_pair)
        eris.vvvv[:] = lib.ddot(Lvv.T, Lvv)
        eris.Loo = Loo.reshape(naux,nocc,nocc)
        eris.Lov = Lov.reshape(naux,nocc,nvir)
        eris.Lvo = Lvo.reshape(naux,nvir,nocc)
        eris.Lvv = lib.unpack_tril(Lvv).reshape(naux,nvir,nvir)
        log.timer('CCSD integral transformation', *cput0)
        return eris
