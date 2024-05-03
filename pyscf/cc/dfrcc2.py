import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__
from pyscf.cc import ccsd
from pyscf.cc import rintermediates as imd
from pyscf.ao2mo import _ao2mo

BLKMIN = getattr(__config__, 'cc_ccsd_blkmin', 4)
MEMORYMIN = getattr(__config__, 'cc_ccsd_memorymin', 2000)

def energy(mycc, t1=None, eris=None):
    '''CC2 correlation energy'''
    if t1 is None: t1 = mycc.t1
    if eris is None: eris = mycc.ao2mo()

    nocc, nvir = t1.shape
    fock = eris.fock
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + mycc.level_shift
    eia = mo_e_o[:,None] - mo_e_v
    Lov = eris.Lov
    Bov = make_Bov(mycc, t1, eris)
    e = np.einsum('ia,ia', fock[:nocc,nocc:], t1) * 2
    for i in range(nocc):
        t2i  =   lib.einsum('La,Ljb->jab', Bov[:, i, :], Bov)
        ejab = lib.direct_sum('a,jb->jab', eia[i, :], eia)
        t2i /= ejab
        taui = t2i + lib.einsum('a,jb->jab', t1[i, :], t1)
        e   += 2*lib.einsum('jab,La,Ljb', taui, Lov[:, i, :], Lov)
        e   -=   lib.einsum('iab,Lia,Lb', taui, Lov, Lov[:, i, :])

    if abs(e.imag) > 1e-4:
        logger.warn(mycc, 'Non-zero imaginary part found in CC2 energy %s', e)
    return e.real

def update_t1(cc, t1, eris):
    # Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004) Eqs.(35)-(36)
    assert(isinstance(eris, ccsd._ChemistsERIs))
    nocc, nvir = t1.shape
    fock = eris.fock
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + cc.level_shift

    eia = mo_e_o[:,None] - mo_e_v

    Loo = eris.Loo
    Lov = eris.Lov
    Lvv = eris.Lvv

    Bov = make_Bov(cc, t1, eris)

    foo = fock[:nocc,:nocc].copy()
    fov = fock[:nocc,nocc:].copy()
    fvv = fock[nocc:,nocc:].copy()

    Fki = np.zeros_like(foo)
    Fac = np.zeros_like(fvv)
    Fkc = np.zeros_like(fov)
    for i in range(nocc):
        t2i        =   lib.einsum('La,Ljb->jab', Bov[:, i, :], Bov)
        ejab       = lib.direct_sum('a,jb->jab', eia[i, :], eia)
        t2i       /= ejab
        Fki[:, i] += 2*lib.einsum('Lkc,Lld,lcd->k', Lov, Lov, t2i)
        Fki[:, i] -=   lib.einsum('Lkd,Llc,lcd->k', Lov, Lov, t2i)
        Fac       +=-2*lib.einsum('Lc,Lld,lad->ac', Lov[:, i, :], Lov, t2i)
        Fac       +=   lib.einsum('Ld,Llc,lad->ac', Lov[:, i, :], Lov, t2i)
    Fki += 2*lib.einsum('Lkc,Lld,ic,ld->ki', Lov, Lov, t1, t1)
    Fki -=   lib.einsum('Lkd,Llc,ic,ld->ki', Lov, Lov, t1, t1)
    Fac -= 2*lib.einsum('Lkc,Lld,ka,ld->ac', Lov, Lov, t1, t1)
    Fac +=   lib.einsum('Lkd,Llc,ka,ld->ac', Lov, Lov, t1, t1)
    Fkc  = 2*np.einsum('Lkc,Lld,ld->kc', Lov, Lov, t1)
    Fkc -=   np.einsum('Lkd,Llc,ld->kc', Lov, Lov, t1)
    Fki += foo
    Fac += fvv
    Fkc += fov
    Foo = Fki
    Fvv = Fac
    Fov = Fkc

    # Move energy terms to the other side
    Foo[np.diag_indices(nocc)] -= mo_e_o
    Fvv[np.diag_indices(nvir)] -= mo_e_v

    # T1 equation
    t1new  =-2*np.einsum('kc,ka,ic->ia', fov, t1, t1)
    t1new +=   np.einsum('ac,ic->ia', Fvv, t1)
    t1new +=  -np.einsum('ki,ka->ia', Foo, t1)
    for i in range(nocc):
        t2i          =   lib.einsum('La,Ljb->jab', Bov[:, i, :], Bov)
        ejab = lib.direct_sum('a,jb->jab', eia[i, :], eia)
        t2i /= ejab
        t1new       += 2*lib.einsum('c,ica->ia', Fov[i, :], t2i)
        t1new[i, :] +=  -lib.einsum('kc,kca->a', Fov, t2i)
        t1new[i, :] += 2*lib.einsum('Lkd,Lac,kcd->a', Lov, Lvv, t2i)
        t1new[i, :] +=  -lib.einsum('Lkc,Lad,kcd->a', Lov, Lvv, t2i)
        t1new       +=-2*lib.einsum('Llc,Li,lac->ia', Lov, Loo[:, i, :], t2i)
        t1new       +=   lib.einsum('Lc,Lli,lac->ia', Lov[:, i, :], Loo, t2i)

    t1new +=   np.einsum('kc,ic,ka->ia', Fov, t1, t1)
    t1new += fov.conj()
    t1new += 2*np.einsum('Lkc,Lia,kc->ia', Lov, Lov, t1)
    t1new +=  -np.einsum('Lki,Lac,kc->ia', Loo, Lvv, t1)
    t1new += 2*lib.einsum('Lkd,Lac,kd,ic->ia', Lov, Lvv, t1, t1)
    t1new +=  -lib.einsum('Lkc,Lad,kd,ic->ia', Lov, Lvv, t1, t1)
    t1new +=-2*lib.einsum('Llc,Lki,lc,ka->ia', Lov, Loo, t1, t1)
    t1new +=   lib.einsum('Lkc,Lli,lc,ka->ia', Lov, Loo, t1, t1)

    t1new /= eia

    return t1new

def make_Bov(cc, t1, eris):
    nocc = eris.nocc
    nmo = eris.fock.shape[0]
    nvir = nmo - nocc
    naux = cc._scf.with_df.get_naoaux()
    C = eris.mo_coeff.copy()
    X = C[:, nocc:] - lib.einsum('ui,ia->ua', C[:, :nocc], t1)
    Y = C[:, :nocc] + lib.einsum('ua,ia->ui', C[:, nocc:], t1)
    C[:, :nocc] = Y
    C[:, nocc:] = X
    Bov = np.empty((naux,nocc,nvir))
    ijslice = (0, nmo, 0, nmo)
    Lpq = None
    p1 = 0
    for eri1 in cc._scf.with_df.loop():
        Lpq = _ao2mo.nr_e2(eri1, C, ijslice, aosym='s2', out=Lpq).reshape(-1,nmo,nmo)
        p0, p1 = p1, p1 + Lpq.shape[0]
        Bov[p0:p1] = Lpq[:,:nocc,nocc:]
    return Bov

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

def update_amps(cc, t1, eris):
    t1new = update_t1(cc, t1, eris)
    return t1new

# t1: ia
# t2: ijab
def kernel(mycc, eris=None, t1=None, max_cycle=50, tol=1e-8,
           tolnormt=1e-6, verbose=None):
    log = logger.new_logger(mycc, verbose)
    if eris is None:
        eris = mycc.ao2mo(mycc.mo_coeff)
    nocc = eris.nocc
    nmo = eris.fock.shape[0]
    nvir = nmo - nocc
    if t1 is None:
        t1 = np.zeros((nocc, nvir), dtype=eris.Loo.dtype)

    cput1 = cput0 = (logger.process_clock(), logger.perf_counter())
    eold = 0
    eccsd = mycc.energy(t1, eris)
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
        eold, eccsd = eccsd, mycc.energy(t1, eris)
        log.info('cycle = %d  E_corr(CC2) = %.15g  dE = %.9g  norm(t1) = %.6g',
                 istep+1, eccsd, eccsd - eold, normt)
        cput1 = log.timer('CC2 iter', *cput1)
        if abs(eccsd-eold) < tol and normt < tolnormt:
            conv = True
            break
    log.timer('CC2', *cput0)
    return conv, eccsd, t1

class DFRCC2(ccsd.CCSD):
    '''Density fitted restricted CC2 without T2 formation

    '''
    energy = energy
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
        eris.Loo = Loo.reshape(naux,nocc,nocc)
        eris.Lov = Lov.reshape(naux,nocc,nvir)
        eris.Lvo = Lvo.reshape(naux,nvir,nocc)
        eris.Lvv = lib.unpack_tril(Lvv).reshape(naux,nvir,nvir)
        log.timer('CC2 integral transformation', *cput0)
        return eris

