import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf import __config__
from pyscf.cc import ccsd
from pyscf.cc.ccsd import _ChemistsERIs

BLKMIN = getattr(__config__, 'cc_ccsd_blkmin', 4)
MEMORYMIN = getattr(__config__, 'cc_ccsd_memorymin', 2000)
'''
CC2 T2 equations
+'iajb'
+'cajb,ic->ijab'
-'ikjb,ka->ijab'
+'iadb,jd->ijab'
-'iajl,lb->ijab'
-'ckjb,ic,ka->ijab'
+'cadb,ic,jd->ijab'
-'cajl,ic,lb->ijab'
-'ikdb,ka,jd->ijab'
+'ikjl,ka,lb->ijab'
-'iadl,jd,lb->ijab'
-'ckdb,ic,ka,jd->ijab'
+'ckjl,ic,ka,lb->ijab'
-'cadl,ic,jd,lb->ijab'
+'ikdl,ka,jd,lb->ijab'
+'ckdl,ic,ak,jd,lb->ijab'
'''

def form_t2i(cc, t1, eris, i):
    Loo = eris.Loo
    Lov = eris.Lov
    Lvv = eris.Lvv
    nocc, nvir = t1.shape
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + cc.level_shift
    eia = mo_e_o[:,None] - mo_e_v
    t2i  = lib.einsum('La,Ljb->jab', Lov[:, i, :], Lov)
    t2i += lib.einsum('Lca,Ljb,c->jab', Lvv, Lov, t1[i])
    t2i -= lib.einsum('Lk,Ljb,ka->jab', Loo[:, i, :], Lov, t1)
    t2i += lib.einsum('La,Ldb,jd->jab', Lov[:, i, :], Lvv, t1)
    t2i -= lib.einsum('La,Ljl,lb->jab', Lov[:, i, :], Loo, t1)
    t2i -= lib.einsum('Lkc,Ljb,c,ka->jab', Lov, Lov, t1[i], t1) # transpose o->v, v->o
    t2i += lib.einsum('Lca,Ldb,c,jd->jab', Lvv, Lvv, t1[i], t1)
    t2i -= lib.einsum('Lca,Ljl,c,lb->jab', Lvv, Loo, t1[i], t1)
    t2i -= lib.einsum('Lk,Ldb,ka,jd->jab', Loo[:, i, :], Lvv, t1, t1)
    t2i += lib.einsum('Lk,Ljl,ka,lb->jab', Loo[:, i, :], Loo, t1, t1)
    t2i -= lib.einsum('La,Lld,jd,lb->jab', Lov[:, i, :], Lov, t1, t1) # transpose o->v, v->o
    t2i -= lib.einsum('Lkc,Ldb,c,ka,jd->jab', Lov, Lvv, t1[i], t1, t1) # transpose
    t2i += lib.einsum('Lkc,Ljl,c,ka,lb->jab', Lov, Loo, t1[i], t1, t1) # transpose
    t2i -= lib.einsum('Lca,Lld,c,jd,lb->jab', Lvv, Lov, t1[i], t1, t1) # transpose
    t2i += lib.einsum('Lk,Lld,ka,jd,lb->jab', Loo[:, i, :], Lov, t1, t1, t1) # transpose
    t2i += lib.einsum('Lkc,Lld,c,ka,jd,lb->jab', Lov, Lov, t1[i], t1, t1, t1) # transpose
    ejab = lib.direct_sum('jb+a->jab', eia, eia[i])
    t2i /= ejab
    return t2i


def energy(cc, t1=None, eris=None):
    '''RCCSD correlation energy'''
    if t1 is None: t1 = cc.t1
    if eris is None: eris = cc.ao2mo()

    nocc, nvir = t1.shape
    fock = eris.fock
    
    Lov = eris.Lov
    Lvv = eris.Lvv

    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + cc.level_shift
    eia = mo_e_o[:,None] - mo_e_v
    
    e = 2*np.einsum('ia,ia', fock[:nocc,nocc:], t1)
    e += 2*lib.einsum('Lia,Ljb,ia,jb', Lov, Lov, t1, t1)
    e +=  -lib.einsum('Lib,Lja,ia,jb', Lov, Lov, t1, t1)
    for i in range(nocc):
        t2i = form_t2i(cc, t1, eris, i)
        e += 2*lib.einsum('jab,La,Ljb', t2i, Lov[:, i, :], Lov)
        e +=  -lib.einsum('jab,Lb,Lja', t2i, Lov[:, i, :], Lov)

    if abs(e.imag) > 1e-4:
        logger.warn(cc, 'Non-zero imaginary part found in RCCSD energy %s', e)
    return e.real

def update_t1(cc, t1, eris):
    # Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004) Eqs.(35)-(36)
    assert(isinstance(eris, ccsd._ChemistsERIs))
    nocc, nvir = t1.shape
    naux = eris.naux
    fock = eris.fock
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + cc.level_shift
    Loo = eris.Loo
    Lov = eris.Lov
    Lvv = eris.Lvv

    fov = fock[:nocc,nocc:].copy()
    foo = fock[:nocc,:nocc].copy()
    fvv = fock[nocc:,nocc:].copy()

    eia = mo_e_o[:,None] - mo_e_v

    Foo = foo.copy()
    Fvv = fvv.copy()
    Fov = fov.copy()
    Fov += 2*lib.einsum('Lkc,Lld,ld->kc', Lov, Lov, t1)
    Fov -=   lib.einsum('Lkd,Llc,ld->kc', Lov, Lov, t1)

    t1new = np.zeros_like(t1)
    # T2 contractions
    for i in range(nocc):
        t2i = form_t2i(cc, t1, eris, i)

        Foo[:, i] += 2*lib.einsum('Lkc,Lld,lcd->k', Lov, Lov, t2i)
        Foo[:, i] -=   lib.einsum('Lkd,Llc,lcd->k', Lov, Lov, t2i)

        vov = lib.einsum('Lc,Lld->cld', Lov[:, i, :], Lov)
        Fvv -= 2*lib.einsum('cld,lad->ac', vov, t2i)
        Fvv +=   lib.einsum('dlc,lad->ac', vov, t2i)

        t1new += 2*lib.einsum('c,ica->ia', Fov[i], t2i)
        t1new[i] += -lib.einsum('kc,kca->a', Fov, t2i)
        t1new[i] += 2*lib.einsum('Lkd,Lac,kcd->a', Lov, Lvv, t2i)
        t1new[i] +=  -lib.einsum('Lkc,Lad,kcd->a', Lov, Lvv, t2i)
        t1new +=-2*lib.einsum('Llc,Li,lac->ia', Lov,Loo[:, i, :], t2i)
        t1new +=   lib.einsum('Lc,Lli,lac->ia', Lov[:, i, :], Loo, t2i)

    Foo += 2*lib.einsum('Lkc,Lld,ic,ld->k', Lov, Lov, t1, t1)
    Foo -=   lib.einsum('Lkd,Llc,ic,ld->k', Lov, Lov, t1, t1)
    Fvv -= 2*lib.einsum('Lkc,Lld,ka,ld->ac', Lov, Lov, t1, t1)
    Fvv +=   lib.einsum('Lkd,Llc,la,ld->ac', Lov, Lov, t1, t1)
    Foo[np.diag_indices(nocc)] -= mo_e_o
    Fvv[np.diag_indices(nvir)] -= mo_e_v

    # T1 equation
    t1new +=-2*np.einsum('kc,ka,ic->ia', fov, t1, t1)
    t1new +=   np.einsum('ac,ic->ia', Fvv, t1)
    t1new +=  -np.einsum('ki,ka->ia', Foo, t1)
    t1new +=   np.einsum('kc,ic,ka->ia', Fov, t1, t1)
    t1new += fov.conj()
    t1new += 2*lib.einsum('Lkc,Lia,kc->ia', Lov, Lov, t1)
    t1new +=  -lib.einsum('Lki,Lac,kc->ia', Loo, Lvv, t1)
    t1new += 2*lib.einsum('Lkd,Lac,kd,ic->ia', Lov, Lvv, t1, t1)
    t1new +=  -lib.einsum('Lkc,Lad,kd,ic->ia', Lov, Lvv, t1, t1)
    t1new +=-2*lib.einsum('Llc,Lki,lc,ka->ia', Lov, Loo, t1, t1)
    t1new +=   lib.einsum('Lkc,Lli,lc,ka->ia', Lov, Loo, t1, t1)

    t1new /= eia

    return t1new

def update_amps(cc, t1, eris):
    t1new = update_t1(cc, t1, eris)
    return t1new

def amplitudes_to_vector(t1, out=None):
    nocc, nvir = t1.shape
    nov = nocc * nvir
    size = nov
    vector = np.ndarray(size, t1.dtype, buffer=out)
    vector = t1.ravel()
    return vector

def vector_to_amplitudes(vector, nmo, nocc):
    nvir = nmo - nocc
    nov = nocc * nvir
    t1 = vector.reshape((nocc, nvir))
    return t1
    
# t1: ia
# t2: ijab
def kernel(mycc, eris=None, t1=None, t2=None, max_cycle=50, tol=1e-8,
           tolnormt=1e-6, verbose=None):
    log = logger.new_logger(mycc, verbose)
    if eris is None:
        eris = mycc.ao2mo(mycc.mo_coeff)
    if t1 is None:
        t1 = mycc.get_init_guess(eris)

    cput1 = cput0 = (logger.process_clock(), logger.perf_counter())
    eold = 0
    eccsd = mycc.energy(t1, eris)
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
        t1new = mycc.update_amps(t1, eris)
        print(t1)
        print(t1new)
        tmpvec = mycc.amplitudes_to_vector(t1new)
        print(tmpvec)
        tmpvec -= mycc.amplitudes_to_vector(t1)
        print(tmpvec)
        normt = np.linalg.norm(tmpvec)
        tmpvec = None
        if mycc.iterative_damping < 1.0:
            alpha = mycc.iterative_damping
            t1new = (1-alpha) * t1 + alpha * t1new
        t1 = t1new
        t1new = None
        t1 = mycc.run_diis(t1, istep, normt, eccsd-eold, adiis)
        eold, eccsd = eccsd, mycc.energy(t1, eris)
        log.info('cycle = %d  E_corr(CCSD) = %.15g  dE = %.9g  norm(t1) = %.6g',
                 istep+1, eccsd, eccsd - eold, normt)
        cput1 = log.timer('CCSD iter', *cput1)
        if abs(eccsd-eold) < tol and normt < tolnormt:
            conv = True
            break
    log.timer('CCSD', *cput0)
    return conv, eccsd, t1

def _make_df_eris(cc, mo_coeff=None):
    eris = _ChemistsERIs()
    eris._common_init_(cc, mo_coeff)
    nocc = eris.nocc
    nmo = eris.fock.shape[0]
    nvir = nmo - nocc
    with_df = cc._scf.with_df
    naux = eris.naux = with_df.get_naoaux()

    Loo = np.empty((naux,nocc,nocc))
    Lov = np.empty((naux,nocc,nvir))
    Lvv = np.empty((naux,nvir,nvir))
    mo = np.asarray(eris.mo_coeff, order='F')
    ijslice = (0, nmo, 0, nmo)
    p1 = 0
    Lpq = None
    for k, eri1 in enumerate(with_df.loop()):
        Lpq = _ao2mo.nr_e2(eri1, mo, ijslice, aosym='s2', mosym='s1', out=Lpq)
        p0, p1 = p1, p1 + Lpq.shape[0]
        Lpq = Lpq.reshape(p1-p0,nmo,nmo)
        Loo[p0:p1] = Lpq[:,:nocc,:nocc]
        Lov[p0:p1] = Lpq[:,:nocc,nocc:]
        Lvv[p0:p1] = Lpq[:,nocc:,nocc:]
    Lpq = None
    eris.Loo = Loo
    eris.Lov = Lov
    eris.Lvv = Lvv

    return eris

class DFRCC2(ccsd.CCSD):
    '''restricted CCSD with IP-EOM, EA-EOM, EE-EOM, and SF-EOM capabilities

    Ground-state CCSD is performed in optimized ccsd.CCSD and EOM is performed here.
    '''
    energy = energy
    kernel = kernel
    update_amps = update_amps

    def update_amps(cc, t1, eris):
        t1new = update_t1(cc, t1, eris)
        return t1new

    def get_init_guess(self, eris=None):
        if eris is None:
            eris = self.ao2mo(self.mo_coeff)
        nocc = self.nocc
        mo_e = eris.mo_energy
        eia = mo_e[:nocc,None] - mo_e[None,nocc:]
        t1 = eris.fock[:nocc,nocc:] / eia
        return t1

    def amplitudes_to_vector(self, t1, out=None):
        return amplitudes_to_vector(t1, out)
    
    def vector_to_amplitudes(self, vec, nmo=None, nocc=None):
        if nocc is None: nocc = self.nocc
        if nmo is None: nmo = self.nmo
        return vector_to_amplitudes(vec, nmo, nocc)

    def run_diis(self, t1, istep, normt, de, adiis):
        if (adiis and
            istep >= self.diis_start_cycle and
            abs(de) < self.diis_start_energy_diff):
            vec = self.amplitudes_to_vector(t1)
            t1 = self.vector_to_amplitudes(adiis.update(vec))
            logger.debug1(self, 'DIIS for step %d', istep)
        return t1

    def ao2mo(self, mo_coeff=None):
        return _make_df_eris(self, mo_coeff)
