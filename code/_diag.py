import numpy as np
from sim import *
for cn in ['A_rule','B_full']:
    k,chk,rec,tr,fc,price = run_case(CONFIGS[cn],'S1',1,BASE,want_rec=True)
    e=eff(BASE,tr['heat'],tr['kap']); imp=rec['imp'].sum()
    dc=rec['sd'].sum()*(1/e['eta_dc']-1); pk=rec['z'].sum()*(1/e['eta_pk']-1); dist=e['lam']*imp
    bess=rec['ch'].sum()-rec['dis'].sum()  # energy lost in BESS incl soc change
    aux=rec['aux'].sum(); dS=k.d_storage.sum()
    print(cn, 'imp %.0f'%imp, {kk:round(v/imp*100,2) for kk,v in dict(dist=dist,dc=dc,pack=pk,bess_net=bess,aux=aux,dS=-dS).items()}, 'loss %.2f'%(k.losses.sum()/imp*100), 'cyc', round(rec['dis'].sum()/E_BU.sum(),2), 'R mean', round(rec['R'].mean(1).sum(),1))
    print(chk)
