import sys, time, numpy as np, pandas as pd
from sim import *
def summ(cfgname, seeds=(1,2,3), scen=('S1','S2','S3','S4')):
    rows=[]
    for sk in scen:
        for sd in seeds:
            k,chk = run_case(CONFIGS[cfgname], sk, sd, BASE)[:2]; k['scenario']=sk; k['seed']=sd; rows.append(k)
    df=pd.concat(rows); m=df.groupby(['scenario','cluster'],sort=False).mean(numeric_only=True).reset_index()
    a=annual(m)
    rel=a.served.sum()/a.demand.sum(); loss=a.losses.sum()/a.imports.sum()
    print(cfgname, 'REL %.3f LOSS %.3f  RE share %.3f curt %.3f'%(rel,loss,a.re_avail.sum()/a.imports.sum(), a.re_curtailed.sum()/a.re_avail.sum()))
    print(m.groupby('scenario',sort=False).apply(lambda g: pd.Series(dict(rel=g.served.sum()/g.demand.sum(), loss=g.losses.sum()/g.imports.sum(), dc=g.dc_served.sum()/g.dc_demand.sum(), sw=g.sw_served.sum()/g.sw_demand.sum()))).round(3).T)
    print((a.served/a.demand).round(3).values, (a.contribution/1e4).round(0).values, (a.revenue/1e4).round(0).values)
    return a
t=time.time(); summ('A_rule'); print('t',time.time()-t)
if len(sys.argv)>1:
    t=time.time(); summ(sys.argv[1], seeds=(1,), scen=('S1',)); print('t',time.time()-t)
