"""Runs every experiment in the report and writes CSV/JSON outputs. Usage: python run_all.py [workers]"""
import sys, os, json, time, ctypes
import numpy as np, pandas as pd
from multiprocessing import Pool
import sim
from sim import *

SEEDS = (1, 2, 3)
SENS = {'Base': {}, 'IEX +10%': dict(iex_mult=1.1), 'IEX +20%': dict(iex_mult=1.2), 'IEX +30%': dict(iex_mult=1.3),
        'RE -10%': dict(re_mult=.9), 'Demand +15%': dict(dem_mult=1.15), 'Degradation cost +50%': dict(deg_mult=1.5),
        'Elasticity halved': dict(eps=BASE['eps'] * .5, eps_net=BASE['eps_net'] * .5),
        'Forecast error +70%': dict(fc_mult=1.7)}
HOURLY_VARS = ['imp', 'reu', 'curt', 'bank', 'q', 'u', 'rtm', 'disc', 'ch', 'dis', 'z', 'w']

def low_priority():
    try: ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    except Exception: pass

def task(args):
    cfg, sk, seed, tag, ovr, hourly = args
    prm = dict(BASE, **ovr)
    k, chk, rec, tr, fc, price = run_case(CONFIGS[cfg], sk, seed, prm, want_rec=True)
    k['config'], k['scenario'], k['seed'], k['case'] = cfg, sk, seed, tag
    chk.update(config=cfg, scenario=sk, seed=seed, case=tag)
    h = None
    if hourly:
        rows = []
        for c in range(NC):
            n = [2 * c, 2 * c + 1]
            d = dict(config=cfg, scenario=sk, cluster=CL[c], hour=H,
                     dc_demand=tr['D'][:, n].sum((0, 1)), sw_demand=tr['Dsw'][n].sum(0),
                     dc_served=rec['sd'][:, n].sum((0, 1)), re_avail=tr['REi'][n].sum(0) + rec['alpha'][c] * tr['Pool'],
                     soc=rec['soc'][n, 1:].sum(0), soc_pct=rec['soc'][n, 1:].sum(0) / E_BU[n].sum(),
                     ready_packs=rec['R'][n, 1:].sum(0), ready_pct=rec['R'][n, 1:].sum(0) / E_PK[n].sum(),
                     ready_core_pct=rec['R'][n[0], 1:] / E_PK[n[0]], pack_reserve=E_PK[n].sum() * BASE['res'],
                     van_moves=rec['xo'][c] + rec['xc'][c], redirected=rec['rd'][:, c].sum(0),
                     iex_dam=tr['pD'], iex_rtm=tr['pR'], retail_price=price[0, n[0]], ride_price=price[1, n[0]])
            for v in HOURLY_VARS: d[v] = rec[v][n].sum(0)
            rows.append(pd.DataFrame(d))
        h = pd.concat(rows)
    return k, chk, h

def forecast_metrics():
    """accuracy of the two forecasters on the synthetic truth (demand, portfolio level per cluster-hour)"""
    out = []
    for mode in ('naive', 'ml'):
        for sk in SCN:
            for seed in SEEDS:
                tr = make_truth(sk, np.random.default_rng(seed), BASE)
                fc = make_fc(tr, mode, np.random.default_rng(seed + 1000), BASE)
                y = cl(tr['D'].sum(0) + tr['Dsw']); f = cl(fc['D'].sum(0) + fc['Dsw'])
                sm = samples(fc, np.random.default_rng(seed + 2000), 200)
                ens = np.array([cl(s['D'].sum(0) + s['Dsw']) for s in sm])
                pin = {}
                for qq in (.1, .5, .9):
                    qv = np.quantile(ens, qq, axis=0); e = y - qv
                    pin[qq] = np.mean(np.maximum(qq * e, (qq - 1) * e))
                cov = np.mean((y >= np.quantile(ens, .1, axis=0)) & (y <= np.quantile(ens, .9, axis=0)))
                out.append(dict(model=mode, scenario=sk, seed=seed, MAE=np.mean(np.abs(y - f)), RMSE=np.sqrt(np.mean((y - f) ** 2)),
                                WAPE=np.abs(y - f).sum() / y.sum(), pinball_P10=pin[.1], pinball_P50=pin[.5], pinball_P90=pin[.9],
                                P10_P90_coverage=cov))
    return pd.DataFrame(out)

def finance(pA, pB):
    capex = {'Data platform, ingestion & integration': 18., 'IoT/BMS telemetry retrofit (packs, chargers, meters)': 22.,
             'EMS/SCADA integration & charger firmware (module staging)': 10., 'ML + optimisation development': 12.,
             'Trading desk set-up, change management, training': 5.}
    capex['Contingency (15 %)'] = .15 * sum(capex.values())
    run = {'Cloud compute & storage': 6., 'Software licences (solver, MLOps, weather feeds)': 3.5,
           'Data science / OR / trading team (30 FTE)': 10.5, 'Annual maintenance of IoT & integrations': 4.5}
    C = sum(capex.values()); dE = pB.ebitda.sum() - pA.ebitda.sum()   # pB already net of run costs
    ramp = [.3, .7, 1, 1, 1]; r = .12
    cf = [-C] + [dE * x for x in ramp]
    npv = sum(v / (1 + r) ** i for i, v in enumerate(cf)); cum = np.cumsum(cf)
    yrs = next((i - 1 + (-cum[i - 1]) / cf[i] for i in range(1, len(cf)) if cum[i] >= 0), None)
    return dict(capex=capex, capex_total=C, run_cost=run, run_total=sum(run.values()), annual_ebitda_uplift=dE,
                ramp=ramp, wacc=r, cashflows=cf, npv_5yr=npv, payback_years=yrs,
                roi_5yr=(sum(cf[1:]) - C) / C)

if __name__ == '__main__':
    t0 = time.time(); W = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    jobs = [(cfg, sk, s, 'Base', {}, cfg in ('A_rule', 'B_full') and s == 1) for cfg in CONFIGS for sk in SCN for s in SEEDS]
    jobs += [(cfg, sk, s, tag, ovr, False) for tag, ovr in SENS.items() if tag != 'Base'
             for cfg in ('A_rule', 'B_full') for sk in SCN for s in SEEDS[:2]]
    if os.environ.get('QUICK'):
        jobs = [j for j in jobs if j[1] in ('S1', 'S4') and j[2] == 1 and j[3] in ('Base', 'IEX +10%')]
    jobs.sort(key=lambda j: j[0] == 'A_rule')
    with Pool(W, initializer=low_priority) as p:
        res = p.map(task, jobs, chunksize=1)
    K = pd.concat([r[0] for r in res]); CH = pd.DataFrame([r[1] for r in res])
    HR = pd.concat([r[2] for r in res if r[2] is not None])
    K.to_csv(os.path.join(CSV, 'daily_kpis_all_runs.csv'), index=False)
    CH.to_csv(os.path.join(CSV, 'validation_checks.csv'), index=False)
    HR.to_csv(os.path.join(CSV, 'hourly_dispatch_A_vs_B.csv'), index=False)

    def annual_pnl(cfg, case):
        d = K[(K.config == cfg) & (K.case == case)].groupby(['scenario', 'cluster'], sort=False).mean(numeric_only=True).reset_index()
        return pnl(annual(d), tech=(cfg != 'A_rule'))
    P = {cfg: annual_pnl(cfg, 'Base') for cfg in CONFIGS}
    pd.concat({c: p for c, p in P.items()}, names=['config', 'cluster']).to_csv(os.path.join(CSV, 'annual_pnl_by_config.csv'))
    scen = K[K.case == 'Base'].groupby(['config', 'scenario', 'cluster'], sort=False).mean(numeric_only=True).reset_index()
    scen.to_csv(os.path.join(CSV, 'scenario_cluster_daily_mean.csv'), index=False)
    def ann(cfg, case, two=False):
        sel = (K.config == cfg) & (K.case == case) & ((K.seed <= 2) | (not two))
        return annual(K[sel].groupby(['scenario', 'cluster'], sort=False).mean(numeric_only=True).reset_index())
    sens = []
    for tag in [t for t in SENS if t in set(K.case)]:
        for cfg in ('A_rule', 'B_full'):
            a = ann(cfg, tag, two=True); p = pnl(a, tech=(cfg != 'A_rule'))
            sens.append(dict(case=tag, config=cfg, revenue=p.revenue.sum(), ebitda=p.ebitda.sum(), margin=p.ebitda.sum() / p.revenue.sum(),
                             min_cluster_margin=p.margin.min(), min_cluster=p.margin.idxmin(),
                             reliability=a.served.sum() / a.demand.sum(), loss=a.losses.sum() / a.imports.sum()))
    port = pd.DataFrame({cfg: dict(reliability=ann(cfg, 'Base').pipe(lambda a: a.served.sum() / a.demand.sum()),
                                   loss=ann(cfg, 'Base').pipe(lambda a: a.losses.sum() / a.imports.sum()),
                                   revenue=P[cfg].revenue.sum(), ebitda=P[cfg].ebitda.sum(),
                                   margin=P[cfg].ebitda.sum() / P[cfg].revenue.sum(), min_margin=P[cfg].margin.min())
                         for cfg in CONFIGS}).T
    port.to_csv(os.path.join(CSV, 'portfolio_kpis_by_config.csv'))
    pd.DataFrame(sens).to_csv(os.path.join(CSV, 'sensitivity.csv'), index=False)
    FM = forecast_metrics(); FM.to_csv(os.path.join(CSV, 'forecast_metrics.csv'), index=False)
    fin = finance(P['A_rule'], P['B_full'])
    json.dump(fin, open(os.path.join(CSV, 'financials.json'), 'w'), indent=2)
    print('runs', len(jobs), 'time %.0f s' % (time.time() - t0))
    print(port.round(4).to_string())
    print(P['A_rule'].round(3).to_string()); print(P['B_full'].round(3).to_string())
    print(pd.DataFrame(sens).round(3).to_string())
    print(CH.drop(columns=['config', 'scenario', 'seed', 'case']).agg(['min', 'max']).T)
    print(json.dumps({k: v for k, v in fin.items() if k not in ('capex', 'run_cost')}, indent=1))
