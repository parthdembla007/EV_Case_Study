"""
EV charging + battery-swapping operator, five Indian metro clusters (each split into a congested CORE zone
and an OUTER zone).  Rule-based baseline (A) vs AI forecasting + stochastic optimisation framework (B).

All data are SYNTHETIC, generated here from the labelled assumptions in the report (Section 3).
Units: energy MWh per 1-hour step (MW x 1 h = MWh), power MW, prices Rs/kWh,
money k-Rs (Rs/kWh x MWh = Rs 1,000).  Annual money is reported in Rs crore (1 Cr = 1e4 k-Rs).
"""
import os
import numpy as np, pandas as pd, pulp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, 'outputs', 'csv'); os.makedirs(CSV, exist_ok=True)
T = 24; H = np.arange(T)
CL = ['Delhi NCR', 'Mumbai MMR', 'Bengaluru', 'Hyderabad', 'Chennai']; NC = 5
NN = 2 * NC; CLN = np.arange(NN) // 2; CORE = (np.arange(NN) % 2 == 0)       # node n = (cluster n//2, zone n%2)
SEG = ['Retail', 'Ride-hailing', 'Logistics']

# ---------------- cluster assets and market facts / assumptions (Section 3) ----------------
P_INST = np.array([336, 288, 240, 180, 156.])                  # MW nameplate, sums to 1,200
P_DC_C = P_INST * np.array([.55, .65, .55, .65, 99 / 156])      # MW DC fast charging, sums to 720 (60 %)
P_SW_C = P_INST - P_DC_C                                        # MW swap-station racks, sums to 480 (40 %)
E_B_C = 300 * P_INST / P_INST.sum()                             # MWh stationary BESS nameplate (300 of 600)
SOH_C = np.array([.93, .90, .94, .95, .91])                     # BESS state of health
E_PK_C = 300 * P_SW_C / P_SW_C.sum()                            # MWh usable swap-pack inventory (300 of 600)
G_MAX_C = .23 * P_INST                                          # MW sanctioned grid connection
THETA_C = np.array([.60, .55, .60, .65, .70])                   # share of load at open-access-eligible HT hubs
D_DAY = 1.3 * np.array([1000, 850, 700, 500, 450.])             # MWh/day nominal demand
SW_SH = np.array([.32, .20, .32, .25, .22])                     # swap share of demand
SEG_SH = np.array([[.30, .50, .20], [.40, .45, .15], [.45, .40, .15], [.35, .40, .25], [.25, .30, .45]])
ACOS = np.array([6.8, 8.6, 7.4, 7.2, 6.9])                      # Rs/kWh DISCOM average cost of supply
PMULT_C = np.array([1.0, 1.05, 1.0, .98, .97])                  # local price level
ZD = np.array([.62, .64, .60, .58, .58])                        # share of demand arising in the core zone
ZC = .36                                                        # share of assets located in the core zone
FIXED_LAKH_PER_MW = np.array([74, 90, 76, 74, 76.])             # Rs lakh / MW / yr fixed opex
TECH_RUN_CR = 24.5                                              # Rs Cr/yr run cost of the framework (B only)

def zsplit(x, share):
    """cluster array -> node array (core gets share, outer the rest)"""
    sh = share[CLN] if np.ndim(share) else share
    return x[CLN] * np.where(CORE, sh, 1 - sh)

P_DC, P_SW, E_B = zsplit(P_DC_C, ZC), zsplit(P_SW_C, ZC), zsplit(E_B_C, ZC)
E_BU = E_B * SOH_C[CLN]; P_B = .5 * E_B; E_PK = zsplit(E_PK_C, ZC); G_MAX = zsplit(G_MAX_C, ZC)
THETA = THETA_C[CLN]; PMULT = PMULT_C[CLN]
XMAX = .05 * E_PK_C                                             # MWh/h of charged packs movable by van, per cluster

BASE = dict(eta_dc=.94, ems=False, eta_pk=.94, lam=.010, a_sb=.003, sleep=0., eta_c=.94, eta_d=.94, sig_b=2e-4, sig_p=1e-4,
            aux_r=.008, aux_b=.002, tau=3.0, kappa=.23, soc_lo=.10, soc_hi=.95, ncyc=1.5,
            res=.05, oa=1.6, c_re=4.3, v_bank=3.8, bank_frac=.03, deg_b=1.4, deg_p=4.5,
            pen=np.array([6., 10., 14.]), pen_sw=10., v_term=6., pen_term=40., ud=.5, vom=.5, c_van=.8,
            rho=np.array([.15, .35, 0.]), delta=1.5,
            p_seg=np.array([19., 15.5, 14.]), p_sw=32., p_comp=22., beta=.35, ramp=.05,
            eps=np.array([-.6, -.8, 0.]), eps_net=np.array([-.3, -.2, 0.]), corr=.45,
            iex_mult=1., re_mult=1., dem_mult=1., fc_mult=1., deg_mult=1., S=4)


# ---------------- hourly profiles ----------------
def bump(mu, w, a):
    d = (H + .5 - mu + 12) % 24 - 12
    return a * np.exp(-.5 * (d / w) ** 2)

SHIFT, EVE, MORN = [0, 1, 0, 0, 0], [1.2, 1., .9, 1., .9], [1., 1., 1.3, 1., 1.]

def shape(k, c):
    m, e = MORN[c], EVE[c]
    y = {0: .25 + m * bump(9, 1.5, .5) + bump(13.5, 2, .35) + e * bump(19.5, 2.2, 1.2),
         1: .45 + bump(4, 1.8, .8) + bump(14, 2, .7) + e * bump(22.5, 1.8, .6),
         2: .25 + bump(1.5, 2.5, 1.) + bump(14, 2, .5),
         3: .15 + m * bump(9.5, 1.8, 1.) + bump(14, 1.5, .45) + e * bump(19, 2.2, 1.1)}[k]
    y = np.roll(y, SHIFT[c]); return y / y.sum()

DN_C = np.array([[D_DAY[c] * (1 - SW_SH[c]) * SEG_SH[c, k] * shape(k, c) for c in range(NC)] for k in range(3)])
DSWN_C = np.array([D_DAY[c] * SW_SH[c] * shape(3, c) for c in range(NC)])
ZDn = np.where(CORE, ZD[CLN], 1 - ZD[CLN])
DN = DN_C[:, CLN, :] * ZDn[None, :, None]; DSWN = DSWN_C[CLN] * ZDn[:, None]

def solar(cf):
    y = np.clip(np.sin(np.pi * (H + .5 - 6) / 12), 0, None) ** 1.3; return 5.0 * cf * y / y.sum()

def wind(cf):
    y = 1 + .35 * np.sin(2 * np.pi * (H + .5 - 10) / 24); return 24 * cf * y / y.sum()

SOL_MWP = np.array([45, 30, 40, 35, 20.]); WIND_MW = np.array([0, 15, 15, 10, 45.])
CLOUD = [.92, .82, .90, .95, .90]; WCF = [0, .25, .30, .25, .38]
REN_C = np.array([SOL_MWP[c] * solar(CLOUD[c]) + WIND_MW[c] * wind(WCF[c]) for c in range(NC)])
POOL_N = 70 * solar(1.0) + 50 * wind(.32)
RE_SCALE = .35 * .80 * D_DAY.sum() / .90 / (REN_C.sum() + POOL_N.sum()); REN_C *= RE_SCALE; POOL_N *= RE_SCALE
ZCn = np.where(CORE, ZC, 1 - ZC)                                # RE and pool delivered pro-rata to HT hubs
PDAM_N = np.array([4.8, 4.5, 4.3, 4.2, 4.3, 4.8, 5.5, 6.0, 5.0, 3.6, 2.9, 2.6, 2.5, 2.6, 2.9, 3.6,
                   4.8, 6.8, 9.2, 9.8, 9.9, 9.5, 8.0, 6.0])
SOLAR_HRS = (H >= 9) & (H < 16)
TAR_C = ACOS[:, None] * np.where(SOLAR_HRS, .7, 1.3)[None, :]; TAR = TAR_C[CLN]
AL0 = np.repeat((D_DAY / D_DAY.sum())[:, None], T, 1)

SCN = {'S1': dict(name='Normal operations'),
       'S2': dict(name='Regional imbalance', dem={0: 1.25, 4: .75}, re={4: 1.6}, pool=1.15, iex=.9),
       'S3': dict(name='RE shortfall + IEX spike', re_all=.5, pool=.5, iex=1.45, dem_all=1.03),
       'S4': dict(name='Summer heatwave', dem={0: 1.3, 1: 1.1, 2: 1.1, 3: 1.25, 4: 1.15}, kap=.88,
                  heat=True, iex=1.3, re_all=.95)}
WEIGHT = {'S1': .65, 'S2': .10, 'S3': .10, 'S4': .15}


# ---------------- synthetic truth and forecasts ----------------
def make_truth(sk, rng, prm):
    sc = SCN[sk]; N = rng.normal
    dm = np.full(NC, sc.get('dem_all', 1.))
    for c, v in sc.get('dem', {}).items(): dm[c] = v
    dm = (dm * prm['dem_mult'] * (1 + N(0, .04, NC)))[CLN]
    D = DN * dm[None, :, None] * (1 + N(0, .05, (3, NN, T)))
    Dsw = DSWN * dm[:, None] * (1 + N(0, .05, (NN, T)))
    rm = np.full(NC, sc.get('re_all', 1.))
    for c, v in sc.get('re', {}).items(): rm[c] = v
    rm = rm * prm['re_mult']
    REc = np.clip(REN_C * rm[:, None] * (1 + N(0, .10, NC))[:, None] * (1 + N(0, .08, (NC, T))), 0, None)
    Pool = np.clip(POOL_N * sc.get('pool', sc.get('re_all', 1.)) * prm['re_mult'] * (1 + N(0, .08)) * (1 + N(0, .06, T)), 0, None)
    pD = np.clip(PDAM_N * sc.get('iex', 1.) * prm['iex_mult'] * (1 + N(0, .06, T)), 0, 10)
    pR = np.clip(pD * (1 + N(0, .10, T)), 0, 10)
    return dict(D=D, Dsw=Dsw, REi=REc[CLN] * ZCn[:, None], Pool=Pool, pD=pD, pR=pR,
                kap=sc.get('kap', 1.), heat=sc.get('heat', False))

def make_fc(tr, mode, rng, prm):
    N = rng.normal; f = prm['fc_mult']
    if mode == 'naive':      # seasonal-naive: same hour last week, blind to today's weather/events
        lv = (1 + N(0, .06, NC))[CLN]
        D = DN * lv[None, :, None] * (1 + N(0, .10, (3, NN, T))); Dsw = DSWN * lv[:, None] * (1 + N(0, .10, (NN, T)))
        REi = REN_C[CLN] * ZCn[:, None] * .95; Pool = POOL_N * .95; pD = PDAM_N.copy(); sig = (.10, .10); L = 12
    else:                    # ML forecast: weather/event-aware, residual error per assumption A18
        lv = (1 + N(0, .03 * f, NC))[CLN]
        D = tr['D'] * lv[None, :, None] * (1 + N(0, .05 * f, (3, NN, T))); Dsw = tr['Dsw'] * lv[:, None] * (1 + N(0, .05 * f, (NN, T)))
        REi = np.clip(tr['REi'] * (1 + N(0, .08 * f, NC))[CLN][:, None] * (1 + N(0, .08 * f, (NC, T)))[CLN], 0, None)
        Pool = np.clip(tr['Pool'] * (1 + N(0, .08 * f)) * (1 + N(0, .06 * f, T)), 0, None)
        pD = np.clip(tr['pD'] * (1 + N(0, .07 * f, T)), 0, 10); sig = (.035 * f, .05 * f); L = 6
    return dict(D=D, Dsw=Dsw, REi=REi, Pool=Pool, pD=pD, pR=pD.copy(), kap=tr['kap'], heat=tr['heat'], sig=sig, L=L)

def samples(fc, rng, S):
    """S equiprobable day-ahead scenarios drawn from the forecaster's error distribution"""
    N = rng.normal; sd, sh = fc['sig']; out = []
    for _ in range(S):
        lv = (1 + N(0, sd, NC))[CLN]
        pD = np.clip(fc['pD'] * (1 + N(0, .08)) * (1 + N(0, .05, T)), 0, 10)
        out.append(dict(D=fc['D'] * lv[None, :, None] * (1 + N(0, sh, (3, NN, T))),
                        Dsw=fc['Dsw'] * lv[:, None] * (1 + N(0, sh, (NN, T))),
                        REi=np.clip(fc['REi'] * (1 + N(0, .10, NC))[CLN][:, None] * (1 + N(0, .08, (NC, T)))[CLN], 0, None),
                        Pool=np.clip(fc['Pool'] * (1 + N(0, .10)), 0, None), pD=pD,
                        pR=np.clip(pD * (1 + N(0, .10, T)), 0, 10), kap=fc['kap'], heat=fc['heat']))
    return out

def blend(tr, fc, t):
    """intraday forecast issued at hour t: nowcast = truth at hour t, day-ahead error decays over L hours"""
    w = np.clip((H - t) / fc['L'], 0, 1); out = dict(kap=tr['kap'], heat=tr['heat'])
    for v in ['D', 'Dsw', 'REi', 'Pool', 'pD']: out[v] = tr[v] + (fc[v] - tr[v]) * w
    out['pR'] = tr['pR'] + (fc['pD'] - tr['pR']) * w
    return out


# ---------------- dynamic pricing and demand response ----------------
def pbar(prm): return prm['p_seg'][:, None] * PMULT[None, :]

def set_prices(mu, prm):
    """retail: +/-20 % band around yesterday's average; ride-hailing fleets: +/-10 % ToU band; logistics: fixed contract"""
    pb = pbar(prm); out = np.repeat(pb[:, :, None], T, 2)
    sig = (mu - mu.mean(1, keepdims=True)) / np.abs(mu.mean(1, keepdims=True))
    for k, (beta, band) in enumerate([(prm['beta'], .20), (prm['beta'] / 2, .10)]):
        x = np.clip(beta * sig, -band, band); x = np.clip(x - x.mean(1, keepdims=True), -band, band)
        for h in range(1, T): x[:, h] = np.clip(x[:, h], x[:, h - 1] - prm['ramp'], x[:, h - 1] + prm['ramp'])
        out[k] = pb[k][:, None] * (1 + x)
        if k == 0: out[k] = np.minimum(out[k], prm['p_comp'])
    return out

def apply_dr(d, price, prm):
    """timing elasticity eps shifts load between hours; eps_net changes daily volume"""
    d = dict(d); D = d['D'].copy(); pb = pbar(prm)
    for k in (0, 1):
        rel = price[k] / pb[k][:, None] - 1
        Dk = D[k] * (1 + prm['eps'][k] * rel)
        pw = (price[k] * D[k]).sum(1) / D[k].sum(1)
        D[k] = Dk * (D[k].sum(1) / Dk.sum(1) * (1 + prm['eps_net'][k] * (pw / pb[k] - 1)))[:, None]
    d['D'] = D; return d


# ---------------- effective technical parameters ----------------
def eff(prm, heat, kap):
    e = dict(prm)
    if prm['ems']: e['eta_dc'] = prm['eta_dc'] + .015                    # power-module staging keeps modules near peak efficiency
    if heat:
        e.update(eta_dc=e['eta_dc'] - .01, eta_c=prm['eta_c'] - .01, eta_d=prm['eta_d'] - .01,
                 aux_r=prm['aux_r'] * 1.6, sig_b=prm['sig_b'] * 2, sig_p=prm['sig_p'] * 2)
    e['cap_dc'] = (1 - prm['res']) * prm['kappa'] * kap * P_DC          # deliverable DC MWh/h after 5 % reserve
    e['rreq'] = prm['res'] * prm['kappa'] * kap * P_DC / e['eta_dc']      # MW reserve backing
    e['rres'] = prm['res'] * E_PK                                         # MWh ready packs held back
    e['sb'] = prm['a_sb'] * P_DC                                         # MW idle-charger standby (all modules awake)
    e['deg_b'] = prm['deg_b'] * prm['deg_mult']; e['deg_p'] = prm['deg_p'] * prm['deg_mult']
    return e

def init_state(prm):
    return dict(soc=.30 * E_BU.copy(), R=.60 * E_PK.copy(), Q=.40 * E_PK.copy(),
                bank=prm['bank_frac'] * D_DAY[CLN] * ZCn, disl=prm['ncyc'] * (prm['soc_hi'] - prm['soc_lo']) * E_BU)


_st0 = init_state(BASE); SOC0, R0 = _st0['soc'], _st0['R']      # day-start state = required day-end state (steady state)


# ---------------- optimisation model: stochastic LP (day-ahead) / MILP (intraday MPC) ----------------
def solve(j0, data, st, fix, e, price, mip, duals=False):
    """hours j0..23 over scenario list `data`. fix: q (NN,T) DAM schedule or None, alpha (NC,T) RE-pool split or None,
    rtm0 (NN) RTM energy already bought for hour j0 or None."""
    J = list(range(j0, T)); S = len(data); lpSum = pulp.lpSum
    m = pulp.LpProblem('ev', pulp.LpMaximize)
    I = [(n, j) for n in range(NN) for j in J]; IC = [(c, j) for c in range(NC) for j in J]
    q = m.add_variable_dicts('q', I, 0) if fix['q'] is None else {(n, j): float(fix['q'][n, j]) for n, j in I}
    if fix['alpha'] is None:
        al = m.add_variable_dicts('al', IC, 0, e['corr'])
        for j in J: m += lpSum(al[c, j] for c in range(NC)) == 1
    else:
        al = {(c, j): float(fix['alpha'][c, j]) for c, j in IC}
    obj, bal, keep = [], {}, []
    for s, d in enumerate(data):
        V = {v: m.add_variable_dicts(f'{v}{s}', I, 0) for v in
             ['reu', 'bank', 'curt', 'u', 'rtm', 'disc', 'ch', 'dis', 'z', 'w', 'rg', 'rb']}
        X = {v: m.add_variable_dicts(f'{v}{s}', IC, 0) for v in ['xo', 'xc']}         # van moves outer->core / core->outer
        rd = m.add_variable_dicts(f'rd{s}', [(k, c, j) for k in (0, 1) for c, j in IC], 0)  # DC redirect core->outer
        sd = m.add_variable_dicts(f'sd{s}', [(k, n, j) for k in range(3) for n, j in I], 0)
        IJ = [(n, j) for n in range(NN) for j in range(j0, T + 1)]
        soc, R, Q = (m.add_variable_dicts(f'{v}{s}', IJ, 0) for v in ('soc', 'R', 'Q'))
        y = m.add_variable_dicts(f'y{s}', I, cat='Binary') if mip else None
        for c in range(NC):
            for j in J:
                m += X['xo'][c, j] + X['xc'][c, j] <= XMAX[c]
                for k in (0, 1): m += rd[k, c, j] <= e['rho'][k] * float(d['D'][k, 2 * c, j])
                obj += [-e['c_van'] * (X['xo'][c, j] + X['xc'][c, j]), -e['delta'] * (rd[0, c, j] + rd[1, c, j])]
        for n in range(NN):
            c = CLN[n]; sgn = 1 if CORE[n] else -1
            m += soc[n, j0] == st['soc'][n]; m += R[n, j0] == st['R'][n]; m += Q[n, j0] == st['Q'][n]
            m += lpSum(V['bank'][n, j] for j in J) <= st['bank'][n]
            m += lpSum(V['dis'][n, j] for j in J) <= st['disl'][n]
            for j in J:
                imp = V['reu'][n, j] + q[n, j] - V['u'][n, j] + V['rtm'][n, j] + V['disc'][n, j]
                m += V['u'][n, j] <= q[n, j]
                m += V['disc'][n, j] >= (1 - THETA[n]) * imp                            # non-OA sites buy from DISCOM
                m += imp + V['rg'][n, j] <= G_MAX[n]                                     # MW connection incl. reserve headroom
                m += V['reu'][n, j] + V['bank'][n, j] + V['curt'][n, j] == float(d['REi'][n, j]) + al[c, j] * float(ZCn[n] * d['Pool'][j])
                load = lpSum(sd[k, n, j] for k in range(3)) / e['eta_dc'] + V['z'][n, j] / e['eta_pk'] \
                    + V['ch'][n, j] + e['aux_r'] * R[n, j] + e['aux_b'] * E_B[n] + e['sb'][n] * (1 - e['sleep'])                     + e['sb'][n] * e['sleep'] / float(e['cap_dc'][n]) * lpSum(sd[k, n, j] for k in range(3))
                nm = f'bal_{s}_{n}_{j}'; m += ((1 - e['lam']) * imp + V['dis'][n, j] - load == 0), nm; bal[(s, n, j)] = nm
                m += soc[n, j + 1] == soc[n, j] * (1 - e['sig_b']) + e['eta_c'] * V['ch'][n, j] - V['dis'][n, j] / e['eta_d']
                m += soc[n, j + 1] >= e['soc_lo'] * E_BU[n]; m += soc[n, j + 1] <= e['soc_hi'] * E_BU[n]
                if mip:
                    m += V['ch'][n, j] <= P_B[n] * y[n, j]; m += V['dis'][n, j] <= P_B[n] * (1 - y[n, j])
                else:
                    m += V['ch'][n, j] <= P_B[n]; m += V['dis'][n, j] <= P_B[n]
                m += V['rg'][n, j] + V['rb'][n, j] >= e['rreq'][n]                     # 5 % reserve: grid headroom + BESS
                m += V['rb'][n, j] + V['dis'][n, j] <= P_B[n]
                m += e['eta_d'] * (soc[n, j + 1] - e['soc_lo'] * E_BU[n]) >= V['rb'][n, j]
                m += lpSum(sd[k, n, j] for k in range(3)) <= float(e['cap_dc'][n])
                for k in range(3):
                    m += sd[k, n, j] <= float(d['D'][k, n, j]) - (sgn * rd[k, c, j] if k < 2 else 0)
                m += V['w'][n, j] <= float(d['Dsw'][n, j]); m += V['w'][n, j] <= R[n, j] - e['rres'][n]
                m += V['z'][n, j] <= Q[n, j] / e['tau']; m += V['z'][n, j] <= P_SW[n]
                mv = sgn * (X['xo'][c, j] - X['xc'][c, j])
                m += R[n, j + 1] == R[n, j] * (1 - e['sig_p']) - V['w'][n, j] + V['z'][n, j] + mv
                m += Q[n, j + 1] == Q[n, j] + e['sig_p'] * R[n, j] + V['w'][n, j] - V['z'][n, j]
                m += R[n, j + 1] >= e['rres'][n]
                if j == j0 and fix.get('rtm0') is not None: m += V['rtm'][n, j] == float(fix['rtm0'][n])
                pD, pR = float(d['pD'][j]), float(d['pR'][j])
                obj += [lpSum(float(price[k, n, j]) * sd[k, n, j] for k in range(3)) + e['p_sw'] * PMULT[n] * V['w'][n, j],
                        -(pD + e['oa']) * q[n, j], e['ud'] * pD * V['u'][n, j], -(pR + e['oa']) * V['rtm'][n, j],
                        -float(TAR[n, j]) * V['disc'][n, j], e['v_bank'] * V['bank'][n, j],
                        -e['deg_b'] * V['dis'][n, j], -e['deg_p'] * V['z'][n, j],
                        lpSum(e['pen'][k] * sd[k, n, j] for k in range(3)) + e['pen_sw'] * V['w'][n, j],
                        -e['vom'] * (lpSum(sd[k, n, j] for k in range(3)) + V['w'][n, j])]
            shs, shr = m.add_variable(f'shs{s}_{n}', 0), m.add_variable(f'shr{s}_{n}', 0)   # end-of-day state shortfall
            m += soc[n, T] + shs >= SOC0[n]; m += R[n, T] + shr >= R0[n]
            obj += [-e['pen_term'] * (e['eta_d'] * shs + shr), e['v_term'] * e['eta_d'] * (soc[n, T] - SOC0[n]) * 0.]
        keep.append((V, X, rd, sd, soc, R))
    m += (1. / S) * lpSum(obj)
    r = m.solve(pulp.HiGHS(msg=False, gapRel=2e-3, timeLimit=120))
    if not r.has_solution:
        raise RuntimeError(f'solver status {r.status} at j0={j0}')
    g = lambda x: x.value() if hasattr(x, 'value') else x
    res = dict(q=np.array([[g(q[n, j]) if j >= j0 else 0. for j in range(T)] for n in range(NN)]),
               alpha=np.array([[g(al[c, j]) if j >= j0 else 0. for j in range(T)] for c in range(NC)]))
    V, X, rd, sd, soc, R = keep[0]
    arr = lambda Dv, idx, rows: np.array([[Dv[idx(r, j)].value() if j >= j0 else 0. for j in range(T)] for r in rows])
    s0 = {v: arr(V[v], lambda n, j: (n, j), range(NN)) for v in V}
    s0.update({v: arr(X[v], lambda c, j: (c, j), range(NC)) for v in X})
    s0['rd'] = np.array([arr(rd, lambda c, j, k=k: (k, c, j), range(NC)) for k in (0, 1)])
    s0['sd'] = np.array([arr(sd, lambda n, j, k=k: (k, n, j), range(NN)) for k in range(3)])
    res['s0'] = s0
    if duals:
        mu = np.zeros((NN, T))
        for (s, n, j), nm in bal.items(): mu[n, j] += abs(m.get_constraint_by_name(nm).pi or 0.)
        res['mu'] = mu
    return res


# ---------------- strategy runners ----------------
NODE_V = ['reu', 'bank', 'curt', 'q', 'u', 'rtm', 'disc', 'ch', 'dis', 'z', 'w', 'imp', 'aux']

def rule_q(fc):
    need = (fc['D'].sum(0) + fc['Dsw']) / .90 - (fc['REi'] + (AL0[CLN] * ZCn[:, None]) * fc['Pool'][None, :])
    return np.minimum(THETA[:, None] * np.clip(need, 0, None), THETA[:, None] * G_MAX[:, None])

def empty_rec():
    r = {v: np.zeros((NN, T)) for v in NODE_V}
    r.update(sd=np.zeros((3, NN, T)), soc=np.zeros((NN, T + 1)), R=np.zeros((NN, T + 1)),
             xo=np.zeros((NC, T)), xc=np.zeros((NC, T)), rd=np.zeros((2, NC, T)))
    return r

def run_rule(tr, fc, prm):
    """incumbent rules: fixed BESS timetable, charge-on-return packs, flat price, pro-rata RE pool, 03:00 van rebalance"""
    e = eff(prm, tr['heat'], tr['kap']); st = init_state(prm); rec = empty_rec()
    q = rule_q(fc); al = AL0; rec['soc'][:, 0], rec['R'][:, 0] = st['soc'], st['R']
    for t in range(T):
        if t == 3:
            for c in range(NC):
                a, b = 2 * c, 2 * c + 1; f = (st['R'][a] + st['R'][b]) / (E_PK[a] + E_PK[b])
                x = float(np.clip(f * E_PK[a] - st['R'][a], -XMAX[c], XMAX[c]))
                x = min(x, st['R'][b] - e['rres'][b]) if x > 0 else max(x, -(st['R'][a] - e['rres'][a]))
                st['R'][a] += x; st['R'][b] -= x; rec['xo' if x > 0 else 'xc'][c, t] = abs(x)
        for n in range(NN):
            soc, R, Q = st['soc'][n], st['R'][n], st['Q'][n]
            Dk = tr['D'][:, n, t]; sd = Dk * min(1., e['cap_dc'][n] / max(Dk.sum(), 1e-9))
            w = min(tr['Dsw'][n, t], max(0., R - e['rres'][n])); z = min(Q / e['tau'], P_SW[n])
            ch = min(P_B[n], max(0., (e['soc_hi'] * E_BU[n] - soc * (1 - e['sig_b'])) / e['eta_c'])) if t <= 5 else 0.
            dis = min(P_B[n], max(0., e['eta_d'] * (soc * (1 - e['sig_b']) - e['soc_lo'] * E_BU[n]) / (23 - t))) if 18 <= t <= 22 else 0.
            aux = e['aux_r'] * R + e['aux_b'] * E_B[n] + e['sb'][n]
            load = lambda: sd.sum() / e['eta_dc'] + z / e['eta_pk'] + ch + aux
            net = load() - dis
            if net < 0: dis += net; net = 0.
            ex = net - (G_MAX[n] - e['rreq'][n]) * (1 - e['lam'])
            if ex > 0:
                r = min(ch, ex); ch -= r; ex -= r
                r = min(z / e['eta_pk'], ex); z -= r * e['eta_pk']; ex -= r
                if ex > 0: sd = sd * max(0., 1 - ex * e['eta_dc'] / sd.sum())
                net = load() - dis
            imp = net / (1 - e['lam']); re_av = tr['REi'][n, t] + al[CLN[n], t] * ZCn[n] * tr['Pool'][t]
            oa = THETA[n] * imp; reu = min(re_av, oa); dam = min(q[n, t], oa - reu)
            sur = re_av - reu; bank = min(sur, st['bank'][n]); st['bank'][n] -= bank
            for v, x in zip(['reu', 'q', 'u', 'disc', 'bank', 'curt', 'imp', 'w', 'z', 'ch', 'dis', 'aux'],
                            [reu, q[n, t], q[n, t] - dam, imp - reu - dam, bank, sur - bank, imp, w, z, ch, dis, aux]):
                rec[v][n, t] = x
            rec['sd'][:, n, t] = sd
            st['soc'][n] = soc * (1 - e['sig_b']) + e['eta_c'] * ch - dis / e['eta_d']
            st['R'][n] = R * (1 - e['sig_p']) - w + z; st['Q'][n] = Q + e['sig_p'] * R + w - z
            rec['soc'][n, t + 1], rec['R'][n, t + 1] = st['soc'][n], st['R'][n]
    rec['alpha'] = al; return rec

def run_opt(tr, fc, prm, price, rng, proc_opt=True, pool_opt=False, redirect=True):
    e = eff(prm, tr['heat'], tr['kap'])
    if not redirect: e['rho'] = np.zeros(3)
    if proc_opt:
        da = solve(0, samples(fc, rng, prm['S']), init_state(prm), dict(q=None, alpha=None if pool_opt else AL0), e, price, mip=False)
        q, al = da['q'], (da['alpha'] if pool_opt else AL0)
    else:
        q, al = rule_q(fc), AL0
    st = init_state(prm); rec = empty_rec(); rec['soc'][:, 0], rec['R'][:, 0] = st['soc'], st['R']
    rtm0 = np.zeros(NN)
    for t in range(T):
        r = solve(t, [blend(tr, fc, t)], st, dict(q=q, alpha=al, rtm0=rtm0), e, price, mip=True)['s0']
        for c in range(NC):
            rec['xo'][c, t], rec['xc'][c, t] = r['xo'][c, t], r['xc'][c, t]; rec['rd'][:, c, t] = r['rd'][:, c, t]
        for n in range(NN):
            soc, R, Q = st['soc'][n], st['R'][n], st['Q'][n]; ch, dis, z, w = (r[v][n, t] for v in ('ch', 'dis', 'z', 'w'))
            rec['sd'][:, n, t] = r['sd'][:, n, t]
            for v in ['reu', 'bank', 'curt', 'u', 'rtm', 'disc', 'ch', 'dis', 'z', 'w']: rec[v][n, t] = r[v][n, t]
            rec['q'][n, t] = q[n, t]; rec['aux'][n, t] = e['aux_r'] * R + e['aux_b'] * E_B[n] + e['sb'][n] * (1 - e['sleep'] + e['sleep'] * r['sd'][:, n, t].sum() / e['cap_dc'][n])
            rec['imp'][n, t] = r['reu'][n, t] + q[n, t] - r['u'][n, t] + r['rtm'][n, t] + r['disc'][n, t]
            st['bank'][n] -= r['bank'][n, t]; st['disl'][n] -= dis
            mv = (1 if CORE[n] else -1) * (r['xo'][CLN[n], t] - r['xc'][CLN[n], t])
            st['soc'][n] = soc * (1 - e['sig_b']) + e['eta_c'] * ch - dis / e['eta_d']
            st['R'][n] = R * (1 - e['sig_p']) - w + z + mv; st['Q'][n] = Q + e['sig_p'] * R + w - z
            rec['soc'][n, t + 1], rec['R'][n, t + 1] = st['soc'][n], st['R'][n]
        rtm0 = r['rtm'][:, t + 1] if t + 1 < T else None          # RTM is bought one hour ahead of delivery
    rec['alpha'] = al; return rec


# ---------------- KPIs, daily P&L, validation ----------------
def cl(x):
    """node -> cluster aggregation (sum over core and outer zones)"""
    x = np.asarray(x); return x.reshape((NC, 2) + x.shape[1:]).sum(1)

def kpis(rec, tr, prm, price):
    e = eff(prm, tr['heat'], tr['kap']); s = lambda x: x.sum(-1)
    dem = cl(s(tr['D'].sum(0)) + s(tr['Dsw'])); srv = cl(s(rec['sd'].sum(0)) + s(rec['w']))
    imp = cl(s(rec['imp'])); dS = cl(e['eta_d'] * (rec['soc'][:, T] - rec['soc'][:, 0]) + (rec['R'][:, T] - rec['R'][:, 0]))
    re_av = cl(s(tr['REi']) + s(rec['alpha'][CLN] * ZCn[:, None] * tr['Pool'][None, :]))
    rev = cl(s((price * rec['sd']).sum(0)) + e['p_sw'] * PMULT * s(rec['w']))
    c_re = e['c_re'] * re_av - e['v_bank'] * cl(s(rec['bank']))
    c_dam = cl(s((tr['pD'][None, :] + e['oa']) * rec['q']) - e['ud'] * s(tr['pD'][None, :] * rec['u']))
    c_rtm = cl(s((tr['pR'][None, :] + e['oa']) * rec['rtm'])); c_disc = cl(s(TAR * rec['disc']))
    c_deg = cl(e['deg_b'] * s(rec['dis']) + e['deg_p'] * s(rec['z'])); c_vom = e['vom'] * srv
    c_log = e['c_van'] * s(rec['xo'] + rec['xc']) + e['delta'] * s(rec['rd'].sum(0))
    v_ds = e['v_term'] * dS
    contrib = rev - c_re - c_dam - c_rtm - c_disc - c_deg - c_vom - c_log + v_ds
    lost = cl(s((price * (tr['D'] - rec['sd'])).sum(0))) + cl(e['p_sw'] * PMULT * s(tr['Dsw'] - rec['w']))
    Rmin = (cl(rec['R'][:, 1:]) / E_PK_C[:, None]).min(1)
    return pd.DataFrame(dict(cluster=CL, demand=dem, served=srv, unserved=dem - srv, reliability=srv / dem,
        dc_demand=cl(s(tr['D'].sum(0))), dc_served=cl(s(rec['sd'].sum(0))), sw_demand=cl(s(tr['Dsw'])), sw_served=cl(s(rec['w'])),
        imports=imp, re_avail=re_av, re_used=cl(s(rec['reu'])), re_banked=cl(s(rec['bank'])), re_curtailed=cl(s(rec['curt'])),
        dam=cl(s(rec['q'])), dam_underdrawn=cl(s(rec['u'])), rtm=cl(s(rec['rtm'])), discom=cl(s(rec['disc'])),
        bess_ch=cl(s(rec['ch'])), bess_dis=cl(s(rec['dis'])), pack_charge=cl(s(rec['z'])), van_moves=s(rec['xo'] + rec['xc']),
        redirected=s(rec['rd'].sum(0)), d_storage=dS, losses=imp - srv - dS, loss_pct=(imp - srv - dS) / imp,
        revenue=rev, cost_re=c_re, cost_dam=c_dam, cost_rtm=c_rtm, cost_discom=c_disc, cost_degradation=c_deg,
        cost_vom=c_vom, cost_logistics=c_log, storage_value=v_ds, contribution=contrib, lost_revenue=lost,
        min_ready_pack_frac=Rmin))

def check(rec, tr, prm, price):
    e = eff(prm, tr['heat'], tr['kap'])
    load = rec['sd'].sum(0) / e['eta_dc'] + rec['z'] / e['eta_pk'] + rec['ch'] + rec['aux']
    pb = pbar(prm)[0][:, None]
    return dict(max_balance_err_MWh=float(np.abs((1 - e['lam']) * rec['imp'] + rec['dis'] - load).max()),
                max_import_over_limit_MW=float((rec['imp'] - G_MAX[:, None]).max()),
                min_soc_margin_MWh=float((rec['soc'][:, 1:] - e['soc_lo'] * E_BU[:, None]).min()),
                max_soc_over_MWh=float((rec['soc'][:, 1:] - e['soc_hi'] * E_BU[:, None]).max()),
                min_pack_reserve_margin_MWh=float((rec['R'][:, 1:] - e['rres'][:, None]).min()),
                simultaneous_ch_dis_hours=int(((rec['ch'] > 1e-4) & (rec['dis'] > 1e-4)).sum()),
                max_dc_over_cap_MWh=float((rec['sd'].sum(0) - e['cap_dc'][:, None]).max()),
                max_retail_price_dev=float(np.abs(price[0] / pb - 1).max()),
                max_daily_cycles=float((rec['dis'].sum(1) / ((e['soc_hi'] - e['soc_lo']) * E_BU)).max()))

CONFIGS = {   # lever-by-lever build-up from the rule-based baseline to the full framework
    'A_rule':      dict(opt=False, fc='naive'),
    'L1_dispatch': dict(opt=True, fc='naive', proc=False, pool=False, pricing=False, redirect=False),
    'L2_network':  dict(opt=True, fc='naive', proc=False, pool=False, pricing=False, redirect=True),
    'L3_procure':  dict(opt=True, fc='naive', proc=True, pool=False, pricing=False, redirect=True),
    'L4_forecast': dict(opt=True, fc='ml', proc=True, pool=False, pricing=False, redirect=True),
    'L5_REpool':   dict(opt=True, fc='ml', proc=True, pool=True, pricing=False, redirect=True),
    'B_full':      dict(opt=True, fc='ml', proc=True, pool=True, pricing=True, redirect=True)}

def run_case(cfg, sk, seed, prm, want_rec=False):
    rng = np.random.default_rng(seed); tr = make_truth(sk, rng, prm)
    fc = make_fc(tr, cfg['fc'], np.random.default_rng(seed + 1000), prm)
    price = np.repeat(pbar(prm)[:, :, None], T, 2)
    if cfg['opt']: prm = dict(prm, ems=True, sleep=.9)                    # EMS: module staging + idle-module sleep
    if not cfg['opt']:
        rec = run_rule(tr, fc, prm)
    else:
        if cfg['pricing']:
            e = eff(prm, tr['heat'], tr['kap'])
            mu = solve(0, samples(fc, np.random.default_rng(seed + 3000), prm['S']), init_state(prm),
                       dict(q=None, alpha=None), e, price, mip=False, duals=True)['mu']
            price = set_prices(mu, prm); tr = apply_dr(tr, price, prm); fc = apply_dr(fc, price, prm)
        rec = run_opt(tr, fc, prm, price, np.random.default_rng(seed + 2000), cfg['proc'], cfg['pool'], cfg['redirect'])
    k = kpis(rec, tr, prm, price)
    out = (k, check(rec, tr, prm, price))
    if want_rec: out += (rec, tr, fc, price)
    return out

def annual(df):
    """df: rows per scenario x cluster of daily values (seed mean) -> annual totals per cluster (day-type weights x 365)"""
    num = [c for c in df.columns if c not in ('cluster', 'scenario', 'seed', 'reliability', 'loss_pct', 'min_ready_pack_frac')]
    w = df['scenario'].map(WEIGHT) * 365
    return df[num].mul(w, axis=0).groupby(df['cluster'], sort=False).sum().loc[CL]

def pnl(a, tech=False):
    """annual P&L per cluster in Rs crore"""
    cr = a / 1e4; out = pd.DataFrame(index=a.index)
    out['revenue'] = cr.revenue
    out['energy_cost'] = cr.cost_re + cr.cost_dam + cr.cost_rtm + cr.cost_discom
    out['degradation'] = cr.cost_degradation; out['vom_logistics'] = cr.cost_vom + cr.cost_logistics
    out['storage_value'] = cr.storage_value
    out['fixed_opex'] = FIXED_LAKH_PER_MW * P_INST / 100
    out['tech_run'] = TECH_RUN_CR * P_INST / P_INST.sum() if tech else 0.
    out['ebitda'] = cr.contribution - out.fixed_opex - out.tech_run
    out['margin'] = out.ebitda / out.revenue
    out['reliability'] = a.served / a.demand; out['loss_pct'] = a.losses / a.imports
    return out
