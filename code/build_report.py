"""Fills report_template.md from outputs/csv and builds PDF (xelatex) + DOCX (pandoc)."""
import os, re, json, subprocess
import numpy as np, pandas as pd
import pypandoc
from sim import *

REP = os.path.join(ROOT, 'report'); FIG = os.path.join(ROOT, 'outputs', 'charts')
pnl_all = pd.read_csv(os.path.join(CSV, 'annual_pnl_by_config.csv'))
port = pd.read_csv(os.path.join(CSV, 'portfolio_kpis_by_config.csv'), index_col=0)
scn = pd.read_csv(os.path.join(CSV, 'scenario_cluster_daily_mean.csv'))
sens = pd.read_csv(os.path.join(CSV, 'sensitivity.csv')); fm = pd.read_csv(os.path.join(CSV, 'forecast_metrics.csv'))
val = pd.read_csv(os.path.join(CSV, 'validation_checks.csv')); fin = json.load(open(os.path.join(CSV, 'financials.json')))
K = pd.read_csv(os.path.join(CSV, 'daily_kpis_all_runs.csv'))
P = {c: g.set_index('cluster').loc[CL] for c, g in pnl_all.groupby('config')}
pc = lambda x, d=1: f'{100 * x:.{d}f}%'
cr = lambda x: f'{x:,.0f}'

def md(df, idx=True):
    df = df.reset_index() if idx else df
    cols = list(df.columns); num = lambda c: all(re.fullmatch(r'[-+*0-9.,%x ]+(pp| /yr| months)?\**', str(v)) for v in df[c] if str(v) != '')
    out = ['| ' + ' | '.join(map(str, cols)) + ' |', '|' + '|'.join(['---' if i == 0 or not num(c) else '---:' for i, c in enumerate(cols)]) + '|']
    for _, r in df.iterrows(): out.append('| ' + ' | '.join(map(str, r.values)) + ' |')
    return '\n'.join(out)

def ann_cfg(cfg):
    d = K[(K.config == cfg) & (K.case == 'Base')].groupby(['scenario', 'cluster'], sort=False).mean(numeric_only=True).reset_index()
    return annual(d)
aA, aB = ann_cfg('A_rule'), ann_cfg('B_full')
A, B = P['A_rule'], P['B_full']
V = dict(relA=pc(port.loc['A_rule', 'reliability']), relB=pc(port.loc['B_full', 'reliability']),
         lossA=pc(port.loc['A_rule', 'loss']), lossB=pc(port.loc['B_full', 'loss']),
         revA=cr(A.revenue.sum()), revB=cr(B.revenue.sum()), ebA=cr(A.ebitda.sum()), ebB=cr(B.ebitda.sum()),
         mA=pc(port.loc['A_rule', 'margin']), mB=pc(port.loc['B_full', 'margin']),
         minmA=pc(A.margin.min()), minclA=A.margin.idxmin(), minmB=pc(B.margin.min()), minclB=B.margin.idxmin(),
         run=f"{fin['run_total']:.1f}", capex=f"{fin['capex_total']:.0f}", npv=cr(fin['npv_5yr']),
         payback=(f"{12 * fin['payback_years']:.0f} months" if fin['payback_years'] < 2 else f"{fin['payback_years']:.1f} years"),
         roi=f"{fin['roi_5yr']:.1f}x", uplift=cr(fin['annual_ebitda_uplift']), quarter=cr(fin['annual_ebitda_uplift'] / 4),
         degb=f"{BASE['deg_b']:.1f}", degp=f"{BASE['deg_p']:.1f}")
dE = B.ebitda.sum() - A.ebitda.sum(); dRev = B.revenue.sum() - A.revenue.sum()
V['share_rev'] = 'Most' if dRev > dE else 'Part'
V['drev'] = cr(dRev); V['dloss'] = f"{100 * (port.loc['A_rule', 'loss'] - port.loc['B_full', 'loss']):.1f}"
V['lossgap'] = f"{max(0, 100 * port.loc['B_full', 'loss'] - 8):.1f}"; V['relDelhiB'] = pc(B.reliability.iloc[0])
V['mMumA'] = pc(A.margin.loc['Mumbai MMR']); V['mMumB'] = pc(B.margin.loc['Mumbai MMR'])
V['val_soc'] = f"{abs(min(0, val.min_soc_margin_MWh.min())):.3f}"; V['val_price'] = pc(val.max_retail_price_dev.max())

# unserved breakdown, baseline
g = scn[scn.config == 'A_rule'].groupby('scenario', sort=False).sum(numeric_only=True)
t = pd.DataFrame({'Scenario (day type, weight)': [f'{s} {SCN[s]["name"]} ({int(100 * WEIGHT[s])}%)' for s in g.index],
                  'Demand (MWh/day)': g.demand.map(cr), 'Unserved DC (MWh/day)': (g.dc_demand - g.dc_served).map(cr),
                  'Unserved swaps (MWh/day)': (g.sw_demand - g.sw_served).map(cr), 'Reliability': (g.served / g.demand).map(pc)})
V['tbl_unserved_A'] = md(t, False)

# assumptions
rows = [
 ('A1', 'Cluster nameplate (MW)', 'Delhi 336, Mumbai 288, Bengaluru 240, Hyderabad 180, Chennai 156 (=1,200)', 'Assumption (total = fact)', 'Proportional to metro EV fleet size and charging density'),
 ('A2', 'Demand (MWh/day, normal day)', ' / '.join(f'{x:,.0f}' for x in D_DAY) + f' (total {D_DAY.sum():,.0f})', 'Assumption', 'About 16% utilisation of nameplate: congested cores, idle outskirts'),
 ('A3', 'DC vs swap (MW)', ' / '.join(f'{a:.0f}+{b:.0f}' for a, b in zip(P_DC_C, P_SW_C)) + ' (720 + 480)', 'Total split = fact', 'Delhi/Bengaluru swap-heavy (e-3W, delivery 2W)'),
 ('A4', 'Stationary vs pack storage', '300 MWh BESS (0.5C, 150 MW) + 300 MWh packs (about 100k packs of 3 kWh)', 'Assumption (total = fact)', 'Even split; BESS at 0.5C is typical for C&I LFP'),
 ('A5', 'Efficiencies', f"DC charger {BASE['eta_dc']:.3f} (0.955 with module staging); pack charging {BASE['eta_pk']:.2f}; BESS {BASE['eta_c']:.2f} each way (RTE 0.88); distribution {BASE['lam']:.1%}", 'Assumption', 'LFP RTE 85-90% [18]; SiC charger peak about 96%'),
 ('A6', 'Self-discharge and standby', f"BESS {BASE['sig_b']:.2%}/h; ready packs {BASE['sig_p']:.2%}/h + cabinet standby {BASE['aux_r']:.1%}/h of stock; BESS HVAC {BASE['aux_b']:.1%}/h; idle chargers {BASE['a_sb']:.1%} of DC MW", 'Assumption', 'Calibrated so baseline losses ≈ 10% (case fact)'),
 ('A7', 'Degradation cost', f"BESS Rs {BASE['deg_b']}/kWh discharged; packs Rs {BASE['deg_p']}/kWh charged; ≤1.5 cycles/day; SOH 0.90-0.95", 'Assumption', 'Pack cost / (cycle life × DoD): Rs 9k/kWh ÷ (6,000 × 0.85); Rs 12k ÷ 2,500 [15][16]'),
 ('A8', 'Renewable availability', 'Intra-state solar+wind per cluster + inter-state pool (70 MWp solar, 50 MW wind, scaled); about 33% of imports in baseline', 'Calibrated to fact (35%)', 'Chennai wind CF 38%, Mumbai cloudier, Delhi solar only'),
 ('A9', 'RE contract price', f"Rs {BASE['c_re']}/kWh landed, take-or-pay; banking credit Rs {BASE['v_bank']}/kWh up to {BASE['bank_frac']:.0%} of daily demand", 'Assumption', 'Recent C&I RE PPA levels plus OA charges; GEOA banking [4]'),
 ('A10', 'DISCOM tariff', 'ACoS Rs ' + ' / '.join(f'{x}' for x in ACOS) + '; 0.7× in solar hours (09-16), 1.3× otherwise', 'Assumption (structure from [5])', 'MoP 2024 EV-tariff guideline structure'),
 ('A11', 'IEX DAM prices', 'Stylised hourly shape: Rs 2.5 (noon) to Rs 9.9 (20:00), mean ≈ Rs 5.4; RTM = DAM ± 10%; ceiling Rs 10', 'Assumption (ceiling = fact [3])', 'Shape follows the solar "duck" pattern; not historical data'),
 ('A12', 'Open-access adder', f"Rs {BASE['oa']}/kWh (transmission, wheeling, CSS, fees); OA-eligible load share " + ' / '.join(f'{x:.0%}' for x in THETA_C), 'Assumption', 'Only HT hubs ≥1 MW (≥100 kW green) can buy OA [4]'),
 ('A13', 'Grid connection limits', f'{G_MAX_C.sum():.0f} MW total (23% of nameplate)', 'Assumption', 'Sanctioned load is usually far below charger nameplate'),
 ('A14', 'Charger power limits', f"Deliverable DC = {BASE['kappa']:.0%} of nameplate (availability × vehicle acceptance × location); 12% heat derate", 'Calibrated', 'Calibrated to 78% reliability (case fact)'),
 ('A15', 'SOC constraints', '10–95% of usable (nameplate × SOH); day-end SOC ≥ day-start', 'Assumption', 'LFP warranty windows'),
 ('A16', 'Pack dynamics', f"Recharge time {BASE['tau']:.0f} h; van capacity 5% of cluster packs per h; core zone hosts {ZC:.0%} of assets but {ZD.min():.0%}-{ZD.max():.0%} of demand", 'Assumption', 'Typical 2W/3W pack charge at 0.33C'),
 ('A17', 'Curtailment', 'RE surplus beyond use and banking cap is curtailed at zero value', 'Assumption', 'No resale without seller registration'),
 ('A18', 'Forecast accuracy', 'Naive: ±6% day level, ±10% hourly, blind to scenario. ML: ±3%/±5% (demand), ±8% (RE), ±7% (price)', 'Assumption', 'Consistent with published day-ahead load-forecast accuracy [8]'),
 ('A19', 'Price elasticity', 'Timing ε: retail −0.6, ride-hailing −0.8, logistics 0; volume ε: −0.3/−0.2/0', 'Assumption', 'Fleets shift readily; retail less; tested at half'),
 ('A20', 'Customer segments (DC share)', 'Retail / ride-hailing / logistics, e.g. Delhi 30/50/20, Chennai 25/30/45; swaps = 20-32% of energy', 'Assumption', 'Delhi ride-hailing heavy; Chennai logistics/port'),
 ('A21', 'Revenue', f"Retail Rs {BASE['p_seg'][0]}, ride-hailing Rs {BASE['p_seg'][1]}, logistics Rs {BASE['p_seg'][2]}, swap Rs {BASE['p_sw']}/kWh (≈ Rs {3 * BASE['p_sw']:.0f} per 3 kWh swap); local index 0.97-1.05", 'Assumption', 'Ex-GST public DC tariffs of about Rs 18-24/kWh incl. GST'),
 ('A22', 'Operating cost', 'Fixed Rs ' + ' / '.join(f'{x:.0f}' for x in FIXED_LAKH_PER_MW) + f" lakh/MW/yr; variable O&M Rs {BASE['vom']}/kWh; van Rs {BASE['c_van']}/kWh moved; redirect incentive Rs {BASE['delta']}/kWh", 'Assumption', 'Site lease, staff, network, insurance, maintenance'),
 ('A23', 'Unserved penalty (goodwill/SLA)', f"Rs {BASE['pen'][0]:.0f} / {BASE['pen'][1]:.0f} / {BASE['pen'][2]:.0f} / {BASE['pen_sw']:.0f} per kWh (retail / ride / logistics / swap)", 'Assumption', 'Decision weight only; not in EBITDA'),
 ('A24', 'Day-type weights', 'Normal 65%, regional imbalance 10%, RE shortfall + price spike 10%, heatwave 15%', 'Assumption', 'About 55 heatwave days per year in north/central India'),
]
V['tbl_assumptions'] = md(pd.DataFrame(rows, columns=['ID', 'Item', 'Value used', 'Type', 'Justification']), False)

f = fm.groupby('model')[['MAE', 'RMSE', 'WAPE', 'pinball_P10', 'pinball_P50', 'pinball_P90', 'P10_P90_coverage']].mean()
f.index = ['Seasonal-naive (A)' if i == 'naive' else 'ML quantile (B)' for i in f.index]
ft = pd.DataFrame({'MAE (MWh/h)': f.MAE.map('{:.2f}'.format), 'RMSE (MWh/h)': f.RMSE.map('{:.2f}'.format), 'WAPE': f.WAPE.map(pc),
                   'Pinball P10': f.pinball_P10.map('{:.2f}'.format), 'Pinball P50': f.pinball_P50.map('{:.2f}'.format),
                   'Pinball P90': f.pinball_P90.map('{:.2f}'.format), 'P10–P90 coverage': f.P10_P90_coverage.map(pc)})
ft.index.name = 'Forecaster (cluster-hour demand)'; V['tbl_forecast'] = md(ft)

def mix(a):
    imp = a.imports.sum(); dam = a.dam.sum() - a.dam_underdrawn.sum()
    cost = (a.cost_re + a.cost_dam + a.cost_rtm + a.cost_discom).sum() / imp
    return [pc(a.re_used.sum() / imp), pc(dam / imp), pc(a.rtm.sum() / imp), pc(a.discom.sum() / imp),
            pc(a.re_curtailed.sum() / a.re_avail.sum()), pc(a.re_banked.sum() / a.re_avail.sum()), f'{cost:.2f}', f'{a.imports.sum() / 1e3:,.0f}']
mt = pd.DataFrame([mix(aA), mix(aB)], index=['Rule-based (A)', 'Framework (B)'],
                  columns=['RE used', 'IEX DAM (net)', 'IEX RTM', 'DISCOM', 'RE curtailed (of avail.)', 'RE banked (of avail.)', 'Rs/kWh imported', 'Imports (GWh/yr)'])
mt.index.name = 'Annual procurement mix [Model output]'; V['tbl_procmix'] = md(mt)

gs = scn[scn.config.isin(['A_rule', 'B_full'])].groupby(['scenario', 'config'], sort=False).sum(numeric_only=True)
rows = []
for s in SCN:
    for cfg, lab in [('A_rule', 'A'), ('B_full', 'B')]:
        d = gs.loc[(s, cfg)]
        rows.append([f'{s} {SCN[s]["name"]}', lab, cr(d.demand), cr(d.re_avail), cr(d.re_used), cr(d.dam - d.dam_underdrawn), cr(d.rtm), cr(d.discom),
                     cr(d.bess_ch), cr(d.bess_dis), cr(d.sw_served), cr(d.unserved), pc(d.served / d.demand), pc(d.losses / d.imports),
                     f'{(d.cost_re + d.cost_dam + d.cost_rtm + d.cost_discom) / 1e4:.2f}', f'{d.revenue / 1e4:.2f}', f'{d.contribution / 1e4:.2f}'])
_cols = ['Scenario', 'Strategy', 'Demand (MWh)', 'RE available (MWh)', 'RE used (MWh)', 'IEX DAM, net (MWh)', 'IEX RTM (MWh)', 'DISCOM (MWh)',
         'BESS charge (MWh)', 'BESS discharge (MWh)', 'Swap energy delivered (MWh)', 'Unserved (MWh)', 'Reliability', 'Losses',
         'Energy cost (Rs Cr)', 'Revenue (Rs Cr)', 'Contribution (Rs Cr)']
_t = pd.DataFrame(rows, columns=_cols); _t.index = [f'{a.split()[0]} {b}' for a, b in zip(_t.Scenario, _t.Strategy)]
_t = _t.drop(columns=['Scenario', 'Strategy']).T; _t.index.name = 'Per day (portfolio)'
V['tbl_scenarios'] = ('*Columns: scenario and strategy (A = rule-based, B = framework). S1 normal, S2 regional imbalance, S3 RE shortfall + IEX spike, '
                      'S4 heatwave. Contribution is before fixed costs.*\n\n' + md(_t))
ct = pd.DataFrame({'Reliability A': A.reliability.map(pc), 'Reliability B': B.reliability.map(pc), 'Losses A': A.loss_pct.map(pc), 'Losses B': B.loss_pct.map(pc),
                   'Revenue A (Rs Cr)': A.revenue.map(cr), 'Revenue B (Rs Cr)': B.revenue.map(cr), 'EBITDA A (Rs Cr)': A.ebitda.map(cr), 'EBITDA B (Rs Cr)': B.ebitda.map(cr),
                   'Margin A': A.margin.map(pc), 'Margin B': B.margin.map(pc)})
ct.index.name = 'Cluster'; V['tbl_cluster'] = md(ct)

def pl(p, lab):
    return pd.Series({'Revenue (charging + swaps)': p.revenue.sum(), 'Energy procurement': -p.energy_cost.sum(), 'Battery degradation': -p.degradation.sum(),
                      'Variable O&M + logistics': -p.vom_logistics.sum(), 'Change in stored energy (valued)': p.storage_value.sum(),
                      'Fixed operating cost': -p.fixed_opex.sum(), 'Framework run cost (cloud, software, team)': -p.tech_run.sum(),
                      'EBITDA': p.ebitda.sum()}, name=lab)
pt = pd.concat([pl(A, 'Rule-based (A)'), pl(B, 'Framework (B)')], axis=1); pt['Change'] = pt.iloc[:, 1] - pt.iloc[:, 0]
pt.loc['EBITDA margin'] = [port.loc['A_rule', 'margin'], port.loc['B_full', 'margin'], port.loc['B_full', 'margin'] - port.loc['A_rule', 'margin']]
lost = pd.Series([aA.lost_revenue.sum() / 1e4, aB.lost_revenue.sum() / 1e4, (aB.lost_revenue.sum() - aA.lost_revenue.sum()) / 1e4], index=pt.columns, name='Memo: revenue lost to unserved demand')
pt = pd.concat([pt, lost.to_frame().T])
ptf = pt.copy().astype(object)
for i in pt.index: ptf.loc[i] = [pc(v) if i == 'EBITDA margin' else f'{v:,.1f}' for v in pt.loc[i]]
ptf.index.name = 'Rs crore / year'; V['tbl_pnl'] = md(ptf)
V['bridge_lines'] = '; '.join(f'{k} {v:+,.0f}' for k, v in pt['Change'].items() if k not in ('EBITDA margin', 'EBITDA', 'Memo: revenue lost to unserved demand')) + f'; **EBITDA {pt.loc["EBITDA", "Change"]:+,.0f}**.'

order = ['A_rule', 'L1_dispatch', 'L2_network', 'L3_procure', 'L4_forecast', 'L5_REpool', 'B_full']
names = ['Rule-based baseline (A)', '+ Optimised BESS, pack charging and van rebalancing; EMS module staging and sleep (hourly MPC)', '+ Fleet redirection from core to outer zones',
         '+ Stochastic day-ahead procurement', '+ ML probabilistic forecasts', '+ Inter-state RE-pool allocation', '+ Dynamic pricing = Framework (B)']
lv = []
for i, (o, nm) in enumerate(zip(order, names)):
    r = port.loc[o]; prv = port.loc[order[i - 1]] if i else None
    lv.append([nm, pc(r.reliability), pc(r.loss), cr(r.ebitda), '' if prv is None else f'{r.ebitda - prv.ebitda:+,.0f}', '' if prv is None else f'{100 * (r.reliability - prv.reliability):+.1f} pp'])
V['tbl_levers'] = md(pd.DataFrame(lv, columns=['Configuration (cumulative)', 'Reliability', 'Losses', 'EBITDA (Rs Cr)', 'Δ EBITDA', 'Δ reliability']), False)

cx = pd.DataFrame([(k, f'{v:.1f}') for k, v in fin['capex'].items()] + [('**Total one-time**', f"**{fin['capex_total']:.1f}**")] +
                  [(f'Run: {k}', f'{v:.1f} /yr') for k, v in fin['run_cost'].items()] + [('**Total run cost**', f"**{fin['run_total']:.1f} /yr**")],
                  columns=['Technology cost item', 'Rs crore'])
V['tbl_capex'] = md(cx, False)

st = sens.copy(); st['Strategy'] = st.config.map({'A_rule': 'A', 'B_full': 'B'})
st = st[['case', 'Strategy', 'reliability', 'loss', 'revenue', 'ebitda', 'margin', 'min_cluster_margin', 'min_cluster']]
for c in ['reliability', 'loss', 'margin', 'min_cluster_margin']: st[c] = st[c].map(pc)
for c in ['revenue', 'ebitda']: st[c] = st[c].map(cr)
st.columns = ['Case', 'Strategy', 'Reliability', 'Losses', 'Revenue (Rs Cr)', 'EBITDA (Rs Cr)', 'Margin', 'Lowest cluster margin', 'Lowest cluster']
V['tbl_sens'] = md(st, False)

CAP = {'01': ('01_capacity_by_cluster.png', 'Figure 1 – Installed capacity by cluster [Assumption]. Bars show nameplate MW split into DC fast charging and swap racks. The black tick marks the sanctioned grid connection, the real power ceiling (23% of nameplate). Text gives the energy storage (MWh) behind each cluster.'),
       '02': ('02_demand_vs_renewables.png', 'Figure 2 – Hourly customer demand vs renewable availability on a normal day (synthetic truth, seed 1). Renewables peak at noon while demand peaks in the evening; the dashed line is the grid connection. This mismatch drives both curtailment and evening shortages.'),
       '03': ('03_iex_price_vs_bess.png', 'Figure 3 – IEX DAM price (top) and portfolio BESS schedule (bottom; discharge positive, charge negative), normal day. The rule-based timetable charges at night; the optimiser charges in cheap solar hours and discharges into the price and congestion peak.'),
       '04': ('04_bess_soc.png', 'Figure 4 – Portfolio BESS state of charge, normal day and heatwave. The optimiser keeps SOC within 10–95%, ends the day where it started (steady state), and holds energy for the evening peak and the 5% reserve.'),
       '05': ('05_swap_inventory.png', 'Figure 5 – Ready (charged) packs at the start of each hour, Delhi NCR core zone. Under charge-on-return (A), every charged pack above the 5% reserve is swapped out within each peak hour and refills are capped by the 3-hour charging time, so the stock plateaus near 30% while swap demand goes unmet from 09:00 to 23:00. The framework (B) builds stock to about 80% before the morning peak and brings charged packs in from the outer zone by van.'),
       '06': ('06_reliability_before_after.png', 'Figure 6 – Annualised service reliability by cluster, before (grey) and after (blue). Target 89.7%.'),
       '07': ('07_losses_before_after.png', 'Figure 7 – Energy losses as a share of imported energy (charger conversion, pack and battery round-trip, standby/self-discharge, distribution). Target 8%.'),
       '08': ('08_ebitda_margin_by_cluster.png', 'Figure 8 – EBITDA margin by cluster after fixed opex and (for B) the framework run cost. Target >12% in every cluster.'),
       '09': ('09_scenario_comparison.png', 'Figure 9 – Scenario-wise portfolio performance: reliability, losses and daily contribution (before fixed costs), rule-based vs framework, mean of 3 stochastic draws.'),
       '10': ('10_sensitivity_iex.png', 'Figure 10 – EBITDA and weakest-cluster margin against IEX price increases of 0–30% (prices capped at Rs 10/kWh).'),
       '11': ('11_sensitivity_tornado.png', 'Figure 11 – Change in framework EBITDA under each sensitivity case, with the weakest cluster margin.'),
       '12': ('12_ebitda_bridge_levers.png', 'Figure 12 – EBITDA bridge by lever, added cumulatively (order-dependent attribution).'),
       '13': ('13_dynamic_pricing.png', 'Figure 13 – Dynamic tariffs in the Delhi NCR core zone. Retail moves within the ±20% band, fleets within a ±10% contractual ToU band, and logistics stays on contract. Prices rise only where the shadow value of energy is high.'),
       '14': ('14_architecture.png', 'Figure 14 – End-to-end decision architecture and operating cadence.')}

def fill(fmt):
    s = open(os.path.join(REP, 'report_template.md'), encoding='utf-8').read()
    for k, (fn, cap) in CAP.items():
        w = '16cm' if fmt == 'docx' else '100%'
        s = s.replace('{{fig_%s}}' % k, f'![{cap}]({os.path.join(FIG, fn).replace(os.sep, "/")}){{width={w}}}')
    for k, v in V.items(): s = s.replace('{{' + k + '}}', str(v))
    left = re.findall(r'\{\{[a-zA-Z_0-9]+\}\}', s); assert not left, left
    return s

HDR = r"""
\usepackage{etoolbox}\AtBeginEnvironment{longtable}{\scriptsize}
\usepackage{xcolor}\definecolor{navy}{HTML}{1F3864}
\usepackage{sectsty}\allsectionsfont{\color{navy}}
\usepackage{fancyhdr}\pagestyle{fancy}\fancyhf{}\rhead{\footnotesize EV energy \& storage decision system}\lhead{\footnotesize Strategy report}\cfoot{\thepage}
\setlength{\headheight}{14pt}
\renewcommand{\arraystretch}{1.15}
\usepackage{caption}\captionsetup{font=small,labelformat=empty,justification=raggedright,singlelinecheck=false}
"""
if __name__ == '__main__':
    os.makedirs(REP, exist_ok=True)
    md_pdf = fill('pdf'); open(os.path.join(REP, 'report_filled.md'), 'w', encoding='utf-8').write(md_pdf)
    open(os.path.join(REP, 'header.tex'), 'w', encoding='utf-8').write(HDR)
    pdf = os.path.join(REP, 'EV_Energy_Decision_System_Report.pdf'); docx = os.path.join(REP, 'EV_Energy_Decision_System_Report.docx')
    pypandoc.convert_text(md_pdf, 'pdf', format='markdown', outputfile=pdf,
        extra_args=['--pdf-engine=xelatex', '-H', os.path.join(REP, 'header.tex'), '--toc', '--toc-depth=1',
                    '-V', 'geometry:margin=2cm', '-V', 'mainfont=Calibri', '-V', 'fontsize=10pt', '-V', 'colorlinks=true',
                    '-V', 'linkcolor=navy', '-V', 'documentclass=article', '-V', 'papersize=a4'])
    ref = os.path.join(REP, 'reference.docx')
    if not os.path.exists(ref):
        subprocess.run([pypandoc.get_pandoc_path(), '-o', ref, '--print-default-data-file', 'reference.docx'], check=True)
        import docx as dx
        from docx.shared import Pt, RGBColor
        d = dx.Document(ref)
        for st in d.styles:
            try:
                if st.type == 1:
                    st.font.name = 'Calibri'
                    if st.name.startswith('Heading') or st.name == 'Title': st.font.color.rgb = RGBColor(0x1F, 0x38, 0x64)
                    if st.name in ('Normal', 'Body Text', 'First Paragraph', 'Compact'): st.font.size = Pt(10)
            except Exception: pass
        d.save(ref)
    pypandoc.convert_text(fill('docx'), 'docx', format='markdown', outputfile=docx, extra_args=['--reference-doc', ref, '--toc', '--toc-depth=1'])
    print('built', pdf, docx)
