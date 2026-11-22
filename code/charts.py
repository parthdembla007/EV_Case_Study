"""All report figures, generated only from the simulation outputs in outputs/csv."""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sim import CL, NC, P_DC_C, P_SW_C, E_B_C, E_PK_C, G_MAX_C, E_BU, SCN, ROOT

CSV = os.path.join(ROOT, 'outputs', 'csv'); FIG = os.path.join(ROOT, 'outputs', 'charts'); os.makedirs(FIG, exist_ok=True)
BLUE, ORANGE, AQUA, YELLOW, MAG = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4'
GREY, INK, INK2, GRID = '#9a9993', '#0b0b0b', '#52514e', '#e6e5e0'
A_C, B_C = GREY, BLUE
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10.5, 'axes.titleweight': 'bold',
                     'axes.edgecolor': '#b5b4ae', 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': .6, 'legend.frameon': False, 'figure.dpi': 110, 'savefig.dpi': 220,
                     'savefig.bbox': 'tight', 'lines.linewidth': 2})
SHORT = ['Delhi', 'Mumbai', 'Bengaluru', 'Hyderabad', 'Chennai']
hr = pd.read_csv(os.path.join(CSV, 'hourly_dispatch_A_vs_B.csv'))
pnl = pd.read_csv(os.path.join(CSV, 'annual_pnl_by_config.csv'))
scn = pd.read_csv(os.path.join(CSV, 'scenario_cluster_daily_mean.csv'))
sens = pd.read_csv(os.path.join(CSV, 'sensitivity.csv'))
port = pd.read_csv(os.path.join(CSV, 'portfolio_kpis_by_config.csv'), index_col=0)
fin = json.load(open(os.path.join(CSV, 'financials.json')))
def save(fig, name): fig.savefig(os.path.join(FIG, name)); plt.close(fig)
def H(cfg, sk): return hr[(hr.config == cfg) & (hr.scenario == sk)]
def portf(cfg, sk): return H(cfg, sk).groupby('hour').sum(numeric_only=True)

# 01 capacity
fig, ax = plt.subplots(figsize=(7.2, 3.0)); y = np.arange(NC)[::-1]
ax.barh(y, P_DC_C, color=BLUE, height=.55, label='DC fast charging (MW)', edgecolor='white', linewidth=2)
ax.barh(y, P_SW_C, left=P_DC_C, color=AQUA, height=.55, label='Battery-swap racks (MW)', edgecolor='white', linewidth=2)
ax.scatter(G_MAX_C, y, marker='|', s=420, color=INK, zorder=5, label='Sanctioned grid connection (MW)')
for i, c in enumerate(range(NC)):
    ax.text(P_DC_C[c] + P_SW_C[c] + 6, y[i], f'BESS {E_B_C[c]:.0f} MWh · packs {E_PK_C[c]:.0f} MWh', va='center', fontsize=8, color=INK2)
ax.set_yticks(y, SHORT); ax.set_xlim(0, 470); ax.set_xlabel('MW (power)'); ax.grid(axis='y', visible=False)
ax.set_title('Installed capacity by cluster: 1,200 MW nameplate, 276 MW grid connection, 600 MWh storage', loc='left')
ax.legend(loc='upper center', bbox_to_anchor=(.5, -.22), ncol=3, fontsize=8); save(fig, '01_capacity_by_cluster.png')

# 02 demand vs renewable
fig, axs = plt.subplots(1, 5, figsize=(11, 2.8), sharey=False)
for i, c in enumerate(CL):
    d = H('A_rule', 'S1')[lambda x: x.cluster == c]
    ax = axs[i]; ax.fill_between(d.hour, d.re_avail, color=YELLOW, alpha=.35, lw=0, label='RE available')
    ax.plot(d.hour, d.dc_demand + d.sw_demand, color=BLUE, label='Customer demand')
    ax.axhline(G_MAX_C[i], color=INK, lw=1, ls='--', label='Grid connection')
    ax.set_title(SHORT[i], loc='left'); ax.set_xticks([0, 6, 12, 18, 23]); ax.set_xlabel('Hour')
    if i == 0: ax.set_ylabel('MWh per hour')
axs[0].legend(fontsize=7.5, loc='upper left')
fig.suptitle('Hourly demand vs renewable availability, normal day (S1): RE peaks at noon, demand peaks in the evening', x=.01, ha='left', fontweight='bold', fontsize=10.5)
fig.tight_layout(); save(fig, '02_demand_vs_renewables.png')

# 03 IEX price vs BESS schedule
fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.3]))
pa, pb = portf('A_rule', 'S1'), portf('B_full', 'S1'); iex = H('B_full', 'S1').groupby('hour').iex_dam.first()
a1.step(iex.index, iex.values, where='mid', color=ORANGE); a1.axhline(10, color=INK2, lw=1, ls=':')
a1.text(0, 10.15, 'CERC price ceiling Rs 10/kWh', fontsize=7.5, color=INK2); a1.set_ylabel('IEX DAM (Rs/kWh)'); a1.set_ylim(0, 11.5)
a1.set_title('IEX day-ahead price (synthetic) and stationary-battery schedule, normal day', loc='left')
w = .38
a2.bar(pb.index + w / 2, pb.dis - pb.ch, width=w, color=B_C, label='Proposed (B): discharge + / charge -')
a2.bar(pa.index - w / 2, pa.dis - pa.ch, width=w, color=A_C, label='Rule-based (A)')
a2.axhline(0, color=INK2, lw=.8); a2.set_ylabel('BESS MWh per hour'); a2.set_xlabel('Hour of day'); a2.legend(fontsize=8, loc='lower left')
a2.set_xticks(range(0, 24, 2)); fig.tight_layout(); save(fig, '03_iex_price_vs_bess.png')

# 04 SOC
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.0), sharey=True)
for ax, sk in zip(axs, ['S1', 'S4']):
    for cfg, col, lab in [('A_rule', A_C, 'Rule-based (A)'), ('B_full', B_C, 'Proposed (B)')]:
        d = H(cfg, sk).groupby('hour').sum(numeric_only=True)
        ax.plot(d.index + 1, 100 * d.soc / E_BU.sum(), color=col, label=lab)
    ax.axhline(10, color=INK2, ls=':', lw=1); ax.axhline(95, color=INK2, ls=':', lw=1)
    ax.set_title(f'{sk}: {SCN[sk]["name"]}', loc='left'); ax.set_xlabel('Hour (end of)'); ax.set_xticks(range(0, 25, 4))
axs[0].set_ylabel('Portfolio BESS SOC (% of usable MWh)'); axs[0].legend(fontsize=8)
fig.suptitle('Stationary battery state of charge over 24 h (dotted lines: 10 % / 95 % SOC limits)', x=.01, ha='left', fontweight='bold', fontsize=10.5)
fig.tight_layout(); save(fig, '04_bess_soc.png')

# 05 swap inventory (Delhi core zone)
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.0), sharey=True)
for ax, sk in zip(axs, ['S1', 'S4']):
    for cfg, col, lab in [('A_rule', A_C, 'Rule-based (A)'), ('B_full', B_C, 'Proposed (B)')]:
        d = H(cfg, sk)[lambda x: x.cluster == CL[0]]
        ax.plot(d.hour + 1, 100 * d.ready_core_pct, color=col, label=lab)
    ax.axhline(5, color=ORANGE, lw=1.2, ls='--'); ax.text(0.3, 6.5, '5 % priority reserve', fontsize=7.5, color=INK2)
    ax.set_title(f'{sk}: {SCN[sk]["name"]}', loc='left'); ax.set_xlabel('Hour (end of)'); ax.set_xticks(range(0, 25, 4))
axs[0].set_ylabel('Ready packs at start of hour (% of inventory)'); axs[0].legend(fontsize=8, loc='upper right')
fig.suptitle('Swap-pack ready inventory, Delhi NCR core zone (where stock-outs happen)', x=.01, ha='left', fontweight='bold', fontsize=10.5)
fig.tight_layout(); save(fig, '05_swap_inventory.png')

# 06-08 dumbbells by cluster
def dumbbell(col, target, xlabel, title, name, pct=True, tgt_lab=''):
    A = pnl[pnl.config == 'A_rule'].set_index('cluster').loc[CL, col]; B = pnl[pnl.config == 'B_full'].set_index('cluster').loc[CL, col]
    pa = port.loc['A_rule', {'reliability': 'reliability', 'loss_pct': 'loss', 'margin': 'margin'}[col]]
    pbb = port.loc['B_full', {'reliability': 'reliability', 'loss_pct': 'loss', 'margin': 'margin'}[col]]
    k = 100 if pct else 1; labels = SHORT + ['Portfolio']; av = list(A * k) + [pa * k]; bv = list(B * k) + [pbb * k]
    fig, ax = plt.subplots(figsize=(7.2, 3.1)); y = np.arange(6)[::-1]
    for i in range(6): ax.plot([av[i], bv[i]], [y[i], y[i]], color='#cfcec8', lw=3, zorder=1)
    ax.scatter(av, y, s=60, color=A_C, zorder=3, label='Rule-based (A)'); ax.scatter(bv, y, s=60, color=B_C, zorder=3, label='Proposed (B)')
    for i in range(6):
        ax.text(av[i], y[i] + .28, f'{av[i]:.1f}', ha='center', fontsize=7.5, color=INK2)
        ax.text(bv[i], y[i] + .28, f'{bv[i]:.1f}', ha='center', fontsize=7.5, color=INK)
    ax.axvline(target, color=ORANGE, ls='--', lw=1.2); ax.text(target, 5.75, tgt_lab, color=ORANGE, fontsize=7.5, ha='center')
    ax.set_yticks(y, labels); ax.set_ylim(-.6, 6.1); ax.set_xlabel(xlabel); ax.grid(axis='y', visible=False)
    ax.set_title(title, loc='left'); ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(.5, -.2), ncol=2); save(fig, name)
dumbbell('reliability', 89.7, 'Service reliability (% of demand delivered, annualised)', 'Reliability before vs after, by cluster', '06_reliability_before_after.png', tgt_lab='target 89.7 %')
dumbbell('loss_pct', 8.0, 'Storage + conversion + distribution losses (% of imported energy)', 'Energy losses before vs after, by cluster', '07_losses_before_after.png', tgt_lab='target 8 %')
dumbbell('margin', 12.0, 'EBITDA margin (%, annual, after framework run-costs)', 'EBITDA margin before vs after, by cluster', '08_ebitda_margin_by_cluster.png', tgt_lab='target 12 %')

# 09 scenarios
g = scn[scn.config.isin(['A_rule', 'B_full'])].groupby(['config', 'scenario']).sum(numeric_only=True)
fig, axs = plt.subplots(1, 3, figsize=(11, 2.9))
mets = [('Reliability (%)', lambda d: 100 * d.served / d.demand), ('Losses (% of imports)', lambda d: 100 * d.losses / d.imports),
        ('Daily contribution (Rs crore/day)', lambda d: d.contribution / 1e4)]
sks = list(SCN); y = np.arange(4)[::-1]
for ax, (lab, f) in zip(axs, mets):
    a = [f(g.loc[('A_rule', s)]) for s in sks]; b = [f(g.loc[('B_full', s)]) for s in sks]
    for i in range(4): ax.plot([a[i], b[i]], [y[i], y[i]], color='#cfcec8', lw=3, zorder=1)
    ax.scatter(a, y, color=A_C, s=55, zorder=3, label='Rule-based (A)'); ax.scatter(b, y, color=B_C, s=55, zorder=3, label='Proposed (B)')
    for i in range(4): ax.text(b[i], y[i] + .27, f'{b[i]:.1f}', fontsize=7.5, ha='center'); ax.text(a[i], y[i] + .27, f'{a[i]:.1f}', fontsize=7.5, ha='center', color=INK2)
    ax.set_yticks(y, [f'{s} {SCN[s]["name"]}' if ax is axs[0] else '' for s in sks]); ax.set_xlabel(lab); ax.grid(axis='y', visible=False); ax.set_ylim(-.6, 3.6)
fig.legend(*axs[0].get_legend_handles_labels(), loc='lower center', ncol=2, fontsize=8, bbox_to_anchor=(.5, -.04))
fig.suptitle('Scenario-wise performance, portfolio of five clusters (mean of 3 stochastic draws)', x=.01, ha='left', fontweight='bold', fontsize=10.5)
fig.tight_layout(); save(fig, '09_scenario_comparison.png')

# 10 IEX sensitivity
fig, axs = plt.subplots(1, 2, figsize=(9, 3.0))
tags = ['Base', 'IEX +10%', 'IEX +20%', 'IEX +30%']; x = [0, 10, 20, 30]
for cfg, col, lab in [('A_rule', A_C, 'Rule-based (A)'), ('B_full', B_C, 'Proposed (B)')]:
    s = sens[sens.config == cfg].set_index('case').loc[tags]
    axs[0].plot(x, s.ebitda, marker='o', color=col, label=lab); axs[1].plot(x, 100 * s.min_cluster_margin, marker='o', color=col, label=lab)
    for xi, v in zip(x, s.ebitda): axs[0].text(xi, v + 12, f'{v:.0f}', ha='center', fontsize=7.5, color=INK2)
axs[1].axhline(12, color=ORANGE, ls='--', lw=1.2); axs[1].text(30, 12.4, '12 %', color=ORANGE, fontsize=7.5, ha='right')
axs[0].set_ylabel('Portfolio EBITDA (Rs crore / yr)'); axs[1].set_ylabel('Lowest cluster EBITDA margin (%)')
for ax in axs: ax.set_xlabel('IEX price increase (%, before the Rs 10/kWh ceiling)'); ax.set_xticks(x)
axs[0].legend(fontsize=8); axs[0].set_title('EBITDA sensitivity to IEX prices', loc='left'); axs[1].set_title('Weakest cluster vs 12 % target', loc='left')
fig.tight_layout(); save(fig, '10_sensitivity_iex.png')

# 11 other sensitivities (B)
base = sens[(sens.config == 'B_full') & (sens.case == 'Base')].iloc[0]
oth = sens[(sens.config == 'B_full') & (sens.case != 'Base')].copy(); oth['d'] = oth.ebitda - base.ebitda; oth = oth.sort_values('d')
fig, ax = plt.subplots(figsize=(7.2, 3.2)); y = np.arange(len(oth))
ax.hlines(y, 0, oth.d, color='#cfcec8', lw=3); ax.scatter(oth.d, y, color=[B_C if v >= 0 else ORANGE for v in oth.d], s=60, zorder=3)
for i, (v, m) in enumerate(zip(oth.d, oth.min_cluster_margin)): ax.text(v + (8 if v >= 0 else -8), i, f'{v:+.0f} Cr · min margin {100 * m:.1f} %', va='center', ha='left' if v >= 0 else 'right', fontsize=7.5)
ax.axvline(0, color=INK2, lw=.8); ax.set_yticks(y, oth.case); ax.grid(axis='y', visible=False)
ax.set_xlabel(f'Change in framework EBITDA vs base (Rs crore / yr; base = {base.ebitda:.0f})'); lim = max(abs(oth.d).max() * 1.9, 50); ax.set_xlim(-lim, lim)
ax.set_title('Proposed framework (B): EBITDA sensitivity to key assumptions', loc='left'); save(fig, '11_sensitivity_tornado.png')

# 12 EBITDA bridge by lever
order = ['A_rule', 'L1_dispatch', 'L2_network', 'L3_procure', 'L4_forecast', 'L5_REpool', 'B_full']
labs = ['Baseline\n(A)', 'BESS, pack &\nvan dispatch', 'Fleet\nredirection', 'Stochastic\nprocurement', 'ML\nforecasts', 'RE-pool\nallocation', 'Dynamic\npricing']
E = [port.loc[o, 'ebitda'] for o in order]; fig, ax = plt.subplots(figsize=(8.6, 3.3))
ax.bar(0, E[0], color=A_C, width=.6); cum = E[0]
for i in range(1, len(E)):
    d = E[i] - E[i - 1]; ax.bar(i, d, bottom=cum, color=B_C if d >= 0 else ORANGE, width=.6); ax.text(i, cum + max(d, 0) + 8, f'{d:+.0f}', ha='center', fontsize=8); cum += d
ax.bar(len(E), E[-1], color=B_C, width=.6); ax.text(0, E[0] + 8, f'{E[0]:.0f}', ha='center', fontsize=8); ax.text(len(E), E[-1] + 8, f'{E[-1]:.0f}', ha='center', fontsize=8, fontweight='bold')
ax.set_xticks(range(len(E) + 1), labs + ['Framework\n(B)'], fontsize=7.8); ax.set_ylabel('Portfolio EBITDA (Rs crore / yr)'); ax.grid(axis='x', visible=False)
ax.set_title('EBITDA bridge: incremental value of each lever (added cumulatively in this order)', loc='left'); save(fig, '12_ebitda_bridge_levers.png')

# 13 retail price band
d = H('B_full', 'S1')[lambda x: x.cluster == CL[0]]
fig, ax = plt.subplots(figsize=(7.2, 2.7)); pbar = 19.0
ax.fill_between([-.5, 23.5], .8 * pbar, 1.2 * pbar, color=GRID, label='±20 % band around yesterday\'s average')
ax.step(d.hour, d.retail_price, where='mid', color=B_C, label='Retail DC price (B)'); ax.step(d.hour, d.ride_price, where='mid', color=AQUA, label='Ride-hailing fleet ToU price (B)')
ax.axhline(pbar, color=A_C, lw=1.5, ls='--', label='Flat retail price (A)'); ax.axhline(14, color=YELLOW, lw=1.5, label='Logistics contract (fixed)')
ax.set_xlim(-.5, 23.5); ax.set_ylim(12, 24); ax.set_xticks(range(0, 24, 2)); ax.set_xlabel('Hour'); ax.set_ylabel('Rs/kWh (ex-GST)')
ax.legend(fontsize=7.3, ncol=3, loc='upper center', bbox_to_anchor=(.5, -.2)); ax.set_title('Dynamic tariffs, Delhi NCR core zone, normal day', loc='left'); save(fig, '13_dynamic_pricing.png')

# 14 architecture diagram
fig, ax = plt.subplots(figsize=(13, 6.4)); ax.set_xlim(0, 117); ax.set_ylim(0, 62); ax.axis('off')
def box(x, y, w, h, title, body, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.3,rounding_size=1.2', fc=fc, ec='#b5b4ae', lw=1))
    ax.text(x + 1, y + h - 1.6, title, fontsize=8.6, fontweight='bold', va='top', color=INK)
    ax.text(x + 1, y + h - 4.6, body, fontsize=7.4, va='top', color=INK2, linespacing=1.35)
def arrow(x1, y1, x2, y2): ax.annotate('', (x2, y2), (x1, y1), arrowprops=dict(arrowstyle='-|>', color=INK2, lw=1.1))
L1, L2, L3, L4 = '#eef4fc', '#eaf7f1', '#fdf3e0', '#fbeef3'
box(1, 40, 24, 20, '1  Data sources', 'Charger OCPP logs, swap-station\nBMS telemetry (SOC, SOH, temp)\nBESS EMS / SCADA, smart meters\nIEX DAM/RTM prices (API)\nIMD + commercial NWP weather\nFleet APIs, calendars, events', L1)
box(28, 40, 24, 20, '2  Ingest + validate', 'Kafka / IoT hub (edge -> cloud)\nSchema + range checks, gap fill\nFeature store (point-in-time)\nData lake (Parquet), 5 y history\nLatency: telemetry 1-5 s', L1)
box(55, 40, 27, 20, '3  Forecasting (every 15-60 min)', 'Demand: global LightGBM quantile\n  (P10/P50/P90 per cluster-zone-hour)\nSolar/wind: NWP + physical model\n  + gradient-boosted residual, quantiles\nIEX price: LEAR/LightGBM quantiles\nConformal calibration of intervals', L2)
box(85, 40, 31, 20, '4  Optimisation engine', 'Day-ahead (D-1, by 09:30):\n  2-stage stochastic LP, S scenarios\n  -> DAM bids, RE-pool schedule, tariffs\nIntraday (hourly MPC, 24 h rolling):\n  MILP -> BESS, pack charging, vans,\n  fleet redirect, RTM bids (1 h ahead)\nSolver: HiGHS / Gurobi, 60 s cap', L2)
box(85, 14, 31, 20, '5  Execution', 'IEX bidding via trader-member\nEMS/SCADA setpoints to BESS + chargers\n  (OCPP 2.0.1 smart charging)\nPack-charging queue per station\nVan routing app, fleet dispatch API\nTariff push to app (± 20 % guard)', L3)
box(55, 14, 27, 20, '6  Edge safety layer', 'Site controller enforces hard limits:\n  grid MW, SOC 10-95 %, thermal\n  5 % priority reserve\nRule-based fallback controller\nruns locally if cloud link fails', L3)
box(28, 14, 24, 20, '7  Monitoring + MLOps', 'KPI dashboard: reliability, losses,\n  EBITDA by cluster, queue, stock-outs\nForecast drift (WAPE, pinball)\nWeekly retrain, champion/challenger\nAudit log of every decision', L4)
box(1, 14, 24, 20, '8  People + governance', 'Control-room operator: override,\n  approve DAM bids above limits\nTrading desk: risk limits\nSecurity: IEC 62443 zones, mTLS,\n  RBAC, CERT-In incident rules', L4)
for a, b in [((25.6, 50), (27.6, 50)), ((52.6, 50), (54.6, 50)), ((82.6, 50), (84.6, 50)), ((100, 39.6), (100, 34.6)), ((84.6, 24), (82.6, 24)), ((54.6, 24), (52.6, 24)), ((27.6, 24), (25.6, 24))]: arrow(*a, *b)
ax.annotate('', (40, 39.6), (40, 34.6), arrowprops=dict(arrowstyle='-|>', color=INK2, lw=1.1, ls='--'))
ax.text(41, 36.4, 'feedback: actuals,\nerrors, retraining', fontsize=7, color=INK2)
ax.text(1, 9, 'Cadence:  telemetry seconds  ·  forecasts 15-60 min  ·  MPC hourly (aligned to 15-min IEX blocks)  ·  DAM bids D-1 10:00-12:00 IST  ·  RTM bids each half-hour session  ·  retrain weekly',
        fontsize=7.6, color=INK)
ax.text(1, 5.5, 'Cloud: forecasting, optimisation, data lake, dashboards.   Edge: site controllers, OCPP, BMS gateways, safety limits, fallback rules.', fontsize=7.6, color=INK2)
ax.set_title('End-to-end decision architecture: one forecast-optimise-execute-learn loop', loc='left', fontsize=11)
save(fig, '14_architecture.png')
print('charts written to', FIG)
