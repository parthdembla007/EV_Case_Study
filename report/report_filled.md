---
title: "From 78% to 90% Reliability: An AI-Driven Energy, Storage and Pricing Decision System"
subtitle: "Strategic recommendation for a national EV charging and battery-swapping operator (Delhi NCR, Mumbai MMR, Bengaluru, Hyderabad, Chennai)"
date: "November 2026"
---

> **Reading guide.** Every number in this report is labelled as one of four kinds. **[Fact]** comes from the case or a cited public source. **[Assumption]** is a hypothetical parameter, justified in Section 3. **[Model output]** is computed by the reproducible Python simulation in `code/` from those assumptions; it is not company data. **[Recommendation]** is our advice. The simulation uses a synthetic, internally consistent dataset because the operator's five-year history was not available to us. All money is in Indian rupees (Rs; 1 crore = Rs 10^7^), excluding GST.

# 1. Executive summary

**The problem.** The operator has plenty of *nameplate* capacity (1,200 MW, 600 MWh) but serves only 78% of demand. Our model shows that the binding limits are not nameplate megawatts. They are four smaller, time-varying limits:

- the sanctioned grid connection, about 276 MW against 1,200 MW of chargers (assumption);
- the share of charger power customers can actually draw at the place and hour they arrive;
- the number of *charged* swap packs at congested stations;
- the clock: a depleted pack needs about three hours before it can be swapped again.

The incumbent rule-based schedule (fixed battery timetable, charge-on-return packs, flat prices, naive forecasts, pro-rata renewable allocation) leaves value on the table in each of these limits at once. That is why surpluses and shortages coexist.

**The solution.** One decision system with a single loop: forecast, optimise, execute, learn.

1. **Probabilistic forecasts.** Gradient-boosted quantile models give P10/P50/P90 demand, renewable output and IEX price for each cluster, zone and hour.
2. **Day-ahead.** A two-stage stochastic programme sets IEX day-ahead bids, the allocation of the shared inter-state renewable pool, and next-day tariffs within the ±20% band.
3. **Intraday.** An hourly rolling-horizon MILP (model predictive control) dispatches the stationary batteries, swap-pack charging queues, pack-rebalancing vans, fleet redirection and one-hour-ahead Real-Time Market purchases. It never touches the 5% priority reserve.
4. **Execution and safety.** An edge safety layer and a rule-based fallback controller execute and guard the decisions.

**The result [Model output].** On the synthetic year (65% normal days, 10% regional-imbalance days, 10% renewable-shortfall/price-spike days, 15% heatwave days), the framework moves the portfolio as follows:

| KPI (annual, portfolio) | Rule-based today (A) | Proposed framework (B) | Executive target |
|---|---:|---:|---:|
| Service reliability | 78.7% | **89.8%** | 89.7% |
| Energy losses (% of imported energy) | 10.1% | **8.3%** | 8.0% |
| Revenue (Rs crore/yr) | 2,814 | 3,194 | – |
| EBITDA (Rs crore/yr, B net of Rs 24.5 Cr framework run cost) | 418 | **787** | – |
| EBITDA margin, portfolio | 14.9% | **24.6%** | >12% |
| EBITDA margin, weakest cluster | 6.6% (Mumbai MMR) | **17.5%** (Mumbai MMR) | >12% in every cluster |

The one-time investment is Rs 77 crore. Payback is 8 months and the five-year NPV at a 12% discount rate is Rs 935 crore. Most of the value comes from **serving demand that is lost today**, not from cheaper electricity: revenue rises by Rs 380 crore a year. Losses fall by 1.8 percentage points to 8.3%, just short of the 8% target. That reduction comes from energy-management controls (power-module staging and sleep, just-in-time pack charging, value-based battery cycling), not from forecasting.

**Recommendations [Recommendation].**

1. Approve a 5-phase, 18-month programme. Start with data integration and a Delhi NCR core-zone pilot, the cluster with the lowest reliability.
2. Treat the swap-pack fleet as **inventory**, not storage. Schedule its charging against forecast swaps, and fund a small van-rebalancing operation.
3. Re-paper fleet contracts as time-of-use agreements with redirection incentives. Keep emergency and priority customers outside dynamic pricing.
4. Build an in-house IEX trading desk (via a trader-member) with risk limits. Run day-ahead bidding from the stochastic optimiser.
5. Plan capacity where software stops. Delhi NCR reaches only 85.2% reliability even with the framework, because its core-zone chargers and packs are physically short at the peak. Use the model's shadow prices to size a grid-connection and pack-inventory expansion there. Mumbai has the thinnest margin (highest fixed cost and DISCOM tariff) but stays above 12% in every case we tested.

**The targets are achievable, not guaranteed.** In the base case the portfolio reaches the reliability and margin targets and comes within 0.3 percentage points of the loss target. Delhi NCR stays below 90% reliability (Section 12). They depend on three things the operator controls: telemetry quality, fleet participation in redirection and time-of-use offers, and holding forecast error near the levels assumed (about 6–8% WAPE day-ahead).

# 2. Business problem and root-cause analysis

## 2.1 The problem in plain words

Customers arrive at specific stations at specific hours. They need either a charger that can deliver power *now* or a *charged* pack on the shelf *now*. The operator buys electricity a day ahead, in 15-minute blocks, before it knows exactly who will come. Renewable energy arrives mostly at noon, while demand peaks in the evening. Batteries can move energy in time, but each kWh stored loses 10–12% and wears the battery. Today these decisions are made by separate teams using fixed rules. The result is unused renewable energy and idle batteries in one place, and queues and empty swap shelves in another.

## 2.2 Units: what each capacity number actually means

| Quantity | Unit | Meaning in this case | Why it does not guarantee service |
|---|---|---|---|
| Charging capacity | MW (power) | Maximum rate chargers can deliver: 720 MW DC + 480 MW swap racks **[Fact]** | Usable only if the grid connection, the vehicle's acceptance rate and the customer's location all line up |
| Grid connection | MW | Sanctioned contract demand at sites; we assume 23% of nameplate (276 MW) **[Assumption]** | Caps the power that can be drawn at any instant |
| Stationary storage | MWh (energy) and MW (power) | 300 MWh of BESS, 0.5C, i.e. 150 MW **[Assumption on split]** | Must be charged *before* the peak; usable energy shrinks with SOC limits and state of health |
| Swap-pack inventory | MWh, or a count of packs | 300 MWh of packs (about 100,000 packs of 3 kWh) **[Assumption on split]** | Only *charged, compatible packs at the right station* are useful. A depleted pack needs about 3 h before re-use |
| Energy availability | MWh per hour | Renewable + DISCOM + IEX energy deliverable in an hour | Arrives at the wrong hour (solar) or is limited by contracts and open-access rules |
| Customer demand | MWh per hour, or swaps per hour | Energy customers want in that hour and zone | Uncertain, weather- and event-driven, and spatially concentrated |

A 1-hour time step makes MW × 1 h = MWh. Every power limit in the model is therefore applied as "MWh per hour", and every energy stock (SOC, ready packs) is carried between hours by a balance equation.

## 2.3 Why surpluses and shortages happen at the same time

1. **Time mismatch.** Renewable output peaks at 11:00–14:00. Retail and ride-hailing demand peaks at 18:00–23:00 (Figure 2). Without timely storage, noon energy is curtailed or banked at a discount while evenings run short.
2. **Space mismatch inside each metro.** We assume about 60% of demand arises in dense core zones that host only 36% of assets. Outer-zone chargers and packs sit idle while core stations queue.
3. **Space mismatch across metros.** Chennai can be windy and quiet while Delhi is in a heatwave. The *physical* grid does not let the company ship energy from Chennai to Delhi. Part of its *contracted* inter-state renewable portfolio can, however, be scheduled to different drawal states day-ahead. A pro-rata rule wastes that flexibility.
4. **Inventory dynamics.** Charge-on-return keeps a large stock of charged packs at quiet hours (paying standby losses) and too few at the 09:00 and 19:00 swap peaks.
5. **Decision silos.** Procurement buys from a naive forecast. Storage follows a timetable. Pricing is flat. None of them sees the others' constraints.

## 2.4 Why reliability is 78% [Model output, calibrated]

We calibrated the synthetic baseline to reproduce the two reported KPIs: 78% reliability and 10% losses. The calibrated baseline gives 78.7% and 10.1%. In the calibrated model, unserved energy splits as follows:

| Scenario (day type, weight) | Demand (MWh/day) | Unserved DC (MWh/day) | Unserved swaps (MWh/day) | Reliability |
|---|---:|---:|---:|---:|
| S1 Normal operations (65%) | 4,539 | 674 | 207 | 80.6% |
| S2 Regional imbalance (10%) | 4,723 | 758 | 255 | 78.5% |
| S3 RE shortfall + IEX spike (10%) | 4,675 | 733 | 225 | 79.5% |
| S4 Summer heatwave (15%) | 5,383 | 1,204 | 335 | 71.4% |

Heatwave days are the worst. Chargers derate, demand rises 10–30%, and the core-zone limits bind for many more hours.

## 2.5 Trade-offs the system must manage

- **Reliability vs profit.** Serving the last few percent of peak demand needs pre-charged batteries, pack inventory and occasionally RTM purchases near the Rs 10/kWh ceiling.
- **Reliability vs battery health.** Deeper and more frequent cycling raises availability but consumes cycle life, worth Rs 1.4 per kWh discharged for the BESS and Rs 4.5 per kWh charged for packs (Section 3).
- **Reliability vs losses.** Every kWh routed through a battery loses about 12% round-trip. Holding many charged packs costs standby energy.
- **Profit vs fairness.** Dynamic prices shift flexible demand, but must stay within ±20%, respect fleet contracts, and must never apply to priority or emergency users.

## 2.6 Why rule-based scheduling is insufficient

Fixed rules ("charge 00:00–06:00, discharge 18:00–23:00"; "charge every returned pack immediately") cannot react to three things. They cannot react to a forecast heatwave or a renewable shortfall. They cannot weigh a Rs 3/kWh arbitrage spread against Rs 1.4/kWh of degradation and 12% losses. And they cannot coordinate limits that are shared: the same grid megawatt cannot charge the BESS, charge packs and serve a car in the same hour. These are coupled, multi-period decisions under uncertainty, which is exactly what stochastic optimisation is built for.

## 2.7 Information missing from the case

The case does not give the following. We assumed each of them (Section 3):

- the cluster-wise split of capacity and demand;
- the grid connection limits;
- the split of the 600 MWh between stationary and pack storage;
- efficiencies;
- DISCOM tariffs and open-access charges;
- pack size, charging time and compatibility;
- customer elasticities;
- fixed operating costs;
- the current EBITDA margin;
- how much of the load sits at sites eligible for open access.

# 3. Assumptions and baseline

## 3.1 Assumption register

| ID | Item | Value used | Type | Justification |
|---|---|---|---|---|
| A1 | Cluster nameplate (MW) | Delhi 336, Mumbai 288, Bengaluru 240, Hyderabad 180, Chennai 156 (=1,200) | Assumption (total = fact) | Proportional to metro EV fleet size and charging density |
| A2 | Demand (MWh/day, normal day) | 1,300 / 1,105 / 910 / 650 / 585 (total 4,550) | Assumption | About 16% utilisation of nameplate: congested cores, idle outskirts |
| A3 | DC vs swap (MW) | 185+151 / 187+101 / 132+108 / 117+63 / 99+57 (720 + 480) | Total split = fact | Delhi/Bengaluru swap-heavy (e-3W, delivery 2W) |
| A4 | Stationary vs pack storage | 300 MWh BESS (0.5C, 150 MW) + 300 MWh packs (about 100k packs of 3 kWh) | Assumption (total = fact) | Even split; BESS at 0.5C is typical for C&I LFP |
| A5 | Efficiencies | DC charger 0.940 (0.955 with module staging); pack charging 0.94; BESS 0.94 each way (RTE 0.88); distribution 1.0% | Assumption | LFP RTE 85-90% [18]; SiC charger peak about 96% |
| A6 | Self-discharge and standby | BESS 0.02%/h; ready packs 0.01%/h + cabinet standby 0.8%/h of stock; BESS HVAC 0.2%/h; idle chargers 0.3% of DC MW | Assumption | Calibrated so baseline losses ≈ 10% (case fact) |
| A7 | Degradation cost | BESS Rs 1.4/kWh discharged; packs Rs 4.5/kWh charged; ≤1.5 cycles/day; SOH 0.90-0.95 | Assumption | Pack cost / (cycle life × DoD): Rs 9k/kWh ÷ (6,000 × 0.85); Rs 12k ÷ 2,500 [15][16] |
| A8 | Renewable availability | Intra-state solar+wind per cluster + inter-state pool (70 MWp solar, 50 MW wind, scaled); about 33% of imports in baseline | Calibrated to fact (35%) | Chennai wind CF 38%, Mumbai cloudier, Delhi solar only |
| A9 | RE contract price | Rs 4.3/kWh landed, take-or-pay; banking credit Rs 3.8/kWh up to 3% of daily demand | Assumption | Recent C&I RE PPA levels plus OA charges; GEOA banking [4] |
| A10 | DISCOM tariff | ACoS Rs 6.8 / 8.6 / 7.4 / 7.2 / 6.9; 0.7× in solar hours (09-16), 1.3× otherwise | Assumption (structure from [5]) | MoP 2024 EV-tariff guideline structure |
| A11 | IEX DAM prices | Stylised hourly shape: Rs 2.5 (noon) to Rs 9.9 (20:00), mean ≈ Rs 5.4; RTM = DAM ± 10%; ceiling Rs 10 | Assumption (ceiling = fact [3]) | Shape follows the solar "duck" pattern; not historical data |
| A12 | Open-access adder | Rs 1.6/kWh (transmission, wheeling, CSS, fees); OA-eligible load share 60% / 55% / 60% / 65% / 70% | Assumption | Only HT hubs ≥1 MW (≥100 kW green) can buy OA [4] |
| A13 | Grid connection limits | 276 MW total (23% of nameplate) | Assumption | Sanctioned load is usually far below charger nameplate |
| A14 | Charger power limits | Deliverable DC = 23% of nameplate (availability × vehicle acceptance × location); 12% heat derate | Calibrated | Calibrated to 78% reliability (case fact) |
| A15 | SOC constraints | 10–95% of usable (nameplate × SOH); day-end SOC ≥ day-start | Assumption | LFP warranty windows |
| A16 | Pack dynamics | Recharge time 3 h; van capacity 5% of cluster packs per h; core zone hosts 36% of assets but 58%-64% of demand | Assumption | Typical 2W/3W pack charge at 0.33C |
| A17 | Curtailment | RE surplus beyond use and banking cap is curtailed at zero value | Assumption | No resale without seller registration |
| A18 | Forecast accuracy | Naive: ±6% day level, ±10% hourly, blind to scenario. ML: ±3%/±5% (demand), ±8% (RE), ±7% (price) | Assumption | Consistent with published day-ahead load-forecast accuracy [8] |
| A19 | Price elasticity | Timing ε: retail −0.6, ride-hailing −0.8, logistics 0; volume ε: −0.3/−0.2/0 | Assumption | Fleets shift readily; retail less; tested at half |
| A20 | Customer segments (DC share) | Retail / ride-hailing / logistics, e.g. Delhi 30/50/20, Chennai 25/30/45; swaps = 20-32% of energy | Assumption | Delhi ride-hailing heavy; Chennai logistics/port |
| A21 | Revenue | Retail Rs 19.0, ride-hailing Rs 15.5, logistics Rs 14.0, swap Rs 32.0/kWh (≈ Rs 96 per 3 kWh swap); local index 0.97-1.05 | Assumption | Ex-GST public DC tariffs of about Rs 18-24/kWh incl. GST |
| A22 | Operating cost | Fixed Rs 74 / 90 / 76 / 74 / 76 lakh/MW/yr; variable O&M Rs 0.5/kWh; van Rs 0.8/kWh moved; redirect incentive Rs 1.5/kWh | Assumption | Site lease, staff, network, insurance, maintenance |
| A23 | Unserved penalty (goodwill/SLA) | Rs 6 / 10 / 14 / 10 per kWh (retail / ride / logistics / swap) | Assumption | Decision weight only; not in EBITDA |
| A24 | Day-type weights | Normal 65%, regional imbalance 10%, RE shortfall + price spike 10%, heatwave 15% | Assumption | About 55 heatwave days per year in north/central India |

## 3.2 Differentiated cluster profiles [Assumption]

| Cluster | Demand character | Renewable character | Scheduling implication |
|---|---|---|---|
| Delhi NCR | Largest; ride-hailing and e-3W swap heavy; strong evening and heat-driven peaks | Solar only (no local wind); haze | Most congested core; biggest gain from pre-charging packs and evening BESS discharge |
| Mumbai MMR | Dense, late evening peak (shifted about 1 h); highest DISCOM cost (Rs 8.6/kWh ACoS) | Cloudier/humid solar; some wind | Highest fixed cost; most sensitive to energy price |
| Bengaluru | Retail + delivery 2W; sharp morning peak | Moderate solar + wind | Morning swap peak needs overnight pack charging |
| Hyderabad | Mid-size, mixed fleets | Best solar irradiance | Midday charging and BESS charging on solar |
| Chennai | Logistics- and port-heavy; overnight depot charging | Strong wind (Tamil Nadu), evening wind peak | Wind surplus some days; donor of inter-state pool energy |

![Figure 1 – Installed capacity by cluster [Assumption]. Bars show nameplate MW split into DC fast charging and swap racks. The black tick marks the sanctioned grid connection, the real power ceiling (23% of nameplate). Text gives the energy storage (MWh) behind each cluster.](D:/EV_Case_Study/outputs/charts/01_capacity_by_cluster.png){width=100%}

![Figure 2 – Hourly customer demand vs renewable availability on a normal day (synthetic truth, seed 1). Renewables peak at noon while demand peaks in the evening; the dashed line is the grid connection. This mismatch drives both curtailment and evening shortages.](D:/EV_Case_Study/outputs/charts/02_demand_vs_renewables.png){width=100%}

# 4. Proposed AI-powered solution: one decision system

The framework is a single closed loop rather than nine separate AI tools. The shared state (prices, SOC, pack stock, reserve, forecasts) flows through five layers:

1. **Sense.** Telemetry from chargers (OCPP), packs (BMS), BESS (EMS/SCADA), meters, weather and IEX.
2. **Predict.** Probabilistic forecasts of demand, renewable output and price, each with quantiles.
3. **Decide day-ahead (D-1).** A stochastic programme fixes the commitments that must be made before the market deadline: DAM bids, inter-state RE schedule, and next-day tariffs.
4. **Decide intraday (hourly).** MPC re-optimises everything still flexible: battery, packs, vans, redirection, RTM bids and DISCOM top-up. It respects the commitments from step 3.
5. **Execute, guard and learn.** Hard limits sit at the edge, a fallback controller takes over if needed, operators can override, and forecast and decision errors feed weekly retraining.

Each of the nine required functions maps to a component of this loop (Table 4.1). They share one data model and one objective: profit with explicit penalties for unserved demand.

| Required function | Where it lives | Decision cadence |
|---|---|---|
| Demand forecasting | Predict: quantile LightGBM (global model) | 15–60 min refresh |
| Renewable forecasting | Predict: NWP + physical model + ML residual | 15–60 min |
| Cluster energy allocation | Day-ahead stochastic programme (inter-state RE pool, DAM) | D-1, revisable within IEGC limits |
| Stationary battery optimisation | Intraday MPC | Hourly (15-min execution) |
| Swap inventory optimisation | Intraday MPC (pack-charging queue, vans) | Hourly |
| Procurement and IEX bidding | Day-ahead programme (DAM) + MPC (RTM) | D-1 10:00–12:00; each RTM session |
| Dynamic pricing | Day-ahead pricing rule using shadow prices | D-1, published with ±20% guard |
| Degradation management | Degradation cost + cycle cap in both models; SOH-adjusted capacity | Daily SOH update |
| Real-time operations | Edge controller + fallback rules + operator override | Seconds |

# 5. Forecasting architecture

## 5.1 Demand forecasting

**Model choice [Recommendation].** We recommend a **global gradient-boosted quantile model** (LightGBM [12]): one model trained on all cluster × zone × segment series, with cluster, zone and segment as features. Three reasons:

1. Tabular, calendar- and weather-driven hourly demand with many related series is where gradient boosting is consistently strong and cheap to retrain.
2. A global model shares patterns across clusters. That will matter when the network grows from 5 to 20+ clusters, because new clusters have short histories.
3. Quantile objectives give P10/P50/P90 directly.

The Temporal Fusion Transformer [10] is the challenger. It natively handles known-future inputs such as weather forecasts and holidays, and gives interpretable attention, but costs more to train and monitor. N-BEATS [11] and LSTM are useful as ensemble members for the smooth base load, but on their own they do not use exogenous weather as naturally. **Do not start with deep learning.** Promote the TFT only if it beats LightGBM on pinball loss in the champion/challenger test.

**Features.**

- Calendar: hour of day, day of week, holiday and festival flags, school terms.
- History: lagged sessions and swaps (t-24 h, t-168 h), rolling 7-day means.
- Weather: temperature, humidity, rainfall, heat index — always *as forecast at issue time*.
- Market: the IEX price forecast; for the pricing layer, the published own tariff.
- Fleets: segment mix, fleet utilisation feeds from ride-hailing and logistics APIs.
- Events and stations: event calendar (matches, concerts), station availability.

**Uncertainty.** Quantile regression gives P10/P50/P90 [7]. Conformalised quantile regression [13] then recalibrates the intervals on a rolling window, so that about 80% of outcomes fall inside P10–P90. The optimiser does not use the quantiles directly. It draws correlated scenarios, with a common day-level shock plus hourly noise, so that tail days (a whole-cluster heatwave) are represented.

**Retraining and leakage control.**

- Retrain weekly, with a nightly refresh of the final trees, and immediately after structural breaks such as a new station or a tariff change.
- Use strict time-based splits with rolling-origin back-tests.
- Build features from data available at forecast issue time only, with point-in-time joins in the feature store. In particular, use *forecast* weather and never observed weather for day-ahead models.
- Never use target-derived station status from the same hour.
- Compute scalers and encoders on the training window only.

**Evaluation.** MAE and RMSE (MWh per hour); WAPE = Σ|y − ŷ| / Σy; pinball loss at P10/P50/P90; P10–P90 coverage. All are reported by cluster, segment and lead time. The synthetic forecasters used in the simulation have the following accuracy. These are **assumed error levels** placed on the synthetic truth, not results of a trained model:

| Forecaster (cluster-hour demand) | MAE (MWh/h) | RMSE (MWh/h) | WAPE | Pinball P10 | Pinball P50 | Pinball P90 | P10–P90 coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| ML quantile (B) | 1.17 | 1.51 | 2.9% | 0.26 | 0.58 | 0.22 | 82.5% |
| Seasonal-naive (A) | 5.15 | 6.66 | 12.6% | 0.91 | 2.54 | 1.72 | 59.1% |

## 5.2 Renewable forecasting

- **Solar.** NWP irradiance (GHI, cloud cover) is converted to plane-of-array irradiance and then to AC output with a physical model (pvlib [14]). This includes the temperature derate of about -0.35 to -0.45 %/°C cell temperature for crystalline silicon, soiling and inverter clipping. A gradient-boosted model then learns the residual from satellite cloud-motion nowcasts (0–6 h) and site history.
- **Wind.** Hub-height wind speed from NWP is passed through the measured turbine power curve: cut-in about 3 m/s, rated about 11–13 m/s, cut-out about 25 m/s. Output is roughly cubic between cut-in and rated, so errors in wind speed are amplified. A quantile model on top captures this non-linearity and ramp risk.
- **Uncertainty.** Solar forecast error is dominated by cloud timing; wind error grows with lead time. Both are reported as quantiles, and the scenario generator correlates them with price (low renewables, high price, as in Scenario 3).

## 5.3 IEX price forecasting

Features: lagged DAM prices for the same block (D-1, D-7), the All-India demand forecast, the renewable forecast (the "duck" shape), hydro and coal availability, temperature, day type and season, and the market-clearing volume. Models: a LASSO-regularised autoregressive model (LEAR) as a robust baseline plus a LightGBM quantile model. This combination is consistent with the open benchmark of Lago et al. [9].

**Limitations.** Prices are bounded by the CERC-directed ceiling of Rs 10/kWh in DAM and RTM [3]. This clips spikes, so the *distribution* matters more than the point forecast. Scarcity events (heatwave plus low wind) are rare and regime-like, and congestion causes market splitting between bid areas. Forecasts are therefore used to rank hours and size hedges, never as certainties. The optimiser's scenario set always contains price-at-cap evenings.

## 5.4 Model selection summary

| Component | Selected model | Why not something more complex |
|---|---|---|
| Demand (all segments) | Global LightGBM quantile + conformal calibration | Strong on tabular exogenous data, fast retrain, explainable (SHAP). TFT only as challenger |
| Swap demand by station | Same global model + hierarchical reconciliation to cluster totals | Sparse station series benefit from pooling |
| Solar / wind | Physical model + ML residual | Physics extrapolates to new plants; pure ML cannot |
| IEX price | LEAR + LightGBM quantile | Deep nets add little over LEAR on DAM data [9] |
| Battery SOH | Semi-empirical cycle/calendar model calibrated to BMS data [15][16] | Interpretability matters for warranty claims |

# 6. Quantitative optimisation model

## 6.1 Structure

The model is a **two-stage stochastic programme** solved day-ahead, followed by a **rolling-horizon MILP** solved every hour:

- **Stage 1 (here-and-now, D-1).** IEX DAM quantities $q_{n,t}$ and the inter-state RE-pool split $\alpha_{c,t}$. Tariffs $p_{n,k,t}$ are fixed by the pricing rule of Section 8 *before* the stochastic programme. This keeps the model linear instead of bilinear in price × demand.
- **Stage 2 (recourse, per scenario s).** Everything else: RTM and DISCOM energy, battery charge and discharge, pack charging, swaps, van moves, redirection, served and unserved demand, reserve.
- **Intraday MPC.** At each hour $t_0$ the stage-2 problem is re-solved for $t_0,\dots,24$ on the latest forecast. Stage-1 decisions are fixed, the RTM quantity for hour $t_0$ is the one bought in the previous hour (RTM gate closure is about one hour before delivery), and binaries enforce non-simultaneous charge and discharge. Only hour $t_0$'s decisions are executed.

## 6.2 Sets and indices

| Symbol | Meaning |
|---|---|
| $c \in \mathcal{C}$ | Clusters {Delhi NCR, Mumbai MMR, Bengaluru, Hyderabad, Chennai} |
| $z \in \{\text{core}, \text{outer}\}$; $n=(c,z) \in \mathcal{N}$ | Zones within a cluster; a node is one cluster-zone |
| $t \in \mathcal{T}=\{1,\dots,24\}$, $\Delta t = 1$ h | Hourly periods (each executed as four 15-min IEX blocks) |
| $k \in \mathcal{K}$ | DC customer segments {retail, ride-hailing, logistics} |
| $r \in \{\text{solar}, \text{wind}\}$ | Renewable sources (aggregated into $G^{RE}$ below) |
| $a \in \{\text{BESS}, \text{packs}\}$ | Storage asset types, modelled separately |
| $f \in \mathcal{F}$ | Pack families (2W 48/60 V, 3W 51 V, ...); the simulation uses one family per zone |
| $s \in \mathcal{S}$, probability $\pi_s$ | Joint weather-demand-price scenarios |

## 6.3 Parameters

| Symbol | Unit | Meaning |
|---|---|---|
| $D_{n,k,t,s}$, $D^{SW}_{n,t,s}$ | MWh/h | DC demand by segment; swap demand (energy-equivalent) |
| $G^{RE}_{n,t,s}$, $P^{pool}_{t,s}$ | MWh/h | Intra-state RE delivered to node; inter-state RE pool output |
| $\lambda^{DA}_{t,s}$, $\lambda^{RT}_{t,s}$ | Rs/kWh | IEX DAM and RTM prices (≤ 10) |
| $\tau^{D}_{c,t}$ | Rs/kWh | DISCOM ToD tariff |
| $c^{OA}$, $c^{RE}$, $v^{bank}$ | Rs/kWh | Open-access charges; RE contract price; banking credit |
| $\bar G_n$ | MW | Sanctioned grid connection |
| $\theta_n$ | – | Share of load at open-access-eligible (HT) sites |
| $\bar C^{DC}_n=\kappa P^{DC}_n$ | MW | Deliverable DC power (nameplate × access factor) |
| $\bar E_n$, $\bar P^{B}_n$ | MWh, MW | Usable BESS energy (nameplate × SOH) and power |
| $\eta^{c},\eta^{d},\eta^{DC},\eta^{PK},\ell$ | – | Charge/discharge, DC charger and pack-charging efficiency, distribution loss |
| $\sigma^{B},\sigma^{P}$ | 1/h | Self-discharge of BESS and of ready packs |
| $a^{R}, a^{B}, a^{SB}_n$ | –, –, MW | Standby: ready-pack cabinets (fraction of stock per h), BESS HVAC, idle chargers |
| $\bar S_n$, $\tau^{ch}$ | MWh, h | Pack inventory; time to recharge a depleted pack |
| $\bar M_c$ | MWh/h | Van capacity for moving charged packs |
| $\rho_k$, $\delta$ | –, Rs/kWh | Max share of core demand redirectable; redirection incentive |
| $\beta^{res}=5\%$ | – | Priority reserve share **[Fact]** |
| $c^{degB}, c^{degP}, N^{cyc}$ | Rs/kWh, cycles/day | Degradation costs; daily cycle cap |
| $\pi^U_k,\pi^U_{SW}$ | Rs/kWh | Goodwill/SLA penalty for unserved demand, on top of lost revenue |

## 6.4 Decision variables (all ≥ 0 unless stated)

| Variable | Unit | Meaning |
|---|---|---|
| $q_{n,t}$ | MWh | IEX DAM purchase (stage 1) |
| $\alpha_{c,t}\in[0,\bar\alpha]$ | – | Share of inter-state RE pool scheduled to cluster $c$ (stage 1) |
| $p_{n,k,t}$ | Rs/kWh | Retail and fleet tariffs (set by the pricing module, then fixed) |
| $u_{n,t,s}$ | MWh | Under-drawal of the DAM schedule (settled as deviation) |
| $x^{RT}_{n,t,s}$, $x^{D}_{n,t,s}$ | MWh | RTM purchase; DISCOM supply |
| $g_{n,t,s}$, $b_{n,t,s}$, $k^{cur}_{n,t,s}$ | MWh | RE used; RE banked; RE curtailed |
| $e^{ch}_{n,t,s}$, $e^{dis}_{n,t,s}$, $y_{n,t,s}\in\{0,1\}$ | MWh, – | BESS charge/discharge (AC side); charge-mode binary |
| $E_{n,t,s}$ | MWh | BESS state of charge |
| $w_{n,t,s}$, $z_{n,t,s}$ | MWh | Energy delivered by swaps; energy charged into depleted packs |
| $R_{n,t,s}$, $Q_{n,t,s}$ | MWh | Ready (charged) and depleted pack stock |
| $m^{oc}_{c,t,s}$, $m^{co}_{c,t,s}$ | MWh | Charged packs moved outer→core and core→outer |
| $\phi_{c,k,t,s}$ | MWh | DC demand redirected from core to outer zone (fleet routing) |
| $d_{n,k,t,s}$ | MWh | DC energy delivered to segment $k$ |
| $r^{G}_{n,t,s}$, $r^{B}_{n,t,s}$ | MW | Reserve held as grid headroom / BESS headroom |
| $I_{n,t,s}$ | MWh | Total import $= g + q - u + x^{RT} + x^{D}$ (defined) |

## 6.5 Objective function

$$\max\ \sum_s \pi_s \Big[\ \sum_{n,t}\Big( \underbrace{\textstyle\sum_k p_{n,k,t}d_{n,k,t,s} + p^{SW}_n w_{n,t,s}}_{\text{revenue}} - \underbrace{(\lambda^{DA}_{t,s}+c^{OA})\,q_{n,t}}_{\text{DAM cost}} + \underbrace{\gamma\lambda^{DA}_{t,s}u_{n,t,s}}_{\text{under-drawal credit}}$$
$$-\underbrace{(\lambda^{RT}_{t,s}+c^{OA})\,x^{RT}_{n,t,s}}_{\text{RTM cost}} - \underbrace{\tau^{D}_{c,t}\,x^{D}_{n,t,s}}_{\text{DISCOM cost}} + \underbrace{v^{bank}b_{n,t,s}}_{\text{banking credit}} - \underbrace{c^{degB}e^{dis}_{n,t,s} - c^{degP}z_{n,t,s}}_{\text{degradation}}$$
$$-\underbrace{\textstyle\sum_k \pi^U_k (D_{n,k,t,s}-d_{n,k,t,s}) - \pi^U_{SW}(D^{SW}_{n,t,s}-w_{n,t,s})}_{\text{unserved-demand penalty}} - \underbrace{c^{vom}\big(\textstyle\sum_k d_{n,k,t,s}+w_{n,t,s}\big)}_{\text{variable O\&M}}\Big)$$
$$-\sum_{c,t}\underbrace{\big(c^{van}(m^{oc}_{c,t,s}+m^{co}_{c,t,s})+\delta\textstyle\sum_k\phi_{c,k,t,s}\big)}_{\text{rebalancing}} - \underbrace{\pi^{term}\textstyle\sum_n(\eta^d\xi^E_{n,s}+\xi^R_{n,s})}_{\text{end-of-day shortfall}}\ \Big]$$

**What each term means economically.**

- *Revenue* is earned only on delivered energy, so lost sessions are automatically lost revenue.
- The *DAM cost* is the clearing price plus the open-access adders: transmission, wheeling, cross-subsidy and additional surcharge, and SLDC fees.
- *Under-drawal* earns only a fraction $\gamma$ of the price, a deviation-settlement-type outcome. This makes over-buying costly.
- *RTM* is the flexible but riskier hour-ahead top-up.
- *DISCOM* energy is the residual supply at time-of-day tariffs. Small non-HT sites have no alternative.
- *Banking* credits surplus green energy within the allowances of the Green Energy Open Access rules [4].
- *Degradation* is the marginal wear cost of throughput.
- The *penalty* represents SLA and churn costs beyond lost revenue. It is highest for logistics contracts.
- *Rebalancing costs* are van logistics and the redirection discount.
- The *terminal penalty* keeps the plan steady-state: packs and BESS end the day at least as full as they started.

Renewable contract cost $c^{RE}$ is take-or-pay on available output. It is a constant in the optimisation but is counted in the P&L.

*Losses* need no separate term. Every lost kWh must be bought, so the energy-balance constraint prices losses at the marginal energy cost. The 8% KPI is checked ex post. Section 11 discusses adding it as an explicit constraint.

## 6.6 Constraints (for all $n=(c,z)$, $t$, $s$; $\Delta t$ = 1 h)

**(1) Node energy balance (site bus, after distribution losses).** One kWh can only be used once:
$$
(1-\ell)\,I_{n,t,s} + e^{dis}_{n,t,s} = \sum_k \frac{d_{n,k,t,s}}{\eta^{DC}} + \frac{z_{n,t,s}}{\eta^{PK}} + e^{ch}_{n,t,s} + a^{R}R_{n,t,s} + a^{B}\bar E_n + a^{SB}_n\Big(1-\zeta+\zeta\frac{\sum_k d_{n,k,t,s}}{\bar C^{DC}_n}\Big)
$$
Here $\zeta$ is the share of idle-charger standby removed by module sleep: 0 under rules, 0.9 under the EMS.

**(2) Renewable availability, banking and curtailment.**
$$g_{n,t,s}+b_{n,t,s}+k^{cur}_{n,t,s}=G^{RE}_{n,t,s}+\alpha_{c,t}\,\omega_n P^{pool}_{t,s},\qquad \sum_t b_{n,t,s}\le \bar B_n,\qquad \sum_c\alpha_{c,t}=1,\ \alpha_{c,t}\le\bar\alpha$$
$\omega_n$ is the node's share of the cluster's open-access-eligible sites. $\bar\alpha=0.45$ reflects corridor and scheduling limits on the inter-state pool.

**(3) Grid import limit and open-access eligibility.**
$$I_{n,t,s}+r^{G}_{n,t,s}\le\bar G_n\cdot\Delta t,\qquad x^{D}_{n,t,s}\ge(1-\theta_n)\,I_{n,t,s},\qquad u_{n,t,s}\le q_{n,t}$$

**(4)–(5) BESS SOC dynamics with efficiencies.**
$$E_{n,t+1,s}=E_{n,t,s}(1-\sigma^{B})+\eta^{c}e^{ch}_{n,t,s}-e^{dis}_{n,t,s}/\eta^{d}$$

**(6) Power limits and (18) non-simultaneity.**
$$e^{ch}_{n,t,s}\le\bar P^{B}_n\,y_{n,t,s},\qquad e^{dis}_{n,t,s}\le\bar P^{B}_n(1-y_{n,t,s})$$
These binaries are used in the intraday MILP. The day-ahead LP relaxes them, and the dispatch actually executed always comes from the MILP.

**(7) SOC window.** $\ \underline{s}\,\bar E_n\le E_{n,t,s}\le\overline{s}\,\bar E_n$, with $\underline s=10\%$, $\overline s=95\%$, and $\bar E_n$ = nameplate × SOH.

**(8) Degradation.** $\ \sum_t e^{dis}_{n,t,s}\le N^{cyc}(\overline s-\underline s)\bar E_n$ ($N^{cyc}$ = 1.5 cycles/day), plus the degradation cost terms in the objective. Temperature effects enter as lower $\eta$ and higher $\sigma$ on heatwave days.

**(9) Swap-pack inventory balance.** Packs are inventory, not dispatchable storage. Energy can only leave a pack through a swap to a customer:
$$R_{n,t+1,s}=R_{n,t,s}(1-\sigma^{P})-w_{n,t,s}+z_{n,t,s}\pm(m^{oc}_{c,t,s}-m^{co}_{c,t,s})$$
$$Q_{n,t+1,s}=Q_{n,t,s}+\sigma^{P}R_{n,t,s}+w_{n,t,s}-z_{n,t,s}$$
Here $R+Q=\bar S_n$ (conserved). The sign is + for the core node and − for the outer node.

**(10) Charging time.** $\ z_{n,t,s}\le Q_{n,t,s}/\tau^{ch}$ and $z_{n,t,s}\le P^{SW}_n\Delta t$. A depleted pack takes $\tau^{ch}$ ≈ 3 h, so at most a third of the depleted stock can be refilled per hour. A pack swapped in hour $t$ becomes available for charging from $t+1$.

**(11) Availability and compatibility.** $\ w_{n,t,s}\le D^{SW}_{n,t,s}$ and $w_{n,t,s}\le R_{n,t,s}-\beta^{res}\bar S_n$. Only packs that are charged at the start of the hour can be swapped. With several pack families the same constraints are written per family $f$, and demand of family $f$ can only be met by $R_{n,f,t}$.

**(12) Demand fulfilment and fleet redirection.**
$$d_{n,k,t,s}\le D_{n,k,t,s}-\phi_{c,k,t,s}\ (\text{core}),\qquad d_{n,k,t,s}\le D_{n,k,t,s}+\phi_{c,k,t,s}\ (\text{outer}),\qquad \phi_{c,k,t,s}\le\rho_k D_{(c,\text{core}),k,t,s}$$
Here $\rho$ = 15% for retail (navigation suggestions), 35% for ride-hailing (dispatch API), and 0 for logistics (depot-based).

**(13) 5% reserve, with no double-counting.**
$$\sum_k d_{n,k,t,s}\le(1-\beta^{res})\bar C^{DC}_n,\qquad R_{n,t+1,s}\ge\beta^{res}\bar S_n$$
$$r^{G}_{n,t,s}+r^{B}_{n,t,s}\ge\beta^{res}\bar C^{DC}_n/\eta^{DC},\qquad r^{B}_{n,t,s}+e^{dis}_{n,t,s}\le\bar P^{B}_n,\qquad \eta^{d}\big(E_{n,t+1,s}-\underline s\bar E_n\big)\ge r^{B}_{n,t,s}\Delta t$$
The reserve MW must be backed by grid headroom not used for imports (constraint 3) or by battery power and energy *left after* that hour's dispatch. The same energy therefore cannot be counted for customers, storage charging and reserve at once.

**(14) Charger capacity.** Covered by the first inequality of (13). $\bar C^{DC}_n$ is derated by 12% on heatwave days.

**(15) Distribution losses.** The factor $(1-\ell)$ in (1).

**(16) Inter-cluster transfer.** Physical transfer between metros is **not modelled as available**. The metros are 600–2,000 km apart and connected only through the inter-state transmission system (ISTS), which the operator cannot dispatch. The only cross-cluster lever is the contractual one in (2): re-scheduling the *inter-state RE pool* day-ahead. Within a metro, energy moves only as *charged packs on vans* (9) or as *customers redirected* to outer sites (12).

**(17) Retail price band.**
$$0.8\,\bar p_{n,k}^{\,d-1}\le p_{n,k,t}\le 1.2\,\bar p_{n,k}^{\,d-1},\qquad |p_{n,k,t}-p_{n,k,t-1}|\le 0.05\,\bar p_{n,k}^{\,d-1}$$
This is enforced in the pricing module (Section 8).

**End-of-day.** $E_{n,25,s}+\xi^{E}_{n,s}\ge E_{n,1}$ and $R_{n,25,s}+\xi^{R}_{n,s}\ge R_{n,1}$, so the next day starts with the same stock.

## 6.7 Size and solve times [Model output]

The day-ahead programme with 4 scenarios has about 28,000 variables. The hourly MILP has about 7,000 variables and up to 240 binaries. Both are solved with the open-source HiGHS solver via PuLP in about 0.3–4 s on a laptop. A commercial solver (Gurobi or CPLEX) is optional and becomes useful beyond 20 clusters or 50+ scenarios.

# 7. Storage and battery management

## 7.1 Stationary batteries

**Policy.**

- Charge in solar hours, when the DISCOM ToD tariff is 0.7 × ACoS and DAM prices are lowest [5], and from surplus renewables.
- Discharge into the 18:00–23:00 peak, when both prices and core-zone congestion bind.
- Hold the SOC needed for the 5% reserve.
- Cycle only when the value spread (avoided DISCOM/RTM cost or recovered revenue) exceeds the 12% round-trip loss plus Rs 1.4/kWh wear.
- Cap cycling at 1.5 per day.
- Track SOH from BMS data and shrink $\bar E_n$ as the battery ages.

![Figure 3 – IEX DAM price (top) and portfolio BESS schedule (bottom; discharge positive, charge negative), normal day. The rule-based timetable charges at night; the optimiser charges in cheap solar hours and discharges into the price and congestion peak.](D:/EV_Case_Study/outputs/charts/03_iex_price_vs_bess.png){width=100%}

![Figure 4 – Portfolio BESS state of charge, normal day and heatwave. The optimiser keeps SOC within 10–95%, ends the day where it started (steady state), and holds energy for the evening peak and the 5% reserve.](D:/EV_Case_Study/outputs/charts/04_bess_soc.png){width=100%}

## 7.2 Swap-pack inventory

**Policy.**

1. Forecast swaps by station and hour (P50 and P90).
2. Track every pack's state (charged, charging, depleted) and family. The model's $R$ and $Q$ are the aggregate of this ledger.
3. Charge **just in time**, so that the P90 swap demand of the next 2–3 hours plus the 5% reserve is ready at the start of each hour. This replaces charge-on-return, which fills shelves at 03:00 and empties them at 09:00 and 19:00.
4. Prioritise core stations: vans move charged packs from outer to core zones before the peaks.
5. Respect compatibility: inventory is planned per pack family.
6. Charge at no more than 0.33C where possible, which limits heat and wear.

![Figure 5 – Ready (charged) packs at the start of each hour, Delhi NCR core zone. Under charge-on-return (A), every charged pack above the 5% reserve is swapped out within each peak hour and refills are capped by the 3-hour charging time, so the stock plateaus near 30% while swap demand goes unmet from 09:00 to 23:00. The framework (B) builds stock to about 80% before the morning peak and brings charged packs in from the outer zone by van.](D:/EV_Case_Study/outputs/charts/05_swap_inventory.png){width=100%}

**How AI and optimisation work together.** Forecasts say *how much and where* demand will be, with uncertainty. The optimiser decides *what to do about it*, given the physics, contracts and prices. The shadow prices it produces (the marginal value of 1 kWh at a site and hour) feed back into pricing.

# 8. Dynamic bidding and customer pricing

## 8.1 Indian market mechanics that constrain the design [Fact]

- **IEX Day-Ahead Market (DAM).** A double-sided closed auction with uniform clearing price, run in 15-minute time blocks (96 per day). Bids are submitted on D-1 between 10:00 and 12:00 IST, and cleared schedules are then sent to the system operator [1][2]. A Green DAM segment is integrated with the DAM.
- **Real-Time Market (RTM).** 48 half-hourly auctions a day. Each clears energy for two 15-minute blocks, delivered about one hour after gate closure [1].
- **Price ceiling.** Rs 10/kWh in DAM and RTM under CERC directions; a separate high-price DAM segment exists [3].
- **Open-access eligibility.** Conventional open access generally needs ≥1 MW contract demand. Green Energy Open Access (GEOA) is available from 100 kW, with banking allowed at least monthly at notified charges [4]. Most small charging sites therefore cannot buy on IEX and must buy from the DISCOM. We assume 55–70% of each cluster's load sits at HT hubs that can.
- **Scheduling.** Schedules follow IEGC 2023 timelines in 15-minute blocks, with limited revisions [6]. Deviations from schedule are settled under the CERC Deviation Settlement Mechanism.
- **EV tariffs.** Under the Ministry of Power's 2024 EV charging guidelines, public charging tariffs are single-part, capped at the average cost of supply until March 2028, with lower tariffs in solar hours and higher ones in non-solar hours [5]. Actual state tariffs vary and must be taken from each SERC order.

**Implication.** The company **cannot trade freely every hour**. It commits most of its OA energy a day ahead, can top up one hour ahead in RTM, and must take the rest from the DISCOM. It cannot sell surplus back to the market without a separate seller registration and must bank or curtail instead.

## 8.2 Procurement strategy [Recommendation, model-tested]

1. **Renewables first.** Contracted solar and wind are take-or-pay, so the marginal cost of using them is zero. Each is scheduled with its P50 forecast, and the inter-state pool is re-allocated day-ahead towards clusters with the highest shadow value.
2. **DAM for the predictable base of OA-eligible load.** Buy roughly the optimiser's recourse-balanced quantity, between the P30 and P50 of net load, rather than the P50 point. Under-drawal recovers only a fraction of cost, while RTM or DISCOM can cover upside.
3. **RTM for intraday corrections.** Bid one hour ahead using the updated nowcast, with price caps near Rs 10.
4. **DISCOM** for non-OA sites and as last resort.
5. **Battery arbitrage** only when the spread beats losses plus wear.
6. **Risk limits.** A daily VaR on RTM exposure, a maximum share of procurement at the price cap, and operator approval for bids outside limits.

| Annual procurement mix [Model output] | RE used | IEX DAM (net) | IEX RTM | DISCOM | RE curtailed (of avail.) | RE banked (of avail.) | Rs/kWh imported | Imports (GWh/yr) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Rule-based (A) | 28.3% | 29.4% | 0.0% | 42.3% | 9.1% | 5.7% | 8.08 | 1,476 |
| Framework (B) | 27.7% | 9.1% | 12.6% | 50.6% | 1.2% | 4.8% | 6.80 | 1,669 |

## 8.3 Customer pricing model

For each zone, hour and segment, the pricing module uses the shadow price $\mu_{n,t}$ of the energy-balance constraint (1) from the day-ahead programme. $\mu_{n,t}$ is the marginal value of one more kWh at that site and hour, so it is high when the grid connection, chargers or packs are binding and low when renewables are in surplus. The tariff is:
$$
p_{n,k,t}=\bar p_{n,k}^{\,d-1}\Big(1+\mathrm{clip}\big[\beta_k\,\tfrac{\mu_{n,t}-\bar\mu_n}{\bar\mu_n}-\overline{x}_n,\ -b_k,\ b_k\big]\Big)
$$
Here $\beta$ = 0.35 for retail and 0.175 for ride-hailing; $\bar\mu_n$ is the daily mean shadow price; and the bands are $b$ = 20% for retail (the case constraint) and 10% for ride-hailing (contractual ToU band). Logistics stays at a fixed contract price, and priority and emergency users stay at the regulated base price. Re-centring by $\overline{x}_n$ keeps the daily average near yesterday's, so the band is respected the next day too. A ramp limit of 5% of $\bar p$ per hour and a cap at a competitor tariff of Rs 22/kWh are also applied.

**Demand response** (used to simulate the effect):
$$D'_{n,k,t}=D_{n,k,t}\big(1+\varepsilon_k(p_{n,k,t}/\bar p_{n,k}-1)\big)\times\text{normaliser}$$
The timing elasticity $\varepsilon$ = −0.6 (retail) and −0.8 (ride-hailing) moves load between hours. A smaller volume elasticity (−0.3 / −0.2) changes total demand. These are **assumptions**; Section 12 halves them.

**Why this avoids volatility and unfairness.**

- Tariffs are set day-ahead and published, so customers can plan.
- Tariffs are bounded, ramp-limited and re-centred.
- Customer lifetime value is protected because fleet contracts carry bands, not spot prices.
- Queueing and reliability enter through $\mu$: congestion raises the price signal only where it actually binds.
- The 5% reserve and the priority price are outside the mechanism.

![Figure 13 – Dynamic tariffs in the Delhi NCR core zone. Retail moves within the ±20% band, fleets within a ±10% contractual ToU band, and logistics stays on contract. Prices rise only where the shadow value of energy is high.](D:/EV_Case_Study/outputs/charts/13_dynamic_pricing.png){width=100%}

# 9. End-to-end technology architecture

![Figure 14 – End-to-end decision architecture and operating cadence.](D:/EV_Case_Study/outputs/charts/14_architecture.png){width=100%}

| Layer | Components | Frequency |
|---|---|---|
| Data sources | OCPP 2.0.1 charger logs, BMS (SOC, SOH, temperature), BESS EMS/SCADA, smart meters, IEX market data, IMD and commercial NWP, fleet APIs, event calendars | Telemetry 1–5 s; market and weather every 15 min |
| Ingestion and validation | IoT hub/Kafka; schema, range and unit checks (MW vs MWh!); gap filling; point-in-time feature store; data lake | Streaming |
| Forecasting | LightGBM quantile models, physical RE models, LEAR price model, conformal calibration | Every 15–60 min; retrain weekly |
| Optimisation | Day-ahead stochastic LP; hourly MILP (HiGHS / Gurobi); 60 s time cap with fallback | D-1 by 09:30 IST (before the 10:00–12:00 DAM window); hourly MPC; RTM bids per half-hour session |
| Execution | IEX bidding via a trader-member; EMS/SCADA set-points; OCPP smart-charging profiles; pack-charging queues; van app; tariff API to customer apps | 15-min blocks |
| Edge safety | Site controller enforcing grid MW, SOC, thermal limits and the 5% reserve; local rule-based fallback | Continuous |
| Monitoring | KPI dashboard (reliability, losses, EBITDA by cluster, queue time, stock-outs); forecast drift; decision audit log | Real time / daily |
| Security | IEC 62443 zones and conduits; mTLS device identity; role-based access; signed firmware; CERT-In incident reporting within 6 h [17] | Continuous |
| Human override | Control-room approval for out-of-limit bids; one-click revert to fallback rules; all overrides logged and used for learning | On demand |

# 10. Simulation and scenario analysis

## 10.1 Method [Model output]

The simulation (`code/sim.py`) represents 10 nodes (5 clusters × core/outer) over 24 hours. For each of four day types it draws a synthetic "truth" (demand, renewables, prices) and a forecast, and runs both strategies through the *same* physics and accounting.

- **A — rule-based (today).** Seasonal-naive forecast; DAM bid = forecast net load × OA share; BESS charges 00:00–06:00 and discharges 18:00–23:00; packs charge on return; one 03:00 van rebalance; flat tariffs; pro-rata RE pool; reserve held as grid headroom.
- **B — proposed framework.** ML forecast (assumed accuracy, Table 5.1); stochastic day-ahead programme (4 scenarios); shadow-price tariffs; hourly MILP MPC with one-hour-ahead RTM; EMS module staging and sleep.

Each day type is run with 3 random draws (seeds). Annual values weight the day types 65/10/10/15% × 365 days. Every KPI below is computed from the same dispatch records: reliability = delivered ÷ demand; losses = (imports − delivered − Δstored energy) ÷ imports; EBITDA = revenue − energy − degradation − variable O&M − logistics + Δstored-energy value − fixed opex − framework run cost.

**Validation [Model output].** The energy balance closes to within 10^−13^ MWh in every node-hour of every run. Imports never exceed the grid limit. SOC stays within limits (largest violation 0.002 MWh, a rounding artefact from self-discharge in the rule-based controller). The pack reserve is never breached. There are zero hours of simultaneous charge and discharge. Retail prices deviate at most 19.4% from yesterday's average (limit 20%). Full results are in `outputs/csv/validation_checks.csv`.

## 10.2 Scenario results (portfolio, mean of 3 draws, per day)

*Columns: scenario and strategy (A = rule-based, B = framework). S1 normal, S2 regional imbalance, S3 RE shortfall + IEX spike, S4 heatwave. Contribution is before fixed costs.*

| Per day (portfolio) | S1 A | S1 B | S2 A | S2 B | S3 A | S3 B | S4 A | S4 B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Demand (MWh) | 4,539 | 4,536 | 4,723 | 4,720 | 4,675 | 4,673 | 5,383 | 5,381 |
| RE available (MWh) | 1,401 | 1,401 | 1,661 | 1,661 | 700 | 700 | 1,331 | 1,331 |
| RE used (MWh) | 1,185 | 1,324 | 1,244 | 1,372 | 700 | 699 | 1,211 | 1,318 |
| IEX DAM, net (MWh) | 1,144 | 447 | 1,098 | 517 | 1,557 | 293 | 1,201 | 299 |
| IEX RTM (MWh) | 0 | 585 | 0 | 638 | 0 | 567 | 0 | 515 |
| DISCOM (MWh) | 1,668 | 2,173 | 1,708 | 2,023 | 1,801 | 3,048 | 1,825 | 2,620 |
| BESS charge (MWh) | 192 | 223 | 192 | 204 | 192 | 193 | 195 | 171 |
| BESS discharge (MWh) | 221 | 197 | 221 | 180 | 221 | 170 | 218 | 147 |
| Swap energy delivered (MWh) | 1,012 | 1,157 | 1,038 | 1,162 | 1,031 | 1,180 | 1,121 | 1,262 |
| Unserved (MWh) | 881 | 358 | 1,013 | 517 | 958 | 416 | 1,538 | 1,031 |
| Reliability | 80.6% | 92.1% | 78.5% | 89.0% | 79.5% | 91.1% | 71.4% | 80.8% |
| Losses | 9.9% | 8.2% | 9.9% | 8.1% | 9.9% | 8.1% | 10.9% | 9.1% |
| Energy cost (Rs Cr) | 3.19 | 3.02 | 3.24 | 3.04 | 3.43 | 3.41 | 3.52 | 3.33 |
| Revenue (Rs Cr) | 7.61 | 8.68 | 7.75 | 8.71 | 7.74 | 8.82 | 8.08 | 9.04 |
| Contribution (Rs Cr) | 3.72 | 4.83 | 3.79 | 4.84 | 3.60 | 4.57 | 3.80 | 4.84 |

![Figure 9 – Scenario-wise portfolio performance: reliability, losses and daily contribution (before fixed costs), rule-based vs framework, mean of 3 stochastic draws.](D:/EV_Case_Study/outputs/charts/09_scenario_comparison.png){width=100%}

**Reading the scenarios.**

- **S1 (normal).** Gains come mainly from pre-positioning packs, redirecting fleets out of the congested core, and evening BESS discharge.
- **S2 (Chennai windy and quiet, Delhi short).** The optimiser shifts the inter-state RE pool towards Delhi and the other short clusters, and banks or stores Chennai's surplus.
- **S3 (renewable shortfall + price spike).** Reliability holds because the binding limits are physical, not energy. The margin falls with energy cost, and the optimiser avoids buying RTM at the cap except where unserved penalties justify it.
- **S4 (heatwave).** The hardest day. Charger derating and 10–30% higher demand make the physical limits bind for many hours, and reliability drops in both strategies. This is where extra grid connection or pack inventory has the highest value.

## 10.3 Cluster results (annualised)

| Cluster | Reliability A | Reliability B | Losses A | Losses B | Revenue A (Rs Cr) | Revenue B (Rs Cr) | EBITDA A (Rs Cr) | EBITDA B (Rs Cr) | Margin A | Margin B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Delhi NCR | 74.0% | 85.2% | 10.0% | 8.3% | 832 | 950 | 168 | 270 | 20.2% | 28.5% |
| Mumbai MMR | 78.0% | 91.1% | 10.3% | 8.2% | 652 | 756 | 43 | 132 | 6.6% | 17.5% |
| Bengaluru | 79.4% | 91.5% | 10.0% | 8.4% | 589 | 674 | 109 | 192 | 18.6% | 28.4% |
| Hyderabad | 86.0% | 95.3% | 10.1% | 8.3% | 398 | 440 | 52 | 102 | 13.2% | 23.2% |
| Chennai | 82.3% | 89.9% | 10.0% | 8.2% | 344 | 374 | 45 | 91 | 13.2% | 24.2% |

![Figure 6 – Annualised service reliability by cluster, before (grey) and after (blue). Target 89.7%.](D:/EV_Case_Study/outputs/charts/06_reliability_before_after.png){width=100%}

![Figure 7 – Energy losses as a share of imported energy (charger conversion, pack and battery round-trip, standby/self-discharge, distribution). Target 8%.](D:/EV_Case_Study/outputs/charts/07_losses_before_after.png){width=100%}

![Figure 8 – EBITDA margin by cluster after fixed opex and (for B) the framework run cost. Target >12% in every cluster.](D:/EV_Case_Study/outputs/charts/08_ebitda_margin_by_cluster.png){width=100%}

## 10.4 Limitations of the simulation

1. Synthetic data. Shapes and magnitudes are plausible but not the company's. The baseline was calibrated to the two reported KPIs only.
2. Two zones per cluster stand in for hundreds of sites. Queueing inside a site is represented by the access factor $\kappa$, not by an explicit queue model.
3. Forecast accuracy is assumed, not trained. Sensitivity to worse forecasts is tested in Section 12.
4. There is one pack family per zone. Hourly steps stand in for 15-minute blocks.
5. Prices are exogenous (price-taker). The operator's volume of about 4–5 GWh/day is small relative to IEX volumes.
6. Each day starts from the same state (steady state), with end-of-day penalties ensuring consistency.

# 11. Financial and economic feasibility

## 11.1 Annual P&L [Model output; Rs crore]

| Rs crore / year | Rule-based (A) | Framework (B) | Change |
|---|---:|---:|---:|
| Revenue (charging + swaps) | 2,813.7 | 3,194.1 | 380.3 |
| Energy procurement | -1,193.2 | -1,135.4 | 57.9 |
| Battery degradation | -179.9 | -199.1 | -19.2 |
| Variable O&M + logistics | -67.7 | -101.2 | -33.5 |
| Change in stored energy (valued) | -12.9 | -4.7 | 8.2 |
| Fixed operating cost | -942.0 | -942.0 | 0.0 |
| Framework run cost (cloud, software, team) | -0.0 | -24.5 | -24.5 |
| EBITDA | 418.0 | 787.2 | 369.2 |
| EBITDA margin | 14.9% | 24.6% | 9.8% |
| Memo: revenue lost to unserved demand | 736.3 | 357.4 | -378.9 |

## 11.2 Where the improvement comes from

**By lever.** Levers are added cumulatively in the order shown. The attribution is therefore order-dependent and indicative.

| Configuration (cumulative) | Reliability | Losses | EBITDA (Rs Cr) | Δ EBITDA | Δ reliability |
|---|---:|---:|---:|---:|---:|
| Rule-based baseline (A) | 78.7% | 10.1% | 418 |  |  |
| + Optimised BESS, pack charging and van rebalancing; EMS module staging and sleep (hourly MPC) | 81.7% | 8.4% | 524 | +106 | +3.0 pp |
| + Fleet redirection from core to outer zones | 89.6% | 8.2% | 655 | +131 | +7.9 pp |
| + Stochastic day-ahead procurement | 89.6% | 8.3% | 763 | +109 | -0.0 pp |
| + ML probabilistic forecasts | 89.6% | 8.3% | 765 | +2 | +0.0 pp |
| + Inter-state RE-pool allocation | 89.6% | 8.3% | 771 | +6 | -0.0 pp |
| + Dynamic pricing = Framework (B) | 89.8% | 8.3% | 787 | +16 | +0.1 pp |

![Figure 12 – EBITDA bridge by lever, added cumulatively (order-dependent attribution).](D:/EV_Case_Study/outputs/charts/12_ebitda_bridge_levers.png){width=100%}

**By P&L line (A → B, Rs crore/yr).** Revenue (charging + swaps) +380; Energy procurement +58; Battery degradation -19; Variable O&M + logistics -34; Change in stored energy (valued) +8; Fixed operating cost +0; Framework run cost (cloud, software, team) -24; **EBITDA +369**.

**Interpretation.**

- *Reduced unserved demand* dominates: it is recovered revenue at about a 55–60% contribution margin.
- *Battery and pack dispatch* and *network rebalancing* (fleet redirection, vans) unlock that demand.
- *ML forecasts* add little on top of hourly MPC in this simulation. Intraday re-optimisation corrects most day-ahead errors, and the price-taker procurement model is not very sensitive to forecast error (see also the +70% forecast-error case). Their real value is in tail risk (heatwaves, renewable droughts) and in station-level pack planning, which this two-zone model understates. Do not justify the programme on forecast accuracy.
- *Renewable utilisation* raises the RE share used and cuts curtailment.
- *Dynamic pricing* flattens peaks and lifts average realised price within the band.
- *Losses* fall through EMS module staging and sleep and through leaner pack inventory, not through ML.

## 11.3 Investment case

| Technology cost item | Rs crore |
|---|---:|
| Data platform, ingestion & integration | 18.0 |
| IoT/BMS telemetry retrofit (packs, chargers, meters) | 22.0 |
| EMS/SCADA integration & charger firmware (module staging) | 10.0 |
| ML + optimisation development | 12.0 |
| Trading desk set-up, change management, training | 5.0 |
| Contingency (15 %) | 10.0 |
| **Total one-time** | **77.0** |
| Run: Cloud compute & storage | 6.0 /yr |
| Run: Software licences (solver, MLOps, weather feeds) | 3.5 /yr |
| Run: Data science / OR / trading team (30 FTE) | 10.5 /yr |
| Run: Annual maintenance of IoT & integrations | 4.5 /yr |
| **Total run cost** | **24.5 /yr** |

| Metric | Value |
|---|---:|
| Annual EBITDA uplift at full run-rate (after Rs 24.5 Cr/yr run cost) | Rs 369 Cr |
| Benefit ramp-up (years 1–5) | 30% / 70% / 100% / 100% / 100% |
| Payback | 8 months |
| 5-year NPV at 12% (assumed WACC) | Rs 935 Cr |
| 5-year ROI (net benefit ÷ capex) | 18.2x |

The payback is short because the investment is small relative to revenue lost to unserved demand today. The investment case is robust even if only a quarter of the modelled uplift is realised. Even then, the annual benefit (about Rs 92 Cr) exceeds the one-time capex.

## 11.4 Conditions under which the targets are achievable

1. **Reliability ≈ 90%** needs all of the following: live pack-ledger telemetry; at least 35% of core ride-hailing demand redirectable via the fleet API; van capacity of about 5% of pack inventory per hour; and P50 forecast WAPE below about 8%.
2. **Losses ≤ 8%** needs charger firmware that supports power-module staging and sleep (verify with OEMs) and pack-charging queues controlled by the EMS. The modelled reduction to 8.3% relies mostly on these controls. The last 0.3 pp needs targeted hardware, for example SiC power-module retrofits at the highest-throughput hubs or low-voltage cable and transformer upgrades at congested sites, or an explicit loss constraint in the optimiser at some cost in EBITDA.
3. **EBITDA > 12% in every cluster** depends most on Mumbai: highest fixed cost per MW and highest DISCOM tariff. Today Mumbai is at 6.6%. With the framework it reaches 17.5% and stays above 12% in every sensitivity case tested (Section 12).

# 12. Sensitivity analysis [Model output]

| Case | Strategy | Reliability | Losses | Revenue (Rs Cr) | EBITDA (Rs Cr) | Margin | Lowest cluster margin | Lowest cluster |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Base | A | 78.7% | 10.1% | 2,814 | 418 | 14.8% | 7.6% | Mumbai MMR |
| Base | B | 89.9% | 8.3% | 3,202 | 791 | 24.7% | 18.4% | Mumbai MMR |
| IEX +10% | A | 78.7% | 10.1% | 2,814 | 391 | 13.9% | 6.6% | Mumbai MMR |
| IEX +10% | B | 89.9% | 8.3% | 3,199 | 773 | 24.2% | 17.7% | Mumbai MMR |
| IEX +20% | A | 78.7% | 10.1% | 2,814 | 368 | 13.1% | 5.9% | Mumbai MMR |
| IEX +20% | B | 89.9% | 8.3% | 3,196 | 757 | 23.7% | 17.3% | Mumbai MMR |
| IEX +30% | A | 78.7% | 10.1% | 2,814 | 347 | 12.3% | 5.2% | Mumbai MMR |
| IEX +30% | B | 89.9% | 8.3% | 3,194 | 747 | 23.4% | 17.0% | Mumbai MMR |
| RE -10% | A | 78.7% | 10.1% | 2,814 | 428 | 15.2% | 7.8% | Mumbai MMR |
| RE -10% | B | 89.9% | 8.3% | 3,203 | 786 | 24.6% | 18.2% | Mumbai MMR |
| Demand +15% | A | 73.8% | 9.8% | 3,041 | 561 | 18.4% | 11.2% | Mumbai MMR |
| Demand +15% | B | 84.5% | 8.2% | 3,443 | 910 | 26.4% | 20.9% | Mumbai MMR |
| Degradation cost +50% | A | 78.7% | 10.1% | 2,814 | 328 | 11.7% | 5.0% | Mumbai MMR |
| Degradation cost +50% | B | 89.9% | 8.3% | 3,204 | 694 | 21.7% | 16.0% | Mumbai MMR |
| Elasticity halved | A | 78.7% | 10.1% | 2,814 | 418 | 14.8% | 7.6% | Mumbai MMR |
| Elasticity halved | B | 89.9% | 8.3% | 3,203 | 791 | 24.7% | 18.5% | Mumbai MMR |
| Forecast error +70% | A | 78.7% | 10.1% | 2,814 | 418 | 14.8% | 7.6% | Mumbai MMR |
| Forecast error +70% | B | 89.9% | 8.3% | 3,201 | 789 | 24.7% | 18.2% | Mumbai MMR |

![Figure 10 – EBITDA and weakest-cluster margin against IEX price increases of 0–30% (prices capped at Rs 10/kWh).](D:/EV_Case_Study/outputs/charts/10_sensitivity_iex.png){width=100%}

![Figure 11 – Change in framework EBITDA under each sensitivity case, with the weakest cluster margin.](D:/EV_Case_Study/outputs/charts/11_sensitivity_tornado.png){width=100%}

**Findings.**

- **IEX prices.** Price increases of 10–30% reduce EBITDA almost linearly. The Rs 10/kWh ceiling clips the evening peak, so the impact is smaller than proportional. Without the ceiling the impact would be larger.
- **Renewable shortfall (10%).** The effect is small and can even be slightly positive. The contracted RE is take-or-pay at Rs 4.3/kWh, above the landed noon DAM price, and part of it is curtailed or banked at a discount, so 10% less of it does little harm. The case for renewables here rests on cost certainty and green claims, not on marginal cost.
- **Demand +15%.** This raises absolute EBITDA (more revenue on the same fixed cost) but *lowers reliability*: the physical limits bind more. This is the clearest signal that capacity investment, not software, is the next constraint.
- **Degradation cost +50%** makes the optimiser cycle less. EBITDA falls modestly.
- **Halved elasticity** removes part of the pricing lever only. The other levers are unaffected.
- **Forecast error +70%** degrades procurement and pack readiness but keeps most of the value. The value is in the *decision system*, not in forecast perfection.

# 13. Implementation roadmap

| Phase | Duration | Deliverables | Teams | Technology | KPIs | Main risks | Expected impact |
|---|---|---|---|---|---|---|---|
| 1. Data integration and baseline analytics | Months 0–3 | Unified data model; pack ledger; meter/OCPP/BMS/IEX feeds; KPI baseline by site; this model re-calibrated on real data | Data engineering, IT/OT, operations | Cloud data lake, Kafka/IoT hub, feature store, dashboard | Data completeness >98%; latency <5 s; reconciled energy balance per site | OT integration delays; poor BMS data | Transparency; quick wins (fix idle chargers) |
| 2. Forecasting + pilot | Months 3–6 | Quantile demand, RE and price models; Delhi NCR core pilot of JIT pack charging and BESS MPC (advisory, then closed loop) | Data science, Delhi operations | LightGBM, pvlib, MLflow, HiGHS | WAPE <8%; pilot stock-outs −50%; reliability +5 pp in pilot | Change resistance; model drift | First reliability gain |
| 3. Storage and procurement optimisation | Months 6–10 | Day-ahead stochastic bidding via trader-member; hourly MPC in all clusters; module staging firmware; vans | OR team, trading desk, EMS vendor | Optimisation service, EMS/SCADA APIs, OCPP smart charging | Procurement cost/kWh −5%; losses −1.5 pp; zero reserve breaches | Regulatory approvals for OA; solver robustness | Most of the loss target; margin gain |
| 4. Dynamic pricing + cross-cluster coordination | Months 10–14 | Shadow-price tariffs with ±20% guard; fleet ToU contracts and redirection API; inter-state RE pool scheduling | Commercial, legal, fleet partnerships | Pricing service, app/API, scheduling interface with SLDC/RLDC | Peak demand −8%; complaints < threshold; RE curtailment <2% | Customer backlash; contract renegotiation | Reliability to about 90% |
| 5. Full deployment + continuous learning | Months 14–18+ | All clusters closed loop; champion/challenger models; automated retraining; playbooks for 20+ clusters | All + MLOps | Kubernetes, model registry, monitoring | KPIs at target for 3 consecutive months | Complacency; scale-up of data volume | Sustained EBITDA >12% |

**Scaling to 20+ clusters.**

- The model decomposes by cluster. The only couplings are the RE pool and company-level risk limits, so it can be solved in parallel, or with Lagrangian or Benders decomposition on the pool constraint.
- The global forecasting model absorbs new clusters with little history.
- Zones per cluster can grow from 2 to about 10 without changing the formulation.
- Scenario count grows with scenario reduction, not proportionally.

# 14. Risk management

| Risk | Mitigation |
|---|---|
| Forecasting errors | Probabilistic forecasts; scenario-based optimisation; conformal calibration; RTM recourse; reserve untouched |
| Extreme weather | Heatwave scenarios in every day-ahead run; pre-cooling of pack cabinets; charger derating in constraints; P90-based pack targets |
| Battery degradation | Degradation cost and cycle cap in objective; SOH-adjusted capacity; warranty-compliant SOC window; thermal limits at edge |
| Cybersecurity | IEC 62443 segmentation; mTLS; least-privilege access; signed OTA firmware; SOC monitoring; CERT-In reporting |
| Grid outages | Edge controller islands to BESS for priority reserve; reserve backed by BESS energy (constraint 13) |
| IEX volatility | Price-capped bids; RTM exposure limits; share of procurement in DAM bounded; Rs 10 ceiling in scenarios |
| Regulatory restrictions | Model respects OA eligibility, banking limits and scheduling rules; regulatory monitoring per SERC; no assumption of free trading |
| Data quality | Unit and range validation (MW vs MWh); energy-balance reconciliation per site; quarantine of bad feeds |
| Customer resistance to pricing | Day-ahead published tariffs; ±20% and 5%/h ramp limits; priority users exempt; fleet bands in contracts; A/B testing |
| Solver failure | 60 s time cap → previous feasible plan → **fallback rule-based controller** below |

**Fallback rule-based controller** (runs at the edge, no cloud needed):

1. Never breach the grid limit, the SOC window or the 5% reserve.
2. Serve customers in priority order: emergency/priority, logistics SLA, ride-hailing, retail.
3. Charge depleted packs earliest-first, keeping ready stock ≥ the P90 swap demand of the next 2 h.
4. Charge the BESS when renewables are in surplus or in solar ToD hours; discharge when site load exceeds 90% of the grid limit.
5. Keep yesterday's tariffs.
6. Keep the last valid DAM schedule; top up from the DISCOM.

# 15. Final recommendations

> **1. Decide as one system.** Replace siloed rules with the forecast–optimise–execute–learn loop. Procurement, storage, packs and pricing should share one objective and one set of constraints.
>
> **2. Manage swap packs as inventory.** Run just-in-time charging against P90 swap forecasts, keep a live pack ledger, and fund core-zone rebalancing vans. This is the single largest reliability lever.
>
> **3. Use the fleets.** Convert ride-hailing and logistics agreements to time-of-use and redirection-incentive contracts. Fleets are the most price- and route-flexible demand.
>
> **4. Professionalise procurement.** Build a trading desk with stochastic day-ahead bidding, one-hour-ahead RTM corrections and explicit risk limits. Schedule the inter-state RE pool to where it is worth most.
>
> **5. Cut losses with controls before capex.** Enable charger module staging and sleep, lean pack inventories, and value-based battery cycling. Then target hardware upgrades where losses remain highest.
>
> **6. Pilot first, then scale.** Start in the Delhi NCR core (lowest reliability, largest cluster), prove the reliability gain within 6 months, and roll out over 18 months.
>
> **7. Plan capacity where software stops.** Under demand growth and heatwaves the binding limits are physical; the Delhi NCR core is already there. Use the model's shadow prices to rank grid-connection upgrades and pack purchases by value per rupee.

# References

1. Indian Energy Exchange (IEX). Day-Ahead Market and Real-Time Market: product specifications and trading timelines. iexindia.com.
2. Central Electricity Regulatory Commission (CERC). Power Market Regulations, 2021.
3. CERC directions on price ceiling (Rs 10/kWh) for DAM and RTM, and introduction of the High-Price DAM segment (2022–2023 orders).
4. Ministry of Power. Electricity (Promoting Renewable Energy Through Green Energy Open Access) Rules, 2022, and amendments.
5. Ministry of Power. Guidelines for Installation and Operation of Electric Vehicle Charging Infrastructure – 2024.
6. CERC. Indian Electricity Grid Code Regulations, 2023; CERC Deviation Settlement Mechanism and Related Matters Regulations (current version). Grid-India (formerly POSOCO) scheduling procedures.
7. Koenker, R., Bassett, G. (1978). Regression quantiles. *Econometrica* 46(1), 33–50.
8. Hong, T., et al. (2016). Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond. *Int. J. Forecasting* 32(3), 896–913.
9. Lago, J., Marcjasz, G., De Schutter, B., Weron, R. (2021). Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark. *Applied Energy* 293, 116983.
10. Lim, B., Arık, S.Ö., Loeff, N., Pfister, T. (2021). Temporal Fusion Transformers for interpretable multi-horizon time series forecasting. *Int. J. Forecasting* 37(4), 1748–1764.
11. Oreshkin, B.N., Carpov, D., Chapados, N., Bengio, Y. (2020). N-BEATS: Neural basis expansion analysis for interpretable time series forecasting. *ICLR*.
12. Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *NeurIPS*.
13. Romano, Y., Patterson, E., Candès, E. (2019). Conformalized quantile regression. *NeurIPS*.
14. Holmgren, W.F., Hansen, C.W., Mikofski, M.A. (2018). pvlib python: a python package for modeling solar energy systems. *J. Open Source Software* 3(29), 884.
15. Xu, B., Oudalov, A., Ulbig, A., Andersson, G., Kirschen, D.S. (2018). Modeling of lithium-ion battery degradation for cell life assessment. *IEEE Trans. Smart Grid* 9(2), 1131–1140.
16. Preger, Y., et al. (2020). Degradation of commercial lithium-ion cells as a function of chemistry and cycling conditions. *J. Electrochem. Soc.* 167, 120532.
17. CERT-In. Directions under Section 70B(6) of the IT Act, 28 April 2022 (6-hour incident reporting).
18. NREL. Annual Technology Baseline: Utility-Scale Battery Storage (round-trip efficiency ≈ 85%).
19. NITI Aayog. Draft Battery Swapping Policy (2022); NITI Aayog et al., Handbook for EV Charging Infrastructure Implementation (2021).
20. Conejo, A.J., Carrión, M., Morales, J.M. (2010). *Decision Making Under Uncertainty in Electricity Markets*. Springer.
21. Open Charge Alliance. OCPP 2.0.1; IEC 62443 Industrial communication networks – Security.

*Regulatory details (tariffs, OA charges, banking limits) differ by state and change frequently. Verify them against the latest CERC/SERC orders before any commercial decision.*

# Appendix: reproducibility

`python code/run_all.py 4` runs all experiments (about 15 minutes on 4 cores). `python code/charts.py` regenerates every figure. `python code/build_report.py` rebuilds this report (PDF and DOCX) from the CSV outputs. Outputs are in `outputs/csv/` (daily KPIs for every run, hourly dispatch, annual P&L by configuration, scenario means, sensitivity, forecast metrics, validation checks, financials) and `outputs/charts/`.
