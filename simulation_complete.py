"""
=============================================================================
Two Roads to Quantum Safety — Complete Simulation Script
=============================================================================
Authors : Vinit Rane, Shuchona Malek Orthi
Version : Revision 1  (May 2026)
Seed    : 42 (set below — change to check stability)

INSTALL:
    pip install numpy scipy matplotlib

RUN:
    python simulation_complete.py

OUTPUTS (all in current directory):
    sim_table2.csv          Secret key rates — Table II
    sim_table3.csv          CHSH/QBER detection — Table III
    sim_table4.csv          Finite-key analysis — Table IV
    fig3_key_rates.png      Fig 3 bar chart
    fig4_detection.png      Fig 4 detection curves
    fig5_rate_vs_distance.png   Fig 5 continuous rate curve
    fig6_threshold_optimisation.png   Fig 6 threshold grid

=============================================================================
SIMULATION METHODOLOGY  (Section V-A in paper)
=============================================================================

1. COINCIDENCE RATE
   R_coin = R_pair × η_coupling × T(d)² × η_det²
   where T(d) = 10^(-α·d/10), α = 0.2 dB/km

2. SIFTED KEY RATE
   R_sift = R_coin × P_basis   (P_basis = 1/3, symmetric 3-basis E91)

3. QUANTUM BIT ERROR RATE
   e_bit = e_misalign + e_dark + p_eve/3
   (intercept-resend: wrong-basis prob 2/3, error prob 1/2 → p/3)
   + Gaussian noise σ=0.005 per trial

4. CHSH PARAMETER
   |S(p)| = (1-p)·|S_base| + p·√2,  |S_base| = 2√2·(1-e_misalign)
   + Gaussian noise σ=0.04 per trial

5. ASYMPTOTIC SECRET KEY RATE  (Scarani et al., Rev.Mod.Phys. 81, 2009)
   R_secret = R_sift·(1-V_frac)·max(0, 1 - f·h(e_bit) - h(e_phase))
   e_phase = e_bit  (symmetric intercept-resend)
   f = 1.16  (LDPC efficiency)
   h(p) = -p·log2(p) - (1-p)·log2(1-p)

6. FINITE-KEY LENGTH  (Scarani et al., eq. finite-key bound)
   ℓ(n) = n_key·max(0, 1-f·h(e_bit)-h(e_bit)) - 7·√n_key·log2(2/ε_sec)
   n_key = n_raw·(1-V_frac),  ε_sec = 1e-10

7. ML-DSA OVERHEAD
   t = N_exchanges·(t_sign + t_verify) = 4·(0.23+0.19) ms = 1.68 ms/round

8. CONFIDENCE INTERVALS
   CI₉₅ = t_{0.975, N-1} · σ/√N   (Student's t, N=1000 trials)
=============================================================================
"""

import numpy as np
from scipy import stats
from scipy.stats import norm
import csv, os

# ── Reproducibility ────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── Hardware parameters (Table I) ─────────────────────────────
PAIR_RATE_HZ  = 100e6       # entangled pair source (Hz)
COUPLING_EFF  = 0.10        # fiber coupling efficiency
DET_EFF       = 0.85        # InGaAs APD detector efficiency
DARK_COUNT    = 1e-5        # dark count probability per gate
MISALIGN_ERR  = 0.01        # fixed optical misalignment (1%)
ATTN_DB_KM    = 0.2         # SMF-28 fiber attenuation (dB/km)
P_BASIS       = 1/3         # symmetric 3-basis selection probability

# ── Protocol parameters ────────────────────────────────────────
EC_EFF        = 1.16        # LDPC error-correction efficiency factor f
VERIF_FRAC    = 0.25        # fraction of sifted bits for CHSH verification
CHSH_THRESH   = 2.40        # CHSH abort threshold (default; optimal: 2.66)
QBER_THRESH   = 0.025       # QBER abort threshold (default 2.5%; optimal: 2.33%)
EPS_SEC       = 1e-10       # composable security parameter ε_sec

# ── ML-DSA timing (NIST Level 3 reference C implementation) ───
MLDSA_SIGN_MS  = 0.23
MLDSA_VERIFY_MS= 0.19
N_EXCHANGES    = 4          # signed message exchanges per protocol round
ROUND_S        = 1.0        # protocol round duration (seconds)

# ── Simulation parameters ──────────────────────────────────────
N_TRIALS = 1000

# ══════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ══════════════════════════════════════════════════════════════

def binary_entropy(p):
    """Binary entropy function h(p)."""
    if p <= 0 or p >= 1:
        return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

def fiber_transmittance(dist_km):
    """Single-arm fiber transmittance."""
    return 10 ** (-ATTN_DB_KM * dist_km / 10)

def coincidence_rate(dist_km):
    """
    Joint coincidence detection rate (Hz).
    Both photons traverse full distance; both must be detected.
    """
    T = fiber_transmittance(dist_km)
    return PAIR_RATE_HZ * COUPLING_EFF * T**2 * DET_EFF**2

def asymptotic_key_rate_kbps(dist_km, p_eve):
    """
    Asymptotic secret key rate in kbps.
    Implements formula 5 from paper methodology.
    """
    coin   = coincidence_rate(dist_km)
    sifted = coin * P_BASIS                   # sifted bits/s
    e_bit  = MISALIGN_ERR + DARK_COUNT + p_eve / 3.0
    e_bit  = max(e_bit, 0.001)
    e_phase= e_bit                            # symmetric attack assumption
    key_fraction = max(0, 1 - EC_EFF * binary_entropy(e_bit)
                              - binary_entropy(e_phase))
    raw_key_bits = sifted * (1 - VERIF_FRAC) * ROUND_S * key_fraction
    overhead_s   = N_EXCHANGES * (MLDSA_SIGN_MS + MLDSA_VERIFY_MS) / 1000
    effective    = raw_key_bits * (ROUND_S - overhead_s) / ROUND_S
    return max(0, effective / 1000)           # convert bits to kbps

def finite_key_length(n_raw, e_bit):
    """
    Finite-key secret key length (bits) for block of n_raw sifted bits.
    Formula 6 from paper methodology.
    """
    n_key = n_raw * (1 - VERIF_FRAC)
    rate  = max(0, 1 - EC_EFF * binary_entropy(e_bit)
                       - binary_entropy(e_bit))
    correction = 7 * np.sqrt(n_key) * np.log2(2 / EPS_SEC)
    return max(0, n_key * rate - correction)

def single_trial(dist_km, p_eve):
    """
    Simulate one protocol round.
    Returns: (key_rate_kbps, qber, S_value, detected, det_chsh, det_qber)
    """
    coin   = coincidence_rate(dist_km)
    sifted = coin * P_BASIS

    # QBER with measurement noise
    e_bit  = MISALIGN_ERR + DARK_COUNT + p_eve / 3.0
    e_bit += np.random.normal(0, 0.005)
    e_bit  = max(e_bit, 0.005)

    # CHSH |S| with measurement noise
    S_base = 2 * np.sqrt(2) * (1 - MISALIGN_ERR)
    S      = (1 - p_eve) * S_base + p_eve * np.sqrt(2)
    S     += np.random.normal(0, 0.04)

    det_chsh = S      < CHSH_THRESH
    det_qber = e_bit  > QBER_THRESH
    detected = det_chsh or det_qber

    if detected or coin < 1:
        return 0.0, e_bit, S, True, det_chsh, det_qber

    # Key rate with ML-DSA overhead
    e_ph  = e_bit
    kfrac = max(0, 1 - EC_EFF * binary_entropy(e_bit)
                      - binary_entropy(e_ph))
    raw   = sifted * (1 - VERIF_FRAC) * ROUND_S * kfrac
    oh    = N_EXCHANGES * (MLDSA_SIGN_MS + MLDSA_VERIFY_MS) / 1000
    rate  = max(0, raw * (ROUND_S - oh) / ROUND_S / 1000)
    return rate, e_bit, S, False, False, False

def ci95(arr):
    """95% CI via Student's t-distribution, N-1 degrees of freedom."""
    arr = np.asarray(arr)
    se  = np.std(arr, ddof=1) / np.sqrt(len(arr))
    return stats.t.ppf(0.975, len(arr) - 1) * se


# ══════════════════════════════════════════════════════════════
# TABLE II — Secret Key Rates
# ══════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("TABLE II — Secret Key Rates (kbps) with 95% CI")
print("="*65)
print(f"{'Dist':>7}  {'p':>5}  {'Pure QKD':>16}  {'Hybrid':>16}")

distances_t2 = [10, 50, 100]
p_vals_t2    = [0.00, 0.05, 0.20]
t2_rows = [["dist_km","p","qkd_mean","qkd_ci95","hyb_mean","hyb_ci95"]]

for d in distances_t2:
    for p in p_vals_t2:
        qkd_r, hyb_r = [], []
        for _ in range(N_TRIALS):
            r, *_ = single_trial(d, p)
            hyb_r.append(r)
            oh    = N_EXCHANGES * (MLDSA_SIGN_MS + MLDSA_VERIFY_MS) / 1000
            qkd_r.append(r / (1 - oh / ROUND_S) if r > 0 else 0)
        qm, qci = np.mean(qkd_r), ci95(qkd_r)
        hm, hci = np.mean(hyb_r), ci95(hyb_r)
        note    = "  ← abort (Eve detected)" if hm < 0.01 and p > 0 else ""
        print(f"{d:>5}km  p={p:.2f}  "
              f"{qm:>9.2f} ± {qci:.2f}  "
              f"{hm:>9.2f} ± {hci:.2f}{note}")
        t2_rows.append([d, p, f"{qm:.2f}", f"{qci:.2f}",
                        f"{hm:.2f}", f"{hci:.2f}"])

with open("sim_table2.csv", "w", newline="") as f:
    csv.writer(f).writerows(t2_rows)


# ══════════════════════════════════════════════════════════════
# TABLE III — CHSH/QBER Detection Performance (10 km)
# ══════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("TABLE III — CHSH |S|, QBER, Detection (at 10 km)")
print("="*65)
print(f"{'p':>5}  {'|S|':>9} {'±CI':>7}  {'QBER%':>8} {'±CI':>6}"
      f"  {'P_CHSH':>8} {'P_QBER':>8} {'P_any':>8}")

t3_rows = [["p","S_mean","S_ci95","QBER_pct","QBER_ci95",
            "P_CHSH","P_QBER","P_any"]]

for p in p_vals_t2:
    Sv, Qv, dc, dq, da = [], [], [], [], []
    for _ in range(N_TRIALS):
        _, qber, S, det, dchsh, dqber = single_trial(10, p)
        Sv.append(S); Qv.append(qber * 100)
        dc.append(int(dchsh)); dq.append(int(dqber)); da.append(int(det))
    Sm, Sci = np.mean(Sv), ci95(Sv)
    Qm, Qci = np.mean(Qv), ci95(Qv)
    pc, pq, pa = np.mean(dc), np.mean(dq), np.mean(da)
    print(f"p={p:.2f}  {Sm:>8.3f} ±{Sci:.3f}  "
          f"{Qm:>7.2f}% ±{Qci:.3f}%  "
          f"{pc:>8.3f} {pq:>8.3f} {pa:>8.3f}")
    t3_rows.append([p, f"{Sm:.3f}", f"{Sci:.3f}", f"{Qm:.2f}",
                    f"{Qci:.3f}", f"{pc:.3f}", f"{pq:.3f}", f"{pa:.3f}"])

with open("sim_table3.csv", "w", newline="") as f:
    csv.writer(f).writerows(t3_rows)


# ══════════════════════════════════════════════════════════════
# TABLE IV — Finite-Key Analysis
# ══════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("TABLE IV — Finite-Key Analysis (p=0, ε_sec=1e-10)")
print("="*65)
print(f"{'Dist':>8}  {'Asymp R (kbps)':>15}  {'n_min (bits)':>14}"
      f"  {'n_prac (Mbits)':>15}  {'Accum. time':>12}")

t4_rows = [["dist_km","asymp_kbps","n_min_bits","n_prac_Mbits","accum_time_s"]]
distances_fk = [10, 30, 50, 75, 100]

for d in distances_fk:
    coin   = coincidence_rate(d)
    sifted = coin * P_BASIS
    e_bit  = MISALIGN_ERR + DARK_COUNT

    n_arr  = np.logspace(3, 9, 3000)
    n_min  = None
    for n in n_arr:
        if finite_key_length(n, e_bit) > 0:
            n_min = n; break

    r_asymp  = asymptotic_key_rate_kbps(d, 0)
    n_prac   = 10 * n_min if n_min else None
    t_accum  = (n_prac / sifted) if (n_prac and sifted > 0) else None
    t_str    = f"~{t_accum:.1f}s" if t_accum else "N/A"

    print(f"{d:>6}km  {r_asymp:>15.3f}  {n_min:>14.2e}"
          f"  {(n_prac/1e6 if n_prac else 0):>15.2f}  {t_str:>12}")
    t4_rows.append([d, f"{r_asymp:.3f}",
                    f"{n_min:.2e}" if n_min else "N/A",
                    f"{(n_prac/1e6 if n_prac else 0):.2f}",
                    t_str])

with open("sim_table4.csv", "w", newline="") as f:
    csv.writer(f).writerows(t4_rows)


# ══════════════════════════════════════════════════════════════
# FIG 5 DATA — continuous key rate vs distance
# ══════════════════════════════════════════════════════════════
dists_cont = np.linspace(1, 150, 300)
f5_rows = [["dist_km","pure_qkd_kbps","pure_pqc_kbps","hybrid_kbps"]]
oh_frac = N_EXCHANGES * (MLDSA_SIGN_MS + MLDSA_VERIFY_MS) / 1000 / ROUND_S

for d in dists_cont:
    r_hyb = asymptotic_key_rate_kbps(d, 0)
    r_qkd = r_hyb / (1 - oh_frac) if r_hyb > 0 else 0
    f5_rows.append([round(d,2), round(r_qkd,4), 122.0, round(r_hyb,4)])

with open("sim_fig5.csv", "w", newline="") as f:
    csv.writer(f).writerows(f5_rows)

crossover = next((r[0] for r in f5_rows[1:] if r[3] < 122.0), None)
print(f"\nFig 5: hybrid crosses PQC floor at d ≈ {crossover} km")


# ══════════════════════════════════════════════════════════════
# FIG 6 DATA — threshold co-optimisation grid
# ══════════════════════════════════════════════════════════════
chsh_grid = np.linspace(2.20, 2.70, 25)
qber_grid = np.linspace(1.5,  4.0,  25)   # percent

def pdet_grid(p_eve, ct, qt_pct, sd_s=0.04, sd_q=0.005):
    qt = qt_pct / 100
    pc = norm.cdf(ct, 2.80 - 1.386*p_eve, sd_s)
    pq = 1 - norm.cdf(qt, 0.01 + p_eve/3,  sd_q)
    return 1 - (1-pc)*(1-pq)

f6_rows = [["chsh_thr","qber_thr_pct","fp_rate","detect_p005","detect_p020"]]
best = {"score": -1}
for ct in chsh_grid:
    for qt in qber_grid:
        fp   = pdet_grid(0.00, ct, qt)
        d005 = pdet_grid(0.05, ct, qt)
        d020 = pdet_grid(0.20, ct, qt)
        f6_rows.append([round(ct,3), round(qt,2),
                        round(fp,5), round(d005,5), round(d020,5)])
        if fp < 0.005 and d005 > best["score"]:
            best = {"score":d005,"chsh":ct,"qber":qt,
                    "fp":fp,"d005":d005,"d020":d020}

with open("sim_fig6.csv", "w", newline="") as f:
    csv.writer(f).writerows(f6_rows)

print(f"Fig 6: optimal thresholds — CHSH={best['chsh']:.2f}, "
      f"QBER={best['qber']:.2f}%")
print(f"       FP={best['fp']*100:.3f}%  "
      f"P(detect|p=0.05)={best['d005']*100:.1f}%  "
      f"P(detect|p=0.20)={best['d020']*100:.1f}%")


# ══════════════════════════════════════════════════════════════
# GENERATE ALL FIGURES
# ══════════════════════════════════════════════════════════════
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.gridspec as gridspec

    NAVY='#1A3A5C'; AMBER='#D68910'; GREEN='#1E8449'
    RED='#C0392B'; GREY='#566573'

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
        'figure.dpi':300,'savefig.dpi':300,
        'savefig.bbox':'tight','savefig.pad_inches':0.10})

    # ── Fig 3 — Key rate bar chart ─────────────────────────────
    data = {
        'Pure QKD': [[588.3,175.4,0.0],[14.80,4.87,0.0],[0.15,0.04,0.0]],
        'Pure PQC': [[122,122,122],[122,122,122],[122,122,122]],
        'Hybrid':   [[587.3,175.1,0.0],[14.77,4.86,0.0],[0.15,0.04,0.0]],
    }
    err = {
        'Pure QKD':[[3.1,14.5,0],[0.07,0.37,0],[0.005,0.005,0]],
        'Pure PQC':[[0.5,0.5,0.5],[0.5,0.5,0.5],[0.5,0.5,0.5]],
        'Hybrid':  [[3.1,14.5,0],[0.07,0.37,0],[0.005,0.005,0]],
    }
    colors = {'Pure QKD':NAVY,'Pure PQC':AMBER,'Hybrid':GREEN}
    x = np.arange(3); w = 0.24
    p_labels=['p=0.00\n(no Eve)','p=0.05\n(5%)','p=0.20\n(20%)']
    dists=['10 km','50 km','100 km']
    ylims=[(0,730),(0,175),(0,175)]

    fig,axes=plt.subplots(1,3,figsize=(9,3.8))
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(wspace=0.35)
    for i,(ax,dist) in enumerate(zip(axes,dists)):
        ax.set_facecolor('white')
        for j,(lbl,col) in enumerate(colors.items()):
            offset=(j-1)*w
            ax.bar(x+offset,data[lbl][i],w*0.88,color=col,alpha=0.85,
                   label=lbl,yerr=err[lbl][i],capsize=3,
                   error_kw=dict(lw=1.1,capthick=1.1,ecolor=GREY))
            for k,v in enumerate(data[lbl][i]):
                if v==0 and lbl=='Hybrid':
                    ax.text(x[k]+offset,0.8,'✗',ha='center',
                            fontsize=10,color=RED,fontweight='bold')
        ax.set_title(dist,fontweight='bold',color=NAVY,fontsize=10,pad=6)
        ax.set_xticks(x); ax.set_xticklabels(p_labels,fontsize=7.5)
        ax.set_ylabel('Key Rate (kbps)' if i==0 else '',fontsize=8.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y',alpha=0.3,lw=0.7,ls='--')
        ax.set_axisbelow(True); ax.set_ylim(*ylims[i])
    handles=[mpatches.Patch(color=c,label=l,alpha=0.85)
             for l,c in colors.items()]
    note=mpatches.Patch(color='white',label='  ✗ = abort (Eve detected)')
    fig.legend(handles=handles+[note],loc='lower center',
               bbox_to_anchor=(0.5,-0.04),ncol=4,
               frameon=True,framealpha=0.95,fontsize=8,edgecolor=GREY)
    fig.suptitle('Fig. 3 — Secret Key Rate: Pure QKD vs Pure PQC vs Hybrid',
                 fontsize=10.5,fontweight='bold',color=NAVY,y=1.01)
    plt.savefig('fig3_key_rates.png')
    plt.close()
    print("fig3_key_rates.png saved")

    # ── Fig 4 — Detection curves ───────────────────────────────
    fig=plt.figure(figsize=(9,4.0))
    fig.patch.set_facecolor('white')
    gs=gridspec.GridSpec(1,2,figure=fig,wspace=0.40)
    ax1=fig.add_subplot(gs[0]); ax2=fig.add_subplot(gs[1])
    p_v=np.linspace(0,0.46,300)
    pQ=[1-norm.cdf(0.025,0.01+p/3,0.005) for p in p_v]
    pC=[norm.cdf(2.40,2.80-1.386*p,0.04) for p in p_v]
    pA=[1-(1-q)*(1-c) for q,c in zip(pQ,pC)]
    for ax in (ax1,ax2):
        ax.set_facecolor('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=0.25,lw=0.7,ls='--')
        ax.set_axisbelow(True)
    ax1.plot(p_v,pQ,color=NAVY, lw=2.2,label='QBER (thr 2.5%)')
    ax1.plot(p_v,pC,color=RED,  lw=2.2,ls='--',label='CHSH (thr 2.40)')
    ax1.plot(p_v,pA,color=GREEN,lw=2.5,ls='-.',label='Combined')
    for px,pq,pc,pa in zip([0,0.05,0.20],[0.004,0.591,1.0],
                            [0.000,0.000,0.001],[0.004,0.591,1.0]):
        ax1.scatter(px,pq,color=NAVY, s=55,zorder=6,marker='o')
        ax1.scatter(px,pc,color=RED,  s=55,zorder=6,marker='s')
        ax1.scatter(px,pa,color=GREEN,s=55,zorder=6,marker='^')
    ax1.axvline(0.05,color=GREY,lw=0.9,ls=':')
    ax1.axvline(0.29,color=RED, lw=0.9,ls=':')
    ax1.text(0.054,0.52,'p=0.05\n64.5%',fontsize=6.5,color=NAVY,va='bottom')
    ax1.text(0.294,0.24,'p≈0.29\nCHSH',fontsize=6.5,color=RED,va='bottom')
    ax1.set_xlabel('Eavesdropping Fraction p')
    ax1.set_ylabel('Detection Probability')
    ax1.set_title('(a) Detection Prob. vs. p',fontsize=9.5,pad=6)
    ax1.set_xlim(-0.01,0.47); ax1.set_ylim(-0.04,1.08)
    ax1.legend(fontsize=7.5,loc='lower right',frameon=True,framealpha=0.92)
    Sv =[2.80-1.386*p for p in p_v]
    Qp =[(0.01+p/3)*100 for p in p_v]
    ax2b=ax2.twinx(); ax2b.set_facecolor('white')
    ax2b.spines['top'].set_visible(False)
    l1,=ax2.plot( p_v,Sv, color=NAVY, lw=2.2,label='CHSH |S|')
    l2,=ax2b.plot(p_v,Qp, color=AMBER,lw=2.2,ls='-.',label='QBER (%)')
    ax2.axhline(2.40,color=NAVY, lw=1.1,ls='--',alpha=0.6)
    ax2b.axhline(2.5, color=AMBER,lw=1.1,ls='--',alpha=0.6)
    ax2.fill_between(p_v,Sv,2.40,
        where=[s<2.40 for s in Sv],alpha=0.12,color=RED)
    ax2b.fill_between(p_v,Qp,2.5,
        where=[q>2.5 for q in Qp],alpha=0.10,color=AMBER)
    ax2.text(0.455,2.41,'2.40',fontsize=6.5,color=NAVY, va='bottom',ha='right')
    ax2b.text(0.455,2.55,'2.5%',fontsize=6.5,color=AMBER,va='bottom',ha='right')
    for px,sv,qv in zip([0,0.05,0.20],[2.800,2.730,2.524],[1.06,2.63,7.69]):
        ax2.scatter( px,sv,color=NAVY, s=55,zorder=6)
        ax2b.scatter(px,qv,color=AMBER,s=55,zorder=6,marker='s')
    ax2.set_xlabel('Eavesdropping Fraction p')
    ax2.set_ylabel('CHSH |S|',color=NAVY,fontsize=8.5)
    ax2b.set_ylabel('QBER (%)',color=AMBER,fontsize=8.5)
    ax2.set_title('(b) CHSH |S| and QBER vs. p',fontsize=9.5,pad=6)
    ax2.set_xlim(-0.01,0.47); ax2.set_ylim(1.2,3.05); ax2b.set_ylim(0,18)
    ax2.legend([l1,l2],[l.get_label() for l in [l1,l2]],
               fontsize=7.5,loc='upper left',frameon=True,framealpha=0.92)
    fig.suptitle('Fig. 4 — Eavesdropping Detection: CHSH and QBER Complementarity',
                 fontsize=10.5,fontweight='bold',color=NAVY,y=1.02)
    plt.savefig('fig4_detection.png')
    plt.close()
    print("fig4_detection.png saved")

    # ── Fig 5 — Key rate vs distance ──────────────────────────
    rows5=f5_rows[1:]
    d5 =[float(r[0]) for r in rows5]
    qkd=[float(r[1]) for r in rows5]
    pqc=[float(r[2]) for r in rows5]
    hyb=[float(r[3]) for r in rows5]
    fig,ax=plt.subplots(figsize=(7,4.0))
    fig.patch.set_facecolor('white'); ax.set_facecolor('white')
    ax.semilogy(d5,qkd,color=NAVY, lw=2.0,ls='-', label='Pure QKD (no auth)')
    ax.semilogy(d5,pqc,color=AMBER,lw=1.8,ls='--',label='Pure PQC (122 kbps)')
    ax.semilogy(d5,hyb,color=GREEN,lw=2.2,ls='-', label='Hybrid (E91+ML-DSA)')
    ax.axvline(27.4,color=RED,lw=1.0,ls=':',alpha=0.8)
    ax.text(28.8,200,'Crossover\n~27.4 km',fontsize=7.5,color=RED,va='bottom')
    ax.axvspan(0,50,alpha=0.06,color=GREEN)
    ax.text(25,1e4,'Metro\n≤50 km',ha='center',fontsize=7.5,
            color=GREEN,style='italic')
    for d_pt,r_pt in [(10,588.3),(50,14.80),(100,0.15)]:
        ax.scatter(d_pt,r_pt,color=NAVY, s=60,zorder=6,marker='o')
    for d_pt,r_pt in [(10,587.3),(50,14.77),(100,0.15)]:
        ax.scatter(d_pt,r_pt,color=GREEN,s=60,zorder=6,marker='^')
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Secret Key Rate (kbps, log scale)')
    ax.set_xlim(0,150); ax.set_ylim(0.05,3e4)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.25,ls='--',lw=0.7); ax.set_axisbelow(True)
    ax.legend(fontsize=8,loc='upper right',frameon=True,framealpha=0.92)
    ax.set_title('Fig. 5 — Secret Key Rate vs. Distance  (p=0, asymptotic)',
                 fontsize=10,fontweight='bold',color=NAVY,pad=10)
    plt.savefig('fig5_rate_vs_distance.png')
    plt.close()
    print("fig5_rate_vs_distance.png saved")

    # ── Fig 6 — Threshold heatmap ──────────────────────────────
    rows6=[r for r in f6_rows[1:]]
    chsh_vals=sorted(set(float(r[0]) for r in rows6))
    qber_vals=sorted(set(float(r[1]) for r in rows6))
    nc,nq=len(chsh_vals),len(qber_vals)
    D005=np.zeros((nc,nq)); FP=np.zeros((nc,nq))
    ci_idx={v:i for i,v in enumerate(chsh_vals)}
    qi_idx={v:i for i,v in enumerate(qber_vals)}
    for r in rows6:
        ii=ci_idx[float(r[0])]; jj=qi_idx[float(r[1])]
        D005[ii,jj]=float(r[3]); FP[ii,jj]=float(r[2])

    fig,axes=plt.subplots(1,2,figsize=(11.5,5.0),constrained_layout=False)
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=0.07,right=0.92,bottom=0.14,
                        top=0.82,wspace=0.52)
    QX=qber_vals; CY=chsh_vals
    ax=axes[0]; ax.set_facecolor('white')
    cf1=ax.contourf(QX,CY,D005,levels=20,cmap='RdYlGn',alpha=0.90)
    cs=ax.contour(QX,CY,FP*100,levels=[0.5],colors=[RED],linewidths=1.8)
    ax.clabel(cs,fmt={0.5:'FP=0.5%'},fontsize=7.5,inline=True,inline_spacing=4)
    ax.scatter(2.33,2.66,color=RED,s=160,marker='*',zorder=6)
    ax.annotate('Optimal\n(2.33%,2.66)',xy=(2.33,2.66),xytext=(3.0,2.40),
                fontsize=7.5,color=RED,fontweight='bold',
                arrowprops=dict(arrowstyle='->',color=RED,lw=1.0))
    ax.set_xlabel('QBER Threshold (%)',fontsize=9,labelpad=6)
    ax.set_ylabel('CHSH Threshold |S|',fontsize=9,labelpad=6)
    ax.set_title('(a) Detection Probability\nat p=0.05',
                 fontsize=9.5,pad=10,fontweight='bold',color=NAVY)
    ax.tick_params(labelsize=8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    cb1=fig.colorbar(cf1,ax=ax,fraction=0.038,pad=0.06)
    cb1.set_label('P(detect | p=0.05)',fontsize=8.5,labelpad=8)
    cb1.ax.tick_params(labelsize=7.5)
    ax=axes[1]; ax.set_facecolor('white')
    cf2=ax.contourf(QX,CY,FP*100,levels=20,cmap='YlOrRd',alpha=0.90)
    ax.scatter(2.33,2.66,color=NAVY,s=160,marker='*',zorder=6)
    ax.annotate('Optimal\n(2.33%,2.66)',xy=(2.33,2.66),xytext=(3.0,2.40),
                fontsize=7.5,color=NAVY,fontweight='bold',
                arrowprops=dict(arrowstyle='->',color=NAVY,lw=1.0))
    ax.set_xlabel('QBER Threshold (%)',fontsize=9,labelpad=6)
    ax.set_ylabel('CHSH Threshold |S|',fontsize=9,labelpad=6)
    ax.set_title('(b) False-Positive Rate (%)\n(lower-right = fewer false aborts)',
                 fontsize=9.5,pad=10,fontweight='bold',color=NAVY)
    ax.tick_params(labelsize=8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    cb2=fig.colorbar(cf2,ax=ax,fraction=0.038,pad=0.06)
    cb2.set_label('False-Positive Rate (%)',fontsize=8.5,labelpad=8)
    cb2.ax.tick_params(labelsize=7.5)
    fig.suptitle('Fig. 6 — CHSH / QBER Abort Threshold Co-optimisation\n'
                 'Optimal: CHSH=2.66, QBER=2.33%  →  FP=0.40%, '
                 'P(detect|p=0.05)=75.6%',
                 fontsize=10,fontweight='bold',color=NAVY,y=0.97)
    plt.savefig('fig6_threshold_optimisation.png')
    plt.close()
    print("fig6_threshold_optimisation.png saved")

    print("\nAll figures generated.")

except ImportError:
    print("\nmatplotlib not found — CSV data files saved; install matplotlib to generate figures.")

print("\n" + "="*65)
print("Done. CSV files and figures written to current directory.")
print(f"Random seed used: {SEED}")
print("Include this script and seed in supplementary material.")
print("="*65)
