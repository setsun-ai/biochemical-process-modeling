from __future__ import annotations
import sys, json, math, time, shutil, os
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp

# Folder with the lecturer-provided lab programs (not redistributed in this repo).
ROOT = Path(os.environ.get('MPB_LABS_DIR', 'Laby'))
LAB2 = next(p for p in ROOT.iterdir() if p.is_dir() and 'Laboratorium 2' in p.name)
LAB3 = next(p for p in ROOT.iterdir() if p.is_dir() and 'Laboratorium 3' in p.name)
LAB4 = next(p for p in ROOT.iterdir() if p.is_dir() and 'Laboratorium 4' in p.name)
OUT = Path(os.environ.get('MPB_OUT_DIR', 'results'))
FIG = OUT/'figures'
DATA = OUT/'data'
TABLES = OUT/'tables'
LOGS = OUT/'logs'
for d in [OUT, FIG, DATA, TABLES, LOGS]: d.mkdir(parents=True, exist_ok=True)

# Clean old outputs only when requested. This allows staged runs without deleting partial results.
if os.environ.get('MPB_CLEAR', '1') == '1':
    for p in list(FIG.glob('*')) + list(DATA.glob('*')) + list(TABLES.glob('*')) + list(LOGS.glob('*')):
        if p.is_file(): p.unlink()
        elif p.is_dir(): shutil.rmtree(p)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()
    return str(path)


def ci95(values):
    arr = np.asarray(values, dtype=float)
    return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]


def summarize_array(values):
    arr = np.asarray(values, dtype=float)
    return {
        'n': int(arr.size),
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'sd': float(np.std(arr, ddof=1)) if arr.size > 1 else float('nan'),
        'ci95': ci95(arr) if arr.size > 1 else [float('nan'), float('nan')]
    }


def fit_ac(mod, csv_path: Path, n_boot=300):
    t_exp, abs_exp = mod.load_experimental_data(csv_path)
    result = least_squares(mod.residuals, x0=[0.2], args=(t_exp, abs_exp, mod.A0, mod.C0), bounds=(0, np.inf))
    k_fit = float(result.x[0])
    abs_fit_exp = mod.simulate_absorbance_C(t_exp, k_fit, mod.A0, mod.C0)
    res = abs_exp - abs_fit_exp
    sse = float(np.sum(res**2))
    rng = np.random.default_rng(12345)
    k_boot = []
    for _ in range(n_boot):
        res_sample = rng.choice(res, size=len(res), replace=True)
        abs_boot = np.clip(abs_fit_exp + res_sample, 0, None)
        try:
            rb = least_squares(mod.residuals, x0=[k_fit], args=(t_exp, abs_boot, mod.A0, mod.C0), bounds=(0, np.inf))
            k_boot.append(float(rb.x[0]))
        except Exception:
            pass
    k_boot = np.asarray(k_boot, dtype=float)
    t_plot = np.linspace(float(t_exp[0]), float(t_exp[-1]), 400)
    A_fit, C_fit = mod.simulate_A_to_C(t_plot, k_fit, mod.A0, mod.C0)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.scatter(t_exp, abs_exp, label='Dane Abs(C)', zorder=3)
    ax.plot(t_plot, C_fit, label=f'Model C(t), k={k_fit:.4f} min⁻¹')
    ax.plot(t_plot, A_fit, linestyle='--', label='A(t)')
    ax.set_xlabel('t [min]')
    ax.set_ylabel('Absorbancja / stężenie [j.u.]')
    ax.set_title('Ćw. 2: dopasowanie A → C')
    ax.legend(); ax.grid(True, alpha=0.3)
    fit_fig = savefig(FIG/'lab2_AC_fit.png')
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.axhline(0, linestyle='--')
    ax.scatter(t_exp, res)
    ax.set_xlabel('t [min]'); ax.set_ylabel('reszta')
    ax.set_title('Ćw. 2: reszty dopasowania A → C')
    ax.grid(True, alpha=0.3)
    res_fig = savefig(FIG/'lab2_AC_residuals.png')
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.hist(k_boot, bins=30, edgecolor='black')
    lo,hi = ci95(k_boot)
    ax.axvline(k_fit, linestyle='--', label=f'fit={k_fit:.4f}')
    ax.axvline(lo, linestyle=':', label=f'2.5%={lo:.4f}')
    ax.axvline(hi, linestyle=':', label=f'97.5%={hi:.4f}')
    ax.set_xlabel('k [min⁻¹]'); ax.set_ylabel('liczność')
    ax.set_title('Ćw. 2: bootstrap parametru k')
    ax.legend(); ax.grid(True, alpha=0.3)
    boot_fig = savefig(FIG/'lab2_AC_bootstrap_k.png')
    return {'k_fit': k_fit, 'SSE': sse, 'bootstrap': summarize_array(k_boot), 'figures': [fit_fig, res_fig, boot_fig], 'residuals': res.tolist()}


def fit_abc_C(mod, csv_path: Path, n_boot=300):
    t_exp, abs_exp = mod.load_experimental_data(csv_path)
    result = mod.fit_multistart(t_exp, abs_exp)
    k1_fit, k2_fit = map(float, result.x)
    abs_fit_exp = mod.simulate_absorbance_C(t_exp, k1_fit, k2_fit, mod.A0, mod.B0, mod.C0)
    res = abs_exp - abs_fit_exp
    sse = float(np.sum(res**2))
    rng = np.random.default_rng(12345)
    boots=[]
    for _ in range(n_boot):
        res_sample = rng.choice(res, size=len(res), replace=True)
        abs_boot = np.clip(abs_fit_exp + res_sample, 0, None)
        try:
            rb = least_squares(mod.residuals, x0=[k1_fit, k2_fit], args=(t_exp, abs_boot, mod.A0, mod.B0, mod.C0), bounds=(0, np.inf))
            boots.append(rb.x)
        except Exception:
            pass
    boots=np.asarray(boots,float)
    t_plot=np.linspace(float(t_exp[0]), float(t_exp[-1]), 500)
    A,B,C=mod.simulate_A_to_B_to_C(t_plot,k1_fit,k2_fit,mod.A0,mod.B0,mod.C0)
    fig, ax=plt.subplots(figsize=(7.5,4.6))
    ax.scatter(t_exp, abs_exp, label='Dane Abs(C)', zorder=3)
    ax.plot(t_plot, C, label=f'Model C(t); k1={k1_fit:.4f}, k2={k2_fit:.4f}')
    ax.plot(t_plot, A, linestyle='--', label='A(t)')
    ax.plot(t_plot, B, linestyle='--', label='B(t)')
    ax.set_xlabel('t [min]'); ax.set_ylabel('Absorbancja / stężenie [j.u.]')
    ax.set_title('Ćw. 2: dopasowanie A → B → C do C(t)')
    ax.legend(); ax.grid(True, alpha=0.3)
    fit_fig=savefig(FIG/'lab2_ABC_C_fit.png')
    fig, ax=plt.subplots(figsize=(6,6))
    ax.scatter(boots[:,0], boots[:,1], alpha=0.45, s=14)
    ax.scatter([k1_fit],[k2_fit],marker='x',s=100,label='fit')
    ax.set_xlabel('k1 [min⁻¹]'); ax.set_ylabel('k2 [min⁻¹]')
    ax.set_title('Ćw. 2: bootstrap k1-k2 dla C(t)')
    ax.legend(); ax.grid(True, alpha=0.3)
    boot_fig=savefig(FIG/'lab2_ABC_C_bootstrap_scatter.png')
    return {'k1_fit':k1_fit,'k2_fit':k2_fit,'k_slow':min(k1_fit,k2_fit),'k_fast':max(k1_fit,k2_fit),'SSE':sse,
            'bootstrap': {'k1': summarize_array(boots[:,0]), 'k2': summarize_array(boots[:,1])}, 'figures':[fit_fig,boot_fig]}


def fit_abc_B(mod, csv_path: Path, n_boot=300, name_prefix='lab2_ABC_B'):
    t_exp, abs_exp = mod.load_experimental_data(csv_path)
    result = least_squares(mod.residuals, x0=[0.3,0.1], args=(t_exp, abs_exp, mod.A0, mod.B0, mod.C0), bounds=(0, np.inf))
    k1_fit,k2_fit=map(float,result.x)
    abs_fit_exp=mod.simulate_absorbance_B(t_exp,k1_fit,k2_fit,mod.A0,mod.B0,mod.C0)
    res=abs_exp-abs_fit_exp
    sse=float(np.sum(res**2))
    rng=np.random.default_rng(12345)
    boots=[]
    for _ in range(n_boot):
        abs_boot=np.clip(abs_fit_exp+rng.choice(res,size=len(res),replace=True),0,None)
        try:
            rb=least_squares(mod.residuals,x0=[k1_fit,k2_fit],args=(t_exp,abs_boot,mod.A0,mod.B0,mod.C0),bounds=(0,np.inf))
            boots.append(rb.x)
        except Exception: pass
    boots=np.asarray(boots,float)
    t_plot=np.linspace(float(t_exp[0]),float(t_exp[-1]),500)
    A,B,C=mod.simulate_A_to_B_to_C(t_plot,k1_fit,k2_fit,mod.A0,mod.B0,mod.C0)
    fig, ax=plt.subplots(figsize=(7.5,4.6))
    ax.scatter(t_exp,abs_exp,label='Dane Abs(B)',zorder=3)
    ax.plot(t_plot,B,label=f'Model B(t); k1={k1_fit:.4f}, k2={k2_fit:.4f}')
    ax.plot(t_plot,A,linestyle='--',label='A(t)')
    ax.plot(t_plot,C,linestyle='--',label='C(t)')
    ax.set_xlabel('t [min]'); ax.set_ylabel('Absorbancja / stężenie [j.u.]')
    ax.set_title('Ćw. 2: dopasowanie A → B → C do B(t)')
    ax.legend(); ax.grid(True,alpha=0.3)
    fit_fig=savefig(FIG/f'{name_prefix}_fit.png')
    fig,ax=plt.subplots(figsize=(6,6))
    ax.scatter(boots[:,0],boots[:,1],alpha=0.45,s=14)
    ax.scatter([k1_fit],[k2_fit],marker='x',s=100,label='fit')
    ax.set_xlabel('k1 [min⁻¹]'); ax.set_ylabel('k2 [min⁻¹]')
    ax.set_title('Ćw. 2: bootstrap k1-k2 dla B(t)')
    ax.legend(); ax.grid(True,alpha=0.3)
    boot_fig=savefig(FIG/f'{name_prefix}_bootstrap_scatter.png')
    return {'k1_fit':k1_fit,'k2_fit':k2_fit,'SSE':sse,'bootstrap':{'k1':summarize_array(boots[:,0]),'k2':summarize_array(boots[:,1])},'figures':[fit_fig,boot_fig]}


def model_compare_analysis(mod, csv_path: Path, label: str, n_boot=60):
    """Fast analytical version of the model-comparison script.
    The original teaching script solves the same ODEs numerically inside every
    least-squares call. Here the closed-form solutions of the same ODEs are used
    to make residual bootstrap feasible; the fitted models, AIC/BIC formulas and
    decision logic are unchanged.
    """
    t_exp, C_exp = mod.load_experimental_data(csv_path)
    t_exp = np.asarray(t_exp, dtype=float)
    C_exp = np.asarray(C_exp, dtype=float)
    A0 = float(getattr(mod, 'A0', 1.0)); B0 = float(getattr(mod, 'B0', 0.0)); C0 = float(getattr(mod, 'C0', 0.0))
    kmax = float(getattr(mod, 'K_MAX', 50.0)) if hasattr(mod, 'K_MAX') else 50.0
    upper1 = [kmax]
    upper2 = [kmax, kmax]

    def c_one(t, k):
        return C0 + A0 * (1.0 - np.exp(-k * t))

    def c_two(t, k1, k2):
        A = A0 * np.exp(-k1 * t)
        if abs(k2-k1) < 1e-8:
            B = A0 * k1 * t * np.exp(-k1*t)
        else:
            B = A0 * k1 / (k2-k1) * (np.exp(-k1*t) - np.exp(-k2*t))
        return C0 + A0 + B0 - A - B

    def calc_aic_bic(n, sse, k_params):
        sse = max(float(sse), 1e-15)
        return float(n*np.log(sse/n)+2*k_params), float(n*np.log(sse/n)+k_params*np.log(n))

    def choose(AIC_one, BIC_one, SSE_one, AIC_two, BIC_two, SSE_two):
        if (AIC_two < AIC_one) and (BIC_two < BIC_one):
            return 'dwuetapowy (A -> B -> C)'
        if (AIC_one < AIC_two) and (BIC_one < BIC_two):
            return 'jednoetapowy (A -> C)'
        return 'nierozstrzygnięty'

    def fit_models_fast(y):
        r1 = least_squares(lambda x: y - c_one(t_exp, x[0]), x0=[0.2], bounds=([0], upper1), max_nfev=250)
        k = float(r1.x[0]); C1 = c_one(t_exp, k); res1 = y-C1; sse1=float(np.sum(res1**2))
        best = None
        starts = ([0.2,0.2],[0.5,0.1],[0.1,0.5])
        for s in starts:
            try:
                r2 = least_squares(lambda x: y - c_two(t_exp, x[0], x[1]), x0=s, bounds=([0,0], upper2), max_nfev=300)
                ss=float(np.sum(r2.fun**2))
                if best is None or ss < best[0]: best=(ss,r2)
            except Exception:
                pass
        if best is None:
            r2 = least_squares(lambda x: y - c_two(t_exp, x[0], x[1]), x0=[0.2,0.2], bounds=([0,0], upper2), max_nfev=300)
        else:
            r2=best[1]
        k1,k2=map(float,r2.x); C2=c_two(t_exp,k1,k2); res2=y-C2; sse2=float(np.sum(res2**2))
        n=len(t_exp); aic1,bic1=calc_aic_bic(n,sse1,1); aic2,bic2=calc_aic_bic(n,sse2,2)
        selected=choose(aic1,bic1,sse1,aic2,bic2,sse2)
        return {'k_fit':k,'k1_fit':k1,'k2_fit':k2,'SSE_one':sse1,'SSE_two':sse2,
                'AIC_one':aic1,'BIC_one':bic1,'AIC_two':aic2,'BIC_two':bic2,
                'res_C_one':res1,'res_C_two':res2,'C_one_exp':C1,'C_two_exp':C2,
                'selected_model':selected,'verdict_color':'','explanation':''}

    fit=fit_models_fast(C_exp)
    selected=fit['selected_model']
    if selected == 'jednoetapowy (A -> C)':
        C_base=fit['C_one_exp']; res_base=fit['res_C_one']; base='one_step'
    else:
        C_base=fit['C_two_exp']; res_base=fit['res_C_two']; base='two_step'
    rng=np.random.default_rng(getattr(mod,'BOOTSTRAP_SEED',54321))
    selected_in=[]; k_boot=[]; k1_boot=[]; k2_boot=[]
    for _ in range(n_boot):
        C_boot=np.clip(C_base+rng.choice(res_base,size=len(res_base),replace=True),0,None)
        try:
            fb=fit_models_fast(C_boot)
            k_boot.append(fb['k_fit']); k1_boot.append(fb['k1_fit']); k2_boot.append(fb['k2_fit'])
            if fb['selected_model']=='jednoetapowy (A -> C)': selected_in.append('one_step')
            elif fb['selected_model']=='dwuetapowy (A -> B -> C)': selected_in.append('two_step')
            else: selected_in.append('undecided')
        except Exception:
            pass
    selected_in=np.asarray(selected_in)
    n_one=int(np.sum(selected_in=='one_step')); n_two=int(np.sum(selected_in=='two_step')); n_und=int(np.sum(selected_in=='undecided'))
    t_plot=np.linspace(float(t_exp[0]),float(t_exp[-1]),500)
    C_one=c_one(t_plot,fit['k_fit'])
    C_two=c_two(t_plot,fit['k1_fit'],fit['k2_fit'])
    fig, ax=plt.subplots(figsize=(7.5,4.6))
    ax.scatter(t_exp,C_exp,label='Dane Abs(C)',zorder=3)
    ax.plot(t_plot,C_one,label=f'A→C, k={fit["k_fit"]:.4f}')
    ax.plot(t_plot,C_two,label=f'A→B→C, k1={fit["k1_fit"]:.3g}, k2={fit["k2_fit"]:.3g}')
    ax.set_xlabel('t [min]'); ax.set_ylabel('Abs(C) [j.u.]')
    kmax_text=f"K_MAX={getattr(mod,'K_MAX','brak')}" if hasattr(mod,'K_MAX') else 'bez K_MAX'
    ax.set_title(f'Ćw. 2: porównanie modeli dla {label} ({kmax_text})')
    ax.legend(); ax.grid(True,alpha=0.3)
    fit_fig=savefig(FIG/f'lab2_compare_{label}_{"kmax" if hasattr(mod,"K_MAX") else "free"}.png')
    fig, ax=plt.subplots(figsize=(6.8,4.0))
    bars=ax.bar(['A→C','A→B→C','nierozstrz.'],[n_one,n_two,n_und])
    for bar,count in zip(bars,[n_one,n_two,n_und]):
        ax.text(bar.get_x()+bar.get_width()/2, count, str(count), ha='center', va='bottom')
    ax.set_ylabel('liczba replik')
    ax.set_title(f'Ćw. 2: wybór modelu w bootstrapie ({label}, {kmax_text})')
    ax.grid(True, axis='y', alpha=0.3)
    boot_fig=savefig(FIG/f'lab2_compare_{label}_{"kmax" if hasattr(mod,"K_MAX") else "free"}_bootchoice.png')
    return {'fit':fit, 'base_model':base, 'bootstrap_counts':{'one_step':n_one,'two_step':n_two,'undecided':n_und,'n':int(len(selected_in))}, 'bootstrap':{'k':summarize_array(k_boot) if len(k_boot)>1 else {},'k1':summarize_array(k1_boot) if len(k1_boot)>1 else {},'k2':summarize_array(k2_boot) if len(k2_boot)>1 else {}}, 'figures':[fit_fig,boot_fig]}

def cstr_ap_fit(mod, csv_path: Path, n_boot=300):
    t_exp, abs_exp = mod.load_experimental_data(csv_path)
    result=least_squares(mod.residuals,x0=[0.3],args=(t_exp,abs_exp),bounds=([0],[mod.K_MAX]))
    k_fit=float(result.x[0])
    abs_fit_exp=mod.simulate_absorbance_P(t_exp,k_fit)
    res=abs_exp-abs_fit_exp
    sse=float(np.sum(res**2))
    rng=np.random.default_rng(12345); boots=[]
    for _ in range(n_boot):
        abs_boot=np.clip(abs_fit_exp+rng.choice(res,size=len(res),replace=True),0,None)
        try:
            rb=least_squares(mod.residuals,x0=[k_fit],args=(t_exp,abs_boot),bounds=([0],[mod.K_MAX]))
            boots.append(float(rb.x[0]))
        except Exception: pass
    boots=np.asarray(boots,float)
    t_plot=np.linspace(float(t_exp[0]),float(t_exp[-1]),500)
    CA,CP=mod.simulate_cstr_A_to_P(t_plot,k_fit)
    fig,ax=plt.subplots(figsize=(7.5,4.6))
    ax.scatter(t_exp,abs_exp,label='Dane Abs(P)',zorder=3)
    ax.plot(t_plot,CP,label=f'Model P(t), k={k_fit:.4f} h⁻¹')
    ax.plot(t_plot,CA,linestyle='--',label='C_A(t)')
    ax.set_xlabel('t [h]'); ax.set_ylabel('Absorbancja / stężenie [j.u.]')
    ax.set_title('Ćw. 3: CSTR A → P, dopasowanie k')
    ax.legend(); ax.grid(True,alpha=0.3)
    fit_fig=savefig(FIG/'lab3_CSTR_AP_fit.png')
    return {'k_fit':k_fit,'SSE':sse,'bootstrap':summarize_array(boots),'figures':[fit_fig]}


def cstr_abc_fit(mod, csv_path: Path, measure='B', n_boot=200):
    """Fit CSTR A->B->C using the analytical solution of the same linear ODE system.
    This replaces repeated numerical solve_ivp calls only for speed; the model is unchanged.
    """
    t_exp, abs_exp=mod.load_experimental_data(csv_path)
    t_exp=np.asarray(t_exp,dtype=float); abs_exp=np.asarray(abs_exp,dtype=float)
    D=float(mod.D); CAin=float(mod.C_A_in_value); CA0=float(mod.C_A0); CB0=float(mod.C_B0); CC0=float(mod.C_C0)
    KMAX=float(mod.K_MAX)

    def simulate_fast(t,k1,k2):
        t=np.asarray(t,dtype=float)
        a=D+k1; b=D+k2
        CA_ss=D*CAin/a if a>0 else CAin
        CA=CA_ss+(CA0-CA_ss)*np.exp(-a*t)
        CB_ss=k1*CA_ss/b if b>0 else 0.0
        if abs(b-a)<1e-8:
            conv=k1*(CA0-CA_ss)*t*np.exp(-a*t)
        else:
            conv=k1*(CA0-CA_ss)*(np.exp(-a*t)-np.exp(-b*t))/(b-a)
        CB=CB_ss+(CB0-CB_ss)*np.exp(-b*t)+conv
        T0=CA0+CB0+CC0
        T=CAin+(T0-CAin)*np.exp(-D*t)
        CC=T-CA-CB
        return CA,CB,CC

    def pred(t,k1,k2):
        CA,CB,CC=simulate_fast(t,k1,k2)
        return CB if measure=='B' else CC
    def resid(x,y):
        return y-pred(t_exp,x[0],x[1])

    starts=([0.3,0.1],[0.4,0.15],[0.1,0.4],[0.8,0.08])
    best=None
    for s in starts:
        try:
            r=least_squares(lambda x: resid(x,abs_exp),x0=s,bounds=([0,0],[KMAX,KMAX]),max_nfev=800)
            ss=float(np.sum(r.fun**2))
            if best is None or ss<best[0]: best=(ss,r)
        except Exception:
            pass
    result=best[1]
    k1_fit,k2_fit=map(float,result.x)
    fit_exp=pred(t_exp,k1_fit,k2_fit); res=abs_exp-fit_exp; sse=float(np.sum(res**2))
    rng=np.random.default_rng(12345); boots=[]
    for _ in range(n_boot):
        yb=np.clip(fit_exp+rng.choice(res,size=len(res),replace=True),0,None)
        try:
            rb=least_squares(lambda x: yb-pred(t_exp,x[0],x[1]),x0=[k1_fit,k2_fit],bounds=([0,0],[KMAX,KMAX]),max_nfev=500)
            boots.append(rb.x)
        except Exception:
            pass
    boots=np.asarray(boots,float)
    t_plot=np.linspace(float(t_exp[0]),float(t_exp[-1]),500)
    CA,CB,CC=simulate_fast(t_plot,k1_fit,k2_fit)
    yfit=CB if measure=='B' else CC
    label=f'Abs({measure})'
    fig,ax=plt.subplots(figsize=(7.5,4.6))
    ax.scatter(t_exp,abs_exp,label=f'Dane {label}',zorder=3)
    ax.plot(t_plot,yfit,label=f'Model {label}; k1={k1_fit:.4f}, k2={k2_fit:.4f}')
    ax.plot(t_plot,CA,linestyle='--',label='C_A(t)')
    ax.plot(t_plot,CB,linestyle='--',label='C_B(t)')
    ax.plot(t_plot,CC,linestyle='--',label='C_C(t)')
    ax.set_xlabel('t [h]'); ax.set_ylabel('Absorbancja / stężenie [j.u.]')
    ax.set_title(f'Ćw. 3: CSTR A → B → C, dopasowanie do {measure}(t)')
    ax.legend(); ax.grid(True,alpha=0.3)
    fit_fig=savefig(FIG/f'lab3_CSTR_ABC_{measure}_fit.png')
    if len(boots)>1:
        fig,ax=plt.subplots(figsize=(6,6))
        ax.scatter(boots[:,0],boots[:,1],alpha=0.45,s=14)
        ax.scatter([k1_fit],[k2_fit],marker='x',s=100,label='fit')
        ax.set_xlabel('k1 [h⁻¹]'); ax.set_ylabel('k2 [h⁻¹]')
        ax.set_title(f'Ćw. 3: bootstrap k1-k2 dla {measure}(t)')
        ax.legend(); ax.grid(True,alpha=0.3)
        boot_fig=savefig(FIG/f'lab3_CSTR_ABC_{measure}_bootstrap_scatter.png')
    else:
        boot_fig=None
    return {'k1_fit':k1_fit,'k2_fit':k2_fit,'SSE':sse,'bootstrap':{'k1':summarize_array(boots[:,0]) if len(boots)>1 else {},'k2':summarize_array(boots[:,1]) if len(boots)>1 else {}},'figures':[f for f in [fit_fig,boot_fig] if f]}

def bioreactor_fit(mod, csv_path: Path, n_boot=20, prefix='lab3_bioreactor'):
    """Fit Monod CSTR bioreactor with bounded least squares and a capped iteration count.
    The ODE model and residual scaling are taken from the provided fit script.
    """
    t_exp,X_exp,S_exp,P_exp=mod.load_experimental_data(csv_path)
    t_exp=np.asarray(t_exp,dtype=float)

    def residual_vec(par, Xd, Sd, Pd):
        X_m,S_m,P_m=mod.simulate_bioreactor(t_exp,float(par[0]),float(par[1]))
        return mod.make_scaled_residuals(Xd,Sd,Pd,X_m,S_m,P_m)

    def fit_fast(Xd,Sd,Pd,x0=None):
        if x0 is None:
            x0=[mod.MU_MAX_GUESS, mod.K_S_GUESS]
        return least_squares(residual_vec,x0=x0,args=(Xd,Sd,Pd),bounds=([mod.MU_MAX_MIN,mod.K_S_MIN],[mod.MU_MAX_MAX,mod.K_S_MAX]),max_nfev=35)

    result=fit_fast(X_exp,S_exp,P_exp)
    mu_fit,KS_fit=map(float,result.x)
    X_fit_exp,S_fit_exp,P_fit_exp=mod.simulate_bioreactor(t_exp,mu_fit,KS_fit)
    scaled_res=mod.make_scaled_residuals(X_exp,S_exp,P_exp,X_fit_exp,S_fit_exp,P_fit_exp)
    sse=float(np.sum(scaled_res**2))
    resX=X_exp-X_fit_exp; resS=S_exp-S_fit_exp; resP=P_exp-P_fit_exp
    rng=np.random.default_rng(mod.BOOTSTRAP_SEED)
    mu_boot=[]; KS_boot=[]
    for _ in range(n_boot):
        Xb=np.clip(X_fit_exp+rng.choice(resX,size=len(resX),replace=True),0,None)
        Sb=np.clip(S_fit_exp+rng.choice(resS,size=len(resS),replace=True),0,None)
        Pb=np.clip(P_fit_exp+rng.choice(resP,size=len(resP),replace=True),0,None)
        try:
            rb=fit_fast(Xb,Sb,Pb,x0=[mu_fit,KS_fit])
            mu_boot.append(float(rb.x[0])); KS_boot.append(float(rb.x[1]))
        except Exception:
            pass
    mu_boot=np.asarray(mu_boot,float); KS_boot=np.asarray(KS_boot,float)
    t_plot=np.linspace(float(t_exp[0]),float(t_exp[-1]),600)
    X_fit,S_fit,P_fit=mod.simulate_bioreactor(t_plot,mu_fit,KS_fit)
    fig,axes=plt.subplots(3,1,figsize=(8.2,8.6),sharex=True)
    axes[0].scatter(t_exp,X_exp,label='X dane',zorder=3); axes[0].plot(t_plot,X_fit,label='X model')
    axes[0].set_ylabel('X [g/L]'); axes[0].legend(); axes[0].grid(True,alpha=0.3)
    axes[1].scatter(t_exp,S_exp,label='S dane',zorder=3); axes[1].plot(t_plot,S_fit,label='S model'); axes[1].axvline(mod.t_change,linestyle=':',label='zmiana S_in')
    axes[1].set_ylabel('S [g/L]'); axes[1].legend(); axes[1].grid(True,alpha=0.3)
    axes[2].scatter(t_exp,P_exp,label='P dane',zorder=3); axes[2].plot(t_plot,P_fit,label='P model'); axes[2].axvline(mod.t_change,linestyle=':')
    axes[2].set_xlabel('t [h]'); axes[2].set_ylabel('P [g/L]'); axes[2].legend(); axes[2].grid(True,alpha=0.3)
    fig.suptitle('Ćw. 3: bioreaktor CSTR, dopasowanie μmax i KS')
    fit_fig=savefig(FIG/f'{prefix}_fit_XSP.png')
    fig,ax=plt.subplots(figsize=(6,5))
    if len(mu_boot)>1:
        ax.scatter(mu_boot,KS_boot,alpha=0.55,s=20)
    ax.scatter([mu_fit],[KS_fit],marker='x',s=100,label='fit')
    ax.set_xlabel('μmax [h⁻¹]'); ax.set_ylabel('K_S [g/L]')
    ax.set_title('Ćw. 3: bootstrap μmax-KS')
    ax.legend(); ax.grid(True,alpha=0.3)
    boot_fig=savefig(FIG/f'{prefix}_bootstrap_mu_KS.png')
    Dcrit_before = mu_fit * mod.S_in_before / (KS_fit + mod.S_in_before)
    Dcrit_after = mu_fit * mod.S_in_after / (KS_fit + mod.S_in_after)
    return {'mu_max_fit':mu_fit,'K_S_fit':KS_fit,'SSE_scaled':sse,'Dcrit_before':float(Dcrit_before),'Dcrit_after':float(Dcrit_after),
            'bootstrap':{'mu_max':summarize_array(mu_boot) if len(mu_boot)>1 else {},'K_S':summarize_array(KS_boot) if len(KS_boot)>1 else {},'n':int(len(mu_boot))},'figures':[fit_fig,boot_fig]}


def run_lab2(results):
    print('Running lab2...')
    ac_meas=load_module(LAB2/'A-to-C-measure.py','lab2_ac_meas')
    abcC_meas=load_module(LAB2/'A-to-B-to-C-measure-C.py','lab2_abcC_meas')
    abcB_meas=load_module(LAB2/'A-to-B-to-C-measure-B.py','lab2_abcB_meas')
    for m in [ac_meas,abcC_meas,abcB_meas]:
        m.ERROR_MIN=0.02; m.ERROR_MAX=0.10
    times=[0,1,1.5,2,3,5,7,10,12,15,20,22,25,30]
    np.random.seed(20260201); ac_meas.clear_table(); [ac_meas.measure_abs(t) for t in times]
    ac_df=ac_meas.get_measurement_dataframe(); ac_csv=DATA/'lab2_AC.csv'; ac_df.to_csv(ac_csv,index=False)
    np.random.seed(20260202); abcC_meas.clear_table(); [abcC_meas.measure_abs(t) for t in times]
    abcC_df=abcC_meas.get_measurement_dataframe(); abcC_csv=DATA/'lab2_ABC_C.csv'; abcC_df.to_csv(abcC_csv,index=False)
    np.random.seed(20260203); abcB_meas.clear_table(); [abcB_meas.measure_abs(t) for t in times]
    abcB_df=abcB_meas.get_measurement_dataframe(); abcB_csv=DATA/'lab2_ABC_B.csv'; abcB_df.to_csv(abcB_csv,index=False)
    ac_df.to_csv(TABLES/'lab2_measurements_AC.csv',index=False)
    abcC_df.to_csv(TABLES/'lab2_measurements_ABC_C.csv',index=False)
    abcB_df.to_csv(TABLES/'lab2_measurements_ABC_B.csv',index=False)
    # plot measured data combined
    fig,ax=plt.subplots(figsize=(7.5,4.6))
    ax.plot(ac_df['t'],ac_df['abs'],marker='o',label='A→C, C(t)')
    ax.plot(abcC_df['t'],abcC_df['abs'],marker='o',label='A→B→C, C(t)')
    ax.plot(abcB_df['t'],abcB_df['abs'],marker='o',label='A→B→C, B(t)')
    ax.set_xlabel('t [min]'); ax.set_ylabel('Absorbancja [j.u.]')
    ax.set_title('Ćw. 2: wygenerowane pomiary absorbancji')
    ax.legend(); ax.grid(True,alpha=0.3)
    meas_fig=savefig(FIG/'lab2_measurements_all.png')
    ac_fit_mod=load_module(LAB2/'AC-dopasowanie-k.py','lab2_ac_fit')
    abcC_fit_mod=load_module(LAB2/'ABC-dopasowanie-k-do-C.py','lab2_abcC_fit')
    abcB_fit_mod=load_module(LAB2/'ABC-dopasowanie-k-do-B.py','lab2_abcB_fit')
    cmp_mod=load_module(LAB2/'porownanie-modeli-AC-ABC.py','lab2_cmp')
    cmp_kmax_mod=load_module(LAB2/'porownanie-modeli-AC-ABC-k-max.py','lab2_cmp_kmax')
    results['lab2']={'parameters':{'AC_k_true':float(ac_meas.k),'ABC_k1_true':float(abcC_meas.k1),'ABC_k2_true':float(abcC_meas.k2),'error_min':0.02,'error_max':0.10,'times_min':times},
                     'data_files':{'AC':str(ac_csv),'ABC_C':str(abcC_csv),'ABC_B':str(abcB_csv)},'figures':[meas_fig],
                     'measurements':{'AC':ac_df.round(6).to_dict(orient='records'),'ABC_C':abcC_df.round(6).to_dict(orient='records'),'ABC_B':abcB_df.round(6).to_dict(orient='records')}}
    results['lab2']['fit_AC']=fit_ac(ac_fit_mod,ac_csv)
    results['lab2']['fit_ABC_C']=fit_abc_C(abcC_fit_mod,abcC_csv)
    results['lab2']['fit_ABC_B']=fit_abc_B(abcB_fit_mod,abcB_csv)
    # model comparison, maybe 1000 boots
    results['lab2']['compare_AC_free']=model_compare_analysis(cmp_mod,ac_csv,'AC')
    results['lab2']['compare_ABC_C_free']=model_compare_analysis(cmp_mod,abcC_csv,'ABC_C')
    results['lab2']['compare_AC_kmax']=model_compare_analysis(cmp_kmax_mod,ac_csv,'AC')
    results['lab2']['compare_ABC_C_kmax']=model_compare_analysis(cmp_kmax_mod,abcC_csv,'ABC_C')
    print('Lab2 done')


def run_lab3(results):
    print('Running lab3...')
    # Part I steady state A->P for k variants
    tau_values=np.linspace(0.1,20,400); C_A_in=1.0
    rows=[]
    fig,ax=plt.subplots(figsize=(7.5,4.6))
    for k in [0.10,0.30,0.60]:
        XA=k*tau_values/(1+k*tau_values)
        ax.plot(tau_values,XA,label=f'k={k:.2f} h⁻¹')
        for tau in [1,5,10,20]:
            x=k*tau/(1+k*tau); rows.append({'k [1/h]':k,'tau [h]':tau,'D [1/h]':1/tau,'X_A':x,'C_A* [mol/L]':C_A_in*(1-x),'C_P* [mol/L]':C_A_in*x})
    ax.set_xlabel('τ [h]'); ax.set_ylabel('X_A [-]')
    ax.set_title('Ćw. 3: CSTR A→P, konwersja w stanie ustalonym')
    ax.legend(); ax.grid(True,alpha=0.3)
    ap_ss_fig=savefig(FIG/'lab3_AP_steady_conversion.png')
    ap_ss_df=pd.DataFrame(rows); ap_ss_df.to_csv(TABLES/'lab3_AP_steady_table.csv',index=False)
    # Dynamic A->P for D variants
    def simulate_ap_dynamic(D,k=0.30,t_end=50):
        C_A_in_before=1.0; C_A_in_after=0.5; t_change=20.0; C_P_in=0.0; C_A0=0; C_P0=0
        def C_A_in_func(t): return C_A_in_before if t < t_change else C_A_in_after
        def ode(t,y):
            CA,CP=y; return [D*(C_A_in_func(t)-CA)-k*CA, D*(C_P_in-CP)+k*CA]
        t_eval=np.linspace(0,t_end,600)
        sol=solve_ivp(ode,(0,t_end),[C_A0,C_P0],t_eval=t_eval,method='RK45')
        CA,CP=sol.y; feed=np.array([C_A_in_func(t) for t in sol.t])
        X=1-CA/feed
        return sol.t,CA,CP,X,feed
    fig,ax=plt.subplots(figsize=(7.5,4.6)); dyn_rows=[]
    for D in [0.05,0.20,0.50]:
        t,CA,CP,X,feed=simulate_ap_dynamic(D)
        ax.plot(t,CP,label=f'D={D:.2f} h⁻¹')
        # final and 95% settling after step approximate: first time after change within 5% final CP
        final=CP[-1]; after=t>=20
        idx=np.where(after & (np.abs(CP-final) <= 0.05*max(abs(final),1e-12)))[0]
        settle=float(t[idx[0]]-20) if idx.size else float('nan')
        dyn_rows.append({'D [1/h]':D,'tau [h]':1/D,'C_P(t_end) [mol/L]':float(final),'X_A(t_end)':float(X[-1]),'czas dojścia po skoku 5% [h]':settle})
    ax.axvline(20,linestyle=':',label='skok zasilania')
    ax.set_xlabel('t [h]'); ax.set_ylabel('C_P [mol/L]')
    ax.set_title('Ćw. 3: odpowiedź dynamiczna CSTR A→P')
    ax.legend(); ax.grid(True,alpha=0.3)
    ap_dyn_fig=savefig(FIG/'lab3_AP_dynamic_D_variants.png')
    ap_dyn_df=pd.DataFrame(dyn_rows); ap_dyn_df.to_csv(TABLES/'lab3_AP_dynamic_table.csv',index=False)
    # ABC steady variants
    D_values=np.linspace(0.01,1.5,800); rows=[]; fig,ax=plt.subplots(figsize=(7.5,4.6))
    for k1,k2,label in [(0.40,0.15,'k1>k2'),(0.15,0.40,'k1<k2'),(0.30,0.30,'k1≈k2')]:
        CB=k1*D_values/((D_values+k1)*(D_values+k2))
        idx=int(np.argmax(CB)); ax.plot(D_values,CB,label=f'{label}: k1={k1}, k2={k2}')
        rows.append({'wariant':label,'k1 [1/h]':k1,'k2 [1/h]':k2,'D_max num [1/h]':float(D_values[idx]),'D_max sqrt(k1*k2) [1/h]':float(math.sqrt(k1*k2)),'C_B,max* [mol/L]':float(CB[idx]),'tau przy max [h]':float(1/D_values[idx])})
    ax.set_xlabel('D [h⁻¹]'); ax.set_ylabel('C_B* [mol/L]')
    ax.set_title('Ćw. 3: maksimum produktu pośredniego B w CSTR')
    ax.legend(); ax.grid(True,alpha=0.3)
    abc_ss_fig=savefig(FIG/'lab3_ABC_steady_Bmax.png')
    abc_ss_df=pd.DataFrame(rows); abc_ss_df.to_csv(TABLES/'lab3_ABC_steady_Bmax_table.csv',index=False)
    # ABC dynamic default
    def simulate_abc_dynamic(D=0.05,k1=0.40,k2=0.15,t_end=50):
        C_A_in_before=1.0; C_A_in_after=0.5; t_change=20.0
        def CAin(t): return C_A_in_before if t<t_change else C_A_in_after
        def ode(t,y):
            CA,CB,CC=y; return [D*(CAin(t)-CA)-k1*CA, D*(0-CB)+k1*CA-k2*CB, D*(0-CC)+k2*CB]
        t_eval=np.linspace(0,t_end,700)
        sol=solve_ivp(ode,(0,t_end),[0,0,0],t_eval=t_eval)
        return sol.t,sol.y[0],sol.y[1],sol.y[2]
    t,CA,CB,CC=simulate_abc_dynamic(); idx=int(np.argmax(CB));
    fig,ax=plt.subplots(figsize=(7.5,4.6)); ax.plot(t,CA,label='C_A'); ax.plot(t,CB,label='C_B'); ax.plot(t,CC,label='C_C'); ax.axvline(20,linestyle=':',label='skok zasilania'); ax.scatter([t[idx]],[CB[idx]],zorder=3,label='max B')
    ax.set_xlabel('t [h]'); ax.set_ylabel('stężenie [mol/L]'); ax.set_title('Ćw. 3: CSTR A→B→C, przebieg dynamiczny')
    ax.legend(); ax.grid(True,alpha=0.3)
    abc_dyn_fig=savefig(FIG/'lab3_ABC_dynamic_default.png')
    abc_dyn_summary={'t_Bmax [h]':float(t[idx]),'C_Bmax [mol/L]':float(CB[idx]),'C_A_end':float(CA[-1]),'C_B_end':float(CB[-1]),'C_C_end':float(CC[-1])}
    # measurements and fits
    ap_meas=load_module(LAB3/'reaktory'/'CSTR-A-to-P-measure-P.py','lab3_ap_meas')
    abcB_meas=load_module(LAB3/'reaktory'/'CSTR-A-to-B-to-C-measure-B.py','lab3_abcB_meas')
    abcC_meas=load_module(LAB3/'reaktory'/'CSTR-A-to-B-to-C-measure-C.py','lab3_abcC_meas')
    times_ap=[0,2,4,6,8,10,15,20,30,40,50]
    times_abc=[0,2,4,6,8,10,15,20,30,40,50,60,70,80]
    np.random.seed(20260301); ap_meas.clear_table(); [ap_meas.measure_abs(t) for t in times_ap]
    ap_df=ap_meas.get_measurement_dataframe(); ap_csv=DATA/'lab3_CSTR_AP.csv'; ap_df.to_csv(ap_csv,index=False)
    np.random.seed(20260302); abcB_meas.clear_table(); [abcB_meas.measure_abs(t) for t in times_abc]
    abcB_df=abcB_meas.get_measurement_dataframe(); abcB_csv=DATA/'lab3_CSTR_ABC_B.csv'; abcB_df.to_csv(abcB_csv,index=False)
    np.random.seed(20260303); abcC_meas.clear_table(); [abcC_meas.measure_abs(t) for t in times_abc]
    abcC_df=abcC_meas.get_measurement_dataframe(); abcC_csv=DATA/'lab3_CSTR_ABC_C.csv'; abcC_df.to_csv(abcC_csv,index=False)
    ap_fit_mod=load_module(LAB3/'reaktory'/'CSTR-A-to-P-fit-k.py','lab3_ap_fit')
    abcB_fit_mod=load_module(LAB3/'reaktory'/'CSTR-A-to-B-to-C-fit-k-do-B.py','lab3_abcB_fit')
    abcC_fit_mod=load_module(LAB3/'reaktory'/'CSTR-A-to-B-to-C-fit-k-do-C.py','lab3_abcC_fit')
    ap_fit=cstr_ap_fit(ap_fit_mod,ap_csv)
    abcB_fit=cstr_abc_fit(abcB_fit_mod,abcB_csv,measure='B')
    abcC_fit=cstr_abc_fit(abcC_fit_mod,abcC_csv,measure='C')
    # bioreactor default and high error
    bio_meas=load_module(LAB3/'bioreaktory'/'bioreaktor-CSTR-measure-XSP.py','lab3_bio_meas')
    bio_fit_mod=load_module(LAB3/'bioreaktory'/'bioreaktor-CSTR-fit-mu-KS.py','lab3_bio_fit')
    times_bio=[0,2,4,6,8,10,15,20,25,30,35,40,50,60]
    np.random.seed(20260304); bio_meas.clear_table(); [bio_meas.measure_sample(t) for t in times_bio]
    bio_df=bio_meas.get_measurement_dataframe(); bio_csv=DATA/'lab3_bioreactor_XSP.csv'; bio_df.to_csv(bio_csv,index=False)
    bio_fit=bioreactor_fit(bio_fit_mod,bio_csv,n_boot=20,prefix='lab3_bioreactor_default')
    # high error scenario: change only measurement errors, not model
    bio_meas.X_ERROR_MIN=0.08; bio_meas.X_ERROR_MAX=0.15
    bio_meas.S_ERROR_MIN=0.06; bio_meas.S_ERROR_MAX=0.12
    bio_meas.P_ERROR_MIN=0.08; bio_meas.P_ERROR_MAX=0.18
    np.random.seed(20260305); bio_meas.clear_table(); [bio_meas.measure_sample(t) for t in times_bio]
    bio_hi_df=bio_meas.get_measurement_dataframe(); bio_hi_csv=DATA/'lab3_bioreactor_XSP_high_error.csv'; bio_hi_df.to_csv(bio_hi_csv,index=False)
    bio_hi_fit=bioreactor_fit(bio_fit_mod,bio_hi_csv,n_boot=0,prefix='lab3_bioreactor_high_error')
    # Washout scenarios using fit module, set D temporary
    D_orig=bio_fit_mod.D
    t_grid=np.linspace(0,60,600); fig,ax=plt.subplots(figsize=(7.5,4.6)); wash_rows=[]
    for D in [0.20,0.45,0.52]:
        bio_fit_mod.D=D
        X,S,P=bio_fit_mod.simulate_bioreactor(t_grid,bio_meas.mu_max,bio_meas.K_S)
        ax.plot(t_grid,X,label=f'D={D:.2f} h⁻¹')
        wash_rows.append({'D [1/h]':D,'X_end [g/L]':float(X[-1]),'S_end [g/L]':float(S[-1]),'P_end [g/L]':float(P[-1]),'Dcrit_before_true [1/h]':float(bio_meas.mu_max*bio_fit_mod.S_in_before/(bio_meas.K_S+bio_fit_mod.S_in_before))})
    bio_fit_mod.D=D_orig
    ax.set_xlabel('t [h]'); ax.set_ylabel('X [g/L]'); ax.set_title('Ćw. 3: wpływ D na ryzyko washoutu')
    ax.legend(); ax.grid(True,alpha=0.3)
    wash_fig=savefig(FIG/'lab3_bioreactor_washout_D_variants.png')
    wash_df=pd.DataFrame(wash_rows); wash_df.to_csv(TABLES/'lab3_bioreactor_washout_table.csv',index=False)
    results['lab3']={'parameters':{'CSTR_AP_true_k':float(ap_meas.k),'CSTR_AP_D':float(ap_meas.D),'CSTR_ABC_k1':float(abcB_meas.k1),'CSTR_ABC_k2':float(abcB_meas.k2),'CSTR_ABC_D':float(abcB_meas.D),'bio_mu_max':float(bio_meas.mu_max),'bio_K_S':float(bio_meas.K_S),'bio_D':0.20},
                     'part1':{'ap_steady_table':ap_ss_df.round(5).to_dict(orient='records'),'ap_dynamic_table':ap_dyn_df.round(5).to_dict(orient='records'),'abc_steady_Bmax_table':abc_ss_df.round(5).to_dict(orient='records'),'abc_dynamic_summary':abc_dyn_summary},
                     'data_files':{'CSTR_AP':str(ap_csv),'CSTR_ABC_B':str(abcB_csv),'CSTR_ABC_C':str(abcC_csv),'bioreactor':str(bio_csv),'bioreactor_high_error':str(bio_hi_csv)},
                     'fit_CSTR_AP':ap_fit,'fit_CSTR_ABC_B':abcB_fit,'fit_CSTR_ABC_C':abcC_fit,'fit_bioreactor':bio_fit,'fit_bioreactor_high_error':bio_hi_fit,
                     'figures':[ap_ss_fig,ap_dyn_fig,abc_ss_fig,abc_dyn_fig,wash_fig]}
    print('Lab3 done')


def run_lab4(results):
    print('Running lab4...')
    sys.path.insert(0, str(LAB4))
    import main as m
    from pca_module import run_pca_workflow, plot_scree, plot_geometry_map, plot_pc23_map
    from triangle_3form import run_triangle_3form_model, plot_3form_fractions, plot_3form_geometry
    from tetrahedron_4form import run_tetrahedron_4form_model, plot_4form_fractions, plot_4form_geometry_static
    from general_simplex import run_general_simplex_model, plot_general_simplex_fractions, plot_general_simplex_geometry_if_possible
    from pka_fit import fit_pka_model, plot_pka_fit, save_pka_report_txt
    assignments={'example-Ledakrin.csv':3,'example-Symadex.csv':4,'example-C-2045.csv':5}
    init_pkas={3:np.array([4.5,8.0]),4:np.array([3.5,6.5,9.0]),5:np.array([2.5,5.0,7.5,9.5])}
    pca_rows=[]; pka_rows=[]; figures=[]; file_results={}
    for name,nforms in assignments.items():
        print('Lab4 file',name,nforms)
        csv=LAB4/'data'/name
        df, variable_labels, sample_labels=m.load_spectral_csv(csv)
        raw_data=df.to_numpy(dtype=float)
        raw_data, variable_labels=m.ensure_ascending_wavelengths(raw_data, variable_labels)
        resolution,start,end=m.get_default_spectral_window(variable_labels)
        probe, vl=m.crop_and_resample_probe(raw_data, variable_labels, start, end, resolution)
        probe=m.center_probe(probe)
        pca=run_pca_workflow(probe, sample_labels, [str(v) for v in vl], npcs_selected=nforms, max_rank=6)
        evr=pca.pca.explained_variance_ratio
        pca_rows.append({'plik':name,'liczba form':nforms,'PC1 [%]':float(evr[0]*100),'PC2 [%]':float(evr[1]*100),'PC3 [%]':float(evr[2]*100),'PC4 [%]':float(evr[3]*100) if len(evr)>3 else float('nan'),'suma wybranych PC [%]':float(np.sum(evr[:nforms])*100)})
        # scree
        scree_path=FIG/f'lab4_{Path(name).stem}_scree.png'
        plot_scree(pca.pca,scree_path,dpi=180,show=False); figures.append(str(scree_path))
        # geometry/fractions/decomp
        if nforms==3:
            result=run_triangle_3form_model(pca.geometry.geometry_scores[:,:2],sample_labels,clamp=True,n_starts=3,random_scale=0.05,seed=123,lambda_outside=1e4,lambda_degenerate=1e6,min_area=1e-8,maxiter=1000)
            frac_path=DATA/f'lab4_{Path(name).stem}_fractions_3form.csv'; result.to_dataframe().to_csv(frac_path,index=True)
            frac_fig=FIG/f'lab4_{Path(name).stem}_fractions_3form.png'; plot_3form_fractions(result,frac_fig,dpi=180,show=False); figures.append(str(frac_fig))
            geom_fig=FIG/f'lab4_{Path(name).stem}_geometry_3form.png'; plot_3form_geometry(result,geom_fig,dpi=180,show=False); figures.append(str(geom_fig))
        elif nforms==4:
            result=run_tetrahedron_4form_model(pca.geometry.geometry_scores[:,:3],sample_labels,clamp=True,n_starts=3,random_scale=0.05,seed=123,lambda_outside=1e4,lambda_degenerate=1e6,min_volume=1e-10,maxiter=2000)
            frac_path=DATA/f'lab4_{Path(name).stem}_fractions_4form.csv'; result.to_dataframe().to_csv(frac_path,index=True)
            frac_fig=FIG/f'lab4_{Path(name).stem}_fractions_4form.png'; plot_4form_fractions(result,frac_fig,dpi=180,show=False); figures.append(str(frac_fig))
            geom_fig=FIG/f'lab4_{Path(name).stem}_geometry_4form_static.png'; plot_4form_geometry_static(result,geom_fig,dpi=180,show=False); figures.append(str(geom_fig))
        else:
            # default maxiter=20000 can be slow; reduced to 3000 without changing model equations
            result=run_general_simplex_model(pca.geometry.geometry_scores[:,:nforms-1],sample_labels,clamp=True,n_starts=1,random_scale=0.05,seed=123,lambda_outside=1e4,lambda_degenerate=1e6,min_volume=1e-12,maxiter=1000)
            frac_path=DATA/f'lab4_{Path(name).stem}_fractions_5form.csv'; result.to_dataframe().to_csv(frac_path,index=True)
            frac_fig=FIG/f'lab4_{Path(name).stem}_fractions_5form.png'; plot_general_simplex_fractions(result,frac_fig,dpi=180,show=False); figures.append(str(frac_fig))
            geom_path=FIG/f'lab4_{Path(name).stem}_geometry_5form.txt'; plot_general_simplex_geometry_if_possible(result,geom_path,dpi=180,show=False); figures.append(str(geom_path))
        frac_df=result.to_dataframe()
        frac_cols=[c for c in frac_df.columns if str(c).startswith('Form_')]
        pH=np.array([float(str(x).replace('X','').replace(',','.')) for x in frac_df.index])
        fit=fit_pka_model(frac_df[frac_cols].to_numpy(float),pH,init_pkas[nforms],labels=frac_cols,reorder_by_maxima=True,do_loo=True)
        pka_plot=FIG/f'lab4_{Path(name).stem}_pka_fit.png'
        plot_pka_fit(fit,pka_plot,dpi=180,show=False)
        figures.append(str(pka_plot))
        report_path=LOGS/f'lab4_{Path(name).stem}_pka_report.txt'
        save_pka_report_txt(fit,report_path,initial_pKas=init_pkas[nforms])
        # save combined table
        fit.to_dataframe().to_csv(DATA/f'lab4_{Path(name).stem}_pka_table.csv')
        row={'plik':name,'liczba form':nforms,'SSE pKa':float(fit.sse),'optymalizacja OK':bool(fit.success)}
        for i,val in enumerate(fit.optimized_pKas,1): row[f'pKa{i}']=float(val)
        if fit.loo_mean_pKas is not None:
            for i,(mval,sdval) in enumerate(zip(fit.loo_mean_pKas,fit.loo_sd_pKas),1):
                row[f'LOO mean pKa{i}']=float(mval); row[f'LOO SD pKa{i}']=float(sdval)
        pka_rows.append(row)
        frac_values=frac_df[frac_cols].to_numpy(float)
        file_results[name]={'n_forms':nforms,'initial_pKas':init_pkas[nforms].tolist(),'pKas':fit.optimized_pKas.tolist(),'pka_sse':float(fit.sse),'loo_mean':None if fit.loo_mean_pKas is None else fit.loo_mean_pKas.tolist(),'loo_sd':None if fit.loo_sd_pKas is None else fit.loo_sd_pKas.tolist(),'evr_first6':evr[:6].tolist(),'fraction_min':float(np.min(frac_values)),'fraction_max':float(np.max(frac_values)),'fraction_sum_min':float(np.min(np.sum(frac_values,axis=1))),'fraction_sum_max':float(np.max(np.sum(frac_values,axis=1))),'n_clamped':int(np.sum(getattr(result,'clamped_mask',np.zeros(len(sample_labels),dtype=bool)))),'decomposition_success':bool(getattr(result,'success',True)),'decomposition_message':str(getattr(result,'message','')),'fractions_csv':str(frac_path),'pka_plot':str(pka_plot),'pka_report':str(report_path)}
    pca_df=pd.DataFrame(pca_rows); pca_df.to_csv(TABLES/'lab4_pca_summary.csv',index=False)
    pka_df=pd.DataFrame(pka_rows); pka_df.to_csv(TABLES/'lab4_pka_summary.csv',index=False)
    # composite pca scree with EVR bars for report
    fig,axes=plt.subplots(1,3,figsize=(12,3.5),sharey=False)
    for ax,row in zip(axes,pca_rows):
        vals=[row['PC1 [%]'],row['PC2 [%]'],row['PC3 [%]'],row['PC4 [%]']]
        ax.bar(['PC1','PC2','PC3','PC4'],vals)
        ax.set_title(row['plik'].replace('example-','').replace('.csv',''))
        ax.set_ylabel('wariancja [%]')
    fig.suptitle('Ćw. 4: udział wariancji pierwszych składowych PCA')
    comp_fig=savefig(FIG/'lab4_pca_evr_summary.png'); figures.append(comp_fig)
    results['lab4']={'assignments':assignments,'pca_summary':pca_df.round(6).to_dict(orient='records'),'pka_summary':pka_df.round(6).replace({np.nan:None}).to_dict(orient='records'),'files':file_results,'figures':figures,'technical_notes':['Dla pliku 5-formowego użyto n_starts=1 i maxiter=1000 w funkcji general_simplex, ponieważ domyślne 20000 było zbyt kosztowne czasowo; nie zmieniono modelu geometrycznego ani funkcji celu.']}
    print('Lab4 done')


def main():
    t0=time.time()
    results={'source_paths':{'zip':'/mnt/data/Modelowanie P. B.zip','lab2':str(LAB2),'lab3':str(LAB3),'lab4':str(LAB4)},'generated_at':'2026-06-02'}
    run_lab2(results)
    run_lab3(results)
    run_lab4(results)
    def _json_default(o):
        import numpy as _np
        if isinstance(o, _np.ndarray): return o.tolist()
        if isinstance(o, (_np.floating,)): return float(o)
        if isinstance(o, (_np.integer,)): return int(o)
        if isinstance(o, (_np.bool_,)): return bool(o)
        return str(o)
    (OUT/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2,default=_json_default),encoding='utf-8')
    print('ALL DONE in',time.time()-t0,'s')

if __name__=='__main__': main()
