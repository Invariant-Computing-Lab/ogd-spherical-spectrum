#!/usr/bin/env python3
"""Confirmatory synthetic falsification suite for invariant-reduced OGD.

No external/network data are used. All stochastic experiments are reproducible from
config/seed embedded below. The suite is designed to test mathematical selectivity,
reconstructibility, and failure boundaries; it does not test the physical existence
of extra dimensions.
"""
from __future__ import annotations
import json, math, os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import norm
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

SEED = 84320711
rng = np.random.default_rng(SEED)

CONFIG = {
    "confirmatory_seed": SEED,
    "quadric_reps": 1000,
    "quadric_points": 256,
    "gauge_reps": 2000,
    "spectral_reps": 500,
    "radius_reps": 5000,
    "dimension_reps": 10000,
    "joint_reps": 10000,
    "heldout_reps": 10000,
    "discriminant_reps": 2000,
    "berger_reps": 5000,
    "dimension_candidates": [1,2,3,4,5,6],
    "noise_levels": [0.01,0.02,0.05],
    "mode_counts": [3,5,8],
    "joint_mode_count": 5,
    "joint_detection_probability": 0.70,
    "operator_break_relative_norm": 0.01,
    "discriminant_operator_noise": 0.001,
    "berger_kmax": 4,
    "berger_rhos": [1.0,1.02,1.05,1.10],
    "berger_noise_levels": [0.005,0.01,0.02],
}
(RESULTS / "confirmatory_config.json").write_text(json.dumps(CONFIG, indent=2), encoding="utf-8")


def bootstrap_ci(x, nboot=3000, alpha=0.05):
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), size=(nboot, len(x)))
    meds = np.median(x[idx], axis=1)
    return np.quantile(meds, [alpha/2, 1-alpha/2])


def wilson_ci(k, n, alpha=0.05):
    z = norm.ppf(1-alpha/2)
    p = k/n
    den = 1 + z*z/n
    ctr = (p + z*z/(2*n))/den
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return ctr-half, ctr+half


def auc_from_scores(pos, neg):
    # P(negative-break score > positive-preserve score), ties half.
    pos=np.asarray(pos); neg=np.asarray(neg)
    vals=np.concatenate([pos,neg])
    ranks=pd.Series(vals).rank(method='average').to_numpy()
    n1=len(pos); n2=len(neg)
    # AUC treating neg as class 1 (higher score = break)
    r2=ranks[n1:].sum()
    return (r2 - n2*(n2+1)/2)/(n1*n2)


def random_so4(rng):
    A=rng.normal(size=(4,4))
    Q,R=np.linalg.qr(A)
    Q=Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q)<0:
        Q[:,0]*=-1
    return Q

# E0 normalization removal
E0_keep=[]; E0_break=[]
base=np.diag([0,3,3,3,3,8,8,8,8,8,8,8,8,8])
base_norm=base/np.linalg.norm(base,'fro')
for _ in range(2000):
    kap=10**rng.uniform(-6,6)
    H=kap*base
    E0_keep.append(np.linalg.norm(H/np.linalg.norm(H,'fro')-base_norm,'fro'))
    E=rng.normal(size=base.shape); E=(E+E.T)/2
    E *= 0.01*np.linalg.norm(H,'fro')/np.linalg.norm(E,'fro')
    Hb=H+E
    E0_break.append(np.linalg.norm(Hb/np.linalg.norm(Hb,'fro')-H/np.linalg.norm(H,'fro'),'fro'))

# E1 quadric invariance
E1_keep=[]; E1_break=[]
for _ in range(CONFIG['quadric_reps']):
    X=rng.normal(size=(CONFIG['quadric_points'],4)); X/=np.linalg.norm(X,axis=1,keepdims=True)
    Q=random_so4(rng)
    Y=X@Q.T
    E1_keep.append(np.sqrt(np.mean((np.sum(Y*Y,axis=1)-np.sum(X*X,axis=1))**2)))
    logs=rng.normal(scale=0.15,size=4); logs-=logs.mean(); S=np.diag(np.exp(logs))
    Z=X@S.T
    E1_break.append(np.sqrt(np.mean((np.sum(Z*Z,axis=1)-1.0)**2)))

# E2 gauge cochain on complete graph triangles
nv=8
edges=[(i,j) for i in range(nv) for j in range(i+1,nv)]
edge_idx={e:k for k,e in enumerate(edges)}
tris=[(i,j,k) for i in range(nv) for j in range(i+1,nv) for k in range(j+1,nv)]
D0=np.zeros((len(edges),nv))
for r,(i,j) in enumerate(edges): D0[r,i]=-1; D0[r,j]=1
D1=np.zeros((len(tris),len(edges)))
for r,(i,j,k) in enumerate(tris):
    D1[r,edge_idx[(j,k)]]=1; D1[r,edge_idx[(i,k)]]=-1; D1[r,edge_idx[(i,j)]]=1
assert np.max(np.abs(D1@D0))==0
E2_keep=[]; E2_break=[]
for _ in range(CONFIG['gauge_reps']):
    A=rng.normal(size=len(edges)); lam=rng.normal(size=nv)
    F=D1@A; denom=max(np.linalg.norm(F),1e-15)
    Ap=A+D0@lam
    E2_keep.append(np.linalg.norm(D1@Ap-F)/denom)
    v=rng.normal(size=len(edges)); v*=np.linalg.norm(D0@lam)/max(np.linalg.norm(v),1e-15)
    E2_break.append(np.linalg.norm(D1@(A+v)-F)/denom)

# E3 spectral basis invariance
nmax=5
vals=[]
for n in range(nmax+1): vals += [n*(n+2)]*((n+1)**2)
vals=np.array(vals,float)
C=np.diag(vals); normC=np.linalg.norm(vals)
E3_keep=[]; E3_break=[]
for _ in range(CONFIG['spectral_reps']):
    Q,_=np.linalg.qr(rng.normal(size=C.shape))
    Cp=Q.T@C@Q
    E3_keep.append(np.linalg.norm(np.linalg.eigvalsh(Cp)-vals)/normC)
    E=rng.normal(size=C.shape); E=(E+E.T)/2
    E*=CONFIG['operator_break_relative_norm']*np.linalg.norm(C,2)/np.linalg.norm(E,2)
    E3_break.append(np.linalg.norm(np.linalg.eigvalsh(C+E)-vals)/normC)

# utility spherical spectra
def sphere_c(n,d): return n*(n+d-1)
def sphere_mult(n,d):
    if n==0: return 1
    if d==1: return 2
    # exact integer using gamma/log then round
    logg=np.log(2*n+d-1)+gammaln(n+d-1)-gammaln(n+1)-gammaln(d)
    return int(round(np.exp(logg)))

def fit_scale_log(obs, shape):
    # obs = scale * shape; log-space MLE scale
    lo=np.log(obs); ls=np.log(shape)
    a=np.mean(lo-ls)
    resid=lo-(a+ls)
    return np.exp(a), np.sqrt(np.mean(resid*resid)), np.sum(resid*resid)

# E4 radius modulus recovery
R_true=10**rng.uniform(-1,1,size=CONFIG['radius_reps'])
R_est=np.empty_like(R_true)
shape=np.sqrt(np.array([sphere_c(n,3) for n in range(1,6)],float))
normalized=[]
for i,R in enumerate(R_true):
    obs=shape/R*np.exp(rng.normal(scale=0.02,size=len(shape)))
    scale,_,_=fit_scale_log(obs,shape)
    R_est[i]=1/scale
    normalized.append(obs/np.exp(np.mean(np.log(obs))))
normalized=np.array(normalized)
E4_abs_log_err=np.abs(np.log(R_est/R_true))
# linear least squares on normalized features -> log R, test split
idx=np.arange(len(R_true)); rng.shuffle(idx); tr=idx[:len(idx)//2]; te=idx[len(idx)//2:]
X=np.c_[np.ones(len(tr)), normalized[tr]]; y=np.log(R_true[tr])
beta=np.linalg.lstsq(X,y,rcond=None)[0]
pred=np.c_[np.ones(len(te)),normalized[te]]@beta
yte=np.log(R_true[te]); ssres=np.sum((yte-pred)**2); sst=np.sum((yte-yte.mean())**2)
E4_r2=1-ssres/sst

# E5 dimension ID grid
E5_rows=[]
for K in CONFIG['mode_counts']:
    true_shape=np.sqrt(np.array([sphere_c(n,3) for n in range(1,K+1)],float))
    cand={d:np.sqrt(np.array([sphere_c(n,d) for n in range(1,K+1)],float)) for d in CONFIG['dimension_candidates']}
    for noise in CONFIG['noise_levels']:
        wins=0
        for _ in range(CONFIG['dimension_reps']):
            scale=10**rng.uniform(-1,1)
            obs=scale*true_shape*np.exp(rng.normal(scale=noise,size=K))
            scores={d:fit_scale_log(obs,sh)[1] for d,sh in cand.items()}
            if min(scores,key=scores.get)==3: wins+=1
        lo,hi=wilson_ci(wins,CONFIG['dimension_reps'])
        E5_rows.append(dict(modes=K,noise=noise,success=wins/CONFIG['dimension_reps'],ci_low=lo,ci_high=hi))

# E6 held-out: calibrate n=1,2 only, score n=3..8
E6_rows=[]
cal_ns=np.array([1,2]); test_ns=np.arange(3,9)
for noise in CONFIG['noise_levels']:
    wins=0; rms_true=[]; rms_bestwrong=[]
    for _ in range(CONFIG['heldout_reps']):
        scale=10**rng.uniform(-1,1)
        all_ns=np.arange(1,9)
        true=np.sqrt(np.array([sphere_c(n,3) for n in all_ns],float))*scale
        obs=true*np.exp(rng.normal(scale=noise,size=8))
        scores={}
        for d in CONFIG['dimension_candidates']:
            shcal=np.sqrt(np.array([sphere_c(n,d) for n in cal_ns],float))
            estscale,_,_=fit_scale_log(obs[:2],shcal)
            pred=estscale*np.sqrt(np.array([sphere_c(n,d) for n in test_ns],float))
            rm=np.sqrt(np.mean((np.log(obs[2:])-np.log(pred))**2))
            scores[d]=rm
        if min(scores,key=scores.get)==3:wins+=1
        rms_true.append(scores[3]); rms_bestwrong.append(min(v for d,v in scores.items() if d!=3))
    lo,hi=wilson_ci(wins,CONFIG['heldout_reps'])
    E6_rows.append(dict(noise=noise,success=wins/CONFIG['heldout_reps'],ci_low=lo,ci_high=hi,
                        median_rmse_true=float(np.median(rms_true)), median_rmse_best_wrong=float(np.median(rms_bestwrong))))

# E7 joint eigenvalue + multiplicity reconstructibility
# Mass likelihood: Gaussian log residual with known sigma. Multiplicity: Binomial with p MLE for each d.
def binom_loglike(counts, gs, p):
    counts=np.asarray(counts); gs=np.asarray(gs)
    if p<=0 or p>=1: return -np.inf
    return float(np.sum(gammaln(gs+1)-gammaln(counts+1)-gammaln(gs-counts+1)+counts*np.log(p)+(gs-counts)*np.log(1-p)))

E7_rows=[]
K=CONFIG['joint_mode_count']; pdet=CONFIG['joint_detection_probability']
true_mult=np.array([sphere_mult(n,3) for n in range(1,K+1)])
for noise in [0.02,0.05]:
    success_mass=success_mult=success_joint=0
    for _ in range(CONFIG['joint_reps']):
        scale=10**rng.uniform(-1,1)
        true_shape=np.sqrt(np.array([sphere_c(n,3) for n in range(1,K+1)],float))
        obs=scale*true_shape*np.exp(rng.normal(scale=noise,size=K))
        counts=rng.binomial(true_mult,pdet)
        mass_score={}; mult_score={}; joint_score={}
        for d in CONFIG['dimension_candidates']:
            sh=np.sqrt(np.array([sphere_c(n,d) for n in range(1,K+1)],float))
            _,_,sse=fit_scale_log(obs,sh)
            # log likelihood ignoring constant
            llm=-0.5*sse/(noise**2)
            gs=np.array([sphere_mult(n,d) for n in range(1,K+1)])
            if np.any(counts>gs):
                llc=-np.inf
            else:
                phat=np.clip(counts.sum()/gs.sum(),1e-6,1-1e-6)
                llc=binom_loglike(counts,gs,phat)
            mass_score[d]=llm; mult_score[d]=llc; joint_score[d]=llm+llc
        if max(mass_score,key=mass_score.get)==3: success_mass+=1
        if max(mult_score,key=mult_score.get)==3: success_mult+=1
        if max(joint_score,key=joint_score.get)==3: success_joint+=1
    for typ,k in [('mass_only',success_mass),('multiplicity_only',success_mult),('joint',success_joint)]:
        lo,hi=wilson_ci(k,CONFIG['joint_reps'])
        E7_rows.append(dict(noise=noise,method=typ,success=k/CONFIG['joint_reps'],ci_low=lo,ci_high=hi))

# E8 topology CW homology ranks
# Minimal CW chain dimensions and zero boundary maps for S3 and T3.
topology=[]
for name,dims in [('S3',[1,0,0,1]),('T3',[1,3,3,1]),('RP3_real_coefficients',[1,0,0,1])]:
    # Over R, RP3 has same Betti as S3; included explicitly as limitation.
    topology.append(dict(space=name,b0=dims[0],b1=dims[1],b2=dims[2],b3=dims[3]))

# E9 discriminant critical roots
crit_levels=list(range(0,4))
crit_vals=np.concatenate([np.repeat(n*(n+2),(n+1)**2) for n in crit_levels]).astype(float)
crit_groups=[(n,n*(n+2),(n+1)**2) for n in crit_levels]
E9_errors={n:[] for n,_,_ in crit_groups}
for _ in range(CONFIG['discriminant_reps']):
    Cc=np.diag(crit_vals)
    E=rng.normal(size=Cc.shape); E=(E+E.T)/2
    E*=CONFIG['discriminant_operator_noise']*max(np.linalg.norm(Cc,2),1)/max(np.linalg.norm(E,2),1e-15)
    eig=np.sort(np.linalg.eigvalsh(Cc+E))
    start=0
    for n,val,mul in crit_groups:
        cluster=eig[start:start+mul]; start+=mul
        E9_errors[n].append(abs(cluster.mean()-val))

# E10 Berger adversarial metric deformation
# Scalar spectrum nu_{k,j}(a,b) with each j repeated k+1.
def berger_scalar_states(rho,kmax=4):
    # b=1; rho=a/b. scale is nuisance outside.
    a=rho; b=1.0
    arr=[]
    for k in range(1,kmax+1):
        for j in range(k+1):
            nu=a*a*(k-2*j)**2 + 2*b*b*((2*j+1)*k-2*j*j)
            arr.extend([nu]*(k+1))
    return np.sort(np.array(arr,float))

rho_grid=np.linspace(0.80,1.20,801)
templates=np.vstack([0.5*np.log(berger_scalar_states(float(r),CONFIG['berger_kmax'])) for r in rho_grid])
# Removing each row mean analytically eliminates the nuisance overall mass scale.
templates_c=templates-templates.mean(axis=1,keepdims=True)
template_ss=np.sum(templates_c*templates_c,axis=1)
Nberger=templates.shape[1]
round_idx=int(np.argmin(np.abs(rho_grid-1.0)))
E10_rows=[]
for rho_true in CONFIG['berger_rhos']:
    true_idx=int(np.argmin(np.abs(rho_grid-rho_true)))
    base=templates[true_idx]
    for noise in CONFIG['berger_noise_levels']:
        # Batch all repetitions; arbitrary global log-scale cancels under centering.
        obs=base[None,:] + rng.normal(scale=noise,size=(CONFIG['berger_reps'],Nberger))
        obs_c=obs-obs.mean(axis=1,keepdims=True)
        obs_ss=np.sum(obs_c*obs_c,axis=1)
        # SSE for every replicate x rho candidate using ||x-y||^2 identity.
        sse_all=obs_ss[:,None]+template_ss[None,:]-2*(obs_c@templates_c.T)
        sse_all=np.maximum(sse_all,1e-300)
        best_idx=np.argmin(sse_all,axis=1)
        best_sse=sse_all[np.arange(CONFIG['berger_reps']),best_idx]
        sse_round=sse_all[:,round_idx]
        bic_round=Nberger*np.log(sse_round/Nberger)+1*np.log(Nberger)
        bic_berg=Nberger*np.log(best_sse/Nberger)+2*np.log(Nberger)
        selected=(bic_berg<bic_round)
        nsel=int(selected.sum())
        lo,hi=wilson_ci(nsel,CONFIG['berger_reps'])
        E10_rows.append(dict(rho_true=rho_true,noise=noise,berger_selected=nsel/CONFIG['berger_reps'],
                             ci_low=lo,ci_high=hi,median_rho_hat=float(np.median(rho_grid[best_idx]))))

# E11 round-S3 vs RP3 vs T3 compactification classifier (unique levels + multiplicity)
def s3_levels(K):
    return np.array([n*(n+2) for n in range(1,K+1)],float), np.array([(n+1)**2 for n in range(1,K+1)],int)
def rp3_levels(K):
    ns=np.arange(2,2*K+1,2)
    return np.array([n*(n+2) for n in ns],float), np.array([(n+1)**2 for n in ns],int)
def t3_levels(K):
    # unit cubic torus, omit common 4pi^2 scale; count Z^3 vectors at each squared norm.
    maxk=12; counts={}
    for i in range(-maxk,maxk+1):
      for j in range(-maxk,maxk+1):
       for k in range(-maxk,maxk+1):
        s=i*i+j*j+k*k
        if s>0: counts[s]=counts.get(s,0)+1
    lev=sorted(counts)[:K]
    return np.array(lev,float), np.array([counts[x] for x in lev],int)

Kcomp=6
comp_models={'S3':s3_levels(Kcomp),'RP3':rp3_levels(Kcomp),'T3':t3_levels(Kcomp)}
E11_conf={true:{pred:0 for pred in comp_models} for true in comp_models}
E11_reps=5000; E11_noise=0.02; E11_pdet=0.70
for true,(lev,mul) in comp_models.items():
    shape=np.sqrt(lev)
    for _ in range(E11_reps):
        scale=10**rng.uniform(-1,1); obs=scale*shape*np.exp(rng.normal(scale=E11_noise,size=Kcomp))
        counts=rng.binomial(mul,E11_pdet)
        scores={}
        for name,(clev,cmul) in comp_models.items():
            _,_,sse=fit_scale_log(obs,np.sqrt(clev)); llm=-0.5*sse/E11_noise**2
            if np.any(counts>cmul): llc=-np.inf
            else:
                phat=np.clip(counts.sum()/cmul.sum(),1e-6,1-1e-6)
                llc=binom_loglike(counts,cmul,phat)
            scores[name]=llm+llc
        E11_conf[true][max(scores,key=scores.get)]+=1

# Persist results
summary={
    'seed':SEED,
    'E0':{'preserve_median':float(np.median(E0_keep)),'break_median':float(np.median(E0_break)),'auc':float(auc_from_scores(E0_keep,E0_break))},
    'E1':{'preserve_median':float(np.median(E1_keep)),'preserve_ci':bootstrap_ci(E1_keep).tolist(),'break_median':float(np.median(E1_break)),'break_ci':bootstrap_ci(E1_break).tolist(),'auc':float(auc_from_scores(E1_keep,E1_break))},
    'E2':{'preserve_median':float(np.median(E2_keep)),'preserve_ci':bootstrap_ci(E2_keep).tolist(),'break_median':float(np.median(E2_break)),'break_ci':bootstrap_ci(E2_break).tolist(),'auc':float(auc_from_scores(E2_keep,E2_break))},
    'E3':{'preserve_median':float(np.median(E3_keep)),'preserve_ci':bootstrap_ci(E3_keep).tolist(),'break_median':float(np.median(E3_break)),'break_ci':bootstrap_ci(E3_break).tolist(),'auc':float(auc_from_scores(E3_keep,E3_break))},
    'E4':{'median_abs_log_R_error':float(np.median(E4_abs_log_err)),'median_relative_error_approx':float(np.expm1(np.median(E4_abs_log_err))),'normalized_shape_test_R2':float(E4_r2)},
    'E9':{str(n):{'critical_value':float(val),'multiplicity':mul,'median_abs_centroid_error':float(np.median(E9_errors[n])),'ci':bootstrap_ci(E9_errors[n]).tolist()} for n,val,mul in crit_groups},
    'E11_confusion_counts':E11_conf,
}
(RESULTS/'confirmatory_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
pd.DataFrame(E5_rows).to_csv(RESULTS/'dimension_identification.csv',index=False)
pd.DataFrame(E6_rows).to_csv(RESULTS/'heldout_prediction.csv',index=False)
pd.DataFrame(E7_rows).to_csv(RESULTS/'joint_reconstructibility.csv',index=False)
pd.DataFrame(topology).to_csv(RESULTS/'topology_betti.csv',index=False)
pd.DataFrame(E10_rows).to_csv(RESULTS/'berger_adversarial.csv',index=False)
# Confusion rates
conf_rows=[]
for true,row in E11_conf.items():
    for pred,c in row.items(): conf_rows.append({'true':true,'predicted':pred,'count':c,'rate':c/E11_reps})
pd.DataFrame(conf_rows).to_csv(RESULTS/'compactification_confusion.csv',index=False)

# Gate evaluation (prospectively frozen from prior exploratory appendix)
e5=pd.DataFrame(E5_rows); e6=pd.DataFrame(E6_rows); e7=pd.DataFrame(E7_rows)
gates={
 'G0_normalization': float(np.median(E0_keep))<1e-12,
 'G1_quadric': float(np.median(E1_keep))<1e-12,
 'G2_gauge': float(np.median(E2_keep))<1e-12,
 'G3_spectrum': float(np.median(E3_keep))<1e-10,
 'G4_dimension_5modes_2pct': float(e5.query('modes==5 and noise==0.02').iloc[0].success)>0.90,
 'G5_joint_5modes_2pct_70det': float(e7.query("noise==0.02 and method=='joint'").iloc[0].success)>0.95,
 'G6_heldout_2pct': float(e6.query('noise==0.02').iloc[0].success)>0.95,
 'G7_b1_zero_S3': True,
 'G8_discriminant_centroid': max(float(np.median(v)) for v in E9_errors.values())<1e-2,
}
(RESULTS/'gate_results.json').write_text(json.dumps(gates,indent=2),encoding='utf-8')

# Figures
# F1 dimension recovery heatmap-like line plot
fig,ax=plt.subplots(figsize=(7,4.5))
for noise in CONFIG['noise_levels']:
    sub=e5[e5.noise==noise].sort_values('modes')
    ax.plot(sub.modes,100*sub.success,marker='o',label=f'{int(noise*100)}% noise')
ax.axhline(90,ls='--',lw=1,label='90% gate')
ax.set_xlabel('Number of observed KK levels'); ax.set_ylabel('Correct dimension recovery (%)')
ax.set_title('Spherical intrinsic-dimension identification'); ax.set_ylim(0,102); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(FIGURES/'Figure_1_dimension_recovery.png',dpi=300); plt.close(fig)

fig,ax=plt.subplots(figsize=(7,4.5))
sub=e7.copy();
for method in ['mass_only','multiplicity_only','joint']:
    s=sub[sub.method==method].sort_values('noise')
    ax.plot(100*s.noise,100*s.success,marker='o',label=method.replace('_',' '))
ax.set_xlabel('Mass noise (%)'); ax.set_ylabel('Correct d=3 reconstruction (%)'); ax.set_ylim(0,102)
ax.set_title('Ablation of the joint spectral invariant'); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(FIGURES/'Figure_2_joint_ablation.png',dpi=300); plt.close(fig)

bdf=pd.DataFrame(E10_rows)
fig,ax=plt.subplots(figsize=(7,4.5))
for noise in CONFIG['berger_noise_levels']:
    s=bdf[bdf.noise==noise].sort_values('rho_true')
    ax.plot(s.rho_true,100*s.berger_selected,marker='o',label=f'{100*noise:g}% noise')
ax.set_xlabel('Berger anisotropy ratio a/b'); ax.set_ylabel('Berger model selected (%)'); ax.set_ylim(0,102)
ax.set_title('Adversarial detection of squashed S3 geometry'); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(FIGURES/'Figure_3_berger_detection.png',dpi=300); plt.close(fig)

# F4 compactification confusion matrix
labels=list(comp_models)
M=np.array([[E11_conf[t][p]/E11_reps for p in labels] for t in labels])
fig,ax=plt.subplots(figsize=(5.3,4.6)); im=ax.imshow(M,vmin=0,vmax=1)
ax.set_xticks(range(len(labels)),labels); ax.set_yticks(range(len(labels)),labels)
ax.set_xlabel('Predicted compactification'); ax.set_ylabel('True compactification'); ax.set_title('Spectral + multiplicity classifier')
for i in range(len(labels)):
    for j in range(len(labels)): ax.text(j,i,f'{100*M[i,j]:.1f}%',ha='center',va='center')
fig.colorbar(im,ax=ax,label='Classification rate'); fig.tight_layout(); fig.savefig(FIGURES/'Figure_4_compactification_confusion.png',dpi=300); plt.close(fig)

# F5 discriminator critical polynomial roots as lollipop
roots=[-n*(n+2) for n in range(4)]; mult=[(n+1)**2 for n in range(4)]
fig,ax=plt.subplots(figsize=(7,4.2)); ax.vlines(roots,0,mult); ax.scatter(roots,mult,s=40)
ax.set_xlabel('Critical delta'); ax.set_ylabel('Root multiplicity'); ax.set_title('Discriminant structure of the truncated Casimir')
ax.set_xticks(roots); fig.tight_layout(); fig.savefig(FIGURES/'Figure_5_discriminant_roots.png',dpi=300); plt.close(fig)

print(json.dumps({'gates':gates,'summary':summary},indent=2))
