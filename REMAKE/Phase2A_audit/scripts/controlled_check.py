"""New audit computation; original sources/data are read-only. Never call historical main()."""
from pathlib import Path
import os,sys,shutil,sqlite3,json,hashlib,importlib.util,time,ast
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
BASE=Path(__file__).resolve().parents[1]; OUT=BASE/'checks'; OUT.mkdir(exist_ok=True)
os.environ['MPLBACKEND']='Agg';os.environ['MPLCONFIGDIR']=str(OUT/'mplconfig')
import numpy as np,pandas as pd,scipy,matplotlib
import matplotlib.pyplot as plt
ROOT=Path(r'D:\Desktop\LinkCodex')
def load(rel,name):
    src=ROOT/rel; dst=OUT/'source_copies'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    spec=importlib.util.spec_from_file_location(name,dst);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
m=load(Path('try/try.py'),'audit_v3')
m2=load(Path('try/try2.py'),'audit_v3rel');mn=load(Path('try/NUTSVerTry.py'),'audit_nuts');m12=load(Path('baselines/CurVer.py'),'audit_v2')
db=Path(r'D:\Desktop\EE5003\data\AP_1p5.db')
with sqlite3.connect(db.as_uri()+'?mode=ro',uri=True) as c:
    c.execute('PRAGMA query_only=ON');df=pd.read_sql_query('SELECT Freq,Zabs,Phase FROM exp_10',c)
    table_frames={t:pd.read_sql_query('SELECT Freq,Zabs,Phase FROM '+t,c) for t in ['exp_10','exp_11','exp_12','exp_7','exp_8','exp_9']}
df=df.dropna().sort_values('Freq');df=df[(df.Freq>0)&(df.Freq<=1e8)]
f=df.Freq.to_numpy(); z=m.mag_phase_to_complex(df.Zabs.to_numpy(),df.Phase.to_numpy())
result={'source':str(ROOT/'try/try.py'),'source_sha256':hashlib.sha256((ROOT/'try/try.py').read_bytes()).hexdigest(),'database':str(db),'database_sha256':hashlib.sha256(db.read_bytes()).hexdigest(),'environment':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__},'n_freq':len(f),'f_min':float(f.min()),'f_max':float(f.max())}
result['model_identity']={}
for mod in [m2,mn]:
    diffs=[]
    for seed in range(4):
        rng=np.random.default_rng(seed);p=m.make_initial_params();lo,hi=m.default_bounds(p);v=np.exp(rng.uniform(np.log(lo),np.log(hi)))
        za=m.simulate_complex(f,m.Params.from_vector(v));zb=mod.simulate_complex(f,mod.Params.from_vector(v));diffs.append(float(np.max(np.abs(za-zb))))
    result['model_identity'][mod.__name__]=diffs
result['symmetry']={}
ref=table_frames['exp_10'].sort_values('Freq')
for t in ['exp_11','exp_12']:
    d=table_frames[t].sort_values('Freq');same=np.array_equal(ref.Freq.to_numpy(),d.Freq.to_numpy())
    if same:
        zd=m.mag_phase_to_complex(d.Zabs.to_numpy(),d.Phase.to_numpy());zr=m.mag_phase_to_complex(ref.Zabs.to_numpy(),ref.Phase.to_numpy());mask=ref.Freq.to_numpy()<=1e8
        result['symmetry'][t]={'same_grid':same,'relative_complex_rms':float(np.sqrt(np.mean(np.abs((zd[mask]-zr[mask])/zr[mask])**2))),'magnitude_ratio_median':float(np.median(np.abs(zd[mask]/zr[mask])))}
    else:result['symmetry'][t]={'same_grid':False}
rsf=[];rsr=[]
for t in ['exp_7','exp_8','exp_9']:
    d=table_frames[t];d=d[(d.Freq>=100)&(d.Freq<=500)];rsf.extend(d.Freq);rsr.extend(m.mag_phase_to_complex(d.Zabs.to_numpy(),d.Phase.to_numpy()).real)
result['Rs_extrapolated_100_500']=float(scipy.stats.linregress(rsf,rsr).intercept)
result['residual_files']={}
residual_curves={}
for name in ['curver_gp_residual.csv','nutsver_gp_residual.csv','anly_meth_gp_residual.csv']:
    p=Path(r'D:\Desktop\tmp')/name
    if not p.exists():continue
    d=pd.read_csv(p); q=d[['f_hz','res_re','res_im']].dropna()
    if len(q)==len(f) and np.allclose(q.f_hz,f,rtol=0,atol=0):
        recovered=z-(q.res_re.to_numpy()+1j*q.res_im.to_numpy());residual_curves[name]=recovered
        result['residual_files'][name]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'metrics_assuming_p14':m.evaluate_raw_space_metrics(recovered,z,14)}
        pd.DataFrame({'Freq':f,'Z_re_recovered':recovered.real,'Z_im_recovered':recovered.imag}).to_csv(OUT/('recovered_'+name),index=False)
print('Prefit evidence',json.dumps(result,ensure_ascii=False),flush=True)
p0=m.make_initial_params();weights=m.compute_freq_weights(f,z,mode='auto',min_w=.3,max_w=4,power=1);sr=max(m.mad(z.real),1e-12);si=max(m.mad(z.imag),1e-12)
scale=max(m.mad(m.make_residual_fn(f,z,weights,sr,si)(np.log(p0.to_vector()))),1e-6)
start=time.perf_counter()
opt,candidates=m.fit_params_global_local(f_fit=f,Z_fit=z,p0=p0,weights=weights,s_re=sr,s_im=si,n_starts=120,top_k=10,seed=0,global_method='de',max_nfev=200,loss='soft_l1',f_scale=scale,de_maxiter=60,de_popsize=10)
zs=m.simulate_complex(f,opt)
result['rerun']={'parameters':opt.as_dict(),'metrics':m.evaluate_raw_space_metrics(zs,z,14),'relative_magnitude_metrics':m2.evaluate_relative_impedance_metrics(zs,z,14),'best_cost':float(candidates[0]['cost']),'seconds':time.perf_counter()-start,'f_scale':float(scale),'settings':'seed=0, n_starts=120, top_k=10, de_maxiter=60, de_popsize=10, max_nfev=200, soft_l1, weights=.3..4, MAD','at_bounds':{}}
lo,hi=m.default_bounds(p0)
for k,v,l,h in zip(m.PARAM_NAMES,opt.to_vector(),lo,hi):
    result['rerun']['at_bounds'][k]={'initial':float(getattr(p0,k)),'lo':float(l),'hi':float(h),'value':float(v),'near_bound':bool(v<=l*1.001 or v>=h*.999)}
result['curve_match']={name:{'max_complex_abs_diff':float(np.max(np.abs(zs-zr))),'rms_complex_diff':float(np.sqrt(np.mean(np.abs(zs-zr)**2)))} for name,zr in residual_curves.items()}
pd.DataFrame({'Freq':f,'Zobs_re':z.real,'Zobs_im':z.imag,'Zsim_re':zs.real,'Zsim_im':zs.imag}).to_csv(OUT/'v3_rerun_curve.csv',index=False)
fp=np.logspace(np.log10(f.min()),np.log10(f.max()),4000);lm,ph=m.simulate_on_freq(fp,opt)
m.plot_compare(f_exp=f,zabs_exp=np.abs(z),phase_exp=df.Phase.to_numpy(),f_sim=fp,zabs_sim=10**lm,phase_sim=ph,title_suffix='(Fitted)')
plt.gcf().savefig(OUT/'v3_rerun_plot.png',dpi=150);plt.close('all')
# Re-evaluate saved V2 parameters, without importing changed workflow modules.
p=json.loads((ROOT/'baseline1/workflow_outputs/stage2_fit_params_exp_10_seed0.json').read_text())
result['saved_workflow_reevaluation']=m12.evaluate_raw_space_metrics(m12.simulate_complex(f,m12.Params(**p['parameters'])),z,12)
# NUTS inner forward algebra only, extracted verbatim, numpy stands in for tensor exp.
tree=ast.parse((ROOT/'try/NUTSVerTry.py').read_text(encoding='utf-8-sig'))
outer=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fit_params_nuts')
inner=next(n for n in outer.body if isinstance(n,ast.FunctionDef) and n.name=='simulate_reim_pt')
ns={'pt':np,'np':np,'Rs':mn.Rs};exec(compile(ast.Module(body=[inner],type_ignores=[]),'isolated_nuts_algebra','exec'),ns)
ar,ai=ns['simulate_reim_pt'](f,np.log(opt.to_vector()));result['nuts_algebra_numpy_max_abs_diff']=float(np.max(np.abs((ar+1j*ai)-zs)))
(OUT/'controlled_results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
print('FINAL',json.dumps(result,ensure_ascii=False),flush=True)
