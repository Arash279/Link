from pathlib import Path
import ast, hashlib, json, subprocess, sys, zipfile, xml.etree.ElementTree as ET
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT=Path(r'D:\Desktop\LinkCodex')
OUT=Path(__file__).resolve().parents[1]/'evidence'
OUT.mkdir(parents=True,exist_ok=True)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=[]
for p in ROOT.rglob('*'):
    if p.is_file() and '.git' not in p.relative_to(ROOT).parts:
        manifest.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),'mtime_ns':p.stat().st_mtime_ns})
for p in Path(r'D:\Desktop\EE5003\data').glob('*'):
    if p.is_file(): manifest.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),'mtime_ns':p.stat().st_mtime_ns})
(OUT/'source_manifest_before.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
models=[]
for p in ROOT.rglob('*.py'):
    s=p.read_text(encoding='utf-8-sig',errors='replace')
    try:t=ast.parse(s)
    except Exception as e: models.append({'path':str(p),'parse_error':str(e)});continue
    item={'path':str(p),'params':None,'functions':{}}
    for n in t.body:
        if isinstance(n,(ast.Assign,ast.AnnAssign)):
            targets=n.targets if isinstance(n,ast.Assign) else [n.target]
            if any(isinstance(x,ast.Name) and x.id=='PARAM_NAMES' for x in targets):
                try:item['params']=ast.literal_eval(n.value)
                except Exception:pass
        if isinstance(n,ast.FunctionDef) and n.name in ['Zmid','Zmr','Zmin','Z_nLls','Zbra','Zcsf0','Zlad','Z_total','Z1_to_Z9','Y_to_Delta','delta_to_Y','make_initial_params','default_bounds','fit_params_global_local','make_residual_fn','compute_freq_weights','sample_freq_points','evaluate_raw_space_metrics','plot_compare','main']:
            item['functions'][n.name]={'line':n.lineno,'end_line':n.end_lineno,'ast_sha256':hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest(),'source':ast.get_source_segment(s,n)}
    if item['params'] or 'Z_total' in item['functions']:models.append(item)
(OUT/'code_inventory.json').write_text(json.dumps(models,ensure_ascii=False,indent=2),encoding='utf-8')
for p in ROOT.rglob('*.pptx'):
    out=[]
    with zipfile.ZipFile(p) as z:
        names=[n for n in z.namelist() if (n.startswith('ppt/slides/slide') or n.startswith('ppt/notesSlides/notesSlide')) and n.endswith('.xml')]
        for n in names:
            e=ET.fromstring(z.read(n));texts=[x.text for x in e.iter() if x.tag.endswith('}t') and x.text]
            out.append('## '+n+'\n'+'\n'.join(texts))
    (OUT/(p.stem+'.pptx.txt')).write_text('\n\n'.join(out),encoding='utf-8')
try:
    from pypdf import PdfReader
    for p in ROOT.rglob('*.pdf'):
        r=PdfReader(p)
        text='SOURCE: '+str(p)+'\nMETADATA: '+str(r.metadata)+'\n'
        text+='\n\n'.join('## PAGE '+str(i+1)+'\n'+(pg.extract_text() or '') for i,pg in enumerate(r.pages))
        (OUT/(p.stem+'.pdf.txt')).write_text(text,encoding='utf-8')
except ImportError as e: print('PDF import unavailable',e)
try:
    from PIL import Image
    imgs=[]
    for p in ROOT.rglob('*.png'):
        im=Image.open(p)
        imgs.append({'path':str(p),'size':im.size,'info':im.info,'pixel_hash':hashlib.sha256(im.convert('RGBA').tobytes()).hexdigest()})
    (OUT/'png_metadata.json').write_text(json.dumps(imgs,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
except ImportError as e: print('PIL import unavailable',e)
for name,args in [('git_history',['log','--all','--date=iso-strict','--format=%h %ad %s','--name-status']),('git_status',['status','--short']),('git_diff',['diff','--','baseline1/CurVer.py','Parameter_Fitting/fit_rr.py'])]:
    v=subprocess.run(['git','--no-optional-locks','-C',str(ROOT),*args],capture_output=True)
    (OUT/(name+'.txt')).write_bytes(v.stdout)
print('Manifest:',len(manifest),'model scripts:',len(models),'evidence directory:',OUT)
