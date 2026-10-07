from pathlib import Path
import json,re,hashlib,sys
sys.stdout.reconfigure(encoding='utf-8')
B=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((B/'evidence/source_manifest_before.json').read_text(encoding='utf-8'))
changes=[]
for x in manifest:
    p=Path(x['path'])
    if not p.exists() or digest(p)!=x['sha256']:changes.append(x['path'])
root=Path(r'D:\Desktop\LinkCodex')
before={x['path'] for x in manifest if str(root) in x['path']}
after={str(p) for p in root.rglob('*') if p.is_file() and '.git' not in p.relative_to(root).parts}
external=json.loads((B/'evidence/external_manifest.json').read_text())
external_changes=[x['path'] for x in external if digest(Path(x['path']))!=x['sha256']]
links=[];sections={}
for name in ['OPEN_QUESTIONS.md','MODEL_EVOLUTION.md','CURRENT_BEST_RECONSTRUCTION.md','HUMAN_REVIEW.md','PROJECT_AUDIT.md']:
    p=B/name
    if not p.exists():links.append({'document':name,'missing_document':True});continue
    s=p.read_text(encoding='utf-8')
    for target in re.findall(r'\]\(<([^>]+)>\)',s):
        target=target.lstrip('/')
        target=re.sub(r':\d+$','',target)
        if not Path(target).exists():links.append({'document':name,'missing_target':target})
    if name=='OPEN_QUESTIONS.md':
        questions=re.split(r'\n## Q\d{2} — ',s)[1:]
        required=['### Why it matters','### Confirmed evidence','### Candidate hypothesis A','### Candidate hypothesis B','### Current best interpretation','### How to verify','### What would falsify','This is a working hypothesis, not a confirmed conclusion.']
        sections={'count':len(questions),'missing':[[i+1,k] for i,q in enumerate(questions) for k in required if k not in q]}
from PIL import Image
import numpy as np
a=np.asarray(Image.open(root/'EE5003report__Copy_/fig_curver.png').convert('RGB'))
c=np.asarray(Image.open(B/'checks/v3_rerun_plot.png').convert('RGB'))
result={'source_files_checked':len(manifest),'source_content_changes':changes,'source_new_files':sorted(after-before),'source_removed_files':sorted(before-after),'external_content_changes':external_changes,'broken_links':links,'question_sections':sections,'rgb_same_shape':a.shape==c.shape,'rgb_equal':bool(np.array_equal(a,c))}
(B/'evidence/final_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
