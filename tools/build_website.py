"""Export trained LDA distributions and corpus into the Topic Atlas frontend."""
import csv
import json
import shutil
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def export_website(model_dir=ROOT/'outputs',corpus_path=ROOT/'data/corpus.csv',web_dir=ROOT/'dist'):
    model_dir,corpus_path,web_dir=map(Path,(model_dir,corpus_path,web_dir))
    model=np.load(model_dir/'lda_model.npz',allow_pickle=False)
    phi,theta,vocab=model['phi'],model['theta'],model['vocabulary'].tolist()
    with corpus_path.open(encoding='utf-8-sig',newline='') as f: rows={r['id']:r for r in csv.DictReader(f)}
    with (model_dir/'preprocessed_corpus.csv').open(encoding='utf-8',newline='') as f: clean={r['id']:r['tokens'] for r in csv.DictReader(f)}
    docs=[{**rows[str(id)],'tokens':clean[str(id)].split(),'theta':theta[i].tolist(),'dominant':int(theta[i].argmax())} for i,id in enumerate(model['document_ids'])]
    topics=[]
    for k in range(len(phi)):
        best=np.argsort(-theta[:,k])[:3].tolist()
        labels=sorted({docs[d].get('category','Topic '+str(k)) for d in best})
        topics.append({'id':k,'label':' / '.join(labels),'top_words':[{'word':vocab[w],'probability':float(phi[k,w])} for w in np.argsort(-phi[k])[:15]],'representatives':[docs[d]['id'] for d in best],'prevalence':float(theta[:,k].mean()),'dominant_count':sum(d['dominant']==k for d in docs)})
    payload={'metrics':json.loads((model_dir/'metrics.json').read_text()),'alpha':float(model['alpha']),'beta':float(model['beta']),'vocabulary':vocab,'phi':phi.tolist(),'topics':topics,'documents':docs}
    (web_dir/'assets').mkdir(parents=True,exist_ok=True)
    (web_dir/'assets/model-data.json').write_text(json.dumps(payload),encoding='utf-8')
    (web_dir/'downloads').mkdir(exist_ok=True)
    for name in ('preprocessed_corpus.csv','document_term_matrix.csv','document_topics.csv','topic_words.csv','metrics.json','interpretation_report.md','lda_model.npz'):
        shutil.copyfile(model_dir/name,web_dir/'downloads'/name)
    print('Website data exported:',len(docs),'documents;',len(topics),'topics.')

if __name__=='__main__': export_website()
