"""Exercise 4: reproducible LDA with collapsed Gibbs sampling (NumPy only)."""
import argparse
import csv
import html
import json
import re
from pathlib import Path
import numpy as np

STOP = set('a an the and or of to in on for with by from is are was were be been this that it its as at our their we they has have had will can into during after before more new using use'.split())

def preprocess(text):
    return [w for w in re.findall(r'[a-z]+', text.lower()) if len(w) > 2 and w not in STOP]

def fit(docs, vocab_size, k=4, iterations=400, seed=42, alpha=0.1, beta=0.01):
    """Average posterior distributions over retained post-burn-in samples."""
    if k < 2 or iterations < 20 or alpha <= 0 or beta <= 0:
        raise ValueError('Require topics >= 2, iterations >= 20, positive priors.')
    rng = np.random.default_rng(seed)
    nd = np.zeros((len(docs), k), dtype=np.int64)
    nw = np.zeros((k, vocab_size), dtype=np.int64)
    nk = np.zeros(k, dtype=np.int64)
    assignments = [rng.integers(k, size=len(d)) for d in docs]
    for d, words in enumerate(docs):
        for w, z in zip(words, assignments[d]):
            nd[d, z] += 1; nw[z, w] += 1; nk[z] += 1
    theta = np.zeros_like(nd, dtype=float)
    phi = np.zeros_like(nw, dtype=float)
    retained = 0
    for step in range(iterations):
        for d, words in enumerate(docs):
            for i, w in enumerate(words):
                z = assignments[d][i]
                nd[d, z] -= 1; nw[z, w] -= 1; nk[z] -= 1
                p = (nd[d] + alpha) * (nw[:, w] + beta) / (nk + vocab_size * beta)
                z = rng.choice(k, p=p / p.sum())
                assignments[d][i] = z
                nd[d, z] += 1; nw[z, w] += 1; nk[z] += 1
        if step >= iterations // 2 and step % 10 == 0:
            theta += (nd + alpha) / (nd.sum(axis=1, keepdims=True) + k * alpha)
            phi += (nw + beta) / (nk[:, None] + vocab_size * beta)
            retained += 1
    assert np.all(nd >= 0) and np.all(nw >= 0)
    assert nd.sum() == nw.sum() == sum(map(len, docs))
    return theta / retained, phi / retained, retained

def write_csv(path, header, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f); writer.writerow(header); writer.writerows(rows)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, default=Path('data/corpus.csv'))
    ap.add_argument('--output', type=Path, default=Path('outputs'))
    ap.add_argument('--topics', type=int, default=4)
    ap.add_argument('--iterations', type=int, default=400)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    with args.input.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not {'id', 'text'} <= set(reader.fieldnames or []):
            raise ValueError('CSV requires id,text columns; category is optional.')
        rows = list(reader)
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Require nonempty corpus with unique document IDs.')
    tokens = [preprocess(r['text']) for r in rows]
    if any(not t for t in tokens):
        raise ValueError('Every document must contain tokens after preprocessing.')
    vocabulary = sorted(set(w for t in tokens for w in t))
    index = {w: i for i, w in enumerate(vocabulary)}
    docs = [[index[w] for w in t] for t in tokens]
    dtm = np.zeros((len(rows), len(vocabulary)), dtype=int)
    for d, words in enumerate(docs):
        np.add.at(dtm[d], words, 1)
    theta, phi, retained = fit(docs, len(vocabulary), args.topics, args.iterations, args.seed)
    assert np.allclose(theta.sum(axis=1), 1) and np.allclose(phi.sum(axis=1), 1)
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    write_csv(out/'preprocessed_corpus.csv', ['id','tokens'], [(r['id'], ' '.join(t)) for r,t in zip(rows,tokens)])
    write_csv(out/'document_term_matrix.csv', ['id']+vocabulary, [(r['id'], *v) for r,v in zip(rows,dtm)])
    write_csv(out/'document_topics.csv', ['id']+[f'topic_{i}' for i in range(args.topics)], [(r['id'],*v) for r,v in zip(rows,theta)])
    write_csv(out/'topic_words.csv', ['topic','rank','word','probability'], [(k,j+1,vocabulary[w],float(phi[k,w])) for k in range(args.topics) for j,w in enumerate(np.argsort(-phi[k])[:10])])
    np.savez_compressed(out/'lda_model.npz', theta=theta, phi=phi, vocabulary=np.array(vocabulary), document_ids=np.array([r['id'] for r in rows]), alpha=0.1, beta=0.01)
    # Training reconstruction, not held-out generalisation performance.
    logp = sum(np.log((theta[d] @ phi)[words]).sum() for d,words in enumerate(docs))
    perplexity = float(np.exp(-logp / dtm.sum()))
    metrics = dict(documents=len(rows), vocabulary=len(vocabulary), tokens=int(dtm.sum()), topics=args.topics, iterations=args.iterations, seed=args.seed, retained_samples=retained, training_reconstruction_perplexity=perplexity)
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    cards = []
    interpretations = []
    for k in range(args.topics):
        top = np.argsort(-phi[k])[:10]
        best = np.argsort(-theta[:,k])[:3]
        categories = sorted({rows[d].get('category','unlabelled') for d in best})
        label = ' / '.join(categories)
        interpretations.append(f"Topic {k}: {label}. Leading words: {', '.join(vocabulary[w] for w in top[:6])}. Representative documents: {', '.join(rows[d]['id'] for d in best)}.")
        bars = ''.join(f'<div class="barrow"><span>{html.escape(vocabulary[w])}</span><div class="track"><div style="width:{phi[k,w]/phi[k,top[0]]*100:.1f}%"></div></div><small>{phi[k,w]:.3f}</small></div>' for w in top)
        examples = ''.join(f'<li><b>{html.escape(rows[d]["id"])}</b> ({theta[d,k]:.1%}): {html.escape(rows[d]["text"])}</li>' for d in best)
        cards.append(f'<article><h2>Topic {k} · {html.escape(label)}</h2>{bars}<h3>Representative documents</h3><ul>{examples}</ul></article>')
    report = f'''# Exercise 4 - Latent Dirichlet Allocation

## Aim and scope
Apply unsupervised LDA to a document collection, estimate probabilistic topic-word and document-topic distributions, and interpret topics. The supplied corpus is 32 original, deliberately separable educational documents in four categories; it is a demonstration dataset, not a real-world benchmark. Categories are excluded from training and used only for post-hoc interpretation. For other CSVs, labels must be assigned by reading top words and examples.

## Pipeline and methodology
Lowercase -> alphabetic tokenisation -> stopword and short-token removal -> vocabulary -> document-term counts -> collapsed Gibbs sampling -> posterior averages -> interpretation. No stemming is used, preserving readable terms but allowing morphological variants to split counts. No external corpus downloads or API keys are required.

LDA assumes each document mixes topics and each topic distributes probability over words. Symmetric Dirichlet priors are alpha=0.1 and beta=0.01. For each token, remove its old assignment and sample a new topic with probability proportional to (n_document,topic + alpha) * (n_topic,word + beta) / (n_topic + V*beta). Counts exclude the current token. Theta=(n_document,topic+alpha)/(document_length+K*alpha); phi=(n_topic,word+beta)/(n_topic+V*beta). The first half of iterations is burn-in; retain every tenth subsequent sample and average theta/phi. Topic IDs have no semantic ordering.

## Actual run
{json.dumps(metrics, indent=2)}

## Topic interpretation
''' + '\n\n'.join(interpretations) + '''

## Analysis and limitations
Top words and representative documents jointly support each topic interpretation. Bar widths are relative to the largest displayed word probability; numeric values are full-vocabulary probabilities. LDA is unsupervised: category labels do not establish predictive accuracy. Training reconstruction perplexity describes fit to observed words, not held-out performance, and must not be reported as test perplexity. Four topics match the intended demonstration scope; this is not an empirically selected optimal K. Small, clean documents make topic separation easier than noisy real data. Bag-of-words loses word order, negation, and context. Finite Gibbs runs can mix slowly; one seed does not demonstrate convergence. Useful extensions include a larger real corpus, several chains/seeds, held-out evaluation, coherence measures, and K sensitivity analysis.

## Verification and deliverables
The run checks nonnegative counts, token conservation, and normalised distributions. Tests cover deterministic sampling, normalisation, empty documents at the CLI, and text preprocessing. Preprocessed corpus, document-term matrix, complete model arrays, document-topic probabilities, top-word lists, metrics, and this report are saved in outputs. report.html presents the actual results for review.
'''
    (out/'interpretation_report.md').write_text(report, encoding='utf-8')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Exercise 4 | LDA Topic Lab</title><style>body{margin:0;background:#f3f5f8;color:#172238;font:16px/1.6 system-ui}main{max-width:1150px;margin:auto;padding:44px 24px}h1{font-size:42px;line-height:1.15}h2{font-size:22px}header p{max-width:850px}.pill{color:#087f74;font-weight:700;letter-spacing:2px}.stats{display:flex;gap:16px;flex-wrap:wrap;margin:28px 0}.stat,article{background:white;border:1px solid #dee4ed;border-radius:14px;padding:24px}.stat b{display:block;font-size:28px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:20px}.barrow{display:grid;grid-template-columns:100px 1fr 45px;gap:10px;align-items:center;margin:7px 0}.track{background:#edf2f6;height:14px;border-radius:4px}.track div{background:#128c83;height:14px;border-radius:4px}small{font-variant-numeric:tabular-nums}li{margin-bottom:12px}footer{margin-top:30px}a{color:#087f74}@media(max-width:500px){.grid{grid-template-columns:1fr}h1{font-size:32px}}</style><main><header><div class="pill">AML23702 · LAB EXERCISE 04</div><h1>Discovering topics with LDA</h1><p>A complete sampling-based topic modelling pipeline: original corpus, preprocessing, document-term counts, collapsed Gibbs sampling, and probabilistic interpretation.</p></header>'''
    page += '<div class="stats">'+''.join(f'<div class="stat"><b>{value}</b>{label}</div>' for label,value in [('Documents',len(rows)),('Vocabulary',len(vocabulary)),('Topics',args.topics),('Gibbs sweeps',args.iterations)])+'</div>'
    page += '<p>Original educational demonstration corpus. Category descriptions below are post-hoc labels from representative documents; categories are never passed to LDA. Probabilities are averaged over '+str(retained)+' retained samples.</p><div class="grid">'+''.join(cards)+'</div>'
    page += '<footer><h2>Interpretation and limitations</h2><p>Inspect word probabilities alongside representative documents. This small, deliberately separable corpus demonstrates the method; it does not establish real-world performance. Bag-of-words ignores context and order. Topic IDs can change between seeds, and one chain does not prove convergence.</p><p><a href="interpretation_report.md">Full methodology and analysis</a> · <a href="topic_words.csv">Topic-word table</a> · <a href="document_topics.csv">Document-topic probabilities</a> · <a href="preprocessed_corpus.csv">Preprocessed corpus</a> · <a href="metrics.json">Run metrics</a></p></footer></main></html>'
    (out/'report.html').write_text(page, encoding='utf-8')
    print(json.dumps(metrics, indent=2))
    print('\n'+'\n'.join(interpretations))

if __name__ == '__main__':
    main()
