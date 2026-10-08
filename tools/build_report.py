"""Generate the Exercise 4 PDF in the supplied lab-report layout."""
import csv
import json
import platform
import sys
from pathlib import Path
from xml.sax.saxutils import escape
import numpy as np
import reportlab
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Preformatted
from reportlab.graphics.shapes import Drawing, String, Rect

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT/'output/pdf/Exp04_Discovering_Topics_with_LDA.pdf'
BLUE = colors.HexColor('#245882')
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyLab', fontName='Times-Roman', fontSize=11, leading=14, spaceAfter=8))
styles.add(ParagraphStyle(name='HeadLab', fontName='Times-Bold', fontSize=15, leading=19, textColor=BLUE, spaceBefore=10, spaceAfter=10, borderWidth=0.7, borderColor=BLUE, borderPadding=4))
styles.add(ParagraphStyle(name='TitleLab', fontName='Times-Bold', fontSize=17, leading=22, textColor=BLUE, alignment=TA_CENTER, spaceAfter=15))
styles.add(ParagraphStyle(name='CellLab', fontName='Times-Roman', fontSize=10, leading=12))
styles.add(ParagraphStyle(name='CodeLab', fontName='Courier', fontSize=8.3, leading=11, backColor=colors.HexColor('#f2f2f2'), borderPadding=9, spaceAfter=12))
story=[]

def p(text): story.append(Paragraph(text,styles['BodyLab']))
def h(text): story.append(Paragraph(text,styles['HeadLab']))
def bullet(text): p('&#8226; '+text)
def table(headers, rows, widths):
    cells = [[Paragraph('<b>'+escape(str(x))+'</b>',styles['CellLab']) for x in headers]]
    cells += [[Paragraph(escape(str(x)),styles['CellLab']) for x in row] for row in rows]
    t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.5,colors.HexColor('#bfc4c8')),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#edf2f7')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
    story.extend([t,Spacer(1,6)])
def page(): story.append(PageBreak())

metrics=json.loads((ROOT/'outputs/metrics.json').read_text())
model=np.load(ROOT/'outputs/lda_model.npz',allow_pickle=False)
theta,phi,vocab=model['theta'],model['phi'],model['vocabulary']
with (ROOT/'data/corpus.csv').open(encoding='utf-8',newline='') as f: corpus=list(csv.DictReader(f))
labels=['Computing and software','Sport and cricket','Healthcare','Space exploration']
def chart():
    drawing=Drawing(480,210)
    for k in range(4):
        x=(k%2)*245; y=195-(k//2)*105
        drawing.add(String(x,y,f'Topic {k}: {labels[k]}',fontName='Times-Bold',fontSize=10,fillColor=BLUE))
        top=np.argsort(-phi[k])[:5]
        for j,w in enumerate(top):
            yy=y-17-j*16
            drawing.add(String(x,yy,str(vocab[w]),fontName='Times-Roman',fontSize=9))
            drawing.add(Rect(x+62,yy-1,125*phi[k,w]/phi[k,top[0]],8,fillColor=BLUE,strokeColor=None))
            drawing.add(String(x+194,yy,f'{phi[k,w]:.3f}',fontName='Times-Roman',fontSize=9))
    return drawing

story.append(Paragraph('LAB EXERCISE: 4',styles['TitleLab']))
h('Aim / Objective')
p('Apply Latent Dirichlet Allocation (LDA) to a document collection and interpret the discovered topics using probabilistic and sampling-based methods. The implemented Python pipeline preprocesses text, builds a document-term count matrix, trains LDA through collapsed Gibbs sampling, and exports topic-word and document-topic probabilities with a visual results page.')
h('Expected Outcomes Addressed')
bullet('<b>Preprocess text data:</b> lowercase, tokenize alphabetic words, remove stopwords and tokens shorter than three letters.')
bullet('<b>Build a document-term representation:</b> create a sorted vocabulary and a 32 x 195 integer count matrix.')
bullet('<b>Train an LDA model:</b> sample latent topic assignments for 400 sweeps with fixed random seed 42 and symmetric Dirichlet priors.')
bullet('<b>Interpret topic-word distributions:</b> inspect high-probability words with representative documents and describe four learned topics.')
h('Tools &amp; Technologies Used')
table(['Category','Tool / version','Purpose'],[
['Language / runtime',f'Python {platform.python_version()}','Execution, CSV and JSON processing, regular expressions and tests'],
['Numerical computation',f'NumPy {np.__version__}','Count arrays, random sampling, probability distributions and model export'],
['Frontend','HTML5 / CSS3','Static topic charts, representative documents and output links'],
['Report generation',f'ReportLab {reportlab.Version}','PDF layout, tables and vector charts'],
['Verification','unittest (standard library)','Four automated correctness and validation checks']],[105,135,240])
p('The training pipeline requires NumPy only. ReportLab is an optional dependency for regenerating this PDF. No hosted API, pretrained model or downloaded dataset is used.')
page()

h('Dataset / Corpus Description')
p('The collection contains 32 original educational documents, eight each about space exploration, cricket, healthcare, and computing. Documents were authored for this demonstration and are stored in data/corpus.csv. They are deliberately separable; this is not a real-world benchmark or evidence of general-purpose topic-modelling quality.')
p('Input columns are id, category and text. Only text is used for training. Category values are used after training to describe representative documents. For custom data, id and text are required; category is optional. Empty post-preprocessing documents and duplicate IDs are rejected.')
table(['Property','Observed value'],[['Documents','32'],['Retained tokens / vocabulary','420 / 195'],['Count matrix','32 documents x 195 vocabulary terms'],['Topic count','4 (chosen for demonstration scope)'],['Provenance','Original local corpus; no external dataset']],[180,300])
p('<b>Sample raw record (computing_01):</b> '+escape(corpus[24]['text']))
from lda_lab import preprocess
p('<b>Preprocessed tokens:</b> '+escape(' '.join(preprocess(corpus[24]['text']))))
h('System Architecture / Pipeline Design')
table(['Stage','Input and operation','Output'],[
['1. Input validation','Read UTF-8 CSV; validate IDs and text','Document records'],
['2. Preprocessing','Lowercase, regex tokenization and stopword filtering','Clean token lists'],
['3. Representation','Build vocabulary and integer token counts','Document-term matrix'],
['4. Sampling','Initialize assignments; Gibbs resample token topics','Posterior count states'],
['5. Estimation','Burn-in; average retained theta and phi samples','Normalized distributions'],
['6. Interpretation','Rank words and representative documents','CSV tables, HTML and report']],[105,230,145])
p('The pipeline uses bag-of-words counts. Word order and grammatical structure are not modelled. Stemming is omitted to preserve readable terms, so related morphological forms can remain separate vocabulary entries.')
page()

h('Backend Implementation')
p('<b>6.1 Approach / Algorithm</b>')
p('LDA assumes that each document has a distribution over latent topics and each topic has a distribution over vocabulary words. Symmetric Dirichlet priors use alpha = 0.1 for document-topic mixtures and beta = 0.01 for topic-word distributions.')
p('For each token, remove its current assignment from all counts. Then sample a topic using the following conditional, where the counts exclude the token currently being updated:')
story.append(Preformatted('P(z_i=k | rest) proportional to\n(n_dk + alpha) * (n_kw + beta) / (n_k + V*beta)',styles['CodeLab']))
p('Here n_dk counts topic k in document d, n_kw counts word w in topic k, n_k is the total count for topic k, and V is vocabulary size. After sampling, increment the new counts. Repeat for 400 complete corpus sweeps.')
p('The first 200 sweeps are burn-in. Retain every tenth subsequent state, producing 20 retained samples. Estimate theta and phi for each retained state, then average the estimates. This reduces dependence on one final assignment but does not establish convergence.')
story.append(Preformatted('theta[d,k] = (n_dk + alpha) / (N_d + K*alpha)\nphi[k,w]   = (n_kw + beta) / (n_k + V*beta)',styles['CodeLab']))
p('<b>6.2 Key Code Snippets</b>')
story.append(Preformatted("# Text preprocessing\ndef preprocess(text):\n    return [w for w in re.findall(r'[a-z]+', text.lower())\n            if len(w) > 2 and w not in STOP]",styles['CodeLab']))
story.append(Preformatted('# Resampling, after decrementing the current counts\np = (nd[d] + alpha) * (nw[:, w] + beta)\np = p / (nk + vocab_size * beta)\nz = rng.choice(k, p=p / p.sum())\nassignments[d][i] = z\nnd[d, z] += 1\nnw[z, w] += 1\nnk[z] += 1',styles['CodeLab']))
p('The complete implementation is in lda_lab.py. The exported compressed model contains theta, phi, vocabulary, document IDs, alpha and beta. Raw text and category labels are not embedded in the model file.')
page()

h('Frontend / UI Implementation')
p('<b>7.1 Framework Choice</b>')
p('The frontend is a standalone HTML5/CSS3 results page generated by lda_lab.py. Open outputs/report.html in a browser after training. It runs locally without JavaScript, a server or an API key.')
p('<b>7.2 Interface Features</b>')
bullet('Summary cards show corpus size, vocabulary size, number of topics and sampling sweeps.')
bullet('Four topic cards display the ten leading words and their probabilities. Bar lengths are scaled to the largest displayed probability within each topic.')
bullet('Three representative documents per topic show actual document-topic probabilities and original text.')
bullet('Links provide the interpretation report, topic-word table, document-topic probabilities, preprocessed corpus and run metrics.')
p('<b>Visual evidence:</b> The chart below is generated directly from the saved model arrays. It is a vector results chart, not a browser screenshot. An automated screenshot of the local HTML page was blocked by the browser URL policy; a captured frontend screenshot remains to be added if required for submission.')
story.append(chart())
h('Sample Input &amp; Output')
best=int(np.argmax(theta[:,0]))
table(['Item','Actual result'],[['Input ID',corpus[best]['id']],['Input text',corpus[best]['text']],['Document-topic probabilities',', '.join(f'Topic {k}: {theta[best,k]:.4f}' for k in range(4))],['Dominant topic','Topic 0: Computing and software']],[125,355])
page()

h('Results (Deliverable-Specific Outputs)')
table(['Topic','Interpretation','Top words and P(word | topic)'],[[k,labels[k],', '.join(f'{vocab[w]} ({phi[k,w]:.3f})' for w in np.argsort(-phi[k])[:6])] for k in range(4)],[40,120,320])
p('Interpretations are assigned after training by examining words and representative documents. Topic IDs are arbitrary and can change between runs. Top-word probabilities do not sum to one because only part of the full vocabulary is shown.')
table(['Deliverable','Generated file'],[['Preprocessed corpus','outputs/preprocessed_corpus.csv'],['Document-term counts','outputs/document_term_matrix.csv'],['Saved model distributions','outputs/lda_model.npz'],['Top ten words per topic','outputs/topic_words.csv'],['Document-topic mixtures','outputs/document_topics.csv'],['Interpretation report','outputs/interpretation_report.md'],['Visual results page','outputs/report.html'],['Run metrics','outputs/metrics.json']],[170,310])
h('Evaluation / Analysis')
table(['Check / metric','Observed result','Meaning'],[['Automated tests','4 passed','Preprocessing, deterministic sampling, normalized positive distributions, invalid parameters and empty-document rejection'],['Count invariants','Passed','Counts remain nonnegative; all 420 tokens are conserved'],['Probability normalization','Passed','Each theta and phi row sums to one'],['Training reconstruction perplexity',f'{metrics["training_reconstruction_perplexity"]:.4f}','In-sample word reconstruction; not held-out evaluation']],[140,90,250])
p('Perplexity reconstructs observed words using averaged theta and phi. It is not held-out performance, accuracy or coherence. Precision, recall and F1 are not claimed: no supervised prediction benchmark was defined.')
page()

h('Challenges Faced &amp; Solutions')
bullet('<b>Dataset availability:</b> No corpus was supplied with the rubric. A small original corpus was included with transparent provenance and a CSV input option for replacement data.')
bullet('<b>Sampling correctness:</b> Updating a token without first decrementing its counts biases its conditional. The implementation removes the current assignment before drawing and restores counts afterward; conservation checks validate the result.')
bullet('<b>Reproducibility:</b> Random assignments can produce different topics. A fixed seed, pinned NumPy dependency, recorded settings and deterministic-sampling test make the submitted run reproducible in the tested environment.')
bullet('<b>Interpretation:</b> Topic IDs carry no semantic labels. Top words and representative document text are reviewed jointly; category metadata is used only after fitting.')
bullet('<b>Misleading evaluation:</b> Training fit can look optimistic. The reconstruction metric is explicitly marked in-sample, with no unsupported accuracy claims.')
p('<b>Limitations and possible extensions</b>')
p('The corpus is small and intentionally clean, so separation is easier than for real news or research articles. Bag-of-words ignores word order, negation and semantic context. Fixed stopwords may remove useful domain terms; morphological variants split counts. Four topics were chosen for the demonstration rather than selected through model comparison. One finite Gibbs chain does not demonstrate convergence or sampling uncertainty.')
p('Extensions include a larger real-world corpus, multiple independent chains, topic-count and prior sensitivity studies, held-out inference and evaluation, and topic coherence analysis. For arbitrary custom CSVs, representative-document category descriptions must be reviewed rather than treated as learned ground truth.')
h('Conclusion')
p('A complete LDA topic-modelling pipeline was implemented and executed using collapsed Gibbs sampling. The 32-document demonstration produced four interpretable themes: computing and software, sport and cricket, healthcare, and space exploration. The submission includes cleaned text, count representation, model arrays, probability tables, a visual frontend and reproducibility checks.')
p('The exercise demonstrates how document-topic mixtures and topic-word probabilities are inferred from token counts under Dirichlet priors. It also distinguishes descriptive topic interpretation from predictive evaluation, and documents the limitations of a small corpus and a single sampling run.')
page()

h('Project Repository (GitHub Link)')
p('<link href="https://github.com/Aditisy/Discovering-topics-with-LDA" color="#245882">https://github.com/Aditisy/Discovering-topics-with-LDA</link>')
table(['Field','Value'],[['Repository','Aditisy/Discovering-topics-with-LDA'],['Submission branch','main'],['Visibility','Check repository settings on GitHub'],['Submission contents','Code, corpus, generated results, tests, README, dependencies and this PDF']],[140,340])
p('<b>Setup and backend execution</b>')
story.append(Preformatted('python -m venv .venv\n.venv\\Scripts\\Activate.ps1\npython -m pip install -r requirements.txt\npython lda_lab.py\npython -m unittest discover -s tests -v',styles['CodeLab']))
p('<b>Frontend execution:</b> open outputs/report.html in a local browser. For a custom corpus, run python lda_lab.py --input my_documents.csv --output outputs_custom --topics 4 --iterations 600 --seed 42.')
p('<b>PDF regeneration:</b> install requirements-report.txt, then run python tools/build_report.py from the repository root. Regenerating uses the current saved model and metrics.')
h('References')
bullet('Blei, D. M., Ng, A. Y., and Jordan, M. I. (2003). Latent Dirichlet Allocation. Journal of Machine Learning Research, 3, 993-1022. <link href="https://www.jmlr.org/papers/v3/blei03a.html" color="#245882">jmlr.org/papers/v3/blei03a.html</link>')
bullet('Griffiths, T. L., and Steyvers, M. (2004). Finding scientific topics. PNAS, 101 (Suppl. 1), 5228-5235. DOI: 10.1073/pnas.0307752101. Sampling approach used as methodological background.')
bullet('NumPy documentation: <link href="https://numpy.org/doc/stable/" color="#245882">numpy.org/doc/stable/</link>. Numerical arrays and random sampling.')
bullet('Python documentation: <link href="https://docs.python.org/3/" color="#245882">docs.python.org/3/</link>. CSV, regular expressions, pathlib and unittest.')
p('Code and corpus were prepared for this exercise. No external dataset or copied third-party code was used. The supplied lab rubric defines the task; the supplied Lab 1 report defines the report section order and visual format.')
page()

story.append(Paragraph('ADVANCE NLP (AML23702)<br/>LAB EXERCISE EVALUATION SHEET',styles['TitleLab']))
table(['Student details','Value'],[['Student Name',''],['USN',''],['Lab Exercise No.','4'],['Experiment Title','Discovering Topics with Latent Dirichlet Allocation']],[145,335])
h('Marks Awarded')
criteria=[('Problem Understanding & Scope','5'),('Pipeline Design & Methodology','10'),('Implementation Correctness','10'),('Use of NLP Tools / Models','5'),('Output Quality / Task Performance','10'),('Analysis & Interpretation','5'),('Documentation & Deliverables','2.5'),('Presentation & Reproducibility','2.5')]
table(['Sl. No.','Evaluation Criteria','Max Marks','Marks Obtained'],[[i+1,c,m,''] for i,(c,m) in enumerate(criteria)]+[['','Total','50','']],[45,260,75,100])
h('Overall Comments')
story.append(Spacer(1,65))
table(['Date of Submission','Signature of Faculty'],[['','']],[240,240])

class NumberedCanvas(__import__('reportlab.pdfgen.canvas',fromlist=['Canvas']).Canvas):
    def __init__(self,*a,**kw): super().__init__(*a,**kw); self.saved=[]
    def showPage(self): self.saved.append(dict(self.__dict__)); self._startPage()
    def save(self):
        total=len(self.saved)
        for state in self.saved:
            self.__dict__.update(state)
            self.setFont('Times-Italic',9); self.setFillColor(colors.grey)
            self.drawString(54,756,'Department of AIML')
            self.drawRightString(558,756,'Advance NLP (AML23702)')
            self.setStrokeColor(colors.HexColor('#bbbbbb')); self.setLineWidth(.4); self.line(54,749,558,749)
            self.setFont('Times-Roman',9); self.drawCentredString(306,32,f'Page {self._pageNumber} of {total}')
            super().showPage()
        super().save()

if __name__=='__main__':
    OUT.parent.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(OUT),pagesize=(612,792),rightMargin=66,leftMargin=66,topMargin=60,bottomMargin=55,title='Lab Exercise 4 - Discovering Topics with LDA',author='Department of AIML')
    doc.build(story,canvasmaker=NumberedCanvas)
    print(OUT)
