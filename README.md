# Exercise 4: Latent Dirichlet Allocation

Advanced Natural Language Processing (AML23702), CO4 / L3.

Apply LDA to a document collection and interpret discovered topics using probabilistic and sampling-based methods. This project implements collapsed Gibbs sampling with NumPy, including burn-in and posterior averaging. All output files are generated from a real run.

## Run

Use Python 3.11 or later in a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python lda_lab.py
python -m unittest discover -s tests -v
```

Open `outputs/report.html` in a browser to review topic-word probability charts and representative documents. The generated interpretation report explains the algorithm, results, assumptions, and limitations.

## Dataset

`data/corpus.csv` contains 32 original educational documents across space, cricket, healthcare, and software. It is intentionally separable and is not an external benchmark. `data/build_corpus.py` regenerates it. Category labels are excluded from training. No private data, online services, or credentials are required.

For another collection, supply a UTF-8 CSV with unique `id` and nonempty `text` columns. An optional `category` column is used only to describe representative documents; review these descriptions manually for heterogeneous data.

```powershell
python lda_lab.py --input my_documents.csv --output outputs_custom --topics 4 --iterations 600 --seed 42
```

## Deliverables and rubric alignment

| Requirement | File / evidence |
|---|---|
| Problem understanding and scope | README and report aim, assumptions and dataset provenance |
| Pipeline and methodology | `lda_lab.py`; report conditional sampling equation and priors |
| Preprocessed corpus | `outputs/preprocessed_corpus.csv` |
| Document-term representation | `outputs/document_term_matrix.csv` |
| LDA model output | `outputs/lda_model.npz` (theta, phi, vocabulary, document IDs and priors) |
| Topic-word lists | `outputs/topic_words.csv` (top ten per topic) |
| Document-topic distributions | `outputs/document_topics.csv` |
| Interpretation and limitations | `outputs/interpretation_report.md` |
| Optional visualisations and demo | `outputs/report.html` |
| Correctness and reproducibility | fixed seed, saved metrics and unit tests |

Inspect the saved model with `numpy.load('outputs/lda_model.npz', allow_pickle=False)`. `phi[k,w]` is P(word w | topic k); `theta[d,k]` is P(topic k | document d). Each row sums to one. LDA learns distributions, not category predictions. The reported training reconstruction perplexity is not held-out evaluation. Four topics reflect the demonstration design; optimal topic selection and convergence studies remain extensions.

## Demo / viva guide

1. Show one raw document beside its preprocessed tokens and count row.
2. Explain that each token receives a sampled latent topic assignment.
3. Explain the document-topic and topic-word count factors in the conditional.
4. Open the HTML report and interpret words jointly with document examples.
5. Discuss bag-of-words limitations, synthetic corpus bias, and finite-chain uncertainty.

## Formatted lab report

The formatted PDF is delivered separately and excluded from GitHub. It follows the supplied sample's section order, department header, blue headings, Times body text, page numbering and evaluation sheet. Student name and USN are blank for completion. Website screenshots are pending capture and will be added as image files, without uploading the PDF.

To regenerate the PDF from the current output arrays, install `requirements-report.txt` and run `python tools/build_report.py` from the repository root.

Repository: https://github.com/Aditisy/Discovering-topics-with-LDA
