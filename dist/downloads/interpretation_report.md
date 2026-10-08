# Exercise 4 - Latent Dirichlet Allocation

## Aim and scope
Apply unsupervised LDA to a document collection, estimate probabilistic topic-word and document-topic distributions, and interpret topics. The supplied corpus is 32 original, deliberately separable educational documents in four categories; it is a demonstration dataset, not a real-world benchmark. Categories are excluded from training and used only for post-hoc interpretation. For other CSVs, labels must be assigned by reading top words and examples.

## Pipeline and methodology
Lowercase -> alphabetic tokenisation -> stopword and short-token removal -> vocabulary -> document-term counts -> collapsed Gibbs sampling -> posterior averages -> interpretation. No stemming is used, preserving readable terms but allowing morphological variants to split counts. No external corpus downloads or API keys are required.

LDA assumes each document mixes topics and each topic distributes probability over words. Symmetric Dirichlet priors are alpha=0.1 and beta=0.01. For each token, remove its old assignment and sample a new topic with probability proportional to (n_document,topic + alpha) * (n_topic,word + beta) / (n_topic + V*beta). Counts exclude the current token. Theta=(n_document,topic+alpha)/(document_length+K*alpha); phi=(n_topic,word+beta)/(n_topic+V*beta). The first half of iterations is burn-in; retain every tenth subsequent sample and average theta/phi. Topic IDs have no semantic ordering.

## Actual run
{
  "documents": 32,
  "vocabulary": 195,
  "tokens": 420,
  "topics": 4,
  "iterations": 400,
  "seed": 42,
  "retained_samples": 20,
  "training_reconstruction_perplexity": 42.089111207276986
}

## Topic interpretation
Topic 0: Computing and software. Leading words: data, server, software, computer, developer, network. Representative documents: computing_01, computing_06, computing_03.

Topic 1: Sport and cricket. Leading words: cricket, team, tournament, match, players, runs. Representative documents: sport_05, sport_04, sport_01.

Topic 2: Healthcare. Leading words: patient, doctor, health, hospital, treatment, medical. Representative documents: healthcare_07, healthcare_01, healthcare_02.

Topic 3: Space exploration. Leading words: orbit, mission, space, satellite, planet, spacecraft. Representative documents: space_01, space_08, space_04.

## Analysis and limitations
Top words and representative documents jointly support each topic interpretation. Bar widths are relative to the largest displayed word probability; numeric values are full-vocabulary probabilities. LDA is unsupervised: category labels do not establish predictive accuracy. Training reconstruction perplexity describes fit to observed words, not held-out performance, and must not be reported as test perplexity. Four topics match the intended demonstration scope; this is not an empirically selected optimal K. Small, clean documents make topic separation easier than noisy real data. Bag-of-words loses word order, negation, and context. Finite Gibbs runs can mix slowly; one seed does not demonstrate convergence. Useful extensions include a larger real corpus, several chains/seeds, held-out evaluation, coherence measures, and K sensitivity analysis.

## Verification and deliverables
The run checks nonnegative counts, token conservation, and normalised distributions. Tests cover deterministic sampling, normalisation, empty documents at the CLI, and text preprocessing. Preprocessed corpus, document-term matrix, complete model arrays, document-topic probabilities, top-word lists, metrics, and this report are saved in outputs. report.html presents the actual results for review.
