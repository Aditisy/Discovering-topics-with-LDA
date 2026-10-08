import csv
import json
import unittest
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

class WebsiteExportTests(unittest.TestCase):
    def test_export_matches_saved_model(self):
        data=json.loads((ROOT/'dist/assets/model-data.json').read_text())
        model=np.load(ROOT/'outputs/lda_model.npz',allow_pickle=False)
        np.testing.assert_allclose(data['phi'],model['phi'])
        self.assertEqual(data['vocabulary'],model['vocabulary'].tolist())
        np.testing.assert_allclose([d['theta'] for d in data['documents']],model['theta'])
        self.assertEqual([d['id'] for d in data['documents']],model['document_ids'].tolist())
        self.assertAlmostEqual(sum(t['prevalence'] for t in data['topics']),1)

    def test_clean_tokens_and_downloads(self):
        data=json.loads((ROOT/'dist/assets/model-data.json').read_text())
        with (ROOT/'outputs/preprocessed_corpus.csv').open(newline='',encoding='utf-8') as f:
            clean={r['id']:r['tokens'].split() for r in csv.DictReader(f)}
        for d in data['documents']:self.assertEqual(d['tokens'],clean[d['id']])
        for p in (ROOT/'dist/downloads').iterdir():self.assertEqual(p.read_bytes(),(ROOT/'outputs'/p.name).read_bytes())

if __name__=='__main__':unittest.main()
