import unittest
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
from lda_lab import preprocess, fit

class LDATest(unittest.TestCase):
    def test_preprocessing(self):
        self.assertEqual(preprocess('The SPACE station, and 123 rockets!'), ['space','station','rockets'])

    def test_sampling_invariants_and_reproducibility(self):
        docs = [[0,0,1,1,0], [2,3,2,3,3], [0,1,2,3]]
        a,b,n = fit(docs, 4, k=2, iterations=60)
        c,d,_ = fit(docs, 4, k=2, iterations=60)
        np.testing.assert_array_equal(a,c)
        np.testing.assert_array_equal(b,d)
        np.testing.assert_allclose(a.sum(axis=1),1)
        np.testing.assert_allclose(b.sum(axis=1),1)
        self.assertTrue(np.all(a>0) and np.all(b>0))
        self.assertEqual(n,3)

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            fit([[0]],1,k=1)

    def test_cli_rejects_empty_document(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            source = Path(directory)/'empty.csv'
            source.write_text('id,text\n1,the and of\n', encoding='utf-8')
            result = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1]/'lda_lab.py'), '--input', str(source)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Every document must contain tokens', result.stderr)

if __name__ == '__main__':
    unittest.main()
