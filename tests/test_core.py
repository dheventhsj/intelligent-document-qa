import unittest
from core import read_file, corpus_id, citation_status, retrieve, answer, ABSTAIN

class CoreTests(unittest.TestCase):
    def test_text_and_unsupported(self):
        self.assertEqual(read_file('a.txt', b'hello')[0]['text'], 'hello')
        with self.assertRaises(ValueError): read_file('a.exe', b'bad')
    def test_index_identity(self):
        self.assertEqual(corpus_id([('a', b'a'), ('b', b'b')], 256, 32), corpus_id([('b', b'b'), ('a', b'a')], 256, 32))
        self.assertNotEqual(corpus_id([('a', b'a')], 256, 32), corpus_id([('a', b'b')], 256, 32))
    def test_citations(self):
        self.assertEqual(citation_status('Claim [1]', 1), 'valid IDs')
        self.assertEqual(citation_status('Claim [2]', 1), 'invalid IDs')
        self.assertEqual(citation_status('Claim', 1), 'missing')
    def test_filter_and_abstain(self):
        class FakeStore:
            def similarity_search_with_score(self, question, k): return [('close', .2), ('far', 1.1)]
        self.assertEqual(retrieve(FakeStore(), 'test', 2, .9), [('close', .2)])
        self.assertEqual(answer('test', [])[0], ABSTAIN)

if __name__ == '__main__': unittest.main()
