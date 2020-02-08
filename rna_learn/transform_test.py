import unittest

from .transform import sequence_embedding


alphabet_dna = ['A', 'T', 'G', 'C']
alphabet_protein = [
    'A', 'C', 'D', 'E', 'F',
    'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R', 
    'S', 'T', 'V', 'W', 'Y',
]


class TestTransform(unittest.TestCase):

    def test_sequence_embedding(self):
        seq_oh = sequence_embedding(['GAAGTC', 'AA'], alphabet_dna)

        self.assertEqual((2, 6, 4), seq_oh.shape)

        self.assertEqual([
            [0., 0., 1., 0.],
            [1., 0., 0., 0.],
            [1., 0., 0., 0.],
            [0., 0., 1., 0.],
            [0., 1., 0., 0.],
            [0., 0., 0., 1.],
        ], seq_oh[0].tolist())

        self.assertEqual([
            [1., 0., 0., 0.],
            [1., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
        ], seq_oh[1].tolist())

        seq_oh = sequence_embedding(['IYL'], alphabet_protein, dtype='float64')

        self.assertEqual((1, 3, 20), seq_oh.shape)

        self.assertEqual([
            [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        ], seq_oh[0].tolist())

        seq_oh = sequence_embedding(['GA-GTC'], alphabet_dna)

        self.assertEqual((1, 6, 4), seq_oh.shape)

        self.assertEqual([
            [0., 0., 1., 0.],
            [1., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 1., 0.],
            [0., 1., 0., 0.],
            [0., 0., 0., 1.],
        ], seq_oh[0].tolist())


if __name__ == '__main__':
    unittest.main()
