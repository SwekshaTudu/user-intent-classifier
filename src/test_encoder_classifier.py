import unittest
import importlib.util

import numpy as np

from encoder_classifier import DEFAULT_ENCODER, MiniLMEncoder, head_features, mean_pool_and_normalize, question_stem


class EncoderTests(unittest.TestCase):
    def test_padding_cannot_influence_sentence_embedding(self):
        tokens = np.array([[[1., 0.], [0., 1.], [100., -100.]]])
        actual = mean_pool_and_normalize(tokens, np.array([[1, 1, 0]]))
        np.testing.assert_allclose(actual, [[2**-0.5, 2**-0.5]], rtol=1e-6)
        tokens[0, 2] = [-9999., 9999.]
        np.testing.assert_allclose(mean_pool_and_normalize(tokens, [[1, 1, 0]]), actual)

    def test_zero_mask_does_not_produce_nan(self):
        result = mean_pool_and_normalize(np.ones((1, 3, 2)), [[0, 0, 0]])
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue((result == 0).all())

    def test_event_list_does_not_overwhelm_question_stem(self):
        stem = 'In which order do these events appear?'
        self.assertEqual(question_stem(stem+'\n(a) A long event.\n(b) Another event.'), stem)
        self.assertEqual(question_stem('What colour is the car?'), 'What colour is the car?')

    def test_signal_features_do_not_mutate_embeddings(self):
        embeddings = np.ones((1, 384), dtype=np.float32)
        combined = head_features(embeddings, ['What colour is the car?'], True)
        self.assertEqual(combined.shape[0], 1)
        self.assertGreater(combined.shape[1], 384)
        np.testing.assert_array_equal(embeddings, np.ones((1, 384)))

    @unittest.skipUnless(importlib.util.find_spec('onnxruntime') and (DEFAULT_ENCODER/'encoder_manifest.json').exists(), 'Optional local encoder assets unavailable')
    def test_downloaded_encoder_produces_finite_unit_vectors(self):
        encoder = MiniLMEncoder(DEFAULT_ENCODER)
        vectors = encoder.encode(['What colour is the car?', 'Describe the main topic.'])
        self.assertEqual(vectors.shape, (2, 384))
        self.assertTrue(np.isfinite(vectors).all())
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1., 1.], atol=1e-5)


if __name__ == '__main__':
    unittest.main()
