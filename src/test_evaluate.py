import unittest

from evaluate import validate_holdout, video_bootstrap_accuracy


class EvaluationTests(unittest.TestCase):
    def test_rejects_test_video_seen_during_development(self):
        train=[{'Question ID':'001-1','Video ID':'train-video'}]
        validation=[{'Question ID':'002-1','Video ID':'validation-video'}]
        test=[{'Question ID':'002-2','Video ID':'validation-video'}]
        with self.assertRaises(ValueError):
            validate_holdout(train,validation,test)

    def test_perfect_predictions_have_perfect_bootstrap_interval(self):
        rows=[{'Video ID':'a','Intent label':'Frame'}, {'Video ID':'a','Intent label':'Clip'}, {'Video ID':'b','Intent label':'Whole-video'}]
        result=video_bootstrap_accuracy(rows,['Frame','Clip','Whole-video'],draws=20)
        self.assertEqual(result['lower'],1.0)
        self.assertEqual(result['upper'],1.0)


if __name__=='__main__':
    unittest.main()
