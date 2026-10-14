import csv
import tempfile
import unittest
from pathlib import Path
from intent_classifier import QuerySignals,check_separation,make_model,read_dataset
from prepare_training_data import recover


class IntentTests(unittest.TestCase):
    def test_rejects_shared_video(self):
        with self.assertRaises(ValueError):check_separation([{'Question ID':'001-1','Video ID':'shared'}],[{'Question ID':'001-2','Video ID':'shared'}])

    def test_rejects_blank_labels_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'data') as directory:
            path=Path(directory)/'invalid.csv'
            for labels in [('',),('Frame','Frame')]:
                with path.open('w',newline='',encoding='utf-8') as f:
                    writer=csv.writer(f);writer.writerow(['Question ID','Video ID','Question','Intent label'])
                    for label in labels:writer.writerow(['001-1','YPaLGP_xr-w','A question?',label])
                with self.assertRaises(ValueError):read_dataset(path)

    def test_validation_does_not_change_vocabulary(self):
        model=make_model()
        model.fit(['red coat colour','blue shirt colour','open box sequence','close door sequence','overall main theme','overall subject summary'],['Frame','Frame','Clip','Clip','Whole-video','Whole-video'])
        before=dict(model.named_steps['features'].vocabulary_)
        self.assertIn(model.predict(['unseenvalidationtoken'])[0],['Frame','Clip','Whole-video'])
        self.assertEqual(before,model.named_steps['features'].vocabulary_)
        self.assertNotIn('unseenvalidationtoken',before)

    def test_recovery_preserves_identity_and_refuses_changed_labels(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'data') as directory:
            root=Path(directory);backup=root/'backup.csv';source=root/'broken.csv'
            with backup.open('w',newline='',encoding='utf-8') as f:
                writer=csv.writer(f);writer.writerow(['Question ID','Video ID','Question','Intent label'])
                writer.writerow(['001-1','YPaLGP_xr-w','What colour, exactly?','Frame'])
            source.write_text('Question ID,Video ID,Question,Intent label\n001-1,YPaLGP_xr-w,What colour, exactly?,Frame,broken subtitle\n',encoding='utf-8')
            self.assertEqual(recover(source,backup)[0]['Question'],'What colour, exactly?')
            source.write_text(source.read_text().replace(',Frame,',',Clip,'),encoding='utf-8')
            with self.assertRaises(ValueError):recover(source,backup)

    def test_event_options_do_not_change_stem_signals(self):
        extractor=QuerySignals()
        stem='In which order do the events appear?'
        plain=extractor.transform([stem]).toarray()
        with_options=extractor.transform([stem+'\n(a) A person wears a red coat.\n(b) Someone explains why.']).toarray()
        self.assertTrue((plain==with_options).all())


if __name__=='__main__':unittest.main()
