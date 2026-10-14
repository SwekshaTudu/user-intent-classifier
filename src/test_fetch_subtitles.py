import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fetch_subtitles import enrich_rows, fetch_one, read_csv, save_csv, validate_srt

SRT = b'1\n00:00:01,000 --> 00:00:03,000\nHello, world!\n\n'
TEST_ROOT = Path(__file__).resolve().parents[1] / 'data'


class SubtitleTests(unittest.TestCase):
    def test_missing_is_not_a_network_error(self):
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as directory, patch('fetch_subtitles.request_bytes') as request:
            result = fetch_one('YPaLGP_xr-w', set(), 'revision', Path(directory))
            self.assertTrue(result['status'].startswith('Unavailable'))
            request.assert_not_called()

    def test_download_and_cache_keep_timestamps(self):
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as directory, patch('fetch_subtitles.request_bytes', return_value=(SRT, {})) as request:
            arguments = ('YPaLGP_xr-w', {'subtitles/YPaLGP_xr-w.srt'}, 'revision', Path(directory))
            first = fetch_one(*arguments)
            second = fetch_one(*arguments)
            self.assertEqual(first, second)
            self.assertIn('00:00:01,000 -->', first['text'])
            self.assertEqual(request.call_count, 1)

    def test_error_does_not_become_missing_or_erase_context(self):
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as directory, patch('fetch_subtitles.request_bytes', side_effect=OSError('offline')):
            result = fetch_one('YPaLGP_xr-w', {'subtitles/YPaLGP_xr-w.srt'}, 'revision', Path(directory))
        self.assertEqual(result['status'], 'Download failed')
        fields, rows = enrich_rows(['Subtitle text'], [{'Video ID':'YPaLGP_xr-w','Subtitle text':'existing'}], {'YPaLGP_xr-w':result})
        self.assertEqual(rows[0]['Subtitle text'], 'existing')

    def test_multiline_csv_round_trip_preserves_labels(self):
        row = {'Video ID':'YPaLGP_xr-w','Question ID':'020-1','Question':'Why, exactly?','Intent label':'Clip','Reason':'Speech, with context.'}
        fields, rows = enrich_rows(list(row), [row], {'YPaLGP_xr-w':{'url':'https://example.test/srt','text':validate_srt(SRT),'status':'Downloaded'}})
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as directory:
            path = save_csv(Path(directory)/'train.csv', fields, rows)
            self.assertEqual(read_csv(path), (fields, rows))
        self.assertEqual(rows[0]['Intent label'], 'Clip')

    def test_html_is_not_cached_as_subtitles(self):
        with self.assertRaises(ValueError):
            validate_srt(b'<html>Service unavailable</html>')


if __name__ == '__main__':
    unittest.main()
