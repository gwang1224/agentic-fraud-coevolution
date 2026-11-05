"""Verify round boundaries, feedback routing, and failure accounting offline."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from coevolution import run_coevolution


class CoevolutionTests(unittest.TestCase):
    def test_feedback_is_from_prior_rounds_and_miss_is_recorded(self):
        sequence = {'outcome': 'successful_fraud', 'is_complete': True,
                    'family': 'generic', 'fallback_count': 1}
        decisions = [{'classification': label, 'summary': 'Observation summary.', 'evidence': []}
                     for label in ('normal', 'successful_fraud')]
        with TemporaryDirectory() as temporary, \
                patch('coevolution.generate_sequence', return_value=sequence) as generate, \
                patch('coevolution.detect_sequence', side_effect=decisions) as detect, \
                patch('coevolution.prepare_observations', return_value=[{'step': 1}]):
            output = Path(temporary) / 'run'
            report = run_coevolution(rounds=2, output_dir=output)
            self.assertEqual(report['successful_fraud_recall'], 0.5)
            self.assertEqual(generate.call_args_list[0].kwargs['feedback'], [])
            self.assertEqual(detect.call_args_list[0].kwargs['feedback'], [])
            self.assertEqual(generate.call_args_list[1].kwargs['feedback'][0]
                             ['detector_classification'], 'normal')
            self.assertEqual(detect.call_args_list[1].kwargs['feedback'][0]
                             ['expected'], 'successful_fraud')
            self.assertEqual(json.loads((output / 'report.json').read_text()), report)
            with self.assertRaises(FileExistsError):
                run_coevolution(output_dir=output)

    def test_provider_failure_is_not_a_missed_detection(self):
        with TemporaryDirectory() as temporary, \
                patch('coevolution.generate_sequence', side_effect=RuntimeError('offline')):
            output = Path(temporary) / 'run'
            with self.assertRaisesRegex(RuntimeError, 'offline'):
                run_coevolution(output_dir=output)
            report = json.loads((output / 'report.json').read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(report['rounds'], [])
            self.assertNotIn('successful_fraud_recall', report)

    def test_invalid_round_count_does_not_create_output(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / 'run'
            for value in (0, -1, True, 101):
                with self.assertRaises(ValueError):
                    run_coevolution(rounds=value, output_dir=output)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
