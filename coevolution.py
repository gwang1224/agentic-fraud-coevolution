"""Run bounded prompt-feedback rounds between the synthetic generator and detector."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from detector_agent import detect_sequence, prepare_observations
from generator_agent import generate_sequence


ROOT = Path(__file__).resolve().parent


def _save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def run_coevolution(*, rounds=3, output_dir=None, generator_model=None,
                    detector_model=None, verbose=False):
    """Adapt prompts using only completed prior rounds; never update model weights.

    Each run owns a fresh directory. Failed rounds are recorded and stop the run;
    a provider error must never count as a detector miss.
    """
    if isinstance(rounds, bool) or not isinstance(rounds, int) or not 1 <= rounds <= 100:
        raise ValueError('rounds must be an integer between 1 and 100.')
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        output_dir = ROOT / 'data' / 'coevolution' / stamp
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / 'report.json'
    report = {'method': 'prompt_feedback', 'status': 'running', 'requested_rounds': rounds,
              'rounds': [], 'limitations': [
                  'Successful-fraud examples only: this is a training trace, not an accuracy benchmark.',
                  'Prompt adaptation only; model weights and Neo4j are unchanged.',
                  'Fallback-generated steps are included and counted separately.',
              ]}
    _save(report_path, report)
    generator_history, detector_history = [], []
    for number in range(1, rounds + 1):
        directory = output_dir / f'round_{number:03d}'
        directory.mkdir()
        if verbose:
            print(f'Coevolution round {number}/{rounds}: {directory}', flush=True)
        # Persist the exact feedback supplied before either model sees this round.
        _save(directory / 'feedback.json', {'generator': generator_history,
                                           'detector': detector_history})
        try:
            sequence = generate_sequence(output=directory / 'sequence.json',
                                         model=generator_model, feedback=generator_history,
                                         verbose=verbose)
            decision = detect_sequence(directory / 'sequence.json',
                                       output=directory / 'detection.json', model=detector_model,
                                       feedback=detector_history, verbose=verbose)
            expected = sequence['outcome']
            if expected != 'successful_fraud' or not sequence.get('is_complete'):
                raise ValueError('Coevolution requires a complete successful-fraud sequence.')
            matched = decision['classification'] == expected
            record = {'round': number, 'expected': expected,
                      'classification': decision['classification'], 'matched': matched,
                      'fallback_count': sequence['fallback_count']}
            report['rounds'].append(record)
            # Bounded histories keep the context from growing without limit.
            generator_history = (generator_history + [{
                'round': number, 'family': sequence['family'],
                'detector_classification': decision['classification'],
                'detector_summary': decision['summary'], 'evidence': decision['evidence'],
                'goal': 'Vary fictional template-compatible examples while keeping observable facts explicit.',
            }])[-3:]
            detector_history = (detector_history + [{
                'round': number, 'observations': prepare_observations(sequence),
                'expected': expected, 'previous_classification': decision['classification'],
                'matched': matched,
            }])[-3:]
            _save(report_path, report)
        except Exception as exc:
            report.update(status='failed', failed_round=number,
                          error=f'{type(exc).__name__}: {exc}')
            _save(report_path, report)
            raise
    report['status'] = 'complete'
    report['successful_fraud_recall'] = sum(row['matched'] for row in report['rounds']) / rounds
    _save(report_path, report)
    if verbose:
        print(f'Coevolution complete; report saved to {report_path}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--output-dir', type=Path, help='A new directory for this run.')
    args = parser.parse_args()
    run_coevolution(rounds=args.rounds, output_dir=args.output_dir, verbose=True)


if __name__ == '__main__':
    main()
