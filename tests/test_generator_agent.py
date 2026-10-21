"""Regression tests for finite, compatible, successful synthetic fraud paths."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from generator_agent import (Action, Transaction, check_completion,
                             generate_next_action, generate_sequence,
                             initialize_state, load_character_actions,
                             scope_available_actions, select_starting_character,
                             update_state, validate_step)


FRAUDSTER = {'name': 'Synthetic Fraud Actor', 'account_id': 'fraud-account',
             'balance': 200}
VICTIM = {'name': 'Alex Morgan', 'account_id': 'victim-account', 'balance': 5000}


def template(name, stage, family='generic', **extra):
    return {'action': name, 'stage': stage, 'family': family,
            'channels': ['sms'], 'is_terminal': False, **extra}


def minimal_templates():
    return {
        True: [template('contact', 'initial_contact', 'bank'),
               template('transfer', 'final_transaction', is_terminal=True,
                        channels=['bank_app'])],
        False: [template('share', 'disclosure')],
    }


def full_templates():
    rows = minimal_templates()
    rows[True].extend([
        template('investment_offer', 'initial_contact', 'investment'),
        template('follow_up', 'follow_up'),
        template('investment_transfer', 'final_transaction', 'investment',
                 is_terminal=True, channels=['bank_app']),
    ])
    rows[False].extend([
        template('bank_reply', 'response', 'bank'),
        template('investment_reply', 'response', 'investment'),
        template('generic_reply', 'response'),
        template('report', 'report', is_terminal=True),
    ])
    return rows


def choose(name, **extra):
    return {'action': name, 'channel': 'sms',
            'description': 'A fictional simulation step.', **extra}


def transfer(amount=3000, name='transfer', **extra):
    return {'action': name, 'payment_method': 'bank_app', 'amount': amount, **extra}


class FakeModel:
    """Plain invoke-only model: it never connects to Ollama."""
    def __init__(self, payloads):
        self.payloads = iter(payloads)
        self.calls = 0
        self.prompts = []

    def invoke(self, prompt):
        self.calls += 1
        self.prompts.append(prompt)
        payload = next(self.payloads)
        if isinstance(payload, Exception):
            raise payload
        return SimpleNamespace(content=payload if isinstance(payload, str)
                               else json.dumps(payload))


class GeneratorTests(unittest.TestCase):
    def initial_state(self, rows=None, victim=None, max_steps=15):
        rows = rows if rows is not None else minimal_templates()
        with patch('generator_agent.load_character_actions',
                   side_effect=lambda name, is_fraudster: deepcopy(rows[is_fraudster])):
            return initialize_state(dict(FRAUDSTER), dict(victim or VICTIM),
                                    max_steps=max_steps)

    def apply(self, state, name, effect, actor=None):
        scope_available_actions(state, FRAUDSTER)
        actor = actor or state['current_character']
        target = VICTIM['name'] if actor == FRAUDSTER['name'] else FRAUDSTER['name']
        step = Action(actor, name, target, 'sms', 'Synthetic simulation step', effect)
        update_state(state, step, FRAUDSTER, VICTIM)
        return step

    def transfer_state(self):
        state = self.initial_state()
        self.apply(state, 'contact', 'contact')
        self.apply(state, 'share', 'disclose')
        scope_available_actions(state, FRAUDSTER)
        return state

    def run_sequence(self, payloads, rows=None, **kwargs):
        rows = rows if rows is not None else minimal_templates()
        model = FakeModel(payloads)
        with TemporaryDirectory() as directory, patch(
                'generator_agent.select_starting_character',
                return_value=(dict(FRAUDSTER), dict(VICTIM))), patch(
                'generator_agent.load_character_actions',
                side_effect=lambda name, is_fraudster: deepcopy(rows[is_fraudster])) as lookup:
            output = Path(directory) / 'sequence.json'
            result = generate_sequence(output=output, model=model, **kwargs)
            self.assertEqual(result, json.loads(output.read_text()))
        return result, model, lookup

    def test_select_participants_from_neo4j_requires_positive_victim_balance(self):
        driver = MagicMock()
        session = driver.session.return_value.__enter__.return_value
        tx = MagicMock()
        tx.run.return_value.single.side_effect = [dict(FRAUDSTER), dict(VICTIM)]
        session.execute_read.side_effect = lambda callback: callback(tx)
        with patch('generator_agent.connect', return_value=(driver, 'test-db')):
            self.assertEqual(select_starting_character(), (FRAUDSTER, VICTIM))
        driver.session.assert_called_once_with(database='test-db')
        queries = [call.args[0] for call in tx.run.call_args_list]
        self.assertIn('p:fraudsters', queries[0])
        self.assertIn('p:victims', queries[1])
        self.assertTrue(all('rand()' in query and '[:OWNS]' in query for query in queries))
        self.assertRegex(queries[1], r'balance\s*>\s*0')

    def test_select_participants_requires_populated_graph(self):
        driver = MagicMock()
        session = driver.session.return_value.__enter__.return_value
        tx = MagicMock()
        tx.run.return_value.single.return_value = None
        session.execute_read.side_effect = lambda callback: callback(tx)
        with patch('generator_agent.connect', return_value=(driver, 'test-db')):
            with self.assertRaises(ValueError):
                select_starting_character()
        driver.__exit__.assert_called_once()

    def test_action_lookup_uses_character_relationship(self):
        driver = MagicMock()
        session = driver.session.return_value.__enter__.return_value
        tx = MagicMock()
        expected = template('reply', 'response')
        tx.run.return_value = [{'action': expected}]
        session.execute_read.side_effect = lambda callback: callback(tx)
        with patch('generator_agent.connect', return_value=(driver, 'test-db')):
            self.assertEqual(load_character_actions('Taylor Rivera', False), [expected])
        query = tx.run.call_args.args[0]
        self.assertIn('p:victims', query)
        self.assertIn('[:PERFORMS]', query)
        self.assertEqual(tx.run.call_args.kwargs, {'name': 'Taylor Rivera'})

    def test_minimal_complete_route_and_balances(self):
        result, model, lookup = self.run_sequence([
            choose('contact'), choose('share'), transfer()])
        self.assertEqual(result['outcome'], 'successful_fraud')
        self.assertTrue(result['is_complete'])
        self.assertEqual(result['stop_reason'], 'final_transaction')
        self.assertEqual([row['action'] for row in result['steps']],
                         ['contact', 'share', 'transfer'])
        self.assertEqual(model.calls, 3)
        self.assertEqual({call.args for call in lookup.call_args_list},
                         {(FRAUDSTER['name'], True), (VICTIM['name'], False)})
        self.assertEqual(lookup.call_count, 2)
        trace = result['generation_trace']
        self.assertTrue(trace[0]['state_after']['contacted'])
        self.assertTrue(trace[1]['state_after']['disclosed'])
        self.assertEqual(trace[-1]['state_after']['balance'], 2000)
        self.assertEqual(trace[-1]['state_after']['fraudster_balance'], 3200)
        self.assertEqual(trace[-1]['state_after']['balance'] +
                         trace[-1]['state_after']['fraudster_balance'],
                         VICTIM['balance'] + FRAUDSTER['balance'])
        transaction = result['steps'][-1]
        self.assertEqual(transaction['source_account'], VICTIM['account_id'])
        self.assertEqual(transaction['destination_account'], FRAUDSTER['account_id'])

    def test_response_follow_up_and_family_are_preserved(self):
        result, model, _ = self.run_sequence([
            choose('investment_offer'), choose('investment_reply'),
            choose('follow_up'), choose('share'),
            transfer(name='investment_transfer')], rows=full_templates())
        self.assertEqual([row['action'] for row in result['steps']], [
            'investment_offer', 'investment_reply', 'follow_up', 'share',
            'investment_transfer'])
        self.assertEqual([row.get('effect') for row in result['steps'][:-1]],
                         ['contact', 'none', 'none', 'disclose'])
        self.assertEqual(model.calls, 5)
        self.assertEqual(result['outcome'], 'successful_fraud')
        accepted = [row for row in result['generation_trace']
                    if row['validation'] == 'accepted']
        self.assertEqual(accepted[1]['available_actions'], ['investment_reply'])
        self.assertEqual(accepted[-1]['available_transactions'], ['investment_transfer'])

    def test_incompatible_choices_are_retried_without_switching_actor(self):
        result, model, _ = self.run_sequence([
            choose('investment_offer'), choose('bank_reply'),
            choose('investment_reply'), choose('follow_up'), choose('share'),
            transfer(name='investment_transfer')], rows=full_templates())
        rejected = [row for row in result['generation_trace']
                    if row['validation'] == 'rejected']
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]['actor'], VICTIM['name'])
        self.assertEqual(rejected[0]['available_actions'], ['investment_reply'])
        self.assertEqual(model.calls, 6)
        self.assertEqual(result['outcome'], 'successful_fraud')

    def test_only_current_stage_is_available_and_stale_lists_cannot_bypass_it(self):
        state = self.initial_state(full_templates())
        self.apply(state, 'investment_offer', 'contact')
        scope_available_actions(state, FRAUDSTER)
        self.assertEqual([row['action'] for row in state['available_actions']],
                         ['investment_reply'])
        # A stale contact template must not allow another initial contact.
        state['available_actions'] = [template('contact', 'initial_contact', 'bank')]
        state['available_effects'] = ['contact']
        with self.assertRaises(ValueError):
            validate_step(Action(VICTIM['name'], 'contact', FRAUDSTER['name'],
                                 'sms', 'Invalid repeated contact', 'contact'),
                          state, FRAUDSTER, VICTIM)

    def test_response_effect_and_fixed_participants_are_derived_in_python(self):
        state = self.initial_state(full_templates())
        self.apply(state, 'contact', 'contact')
        scope_available_actions(state, FRAUDSTER)
        model = FakeModel([choose('bank_reply', effect='disclose',
                                  actor='invented actor', target='invented target')])
        step = generate_next_action(state, None, FRAUDSTER, VICTIM, model)
        self.assertEqual(step.effect, 'none')
        self.assertEqual((step.actor, step.target), (VICTIM['name'], FRAUDSTER['name']))
        validate_step(step, state, FRAUDSTER, VICTIM)
        self.assertFalse(state['graph']['disclosed'])
        self.assertEqual(len(state['generated_steps']), 1)

    def test_model_cannot_override_transfer_accounts(self):
        state = self.transfer_state()
        step = generate_next_action(state, None, FRAUDSTER, VICTIM, FakeModel([
            transfer(source_account='wrong', destination_account='wrong')]))
        self.assertEqual(step.source_account, VICTIM['account_id'])
        self.assertEqual(step.destination_account, FRAUDSTER['account_id'])
        validate_step(step, state, FRAUDSTER, VICTIM)

    def test_model_schema_offers_only_current_family_action_and_its_channels(self):
        from langchain_ollama import ChatOllama

        state = self.initial_state(full_templates())
        self.apply(state, 'investment_offer', 'contact')
        model = ChatOllama(model='test-model')
        response = SimpleNamespace(content=json.dumps(choose('investment_reply')))
        with patch.object(ChatOllama, 'bind') as bind:
            bind.return_value.invoke.return_value = response
            step = generate_next_action(state, None, FRAUDSTER, VICTIM, model)
        branches = bind.call_args.kwargs['format']['anyOf']
        self.assertEqual(len(branches), 1)
        self.assertEqual(branches[0]['properties']['action']['enum'], ['investment_reply'])
        self.assertEqual(branches[0]['properties']['channel']['enum'], ['sms'])
        self.assertEqual(set(branches[0]['properties']), {'action', 'channel', 'description'})
        self.assertEqual(step.effect, 'none')

    def test_transfer_schema_omits_action_fields_and_bounds_amount(self):
        from langchain_ollama import ChatOllama

        state = self.transfer_state()
        model = ChatOllama(model='test-model')
        response = SimpleNamespace(content=json.dumps(transfer()))
        with patch.object(ChatOllama, 'bind') as bind:
            bind.return_value.invoke.return_value = response
            generate_next_action(state, None, FRAUDSTER, VICTIM, model)
        schema = bind.call_args.kwargs['format']
        self.assertEqual(set(schema['properties']), {'action', 'payment_method', 'amount'})
        self.assertEqual(schema['properties']['action']['enum'], ['transfer'])
        self.assertEqual(schema['properties']['amount']['maximum'], VICTIM['balance'])
        self.assertEqual(schema['properties']['amount']['exclusiveMinimum'], 0)

    def test_transfer_turn_excludes_ordinary_actions(self):
        state = self.transfer_state()
        self.assertEqual(state['available_actions'], [])
        self.assertEqual(state['available_effects'], [])
        self.assertEqual([row['action'] for row in state['available_transactions']],
                         ['transfer'])
        with self.assertRaises(ValueError):
            validate_step(Action(FRAUDSTER['name'], 'contact', VICTIM['name'],
                                 'sms', 'Out of order contact', 'contact'),
                          state, FRAUDSTER, VICTIM)

    def test_transfer_amount_and_destination_are_validated(self):
        state = self.transfer_state()
        for amount in (0, -1, True, float('nan'), float('inf'), 5001, '100'):
            with self.subTest(amount=amount), self.assertRaises(ValueError):
                validate_step(Transaction(VICTIM['account_id'], 'bank_app',
                                          FRAUDSTER['account_id'], amount, 'transfer'),
                              state, FRAUDSTER, VICTIM)
        for source, destination in [('wrong', FRAUDSTER['account_id']),
                                    (VICTIM['account_id'], 'wrong')]:
            with self.subTest(source=source, destination=destination), self.assertRaises(ValueError):
                validate_step(Transaction(source, 'bank_app', destination, 1, 'transfer'),
                              state, FRAUDSTER, VICTIM)
        validate_step(Transaction(VICTIM['account_id'], 'bank_app',
                                  FRAUDSTER['account_id'], 5000, 'transfer'),
                      state, FRAUDSTER, VICTIM)

    def test_invalid_amount_is_retried_then_corrected(self):
        result, model, _ = self.run_sequence([
            choose('contact'), choose('share'), transfer(6000), transfer(3000)])
        self.assertEqual(result['outcome'], 'successful_fraud')
        self.assertEqual(model.calls, 4)
        self.assertEqual(sum(row['validation'] == 'rejected'
                             for row in result['generation_trace']), 1)

    def test_invalid_model_choices_fall_back_and_finish_instead_of_exporting_incomplete(self):
        result, model, _ = self.run_sequence([choose('unknown')] * 30)
        self.assertEqual(model.calls, 9)
        self.assertEqual(result['outcome'], 'successful_fraud')
        self.assertTrue(result['is_complete'])
        accepted = [row for row in result['generation_trace']
                    if row['validation'] == 'accepted']
        self.assertEqual(len(accepted), 3)
        self.assertTrue(all(row['selection_source'] == 'fallback' for row in accepted))
        self.assertTrue(0 < result['steps'][-1]['amount'] <= VICTIM['balance'])
        self.assertEqual(len([row for row in result['generation_trace']
                              if row['validation'] == 'rejected']), 9)

    def test_repeated_contact_from_model_cannot_create_an_endless_sequence(self):
        result, model, _ = self.run_sequence([choose('contact')] * 40, rows=full_templates())
        self.assertEqual(result['outcome'], 'successful_fraud')
        self.assertEqual(len(result['steps']), 5)
        self.assertEqual([row.get('effect') for row in result['steps'][:-1]],
                         ['contact', 'none', 'none', 'disclose'])
        self.assertLessEqual(model.calls, 15)
        self.assertEqual(result['steps'][-1]['type'], 'transaction')

    def test_malformed_json_uses_bounded_fallback(self):
        result, model, _ = self.run_sequence(['not json'] * 20)
        self.assertEqual(result['outcome'], 'successful_fraud')
        self.assertEqual(model.calls, 9)

    def test_model_transport_failures_are_traced_and_fall_back(self):
        result, model, _ = self.run_sequence([ConnectionError('Ollama unavailable')] * 20)
        self.assertEqual(result['outcome'], 'successful_fraud')
        self.assertEqual(model.calls, 9)
        rejected = [entry for entry in result['generation_trace']
                    if entry['validation'] == 'rejected']
        self.assertEqual(len(rejected), 9)
        self.assertTrue(all('ConnectionError' in entry['validation_error'] for entry in rejected))

    def test_no_scenario_is_required(self):
        result, model, _ = self.run_sequence([
            choose('contact'), choose('share'), transfer()])
        self.assertIsNone(result['scenario'])
        self.assertNotIn('"scenario":', model.prompts[0])
        self.assertEqual(result['outcome'], 'successful_fraud')

    def test_compromise_route_waits_for_access_and_acknowledgment(self):
        rows = full_templates()
        rows[True] = [row for row in rows[True] if not row['is_terminal']]
        rows[True].extend([
            template('access', 'account_access', compromises_account=True),
            template('compromised_transfer', 'final_transaction', is_terminal=True,
                     requires_compromise=True, channels=['bank_app'])])
        rows[False].append(template('acknowledge', 'acknowledgment'))
        result, model, _ = self.run_sequence([
            choose('contact'), choose('bank_reply'), choose('follow_up'),
            choose('share'), choose('access'), choose('acknowledge'),
            transfer(name='compromised_transfer')], rows=rows)
        self.assertEqual(len(result['steps']), 7)
        self.assertEqual(result['steps'][4]['effect'], 'compromise')
        accepted = [row for row in result['generation_trace']
                    if row['validation'] == 'accepted']
        self.assertFalse(accepted[3]['state_after']['compromised'])
        self.assertTrue(accepted[4]['state_after']['compromised'])
        self.assertEqual(result['outcome'], 'successful_fraud')
        self.assertEqual(model.calls, 7)

    def test_initialization_rejects_nonpositive_or_invalid_balance(self):
        for balance in (0, -10, True, '5000', float('inf'), float('nan')):
            with self.subTest(balance=balance), self.assertRaises(ValueError):
                self.initial_state(victim={**VICTIM, 'balance': balance})

    def test_initialization_rejects_missing_disclosure_or_compatible_transfer(self):
        rows = minimal_templates()
        rows[False] = [template('reply', 'response', 'bank')]
        with self.assertRaises(ValueError):
            self.initial_state(rows)
        rows = minimal_templates()
        rows[True][-1]['family'] = 'unrelated_family'
        with self.assertRaises(ValueError):
            self.initial_state(rows)

    def test_initialization_rejects_missing_family_metadata(self):
        rows = minimal_templates()
        del rows[True][0]['family']
        with self.assertRaisesRegex(ValueError, 'family'):
            self.initial_state(rows)

    def test_contradictory_compromise_metadata_fails_before_model_or_output(self):
        for is_fraudster in (True, False):
            rows = minimal_templates()
            rows[is_fraudster][0]['compromises_account'] = True
            model = FakeModel([])
            with self.subTest(is_fraudster=is_fraudster), TemporaryDirectory() as directory, patch(
                    'generator_agent.select_starting_character',
                    return_value=(FRAUDSTER, VICTIM)), patch(
                    'generator_agent.load_character_actions',
                    side_effect=lambda name, actor_is_fraudster: rows[actor_is_fraudster]):
                output = Path(directory) / 'sequence.json'
                with self.assertRaises(ValueError):
                    generate_sequence(output=output, model=model)
                self.assertEqual(model.calls, 0)
                self.assertFalse(output.exists())

    def test_initialization_rejects_identical_participants_or_accounts(self):
        for changed in ({'name': FRAUDSTER['name']},
                        {'account_id': FRAUDSTER['account_id']}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.initial_state(victim={**VICTIM, **changed})

    def test_dead_end_opening_is_excluded_before_model_selection(self):
        rows = minimal_templates()
        rows[True].append(template('dead_end_contact', 'initial_contact', 'unrelated'))
        rows[True][1]['family'] = 'bank'
        state = self.initial_state(rows)
        scope_available_actions(state, FRAUDSTER)
        self.assertEqual([row['action'] for row in state['available_actions']], ['contact'])

    def test_compromise_path_requires_both_access_and_acknowledgment_templates(self):
        for missing in ('access', 'acknowledgment'):
            rows = minimal_templates()
            rows[True][1]['requires_compromise'] = True
            if missing != 'access':
                rows[True].append(template('access', 'account_access',
                                           compromises_account=True))
            if missing != 'acknowledgment':
                rows[False].append(template('acknowledge', 'acknowledgment'))
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                self.initial_state(rows)

    def test_step_budget_must_fit_a_complete_route(self):
        with self.assertRaises(ValueError):
            self.initial_state(max_steps=2)
        state = self.initial_state(max_steps=3)
        self.assertEqual(state['current_character'], FRAUDSTER['name'])
        self.assertFalse(check_completion(state))
        with self.assertRaises(ValueError):
            self.initial_state(rows=full_templates(), max_steps=4)

    def test_misconfiguration_fails_before_model_or_output(self):
        model = FakeModel([])
        rows = minimal_templates()
        rows[False] = []
        with TemporaryDirectory() as directory, patch(
                'generator_agent.select_starting_character',
                return_value=(FRAUDSTER, VICTIM)), patch(
                'generator_agent.load_character_actions',
                side_effect=lambda name, is_fraudster: rows[is_fraudster]):
            output = Path(directory) / 'sequence.json'
            with self.assertRaises(ValueError):
                generate_sequence(output=output, model=model)
            self.assertFalse(output.exists())
            self.assertEqual(model.calls, 0)
            output.write_text('Existing result must survive preflight failure.\n')
            with self.assertRaises(ValueError):
                generate_sequence(output=output, model=model)
            self.assertEqual(output.read_text(),
                             'Existing result must survive preflight failure.\n')

    def test_successful_transfer_completes_state_and_prevents_more_steps(self):
        state = self.transfer_state()
        step = Transaction(VICTIM['account_id'], 'bank_app',
                           FRAUDSTER['account_id'], 1000, 'transfer')
        update_state(state, step, FRAUDSTER, VICTIM)
        self.assertTrue(check_completion(state))
        self.assertEqual(state['graph']['balance'], 4000)
        self.assertEqual(state['current_action'], step)
        with self.assertRaises(ValueError):
            update_state(state, step, FRAUDSTER, VICTIM)

    def test_every_shipped_contact_family_reaches_a_compatible_transfer(self):
        from database.build_graph import load_data

        data = load_data(Path(__file__).resolve().parents[1] / 'data' / 'graph')
        rows = {True: [row['props'] for row in data['fraudster_actions']],
                False: [row['props'] for row in data['victim_actions']]}
        contacts = [row for row in rows[True] if row['stage'] == 'initial_contact']
        self.assertTrue(contacts)
        by_name = {row['action']: row for side in rows.values() for row in side}
        for contact in contacts:
            with self.subTest(family=contact['family']):
                # Only the opening is supplied by the model. Stage-constrained
                # fallback must still find a complete compatible continuation.
                result, _, _ = self.run_sequence([
                    choose(contact['action'], channel=contact['channels'][0]),
                    *[choose('unavailable_action')] * 30,
                ], rows=rows)
                self.assertEqual(result['outcome'], 'successful_fraud')
                self.assertEqual(result['steps'][0]['action'], contact['action'])
                self.assertEqual(result['steps'][-1]['type'], 'transaction')
                for step in result['steps']:
                    self.assertIn(by_name[step['action']]['family'],
                                  (contact['family'], 'generic'))


if __name__ == '__main__':
    unittest.main()
