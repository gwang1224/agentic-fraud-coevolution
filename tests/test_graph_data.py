import json
from pathlib import Path
import tempfile
import unittest

from database.build_graph import GROUPS, load_data
from database.connection import ROOT


class GraphDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        for group in GROUPS:
            source = ROOT / "data" / "graph" / f"{group}.json"
            (self.directory / source.name).write_text(source.read_text())

    def change(self, group, edit):
        path = self.directory / f"{group}.json"
        payload = json.loads(path.read_text())
        edit(payload)
        path.write_text(json.dumps(payload))

    def test_legacy_ids_do_not_collide_between_roles(self):
        data = load_data(self.directory)
        self.assertNotEqual(data['victims'][0]['account_id'],
                            data['fraudsters'][0]['account_id'])

    def test_duplicate_people_rejected(self):
        self.change('victims', lambda p: p['victims'].append(p['victims'][0]))
        with self.assertRaisesRegex(ValueError, 'duplicate name'):
            load_data(self.directory)

    def test_nonfinite_balance_rejected(self):
        self.change('victims', lambda p: p['victims'][0]['account'].update(balance=float('nan')))
        with self.assertRaisesRegex(ValueError, 'balance'):
            load_data(self.directory)

    def test_string_boolean_rejected(self):
        self.change('victim_actions', lambda p: p['victim_actions'][0].update(is_terminal='false'))
        with self.assertRaisesRegex(ValueError, 'boolean'):
            load_data(self.directory)

    def test_legitimate_actions_are_ignored(self):
        original = load_data(self.directory)['victim_actions']
        self.change('victim_actions', lambda p: p.update(legit_actions=[{'name': 'balance_check'}]))
        self.assertEqual(load_data(self.directory)['victim_actions'], original)

    def test_action_family_is_preserved_in_import_properties(self):
        data = load_data(self.directory)
        templates = {row['props']['action']: row['props']
                     for group in ('fraudster_actions', 'victim_actions')
                     for row in data[group]}
        self.assertEqual(templates['investment_offer']['family'], 'investment')
        self.assertEqual(templates['accept_deceptive_investment']['family'], 'investment')
        self.assertEqual(templates['fraudulent_investment_payment']['family'], 'investment')
        self.assertEqual(templates['disclose_payment_information']['family'], 'generic')

    def test_legacy_template_without_family_defaults_to_generic(self):
        self.change('fraudster_actions', lambda p: p['fraudster_actions'][0].pop('family'))
        props = load_data(self.directory)['fraudster_actions'][0]['props']
        self.assertEqual(props['family'], 'generic')

    def test_invalid_family_rejected(self):
        for family in ('', '   ', None, ['investment']):
            with self.subTest(family=family):
                self.change('fraudster_actions',
                            lambda p: p['fraudster_actions'][0].update(family=family))
                with self.assertRaisesRegex(ValueError, 'family must be a nonempty string'):
                    load_data(self.directory)

    def test_stage_for_wrong_role_is_rejected(self):
        self.change('fraudster_actions',
                    lambda p: p['fraudster_actions'][0].update(stage='disclosure'))
        with self.assertRaisesRegex(ValueError, 'stage must be one of .* for fraudster_actions'):
            load_data(self.directory)

    def test_every_builtin_contact_family_has_a_complete_success_path(self):
        data = load_data(self.directory)
        fraudster = [row['props'] for row in data['fraudster_actions']]
        victim = [row['props'] for row in data['victim_actions']]
        contacts = [row for row in fraudster if row.get('stage') == 'initial_contact']
        self.assertTrue(contacts)
        for contact in contacts:
            family = contact['family']
            with self.subTest(contact=contact['action'], family=family):
                self.assertNotEqual(family, 'generic')
                compatible = lambda row: row['family'] in ('generic', family)
                for templates, stage in ((victim, 'response'), (fraudster, 'follow_up'),
                                         (victim, 'disclosure')):
                    self.assertTrue(any(compatible(row) and row.get('stage') == stage
                                        and row['channels'] for row in templates), stage)
                self.assertTrue(any(compatible(row) and row.get('is_terminal')
                                    and not row.get('requires_compromise') and row['channels']
                                    for row in fraudster), 'transfer')

    def test_compromise_route_has_an_acknowledgment_before_transfer(self):
        data = load_data(self.directory)
        fraudster = [row['props'] for row in data['fraudster_actions']]
        victim = [row['props'] for row in data['victim_actions']]
        self.assertTrue(any(row.get('stage') == 'account_access'
                            and row.get('compromises_account')
                            and row['family'] == 'generic' for row in fraudster))
        self.assertTrue(any(row.get('stage') == 'acknowledgment'
                            and row['family'] == 'generic' for row in victim))
        self.assertTrue(any(row.get('requires_compromise') and row.get('is_terminal')
                            and row['family'] == 'generic' for row in fraudster))


if __name__ == '__main__':
    unittest.main()
