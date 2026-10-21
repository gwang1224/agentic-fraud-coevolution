"""Generate complete, synthetic fraud sequences from Neo4j action templates."""

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from database.connection import ROOT, connect
from prompts import GENERATOR_PROMPT, RETRY_PROMPT


@dataclass(frozen=True)
class Action:
    actor: str
    action: str
    target: str
    channel: str
    description: str
    effect: str


@dataclass(frozen=True)
class Transaction:
    source_account: str
    payment_method: str
    destination_account: str
    amount: float
    action: str | None = None


class GeneratorState(TypedDict):
    # Shared LangGraph state: plans narrow as steps are accepted; stage_index advances.
    participants: dict
    plans: list[dict]
    stage_index: int
    generated_steps: list[Action | Transaction]
    llm_prompt: str
    graph: dict
    # `graph` is an in-memory simulation snapshot, not a live Neo4j connection.
    current_character: str
    available_actions: list[dict]
    available_effects: list[str]
    available_transactions: list[dict]
    current_action: Action | Transaction | None
    is_complete: bool
    attempt: int
    error: str
    stop_reason: str
    generation_trace: list[dict]


def select_starting_character():
    """Return a random fraudster and a funded victim, with their accounts."""
    def select_participants(tx):
        participants = []
        for role in ('fraudsters', 'victims'):
            # Labels come from this fixed role list; account funding filters only victims.
            record = tx.run(f"""
                MATCH (p:{role})-[:OWNS]->(a:account)
                WHERE $role <> 'victims' OR a.balance > 0
                WITH p, a, rand() AS random_order
                ORDER BY random_order
                LIMIT 1
                RETURN p.name AS name, a.id AS account_id,
                       a.acc_name AS acc_name, a.bank AS bank, a.balance AS balance
            """, role=role).single()
            if record is None:
                raise ValueError(f'Neo4j requires at least one {role} node with '
                                 'an account (victim balance must be positive).')
            participants.append(dict(record))
        return tuple(participants)

    driver, database = connect()
    with driver:
        with driver.session(database=database) as session:
            return session.execute_read(select_participants)


def load_character_actions(character, is_fraudster):
    """Read only templates connected to this person through PERFORMS."""
    role = 'fraudsters' if is_fraudster else 'victims'
    driver, database = connect()
    with driver:
        with driver.session(database=database) as session:
            return session.execute_read(lambda tx: [
                dict(record['action']) for record in tx.run(f"""
                    MATCH (p:{role} {{name: $name}})-[:PERFORMS]->(a:action)
                    RETURN DISTINCT properties(a) AS action
                    ORDER BY action.action
                """, name=character)
            ])


def template_effect(template):
    """Effects are determined by template metadata, never by model text."""
    if template.get('compromises_account'):
        return 'compromise'
    return {'initial_contact': 'contact', 'disclosure': 'disclose'}.get(
        template.get('stage'), 'none')


def _compatible(rows, family, stage, *, terminal=False):
    # Prefer family-specific templates; generic ones fill stages without a specific match.
    candidates = [row for row in rows
                  if row.get('stage') == stage
                  and bool(row.get('is_terminal')) == terminal
                  and row['family'] in (family, 'generic')]
    specific = [row for row in candidates if row['family'] == family]
    return specific or candidates


def build_plans(fraudster_actions, victim_actions, max_steps):
    """Retain only bounded paths whose every stage has a compatible template."""
    for row in fraudster_actions + victim_actions:
        if not isinstance(row.get('family'), str) or not row['family'].strip():
            raise ValueError('Action templates need family metadata. Reimport with '
                             'python -m database.build_graph.')
        if row.get('compromises_account') and row.get('stage') != 'account_access':
            raise ValueError('Only account_access templates may set compromises_account.')
        if row.get('stage') == 'account_access' and not row.get('compromises_account'):
            raise ValueError('An account_access template must set compromises_account.')
        if row.get('is_terminal') and row.get('stage') not in ('final_transaction', 'report'):
            raise ValueError('Terminal templates require a final_transaction or report stage.')
    # Each plan contains a list of stages, each holding interchangeable template choices.
    # Verify a full route before letting the model select its first action.
    plans = []
    for contact in fraudster_actions:
        if contact.get('stage') != 'initial_contact' or contact.get('is_terminal'):
            continue
        family = contact['family']
        disclosures = _compatible(victim_actions, family, 'disclosure')
        transfers = _compatible(fraudster_actions, family, 'final_transaction', terminal=True)
        if not disclosures or not transfers:
            continue
        stages = [[contact]]
        responses = _compatible(victim_actions, family, 'response')
        followups = _compatible(fraudster_actions, family, 'follow_up')
        # A response is followed by a request, not another initial contact.
        if responses and followups:
            stages.extend([responses, followups])
        stages.append(disclosures)
        direct = [row for row in transfers if not row.get('requires_compromise')]
        if direct:
            # Prefer the shorter direct route whenever a transfer permits it.
            stages.append(direct)
        else:
            compromise = [row for row in _compatible(fraudster_actions, family, 'account_access')
                          if row.get('compromises_account')]
            acknowledgment = _compatible(victim_actions, family, 'acknowledgment')
            if not compromise or not acknowledgment:
                continue
            # The victim acknowledgment preserves alternating turns before the transfer.
            stages.extend([compromise, acknowledgment, transfers])
        if len(stages) <= max_steps:
            plans.append({'family': family, 'stages': stages})
    if not plans:
        raise ValueError('No complete contact-to-transfer path fits the connected templates '
                         f'and max_steps={max_steps}. Check family/stage metadata, disclosure '
                         'and transfer templates, or increase max_steps.')
    return plans


def initialize_state(fraudster, victim, max_steps=15):
    """Validate participants and snapshot a set of complete paths before generation."""
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not 3 <= max_steps <= 15:
        raise ValueError('max_steps must be between 3 and 15 for a complete sequence.')
    for person in (fraudster, victim):
        for field in ('name', 'account_id'):
            if not isinstance(person.get(field), str) or not person[field].strip():
                raise ValueError(f'Each participant requires a nonempty {field}.')
    if fraudster['name'] == victim['name'] or fraudster['account_id'] == victim['account_id']:
        raise ValueError('The victim and fraudster must have distinct names and accounts.')
    balance = victim.get('balance')
    if (isinstance(balance, bool) or not isinstance(balance, (int, float))
            or not math.isfinite(balance) or balance <= 0):
        raise ValueError('The victim balance must be finite and positive.')
    destination_balance = fraudster.get('balance', 0)
    if (isinstance(destination_balance, bool) or not isinstance(destination_balance, (int, float))
            or not math.isfinite(destination_balance) or destination_balance < 0):
        raise ValueError('The fraudster balance must be finite and nonnegative.')
    # Snapshot connected templates once; later steps do not reread or modify Neo4j.
    plans = build_plans(load_character_actions(fraudster['name'], True),
                        load_character_actions(victim['name'], False), max_steps)
    return {
        'participants': {'fraudster': dict(fraudster), 'victim': dict(victim)},
        'plans': plans, 'stage_index': 0, 'generated_steps': [], 'llm_prompt': '',
        'graph': {'contacted': False, 'disclosed': False, 'compromised': False,
                  'balance': balance, 'fraudster_balance': destination_balance,
                  'family': None, 'stage': 'initial_contact'},
        'current_character': fraudster['name'], 'available_actions': [],
        'available_effects': [], 'available_transactions': [], 'current_action': None,
        'is_complete': False, 'attempt': 0, 'error': '', 'stop_reason': '',
        'generation_trace': [],
    }


def _eligible_templates(state):
    """Recompute eligibility from retained paths, independently of cached choices."""
    # Multiple surviving plans can share a template; offer each action name only once.
    unique = {}
    for plan in state['plans']:
        if state['stage_index'] < len(plan['stages']):
            for row in plan['stages'][state['stage_index']]:
                unique[row['action']] = row
    return list(unique.values())


def scope_available_actions(state, fraudster):
    """Offer only the current stage of complete, compatible paths."""
    actions = _eligible_templates(state)
    transfers = [row for row in actions if row.get('stage') == 'final_transaction']
    state['available_transactions'] = transfers
    state['available_actions'] = [] if transfers else actions
    state['available_effects'] = sorted({template_effect(row)
                                         for row in state['available_actions']})
    if actions:
        state['graph']['stage'] = actions[0]['stage']
    return state['available_actions']


def check_completion(state):
    """Only a successful, accepted transfer completes this generator's objective."""
    state['is_complete'] = bool(state['generated_steps'] and
                               isinstance(state['generated_steps'][-1], Transaction))
    return state['is_complete']


def validate_step(step, state, fraudster, victim):
    """Validate against the actual path, rather than trusting the model or cached lists."""
    if state['generated_steps'] and isinstance(state['generated_steps'][-1], Transaction):
        raise ValueError('The sequence has already terminated.')
    templates = {row['action']: row for row in _eligible_templates(state)}
    template = templates.get(getattr(step, 'action', None))
    if template is None:
        raise ValueError('Action must match an eligible template for the current stage and family.')
    index = state['stage_index']
    # Every constructed plan starts with the fraudster and alternates roles by index.
    expected_actor = fraudster['name'] if index % 2 == 0 else victim['name']
    if state['current_character'] != expected_actor:
        raise ValueError('Current character does not match the required stage actor.')
    if isinstance(step, Transaction):
        if expected_actor != fraudster['name'] or template.get('stage') != 'final_transaction':
            raise ValueError('A transfer requires the fraudster final_transaction stage.')
        if not template.get('is_terminal') or not state['graph']['disclosed']:
            raise ValueError('A transfer requires a terminal template and prior disclosure.')
        if template.get('requires_compromise') and not state['graph']['compromised']:
            raise ValueError('This transfer requires prior account compromise.')
        if step.source_account != victim['account_id'] or step.destination_account != fraudster['account_id']:
            raise ValueError('Transfer must go from the selected victim to the selected fraudster.')
        if not isinstance(step.payment_method, str) or not step.payment_method.strip():
            raise ValueError('payment_method must be nonempty text.')
        if (isinstance(step.amount, bool) or not isinstance(step.amount, (int, float))
                or not math.isfinite(step.amount) or not 0 < step.amount <= state['graph']['balance']):
            raise ValueError('amount must be finite, positive, and within the victim balance.')
        return
    if not isinstance(step, Action) or template.get('stage') == 'final_transaction':
        raise ValueError('The current stage requires the correct Action or Transaction type.')
    if any(not isinstance(value, str) or not value.strip() for value in asdict(step).values()):
        raise ValueError('All action fields must be nonempty strings.')
    target = victim['name'] if expected_actor == fraudster['name'] else fraudster['name']
    if step.actor != expected_actor or step.target != target:
        raise ValueError('Action participants must match the current turn.')
    if template.get('channels') and step.channel not in template['channels']:
        raise ValueError('channel must match the template channels.')
    if step.effect != template_effect(template):
        raise ValueError('effect must match the selected template.')
    if template.get('stage') != 'initial_contact' and not state['graph']['contacted']:
        raise ValueError('This action requires prior contact.')
    if step.effect == 'compromise' and not state['graph']['disclosed']:
        raise ValueError('Account compromise requires prior disclosure.')
    if step in state['generated_steps']:
        raise ValueError('Duplicate actions are not allowed.')


def update_state(state, step, fraudster, victim):
    """Apply a validated step and advance to the next stage of the selected path."""
    validate_step(step, state, fraudster, victim)
    index = state['stage_index']
    # Commit to paths containing this choice, preventing later switches of premise.
    state['plans'] = [plan for plan in state['plans']
                      if any(row['action'] == step.action for row in plan['stages'][index])]
    state['graph']['family'] = state['plans'][0]['family']
    state['current_action'] = step
    state['generated_steps'].append(step)
    state['stage_index'] += 1
    if isinstance(step, Transaction):
        # Balance changes are simulated only; the source database remains unchanged.
        state['graph']['balance'] -= step.amount
        state['graph']['fraudster_balance'] += step.amount
        state['graph']['stage'] = 'complete'
        state['stop_reason'] = 'final_transaction'
    else:
        flag = {'contact': 'contacted', 'disclose': 'disclosed',
                'compromise': 'compromised'}.get(step.effect)
        if flag:
            state['graph'][flag] = True
        state['current_character'] = step.target
    check_completion(state)
    return state


def _response_schema(templates, is_transfer, balance):
    """Constrain each template to the fields and channels it supports on this turn."""
    properties = {'action': {'type': 'string', 'enum': [row['action'] for row in templates]}}
    if is_transfer:
        properties.update(payment_method={'type': 'string', 'minLength': 1},
                          amount={'type': 'number', 'exclusiveMinimum': 0, 'maximum': balance})
        return {'type': 'object', 'properties': properties, 'required': list(properties),
                'additionalProperties': False}
    # A separate schema branch per template keeps channels tied to that exact action.
    branches = []
    for row in templates:
        fields = {'action': {'type': 'string', 'enum': [row['action']]},
                  'channel': {'type': 'string', 'minLength': 1},
                  'description': {'type': 'string', 'minLength': 1}}
        if row.get('channels'):
            fields['channel']['enum'] = row['channels']
        branches.append({'type': 'object', 'properties': fields, 'required': list(fields),
                         'additionalProperties': False})
    return {'anyOf': branches}


def generate_next_action(state, scenario, fraudster, victim, model, error=''):
    """The LLM selects within a stage; Python owns identities, effects and routing."""
    templates = _eligible_templates(state)
    if not templates:
        raise ValueError('No eligible templates are available on this turn.')
    is_transfer = templates[0]['stage'] == 'final_transaction'
    schema = _response_schema(templates, is_transfer, state['graph']['balance'])
    kind = 'transaction' if is_transfer else 'action'
    context = {'fraudster': fraudster, 'victim': victim,
               'current_character': state['current_character'], 'state': state['graph'],
               'templates': templates, 'required_type': kind, 'response_schema': schema,
               'steps': [asdict(step) for step in state['generated_steps']]}
    if scenario is not None:
        context['scenario'] = scenario
    state['llm_prompt'] = GENERATOR_PROMPT + json.dumps(context)
    prompt = state['llm_prompt'] + (RETRY_PROMPT.format(error=error) if error else '')
    # Real Ollama calls use constrained JSON output; injected test models use invoke directly.
    turn_model = model.bind(format=schema) if isinstance(model, ChatOllama) else model
    try:
        response = turn_model.invoke(prompt)
    except Exception as exc:
        # Provider failures are retried and visibly recorded, not confused with success.
        raise ValueError(f'Model request failed ({type(exc).__name__}).') from exc
    payload = json.loads(response.content)
    if not isinstance(payload, dict):
        raise ValueError('Response must be a JSON object.')
    if payload.pop('type', kind) != kind:
        raise ValueError(f'This stage requires {kind}.')
    selected = next((row for row in templates if row['action'] == payload.get('action')), None)
    if selected is None:
        raise ValueError('Action must match an eligible template for the current stage and family.')
    # Ignore any model attempts to override fields owned by the simulation.
    for field in ('actor', 'target', 'source_account', 'destination_account', 'effect'):
        payload.pop(field, None)
    if is_transfer:
        # The model chooses transfer details; Python fixes the source and destination.
        return Transaction(source_account=victim['account_id'],
                           destination_account=fraudster['account_id'], **payload)
    actor = state['current_character']
    target = victim['name'] if actor == fraudster['name'] else fraudster['name']
    return Action(actor=actor, target=target, effect=template_effect(selected), **payload)


def fallback_step(state, fraudster, victim):
    """Construct a valid template step when the model exhausts its three attempts."""
    template = _eligible_templates(state)[0]
    if template['stage'] == 'final_transaction':
        balance = state['graph']['balance']
        # Deterministic fallback amount stays positive and never exceeds available funds.
        amount = min(balance, 1000)
        return Transaction(victim['account_id'], 'simulated_bank_transfer',
                           fraudster['account_id'], amount, template['action'])
    actor = state['current_character']
    target = victim['name'] if actor == fraudster['name'] else fraudster['name']
    return Action(actor, template['action'], target,
                  (template.get('channels') or ['simulation'])[0],
                  template.get('description') or f'Synthetic {template["stage"]} step.',
                  template_effect(template))


def generate_sequence(scenario=None, *, output=ROOT / 'data' / 'generated' / 'sequence.json',
                      model=None, max_steps=15, verbose=False):
    """Return/export a complete transfer sequence or raise for invalid configuration."""
    if scenario is not None and (not isinstance(scenario, str) or not scenario.strip()):
        raise ValueError('scenario must be nonempty text when provided.')
    fraudster, victim = select_starting_character()
    state = initialize_state(fraudster, victim, max_steps)
    # Replay accepted steps against the original plans before writing any output.
    initial_state = deepcopy(state)
    load_dotenv(ROOT / '.env')
    if model is None:
        model = ChatOllama(model=os.getenv('OLLAMA_MODEL', 'llama3.2'),
                           base_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
                           temperature=0, num_predict=384,
                           client_kwargs={'timeout': 60})

    def progress(message):
        if verbose:
            print(message, flush=True)

    progress(f"Generating complete synthetic sequence: {fraudster['name']} -> {victim['name']}")

    def scope_node(state):
        scope_available_actions(state, fraudster)
        # Retry counts and error feedback belong to one stage, not the whole sequence.
        state.update(attempt=0, error='')
        return state

    def generate_node(state):
        state['attempt'] += 1
        number = len(state['generated_steps']) + 1
        progress(f"[Step {number}, {state['graph']['stage']}, attempt {state['attempt']}/3] "
                 f"{state['current_character']}")
        entry = {'step': number, 'attempt': state['attempt'],
                 'actor': state['current_character'], 'stage': state['graph']['stage'],
                 'state_before': dict(state['graph']),
                 'available_actions': [row['action'] for row in state['available_actions']],
                 'available_transactions': [row['action'] for row in state['available_transactions']],
                 'allowed_effects': list(state['available_effects']), 'selection_source': 'llm'}
        # Clear the previous step so a rejected proposal cannot accidentally be applied.
        state['current_action'] = None
        try:
            step = generate_next_action(state, scenario, fraudster, victim, model, state['error'])
            entry['candidate'] = asdict(step)
            validate_step(step, state, fraudster, victim)
            entry['validation'] = 'accepted'
            state.update(current_action=step, error='')
        except (ValueError, TypeError) as exc:
            state['error'] = str(exc)
            entry.update(validation='rejected', validation_error=state['error'])
            progress(f"  Rejected: {state['error']}")
        state['generation_trace'].append(entry)
        if state['current_action'] is None and state['attempt'] >= 3:
            # Preserve all rejected attempts and mark the fallback as a separate selection.
            step = fallback_step(state, fraudster, victim)
            validate_step(step, state, fraudster, victim)
            state['generation_trace'].append({**entry, 'candidate': asdict(step),
                'selection_source': 'fallback', 'validation': 'accepted',
                'fallback_reason': state['error']})
            state['generation_trace'][-1].pop('validation_error', None)
            state.update(current_action=step, error='')
            progress('  Using a validated template fallback after three failed attempts.')
        return state

    def apply_node(state):
        update_state(state, state['current_action'], fraudster, victim)
        state['generation_trace'][-1]['state_after'] = dict(state['graph'])
        progress(f"  Accepted: {state['current_action'].action}")
        return state

    # scope -> generate -> apply; invalid proposals retry generation without advancing.
    workflow = StateGraph(GeneratorState)
    workflow.add_node('scope_actions', scope_node)
    workflow.add_node('generate_action', generate_node)
    workflow.add_node('apply_action', apply_node)
    workflow.add_edge(START, 'scope_actions')
    workflow.add_edge('scope_actions', 'generate_action')
    workflow.add_conditional_edges('generate_action',
        lambda state: 'apply_action' if state['current_action'] is not None else 'generate_action',
        ['apply_action', 'generate_action'])
    workflow.add_conditional_edges('apply_action',
        lambda state: END if state['is_complete'] else 'scope_actions', ['scope_actions', END])
    # Allow room for three generation attempts plus scoping/application on every stage.
    state = workflow.compile().invoke(state, {'recursion_limit': max_steps * 6 + 10})
    # Independently replay the accepted sequence from the untouched initial snapshot.
    for step in state['generated_steps']:
        scope_available_actions(initial_state, fraudster)
        update_state(initial_state, step, fraudster, victim)
    if not initial_state['is_complete'] or len(state['generated_steps']) > max_steps:
        raise RuntimeError('Generation did not produce a complete, valid transfer sequence.')
    # Export only after replay confirms a complete path. Trace entries are an audit log,
    # not model chain-of-thought; fallback_count distinguishes model and template choices.
    trace = state['generation_trace']
    result = {'scenario': scenario, 'target_outcome': 'successful_fraud',
              'outcome': 'successful_fraud', 'is_complete': True, 'stop_reason': 'final_transaction',
              'participants': state['participants'], 'family': state['graph']['family'],
              'final_balances': {'victim': state['graph']['balance'],
                                 'fraudster': state['graph']['fraudster_balance']},
              'fallback_count': sum(entry['selection_source'] == 'fallback' for entry in trace),
              'generation_trace': trace,
              'steps': [{'type': 'transaction' if isinstance(step, Transaction) else 'action',
                         **asdict(step)} for step in state['generated_steps']]}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    progress(f"Finished: successful_fraud; {len(result['steps'])} steps saved to {output}; "
             f"{result['fallback_count']} fallback selections.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scenario', nargs='?', default=None,
                        help='Optional scenario; omit to let the model choose the premise')
    parser.add_argument('--output', type=Path, default=ROOT / 'data' / 'generated' / 'sequence.json')
    parser.add_argument('--quiet', action='store_true', help='Hide per-step progress messages')
    args = parser.parse_args()
    result = generate_sequence(args.scenario, output=args.output, verbose=not args.quiet)
    if args.quiet:
        print(f"Saved {len(result['steps'])} steps to {args.output} (successful_fraud; "
              f"{result['fallback_count']} fallback selections).")


if __name__ == '__main__':
    main()
