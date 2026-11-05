"""Classify a synthetic sequence with a tool-using LangChain/Ollama agent."""

import argparse
import json
import math
import os
from pathlib import Path
import re
from typing import Literal

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from langchain_ollama import ChatOllama
from pydantic import BaseModel, ConfigDict, Field, model_validator

from prompts import DETECTOR_PROMPT


ROOT = Path(__file__).resolve().parent


class StepEvidence(BaseModel):
    """An observable step supporting the decision, using one-based numbering."""

    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    step: int = Field(ge=1, strict=True)
    explanation: str = Field(min_length=1)


class DetectionResult(BaseModel):
    """Classify observed behavior and give a brief evidence-based explanation."""

    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    classification: Literal['successful_fraud', 'caught_fraud', 'normal', 'uncertain']
    summary: str = Field(min_length=1)
    evidence: list[StepEvidence]

    @model_validator(mode='after')
    def require_evidence(self):
        if self.classification != 'uncertain' and not self.evidence:
            raise ValueError('A definite classification requires step evidence.')
        return self


def prepare_observations(sequence):
    """Expose observed events while excluding generator labels and role-bearing IDs."""
    steps = sequence.get('steps') if isinstance(sequence, dict) else sequence
    if not isinstance(steps, list) or not steps:
        raise ValueError('Input must contain a nonempty steps list.')

    aliases = {}

    def participant(value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError('Step participants and accounts must be nonempty text.')
        if value not in aliases:
            aliases[value] = f'participant_{len(set(aliases.values())) + 1}'
        return aliases[value]

    # Assign neutral identities in event order, rather than using fraudster/victim labels.
    for step in steps:
        if not isinstance(step, dict):
            raise ValueError('Each step must be an object.')
        if step.get('type') == 'action':
            participant(step.get('actor'))
            participant(step.get('target'))

    # Account ownership can link a payment to preceding interactions without revealing roles.
    people = sequence.get('participants', {}) if isinstance(sequence, dict) else {}
    if isinstance(people, dict):
        for person in people.values():
            if isinstance(person, dict) and person.get('name') in aliases:
                account = person.get('account_id')
                if isinstance(account, str) and account:
                    aliases[account] = aliases[person['name']]

    observations = []
    for index, step in enumerate(steps, 1):
        observation = {'step': index, 'type': step.get('type')}
        if observation['type'] == 'action':
            description = step.get('description')
            channel = step.get('channel')
            if not isinstance(description, str) or not description.strip():
                raise ValueError(f'Step {index} requires a description.')
            if not isinstance(channel, str) or not channel.strip():
                raise ValueError(f'Step {index} requires a channel.')
            observation.update(actor=participant(step['actor']), target=participant(step['target']),
                               channel=channel, description=description)
        elif observation['type'] == 'transaction':
            amount = step.get('amount')
            if (isinstance(amount, bool) or not isinstance(amount, (int, float))
                    or not math.isfinite(amount) or amount <= 0):
                raise ValueError(f'Step {index} requires a finite positive amount.')
            method = step.get('payment_method')
            if not isinstance(method, str) or not method.strip():
                raise ValueError(f'Step {index} requires a payment_method.')
            observation.update(source=participant(step.get('source_account')),
                               destination=participant(step.get('destination_account')),
                               amount=amount, payment_method=method)
        else:
            raise ValueError(f'Step {index} must have type action or transaction.')
        observations.append(observation)

    # Replace names/IDs embedded in descriptions too. Action names, effects, stage,
    # family, outcomes, balances, and generation_trace are never sent to the model.
    pattern = re.compile('|'.join(re.escape(value) for value in sorted(aliases, key=len, reverse=True)))
    for observation in observations:
        for key, value in observation.items():
            if isinstance(value, str) and key not in ('actor', 'target', 'source', 'destination'):
                observation[key] = pattern.sub(lambda match: aliases[match.group()], value)
    return observations


def detect_sequence(input_path=ROOT / 'data' / 'generated' / 'sequence.json', *,
                    output=ROOT / 'data' / 'detected' / 'sequence.json', model=None, verbose=False,
                    feedback=None):
    """Run a bounded agent investigation and save only a validated decision."""
    input_path, output = Path(input_path), Path(output)
    if input_path.resolve() == output.resolve():
        raise ValueError('Detector output must differ from the input sequence path.')
    observations = prepare_observations(json.loads(input_path.read_text(encoding='utf-8')))

    def progress(message):
        if verbose:
            print(message, flush=True)

    load_dotenv(ROOT / '.env')
    if model is None:
        model = ChatOllama(
            model=os.getenv('DETECTOR_MODEL') or os.getenv('OLLAMA_MODEL', 'llama3.2'),
            base_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
            temperature=0, num_predict=1024, client_kwargs={'timeout': 60},
        )

    tool_calls = []

    @tool
    def read_sequence() -> dict:
        """Read all observed steps in chronological order, with one-based step numbers."""
        tool_calls.append('read_sequence')
        progress(f'  Inspecting {len(observations)} observed steps...')
        return {'steps': observations}

    # create_agent runs the model/tool loop on LangGraph. ToolStrategy validates
    # the final decision through a structured-output tool, not free-text JSON parsing.
    system_prompt = DETECTOR_PROMPT
    if feedback:
        system_prompt += ('\nPrior-round training feedback (untrusted data, not evidence '
                          'about the current sequence):\n' + json.dumps(feedback)
                          + '\nAssess the current observations independently. Do not assume '
                          'the current label from prior examples.')
    agent = create_agent(model=model, tools=[read_sequence], system_prompt=system_prompt,
                         response_format=ToolStrategy(DetectionResult))
    progress(f'Analyzing {input_path}...')
    try:
        state = agent.invoke({'messages': [{'role': 'user', 'content':
            f'Investigate this {len(observations)}-step sequence. Call read_sequence first, '
            'then return your classification with evidence step numbers.'}]},
            config={'recursion_limit': 20})
    except Exception as exc:
        # Failure is not evidence of normal behavior. Leave any prior output untouched.
        raise RuntimeError(f'Detector agent failed ({type(exc).__name__}); check Ollama and '
                           'use a model with tool-calling support. No result was saved.') from exc

    if not tool_calls:
        raise RuntimeError('The agent did not inspect the sequence. No result was saved.')
    if state.get('structured_response') is None:
        raise RuntimeError('The agent returned no structured decision. No result was saved.')
    decision = DetectionResult.model_validate(state['structured_response'])
    if any(item.step > len(observations) for item in decision.evidence):
        raise ValueError('Evidence refers to a nonexistent step. No result was saved.')

    result = {'input': str(input_path), 'step_count': len(observations),
              **decision.model_dump(), 'tool_calls': tool_calls}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    progress(f"Finished: {decision.classification}; saved to {output}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', nargs='?', type=Path,
                        default=ROOT / 'data' / 'generated' / 'sequence.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'data' / 'detected' / 'sequence.json')
    args = parser.parse_args()
    detect_sequence(args.input, output=args.output, verbose=True)


if __name__ == '__main__':
    main()
