"""Exercise the actual LangChain agent loop without Ollama or Neo4j."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from detector_agent import detect_sequence, prepare_observations


SEQUENCE = {
    'outcome': 'successful_fraud', 'family': 'invoice',
    'participants': {
        'fraudster': {'name': 'Synthetic Invoice Actor', 'account_id': 'fraudsters:Invoice'},
        'victim': {'name': 'Alex Morgan', 'account_id': 'victims:Alex'},
    },
    'generation_trace': [{'validation_error': 'DO NOT SEND THIS METADATA'}],
    'steps': [
        {'type': 'action', 'actor': 'Synthetic Invoice Actor', 'target': 'Alex Morgan',
         'action': 'invoice_impersonation', 'channel': 'email', 'effect': 'contact',
         'description': 'Synthetic Invoice Actor sends Alex Morgan a deceptive invoice.'},
        {'type': 'transaction', 'source_account': 'victims:Alex',
         'destination_account': 'fraudsters:Invoice', 'payment_method': 'bank_transfer',
         'amount': 100, 'action': 'fraudulent_invoice_payment'},
    ],
}


def tool_call(name, args, call_id):
    return AIMessage(content='', tool_calls=[{
        'name': name, 'args': args, 'id': call_id, 'type': 'tool_call',
    }])


def decision(classification='successful_fraud', evidence=None):
    return {
        'classification': classification, 'summary': 'A brief observation-based decision.',
        'evidence': evidence if evidence is not None else [
            {'step': 1, 'explanation': 'The interaction provides evidence for this assessment.'},
            {'step': 2, 'explanation': 'The payment establishes the observed transfer.'},
        ],
    }


class ScriptedChatModel(BaseChatModel):
    """Return scripted tool calls through LangChain's real model interface."""

    responses: list[AIMessage]
    calls: int = 0
    received_messages: list = Field(default_factory=list)

    @property
    def _llm_type(self):
        return 'scripted-detector-test'

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.received_messages.append(list(messages))
        response = self.responses[self.calls]
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=response)])


class DetectorTests(unittest.TestCase):
    def run_detection(self, payload=None, responses=None):
        model = ScriptedChatModel(responses=responses or [
            tool_call('read_sequence', {}, 'inspect'),
            tool_call('DetectionResult', decision(), 'result'),
        ])
        with TemporaryDirectory() as directory:
            source = Path(directory) / 'sequence.json'
            output = Path(directory) / 'detected' / 'result.json'
            source.write_text(json.dumps(payload if payload is not None else SEQUENCE))
            original = source.read_text()
            result = detect_sequence(source, output=output, model=model)
            self.assertEqual(json.loads(output.read_text()), result)
            self.assertEqual(source.read_text(), original)
        return result, model

    def test_actual_agent_reads_tool_then_returns_structured_result(self):
        result, model = self.run_detection()
        self.assertEqual(result['classification'], 'successful_fraud')
        self.assertEqual(result['tool_calls'], ['read_sequence'])
        self.assertEqual(model.calls, 2)
        tool_messages = [message for message in model.received_messages[-1]
                         if isinstance(message, ToolMessage)]
        self.assertEqual(tool_messages[0].name, 'read_sequence')
        data = json.loads(tool_messages[0].content)
        self.assertEqual(data['steps'][1]['source'], data['steps'][0]['target'])
        self.assertEqual(data['steps'][1]['destination'], data['steps'][0]['actor'])

    def test_generator_labels_and_role_names_are_not_exposed(self):
        result, model = self.run_detection()
        observed = next(message.content for message in model.received_messages[-1]
                        if isinstance(message, ToolMessage) and message.name == 'read_sequence')
        for forbidden in ('outcome', 'family', 'generation_trace', 'fraudsters:', 'victims:',
                          'Synthetic Invoice Actor', 'Alex Morgan', 'invoice_impersonation',
                          'fraudulent_invoice_payment', 'DO NOT SEND THIS METADATA'):
            self.assertNotIn(forbidden, observed)
        changed = deepcopy(SEQUENCE)
        changed.update(outcome='normal', target_outcome='normal', family='anything',
                       generation_trace=[{'classification': 'normal'}])
        self.assertEqual(prepare_observations(SEQUENCE), prepare_observations(changed))

    def test_all_classifications_and_uncertain_without_evidence_are_supported(self):
        for classification in ('successful_fraud', 'caught_fraud', 'normal', 'uncertain'):
            with self.subTest(classification=classification):
                payload = decision(classification, [] if classification == 'uncertain' else None)
                result, _ = self.run_detection(responses=[
                    tool_call('read_sequence', {}, 'inspect'),
                    tool_call('DetectionResult', payload, 'result'),
                ])
                self.assertEqual(result['classification'], classification)

    def test_structured_output_error_is_corrected_by_agent(self):
        result, model = self.run_detection(responses=[
            tool_call('read_sequence', {}, 'inspect'),
            tool_call('DetectionResult', decision(evidence=[]), 'invalid'),
            tool_call('DetectionResult', decision(), 'corrected'),
        ])
        self.assertEqual(result['classification'], 'successful_fraud')
        self.assertEqual(model.calls, 3)
        self.assertTrue(any(isinstance(message, ToolMessage) and 'requires step evidence' in message.content
                            for message in model.received_messages[-1]))

    def test_invalid_result_does_not_replace_existing_output(self):
        for responses, exception in (
            ([tool_call('DetectionResult', decision(), 'unread')], RuntimeError),
            ([tool_call('read_sequence', {}, 'inspect'),
              tool_call('DetectionResult', decision(evidence=[
                  {'step': 3, 'explanation': 'Nonexistent evidence.'}]), 'result')], ValueError),
            ([AIMessage(content='Unstructured answer')], RuntimeError),
        ):
            with self.subTest(responses=responses), TemporaryDirectory() as directory:
                source, output = Path(directory) / 'input.json', Path(directory) / 'output.json'
                source.write_text(json.dumps(SEQUENCE))
                output.write_text('Existing output')
                with self.assertRaises(exception):
                    detect_sequence(source, output=output, model=ScriptedChatModel(responses=responses))
                self.assertEqual(output.read_text(), 'Existing output')

    def test_tool_loop_is_bounded(self):
        with self.assertRaisesRegex(RuntimeError, 'GraphRecursionError'):
            self.run_detection(responses=[tool_call('read_sequence', {}, str(i)) for i in range(30)])

    def test_invalid_inputs_are_rejected(self):
        for payload in ({}, {'steps': []}, {'steps': [None]}, {'steps': [{'type': 'unknown'}]}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                prepare_observations(payload)
        for amount in (True, '100', 0, -1, float('inf'), float('nan')):
            payload = deepcopy(SEQUENCE)
            payload['steps'][1]['amount'] = amount
            with self.subTest(amount=amount), self.assertRaises(ValueError):
                prepare_observations(payload)

    def test_input_cannot_be_overwritten(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / 'input.json'
            source.write_text(json.dumps(SEQUENCE))
            with self.assertRaisesRegex(ValueError, 'must differ'):
                detect_sequence(source, output=source)

    def test_bare_steps_list_is_supported(self):
        result, _ = self.run_detection(payload=SEQUENCE['steps'])
        self.assertEqual(result['step_count'], 2)


if __name__ == '__main__':
    unittest.main()
