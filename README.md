# Fraud simulation graph

Imports the same four JSON files as the previous loader into Neo4j 5 or 6.
The included synthetic data provides people, accounts, and compatible action templates.

```text
(fraudsters)-[:OWNS]->(account)
(victims)-[:OWNS]->(account)
(fraudsters)-[:PERFORMS]->(action)-[:TARGETS]->(victims)
(victims)-[:PERFORMS]->(action)-[:TARGETS]->(fraudsters)
```

Actions represent available simulation templates, not recorded events: each is
linked to every actor and target of its role, matching the previous code.
`initiator` and `target` are descriptive properties, not person identifiers.
`legit_actions` in `victim_actions.json` is deliberately ignored.

## Setup and import

Start a Neo4j instance using Neo4j Desktop or use an existing server/Aura database.
Set its connection details in `.env` using `.env.example` as a reference. Preserve
your existing `.env` if it is already configured. Aura normally uses a
`neo4j+s://` URI supplied by its console.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m database.build_graph --validate-only
python -m database.build_graph
```

Use `--data-dir /path/to/graph` for your previous JSON assets. Paths default to
this repository's `data/graph`, independent of the working directory. Run the
module from the repository root.

People require `name` and an `account` containing `acc_name`, `bank`, and a
nonnegative numeric `balance`. Actions require `name`; optional fields are
`common_channels` (string array), `description`, `stage`, `family`, `initiator`,
`target` (strings), `compromises_account`, `requires_compromise`, and `is_terminal`
(booleans). The loader defaults omitted `family` values to `generic`. Generator
templates need explicit stages and an imported family value.

Imports validate all four files first, create uniqueness constraints, then write
data in one transaction. Repeating an import updates matching nodes without
duplicates and resets account balances to the input values. People are identified
by role and name, accounts by role and owner name, and actions by role and name;
numeric IDs from older generators are ignored because they may repeat. This
model supports one account per person. Rename operations create new identities.
Imports are additive: removing a row from JSON does not remove it from Neo4j.
By default the loader preserves existing data. An existing graph built with the old
`CREATE` loader may contain duplicate people that prevent constraint creation;
use a fresh database or resolve those duplicates before importing.

To delete **all nodes and relationships in the configured database** (including
old transaction nodes) and rebuild from the four JSON files:

```sh
python -m database.build_graph --replace
```

The `replace_graph(tx, data)` function deletes and imports in the same transaction,
so an import failure restores the previous graph. Input validation runs before
deletion. Existing indexes and constraints remain; missing uniqueness constraints
are created after the replacement commits, allowing old duplicate nodes to be
removed first. `--replace --validate-only` only validates and does not delete data.

## View the graph

In Neo4j Browser, connect to the same database and run:

```cypher
MATCH (p)-[r:OWNS|PERFORMS|TARGETS]->(q)
RETURN p, r, q LIMIT 100;
```

```cypher
MATCH (f:fraudsters)-[:PERFORMS]->(a:action)-[:TARGETS]->(v:victims)
RETURN f.name AS fraudster, a.action AS action, v.name AS victim;
```

Run local validation tests with `python -m unittest discover -s tests -v`.

## Generate a fraud sequence

The generator exports successful synthetic sequences ending in a positive transfer
from the selected victim account to the selected fraudster account.

1. Select a random fraudster and a victim with a positive account balance.
2. Snapshot both participants' connected `PERFORMS` templates from Neo4j.
3. Preflight complete, compatible paths before asking the model for any choices.
4. Let the model choose only among templates for the current stage and family.
5. Validate and advance each stage, then check completion and the step limit before saving.

The default path has five steps:

```text
fraudster contact -> victim response -> fraudster follow_up
                 -> victim disclosure -> fraudster transfer
```

Custom template sets can use a three-step contact/disclosure/transfer path. When
only compromise-required transfers are compatible, account compromise and a
victim acknowledgment are inserted before the transfer. All paths alternate the
actors and fit within `max_steps` (default 15). Initial contacts cannot repeat on
later turns. Compatible templates share `family` or use `generic`; a specific
family's templates are preferred over generic alternatives.

Python assigns the participants, account IDs, and effects. The model chooses the
action, channel, short description, or transfer amount/payment method. Ollama
receives a structured schema limited to the current stage. Invalid model output
or a failed request is retried up to three times; if all fail, Python constructs
and validates a template fallback for that stage. `selection_source`,
`fallback_reason`, and `fallback_count` make this visible. A completed sequence may
therefore contain both model-selected and deterministic template steps.

Missing compatible templates, stale metadata, an invalid balance, or a step budget
that cannot fit a complete path raises an error before generation and leaves any
existing output file untouched. The generator does not save incomplete runs or
victim-report endings. Free-text descriptions still require review for realism.
No JSON graph files are read during generation. Neo4j is read only during generation;
simulated debits and credits stay in memory and the saved result.

After updating this version, import the template metadata and new stages once:

```sh
python -m database.build_graph --validate-only
python -m database.build_graph
```

This additive import preserves graph nodes and relationships, updates matching
templates, adds the new stages, and resets imported balances to their JSON values.
Do not use `--replace` for this update.

With Neo4j and Ollama running, and the model named by `OLLAMA_MODEL` installed:

```sh
python generator_agent.py --output data/generated/full_sequence.json
```

The model chooses the first contact from valid templates. The output includes `steps`, participants,
the chosen family, final balances, fallback count, and a `generation_trace` of
accepted/rejected candidates and state changes. Successful output always has
`outcome: "successful_fraud"` and `stop_reason: "final_transaction"`.

Driver reference: [Neo4j managed transactions](https://neo4j.com/docs/python-manual/current/transactions/).

## Detect fraud in a sequence

`detector_agent.py` uses LangChain's `create_agent` with Ollama. The agent calls
the read-only `read_sequence` tool, examines the observed events, and returns a
validated decision with a short summary and evidence referencing one-based step
numbers. Its instructions are in `DETECTOR_PROMPT` in `prompts.py`.

```sh
python -m pip install -r requirements.txt
python detector_agent.py data/generated/full_sequence.json --output data/detected/full_sequence.json
```

Without arguments, input defaults to `data/generated/sequence.json` and output to
`data/detected/sequence.json`. Existing detector output at the chosen path is
overwritten only after a valid decision. The input sequence is never overwritten.
Neo4j is not required for detection.

The detector uses `DETECTOR_MODEL` when set, otherwise `OLLAMA_MODEL`, defaulting
to `llama3.2`. `OLLAMA_BASE_URL` selects the Ollama server. The installed model must
support tool calling. Unlike the generator's JSON-only calls, this agent needs
native tool calls for both reading events and returning structured output.

Possible classifications:

| Classification | Meaning |
|---|---|
| `successful_fraud` | A fraudulent attempt followed by an observed fraudulent payment |
| `caught_fraud` | A fraudulent attempt stopped before its payment |
| `normal` | Observations support a legitimate interaction/payment |
| `uncertain` | Insufficient or conflicting evidence to establish another outcome |

Example output fields:

```json
{
  "classification": "successful_fraud",
  "summary": "A deceptive payment request was followed by a transfer.",
  "evidence": [
    {"step": 1, "explanation": "The observed request misrepresents its purpose."},
    {"step": 5, "explanation": "The sequence records the subsequent payment."}
  ],
  "tool_calls": ["read_sequence"]
}
```

The model sees descriptions, channels, payments, and neutral participant aliases.
Generator outcome labels, family/stage metadata, action names, effects, and audit
traces are excluded. Account ownership is retained through neutral aliases when
participant metadata provides it. This reduces obvious label leakage, but current
synthetic descriptions still explicitly mention deception/fraud: they are not an
unbiased real-world benchmark. The tests verify agent execution and validation,
not the model's detection accuracy. Include normal and intercepted examples in a
separate evaluation set before comparing detector performance.

The saved explanation is a concise decision summary, not private chain-of-thought.
The agent has a bounded execution loop; provider failures, missing decisions, and
invalid evidence fail explicitly instead of fabricating a classification.

Run detector tests without external services:

```sh
python -m unittest discover -s tests -p 'test_detector_agent.py' -v
```

## Start prompt-feedback coevolution

With Neo4j populated and Ollama running, run a bounded experiment:

```sh
.venv/bin/python coevolution.py --rounds 3
```

Each round generates a complete sequence, runs the detector, compares its decision
with the simulator's outcome, and supplies the last three completed rounds as
feedback to the next round's prompts. The generator receives prior detector
summaries and evidence to vary its fictional examples. The detector receives
prior observations and expected labels; current-round labels remain hidden from
it. Template constraints and sequence validation still apply. This starter adapts
prompt context, not model weights, and does not modify Neo4j.

Runs create a new directory under `data/coevolution/` with per-round
`sequence.json`, `detection.json`, and `feedback.json`, plus a run-level
`report.json`. Use `--output-dir PATH` to choose a new directory; existing run
directories are never reused. A failed round stops the run and records the error
without counting it as a missed detection. Fallback steps are counted explicitly.

The report's successful-fraud recall is a training statistic on fraud-only
examples, not overall accuracy or evidence of improvement. A fixed held-out set
with legitimate and intercepted interactions is needed to evaluate adaptation
and false positives independently.
