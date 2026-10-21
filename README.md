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
5. Validate and advance the stage. Replay the accepted sequence before saving it.

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

No scenario is required. An optional positional scenario can guide the first
contact choice among valid templates. The output includes `steps`, participants,
the chosen family, final balances, fallback count, and a `generation_trace` of
accepted/rejected candidates and state changes. Successful output always has
`outcome: "successful_fraud"` and `stop_reason: "final_transaction"`.

Driver reference: [Neo4j managed transactions](https://neo4j.com/docs/python-manual/current/transactions/).
