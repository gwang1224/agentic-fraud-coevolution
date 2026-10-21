"""Validate and import the four legacy JSON assets. Run as a module."""

import argparse
import json
import math
from pathlib import Path

from database.connection import ROOT, connect

GROUPS = ("victims", "fraudsters", "fraudster_actions", "victim_actions")
ACTION_STAGES = {
    "fraudster_actions": {"initial_contact", "follow_up", "account_access", "final_transaction"},
    "victim_actions": {"response", "disclosure", "acknowledgment", "report"},
}
CONSTRAINTS = tuple(
    f"CREATE CONSTRAINT {label}_identity IF NOT EXISTS "
    f"FOR (n:{label}) REQUIRE n.{key} IS UNIQUE"
    for label, key in (("victims", "name"), ("fraudsters", "name"),
                       ("account", "id"), ("action", "id"))
)


def required_text(value, context):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a nonempty string")
    return value.strip()


def load_data(directory):
    """Normalize all inputs before any database writes; ignore legit_actions."""
    result = {}
    for group in GROUPS:
        path = Path(directory) / f"{group}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get(group), list):
            raise ValueError(f"{path} must contain a '{group}' array")
        rows, names = [], set()
        for item in payload[group]:
            if not isinstance(item, dict):
                raise ValueError(f"{path}: each row must be an object")
            name = required_text(item.get("name"), f"{path}: name")
            if name in names:
                raise ValueError(f"{path}: duplicate name {name!r}")
            names.add(name)
            if group in ("victims", "fraudsters"):
                account = item.get("account")
                if not isinstance(account, dict):
                    raise ValueError(f"{name}: account must be an object")
                balance = account.get("balance")
                if (isinstance(balance, bool) or not isinstance(balance, (int, float))
                        or not math.isfinite(balance) or balance < 0):
                    raise ValueError(f"{name}: balance must be a finite nonnegative number")
                rows.append({"name": name, "account_id": f"{group}:{name}",
                             "acc_name": required_text(account.get("acc_name"), f"{name}: acc_name"),
                             "bank": required_text(account.get("bank"), f"{name}: bank"),
                             "balance": balance})
            else:
                channels = item.get("common_channels", [])
                if not isinstance(channels, list) or any(not isinstance(c, str) for c in channels):
                    raise ValueError(f"{name}: common_channels must be a list of strings")
                props = {
                    "action": name, "channels": channels, "kind": group,
                    # Legacy custom templates apply to any fraud family until
                    # the author assigns a more specific compatibility family.
                    "family": required_text(item.get("family", "generic"), f"{name}: family"),
                }
                for key in ("description", "stage", "initiator", "target"):
                    if key in item:
                        props[key] = required_text(item[key], f"{name}: {key}")
                if "stage" in props and props["stage"] not in ACTION_STAGES[group]:
                    stages = ", ".join(sorted(ACTION_STAGES[group]))
                    raise ValueError(f"{name}: stage must be one of {stages} for {group}")
                for key in ("compromises_account", "is_terminal", "requires_compromise"):
                    if key in item:
                        if not isinstance(item[key], bool):
                            raise ValueError(f"{name}: {key} must be a boolean")
                        props[key] = item[key]
                rows.append({"id": f"{group}:{name}", "props": props})
        result[group] = rows
    return result


def import_data(tx, data):
    """Import the complete dataset in one transaction (also safe on retry)."""
    for role in ("victims", "fraudsters"):
        tx.run(f"""
            UNWIND $rows AS row
            MERGE (p:{role} {{name: row.name}})
            MERGE (a:account {{id: row.account_id}})
            SET a.acc_name = row.acc_name, a.owner = row.name,
                a.bank = row.bank, a.balance = row.balance
            MERGE (p)-[:OWNS]->(a)
        """, rows=data[role]).consume()
    for group, actor, target in (("fraudster_actions", "fraudsters", "victims"),
                                 ("victim_actions", "victims", "fraudsters")):
        tx.run("""
            UNWIND $rows AS row
            MERGE (a:action {id: row.id})
            SET a += row.props
        """, rows=data[group]).consume()
        # Separate matches avoid materializing the actor x target Cartesian product.
        tx.run(f"""
            MATCH (p:{actor}), (a:action {{kind: $kind}})
            MERGE (p)-[:PERFORMS]->(a)
        """, kind=group).consume()
        tx.run(f"""
            MATCH (a:action {{kind: $kind}}), (p:{target})
            MERGE (a)-[:TARGETS]->(p)
        """, kind=group).consume()


def replace_graph(tx, data):
    """Delete every node/relationship and import validated data atomically.

    Pass this callback to session.execute_write. If import_data fails, the
    deletion rolls back too. Existing indexes and constraints are preserved.
    """
    tx.run("MATCH (n) DETACH DELETE n").consume()
    import_data(tx, data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "graph")
    parser.add_argument("--validate-only", action="store_true", help="Validate JSON without connecting")
    parser.add_argument("--replace", action="store_true",
                        help="Delete ALL nodes and relationships in the configured database, then import")
    args = parser.parse_args()
    data = load_data(args.data_dir)
    counts = {key: len(rows) for key, rows in data.items()}
    if args.validate_only:
        print("Validated:", json.dumps(counts))
        return
    driver, database = connect()
    with driver:
        with driver.session(database=database) as session:
            if args.replace:
                # Old duplicate nodes must be removed before adding constraints.
                session.execute_write(replace_graph, data)
            for query in CONSTRAINTS:
                session.run(query).consume()
            if not args.replace:
                session.execute_write(import_data, data)
    print("Imported:", json.dumps(counts))


if __name__ == "__main__":
    main()
