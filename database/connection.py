"""Connection settings shared by graph commands."""

import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parents[1]


def connect():
    """Return a verified driver and database name; never log credentials."""
    load_dotenv(ROOT / ".env")
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    if not password:
        raise ValueError("Set NEO4J_PASSWORD in .env or your environment")
    driver = GraphDatabase.driver(uri, auth=(user, password), connection_timeout=10)
    try:
        driver.verify_connectivity()
    except Exception:
        driver.close()
        raise
    return driver, os.getenv("NEO4J_DATABASE", "neo4j")
