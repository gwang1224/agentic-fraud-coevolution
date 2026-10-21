"""Inspect generator state while manually testing one turn.

Run from the project root with: .venv/bin/python test.py
"""

import os
from pprint import pprint

import generator_agent as g


def print_state(label, state):
    print(f"\n{'=' * 24} {label} {'=' * 24}")
    pprint(state, sort_dicts=False, width=100, depth=None)


def main():
    fraudster, victim = g.select_starting_character()
    print("Selected participants (the generator state is initialized next):")
    pprint({"fraudster": fraudster, "victim": victim}, sort_dicts=False)

    state = g.initialize_state(fraudster, victim)
    print_state("Initial state", state)

    g.scope_available_actions(state, fraudster)
    print_state("After scoping actions", state)

    #Enable this section when ready to test an Ollama response.
    model = g.ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "llama3.2"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,
        format="json",
    )
    step = g.generate_next_action(
        state, None, fraudster, victim, model
    )
    print("Candidate:")
    pprint(step)
    print_state("After generating candidate", state)
    
    g.validate_step(step, state, fraudster, victim)
    print_state("After validating candidate", state)
    #
    g.update_state(state, step, fraudster, victim)
    print_state("After applying candidate", state)
    #
    # g.check_completion(state)
    # print_state("After checking completion", state)


if __name__ == "__main__":
    main()
