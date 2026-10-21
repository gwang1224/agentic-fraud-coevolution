"""Editable model instructions for synthetic sequence generation."""

GENERATOR_PROMPT = """Select exactly one next step for a fictional, successful fraud sequence.
Return one JSON object matching response_schema, with no Markdown or explanation.
Choose an exact action name from templates. Only templates that can continue the
current stage to a successful transfer are offered. Do not select earlier stages.

Python owns the stage order, chosen family, actor, target, account IDs and effects.
Do not return type, actor, target, account IDs or effect. The required fields are
specified by response_schema. A response step acknowledges the preceding contact;
a follow_up continues that same premise; a disclosure provides fictional information;
a final_transaction completes the simulated transfer. Do not introduce unrelated
fraud premises or switch the family established by the first accepted contact.

For the first contact, choose a premise from the supplied templates. If an optional
scenario is present, prefer a compatible template. On later turns, refer to the
accepted steps and preserve that premise in a brief description. Do not simply
copy an unrelated template's description or invent participants.

If channel is required, use one supported by the selected template, preferably
continuing the previous channel when available. If description is required, write
one brief synthetic sentence about this step. If payment_method and amount are
required, describe the simulated payment method and choose a positive numeric
amount within the victim's current balance. Do not quote numbers as strings.

All sequences must end with a transfer from the selected victim to the selected
fraudster. No report/refusal ending is allowed in this dataset. Keep descriptions
abstract and fictional, with no real credentials or operational fraud instructions.
Generate only the current step, not future steps or a whole sequence.

The current simulation context follows as JSON:
"""

RETRY_PROMPT = "\nThe previous proposal was rejected: {error}\nSelect a permitted choice for this stage."
