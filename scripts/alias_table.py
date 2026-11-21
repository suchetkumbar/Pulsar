"""Print the analytic aliasing table for mains flicker and ceiling-fan harmonics (docs 04 §4.2).

uv run python scripts/alias_table.py
"""

from pulsar.signal.alias import alias_table_markdown

if __name__ == "__main__":
    print(alias_table_markdown())
    print("\n**bold** = aliased into the heart-rate band (0.7-3.5 Hz)")
