"""Validation harness: source injection, incremental replay and summary
statistics (completeness, latency, purity) for the detection pipeline.

Nothing here is used by the production daemon. It exists so that the
numbers a methods paper needs can be regenerated from the shipped
fixtures with one command each (tools/inject_recover.py,
tools/replay_driver.py) instead of being reconstructed from case notes.
"""
