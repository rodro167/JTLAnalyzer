"""REST API layer for JTL Analyzer.

Exposes the specialist agents over HTTP. This layer deliberately bypasses the
orchestrator: the LLM-generated plan is invariant in practice (loader followed
by the four specialists), so invoking an LLM per request would add latency,
cost, and non-determinism for no benefit. The orchestrator remains the CLI's
execution path.
"""
