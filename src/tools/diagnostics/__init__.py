"""Tools that read serialized planning and step evidence.

Nothing here runs a plan. Each module takes what a run already wrote -- a
plan the store holds, a traced step's diagnostics -- and answers a question
about it, so the evidence of any run can be read again without the machine
that produced it.
"""
