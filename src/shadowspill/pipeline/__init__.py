"""The ordered work that turns a captured step into a plan, minus the framework.

A frontend captures a step, compiles its tasks and measures them; everything
after that is here. `common` is what every phase shares -- its clock, the budget
and capacity arithmetic the search is configured from, and the two refusals a
plan can raise. `admission` is what a plan needs of a live runtime: its layout,
the budget it seals, the selection it certifies, and admitting one run of it.
`reporting` is the plan report.

the frontend's planning package composes the phases and calls this.
"""
