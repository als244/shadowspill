"""Structural task compilation, Inductor manifests, and physical layout.

Task measurement and representative values live in :mod:`shadowspill.pytorch.profiling`.
This initializer remains side-effect free so compiler internals and profiling
records keep an explicit, cycle-free dependency boundary.
"""
