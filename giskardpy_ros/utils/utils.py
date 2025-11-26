from __future__ import division

import os

import xacro

from giskardpy.middleware import get_middleware


def load_xacro(path: str) -> str:
    path = get_middleware().resolve_iri(path)
    doc = xacro.process_file(path, mappings={"radius": "0.9"})
    return doc.toprettyxml(indent="  ")


def is_in_github_workflow():
    return "GITHUB_WORKFLOW" in os.environ
