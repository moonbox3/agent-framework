# Copyright (c) Microsoft. All rights reserved.

"""
Run the conditional workflow sample.

Usage:
    python main.py

Demonstrates conditional branching based on age input.
"""

from pathlib import Path

from agent_framework.devui import serve
from agent_framework_declarative import WorkflowFactory


def main():
    """Run the conditional workflow with various age inputs."""
    # Create a workflow factory
    factory = WorkflowFactory()

    # Load the workflow from YAML
    workflow_path = Path(__file__).parent / "workflow.yaml"
    workflow = factory.create_workflow_from_yaml_path(workflow_path)

    serve(entities=[workflow], auto_open=True)


if __name__ == "__main__":
    main()
