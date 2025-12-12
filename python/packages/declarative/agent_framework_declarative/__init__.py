# Copyright (c) Microsoft. All rights reserved.

from importlib import metadata

from ._loader import AgentFactory, DeclarativeLoaderError, ProviderLookupError, ProviderTypeMapping
from ._workflows import DeclarativeWorkflowError, WorkflowFactory, WorkflowState
from ._workflows._graph import ExternalInputRequest, ExternalInputResponse

try:
    __version__ = metadata.version(__name__)
except metadata.PackageNotFoundError:
    __version__ = "0.0.0"  # Fallback for development mode

__all__ = [
    "AgentFactory",
    "DeclarativeLoaderError",
    "DeclarativeWorkflowError",
    "ExternalInputRequest",
    "ExternalInputResponse",
    "ProviderLookupError",
    "ProviderTypeMapping",
    "WorkflowFactory",
    "WorkflowState",
    "__version__",
]
