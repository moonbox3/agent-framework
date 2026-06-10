# Copyright (c) Microsoft. All rights reserved.

"""Tests for FastAPI endpoint creation (_endpoint.py)."""

import json
from typing import Any

import pytest
from ag_ui.core import RunStartedEvent
from agent_framework import (
    Agent,
    ChatResponseUpdate,
    Content,
    WorkflowBuilder,
    WorkflowContext,
    executor,
)
from agent_framework.orchestrations import SequentialBuilder
from fastapi import FastAPI, Header, HTTPException
from fastapi.params import Depends
from fastapi.testclient import TestClient

from agent_framework_ag_ui import InMemoryAGUIThreadSnapshotStore, add_agent_framework_fastapi_endpoint
from agent_framework_ag_ui._agent import AgentFrameworkAgent
from agent_framework_ag_ui._workflow import AgentFrameworkWorkflow


def _decode_sse_events(response: Any) -> list[dict[str, Any]]:
    content = response.content.decode("utf-8")
    return [json.loads(line[6:]) for line in content.splitlines() if line.startswith("data: ")]


def _latest_messages_snapshot(response: Any) -> list[dict[str, Any]]:
    snapshots = [
        event["messages"] for event in _decode_sse_events(response) if event.get("type") == "MESSAGES_SNAPSHOT"
    ]
    assert snapshots
    return snapshots[-1]


@pytest.fixture
def build_chat_client(streaming_chat_client_stub, stream_from_updates_fixture):
    """Create a typed chat client stub for endpoint tests."""

    def _build(response_text: str = "Test response"):
        updates = [ChatResponseUpdate(contents=[Content.from_text(text=response_text)])]
        return streaming_chat_client_stub(stream_from_updates_fixture(updates))

    return _build


async def test_add_endpoint_with_agent_protocol(build_chat_client):
    """Test adding endpoint with raw SupportsAgentRun."""
    app = FastAPI()
    agent = Agent(name="test", instructions="Test agent", client=build_chat_client())

    add_agent_framework_fastapi_endpoint(app, agent, path="/test-agent")

    client = TestClient(app)
    response = client.post("/test-agent", json={"messages": [{"role": "user", "content": "Hello"}]})

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"


async def test_add_endpoint_with_wrapped_agent(build_chat_client):
    """Test adding endpoint with pre-wrapped AgentFrameworkAgent."""
    app = FastAPI()
    agent = Agent(name="test", instructions="Test agent", client=build_chat_client())
    wrapped_agent = AgentFrameworkAgent(agent=agent, name="wrapped")

    add_agent_framework_fastapi_endpoint(app, wrapped_agent, path="/wrapped-agent")

    client = TestClient(app)
    response = client.post("/wrapped-agent", json={"messages": [{"role": "user", "content": "Hello"}]})

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"


async def test_add_endpoint_with_workflow_protocol():
    """Test adding endpoint with native Workflow support."""

    @executor(id="start")
    async def start(message: Any, ctx: WorkflowContext) -> None:
        await ctx.yield_output("Workflow response")

    app = FastAPI()
    workflow = WorkflowBuilder(start_executor=start).build()

    add_agent_framework_fastapi_endpoint(app, workflow, path="/workflow")

    client = TestClient(app)
    response = client.post("/workflow", json={"messages": [{"role": "user", "content": "Hello"}]})

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    content = response.content.decode("utf-8")
    lines = [line for line in content.split("\n") if line.startswith("data: ")]
    event_types = [json.loads(line[6:]).get("type") for line in lines]
    assert "RUN_STARTED" in event_types
    assert "TEXT_MESSAGE_CONTENT" in event_types
    assert "RUN_FINISHED" in event_types


async def test_endpoint_with_state_schema(build_chat_client):
    """Test endpoint with state_schema parameter."""
    app = FastAPI()
    agent = Agent(name="test", instructions="Test agent", client=build_chat_client())
    state_schema = {"document": {"type": "string"}}

    add_agent_framework_fastapi_endpoint(app, agent, path="/stateful", state_schema=state_schema)

    client = TestClient(app)
    response = client.post(
        "/stateful", json={"messages": [{"role": "user", "content": "Hello"}], "state": {"document": ""}}
    )

    assert response.status_code == 200


async def test_endpoint_with_default_state_seed(build_chat_client):
    """Test endpoint seeds default state when client omits it."""
    app = FastAPI()
    agent = Agent(name="test", instructions="Test agent", client=build_chat_client())
    state_schema = {"proverbs": {"type": "array"}}
    default_state = {"proverbs": ["Keep the original."]}

    add_agent_framework_fastapi_endpoint(
        app,
        agent,
        path="/default-state",
        state_schema=state_schema,
        default_state=default_state,
    )

    client = TestClient(app)
    response = client.post("/default-state", json={"messages": [{"role": "user", "content": "Hello"}]})

    assert response.status_code == 200

    content = response.content.decode("utf-8")
    lines = [line for line in content.split("\n") if line.startswith("data: ")]
    snapshots = [json.loads(line[6:]) for line in lines if json.loads(line[6:]).get("type") == "STATE_SNAPSHOT"]
    assert snapshots, "Expected a STATE_SNAPSHOT event"
    assert snapshots[0]["snapshot"]["proverbs"] == default_state["proverbs"]


async def test_endpoint_with_predict_state_config(build_chat_client):
    """Test endpoint with predict_state_config parameter."""
    app = FastAPI()
    agent = Agent(name="test", instructions="Test agent", client=build_chat_client())
    predict_config = {"document": {"tool": "write_doc", "tool_argument": "content"}}

    add_agent_framework_fastapi_endpoint(app, agent, path="/predictive", predict_state_config=predict_config)

    client = TestClient(app)
    response = client.post("/predictive", json={"messages": [{"role": "user", "content": "Hello"}]})

    assert response.status_code == 200


async def test_endpoint_request_logging(build_chat_client):
    """Test that endpoint logs request details."""
    app = FastAPI()
    agent = Agent(name="test", instructions="Test agent", client=build_chat_client())

    add_agent_framework_fastapi_endpoint(app, agent, path="/logged")

    client = TestClient(app)
    response = client.post(
        "/logged",
        json={
            "messages": [{"role": "user", "content": "Test"}],
            "run_id": "run-123",
            "thread_id": "thread-456",
        },
    )

    assert response.status_code == 200


async def test_endpoint_event_streaming(build_chat_client):
    """Test that endpoint streams events correctly."""
    app = FastAPI()
    agent = Agent(name="test", instructions="Test agent", client=build_chat_client("Streamed response"))

    add_agent_framework_fastapi_endpoint(app, agent, path="/stream")

    client = TestClient(app)
    response = client.post("/stream", json={"messages": [{"role": "user", "content": "Hello"}]})

    assert response.status_code == 200

    content = response.content.decode("utf-8")
    lines = [line for line in content.split("\n") if line.strip()]

    found_run_started = False
    found_text_content = False
    found_run_finished = False

    for line in lines:
        if line.startswith("data: "):
            event_data = json.loads(line[6:])
            if event_data.get("type") == "RUN_STARTED":
                found_run_started = True
            elif event_data.get("type") == "TEXT_MESSAGE_CONTENT":
                found_text_content = True
            elif event_data.get("type") == "RUN_FINISHED":
                found_run_finished = True

    assert found_run_started
    assert found_text_content
    assert found_run_finished


async def test_endpoint_with_workflow_as_agent_stream_output(build_chat_client):
    """Test endpoint handles workflow-as-agent stream outputs."""
    app = FastAPI()
    brainstorm_agent = Agent(name="brainstorm", instructions="Brainstorm ideas", client=build_chat_client("Idea"))
    reviewer_agent = Agent(name="reviewer", instructions="Review ideas", client=build_chat_client("Review"))
    agent = SequentialBuilder(participants=[brainstorm_agent, reviewer_agent]).build().as_agent()

    add_agent_framework_fastapi_endpoint(app, agent, path="/workflow-like")

    client = TestClient(app)
    response = client.post("/workflow-like", json={"messages": [{"role": "user", "content": "Hello"}]})

    assert response.status_code == 200
    content = response.content.decode("utf-8")
    lines = [line for line in content.split("\n") if line.startswith("data: ")]
    event_types = [json.loads(line[6:]).get("type") for line in lines]

    assert "RUN_STARTED" in event_types
    assert "TEXT_MESSAGE_CONTENT" in event_types
    assert "RUN_FINISHED" in event_types


async def test_endpoint_error_handling(build_chat_client):
    """Test endpoint error handling during request parsing."""
    app = FastAPI()
    agent = Agent(name="test", instructions="Test agent", client=build_chat_client())

    add_agent_framework_fastapi_endpoint(app, agent, path="/failing")

    client = TestClient(app)

    # Send invalid JSON to trigger parsing error before streaming
    response = client.post("/failing", data=b"invalid json", headers={"content-type": "application/json"})  # type: ignore

    # Pydantic validation now returns 422 for invalid request body
    assert response.status_code == 422


async def test_endpoint_multiple_paths(build_chat_client):
    """Test adding multiple endpoints with different paths."""
    app = FastAPI()
    agent1 = Agent(name="agent1", instructions="First agent", client=build_chat_client("Response 1"))
    agent2 = Agent(name="agent2", instructions="Second agent", client=build_chat_client("Response 2"))

    add_agent_framework_fastapi_endpoint(app, agent1, path="/agent1")
    add_agent_framework_fastapi_endpoint(app, agent2, path="/agent2")

    client = TestClient(app)

    response1 = client.post("/agent1", json={"messages": [{"role": "user", "content": "Hi"}]})
    response2 = client.post("/agent2", json={"messages": [{"role": "user", "content": "Hi"}]})

    assert response1.status_code == 200
    assert response2.status_code == 200


async def test_endpoint_default_path(build_chat_client):
    """Test endpoint with default path."""
    app = FastAPI()
    agent = Agent(name="test", instructions="Test agent", client=build_chat_client())

    add_agent_framework_fastapi_endpoint(app, agent)

    client = TestClient(app)
    response = client.post("/", json={"messages": [{"role": "user", "content": "Hello"}]})

    assert response.status_code == 200


async def test_endpoint_response_headers(build_chat_client):
    """Test that endpoint sets correct response headers."""
    app = FastAPI()
    agent = Agent(name="test", instructions="Test agent", client=build_chat_client())

    add_agent_framework_fastapi_endpoint(app, agent, path="/headers")

    client = TestClient(app)
    response = client.post("/headers", json={"messages": [{"role": "user", "content": "Test"}]})

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    assert "cache-control" in response.headers
    assert response.headers["cache-control"] == "no-cache"


async def test_endpoint_empty_messages(streaming_chat_client_stub):
    """Empty messages keep the existing no-op run behavior when snapshot persistence is not configured."""
    app = FastAPI()
    call_count = 0

    async def stream_fn(messages: Any, options: Any, **kwargs: Any):
        nonlocal call_count
        del messages, options, kwargs
        call_count += 1
        yield ChatResponseUpdate(contents=[Content.from_text(text="Should not run")])

    agent = Agent(name="test", instructions="Test agent", client=streaming_chat_client_stub(stream_fn))

    add_agent_framework_fastapi_endpoint(app, agent, path="/empty")

    client = TestClient(app)
    response = client.post("/empty", json={"messages": []})

    assert response.status_code == 200
    assert call_count == 0
    assert [event.get("type") for event in _decode_sse_events(response)] == ["RUN_STARTED", "RUN_FINISHED"]


async def test_endpoint_complex_input(build_chat_client):
    """Test endpoint with complex input data."""
    app = FastAPI()
    agent = Agent(name="test", instructions="Test agent", client=build_chat_client())

    add_agent_framework_fastapi_endpoint(app, agent, path="/complex")

    client = TestClient(app)
    response = client.post(
        "/complex",
        json={
            "messages": [
                {"role": "user", "content": "First message", "id": "msg-1"},
                {"role": "assistant", "content": "Response", "id": "msg-2"},
                {"role": "user", "content": "Follow-up", "id": "msg-3"},
            ],
            "run_id": "complex-run-123",
            "thread_id": "complex-thread-456",
            "state": {"custom_field": "value"},
        },
    )

    assert response.status_code == 200


async def test_endpoint_openapi_schema(build_chat_client):
    """Test that endpoint generates proper OpenAPI schema with request model."""
    app = FastAPI()
    agent = Agent(name="test", instructions="Test agent", client=build_chat_client())

    add_agent_framework_fastapi_endpoint(app, agent, path="/schema-test")

    client = TestClient(app)
    response = client.get("/openapi.json")

    assert response.status_code == 200
    openapi_spec = response.json()

    # Verify the endpoint exists in the schema
    assert "/schema-test" in openapi_spec["paths"]
    endpoint_spec = openapi_spec["paths"]["/schema-test"]["post"]

    # Verify request body schema is defined
    assert "requestBody" in endpoint_spec
    request_body = endpoint_spec["requestBody"]
    assert "content" in request_body
    assert "application/json" in request_body["content"]

    # Verify schema references AGUIRequest model
    schema_ref = request_body["content"]["application/json"]["schema"]
    assert "$ref" in schema_ref
    assert "AGUIRequest" in schema_ref["$ref"]

    # Verify AGUIRequest model is in components
    assert "components" in openapi_spec
    assert "schemas" in openapi_spec["components"]
    assert "AGUIRequest" in openapi_spec["components"]["schemas"]

    # Verify AGUIRequest has required fields
    agui_request_schema = openapi_spec["components"]["schemas"]["AGUIRequest"]
    assert "properties" in agui_request_schema
    assert "messages" in agui_request_schema["properties"]
    assert "run_id" in agui_request_schema["properties"]
    assert "thread_id" in agui_request_schema["properties"]
    assert "state" in agui_request_schema["properties"]
    assert "required" in agui_request_schema
    assert "messages" in agui_request_schema["required"]


async def test_endpoint_default_tags(build_chat_client):
    """Test that endpoint uses default 'AG-UI' tag."""
    app = FastAPI()
    agent = Agent(name="test", instructions="Test agent", client=build_chat_client())

    add_agent_framework_fastapi_endpoint(app, agent, path="/default-tags")

    client = TestClient(app)
    response = client.get("/openapi.json")

    assert response.status_code == 200
    openapi_spec = response.json()

    endpoint_spec = openapi_spec["paths"]["/default-tags"]["post"]
    assert "tags" in endpoint_spec
    assert endpoint_spec["tags"] == ["AG-UI"]


async def test_endpoint_custom_tags(build_chat_client):
    """Test that endpoint accepts custom tags."""
    app = FastAPI()
    agent = Agent(name="test", instructions="Test agent", client=build_chat_client())

    add_agent_framework_fastapi_endpoint(app, agent, path="/custom-tags", tags=["Custom", "Agent"])

    client = TestClient(app)
    response = client.get("/openapi.json")

    assert response.status_code == 200
    openapi_spec = response.json()

    endpoint_spec = openapi_spec["paths"]["/custom-tags"]["post"]
    assert "tags" in endpoint_spec
    assert endpoint_spec["tags"] == ["Custom", "Agent"]


async def test_endpoint_missing_required_field(build_chat_client):
    """Test that endpoint validates required fields with Pydantic."""
    app = FastAPI()
    agent = Agent(name="test", instructions="Test agent", client=build_chat_client())

    add_agent_framework_fastapi_endpoint(app, agent, path="/validation")

    client = TestClient(app)

    # Missing required 'messages' field should trigger validation error
    response = client.post("/validation", json={"run_id": "test-123"})

    assert response.status_code == 422
    error_detail = response.json()
    assert "detail" in error_detail


async def test_endpoint_internal_error_handling(build_chat_client):
    """Test endpoint error handling when an exception occurs before streaming starts."""
    from unittest.mock import patch

    app = FastAPI()
    agent = Agent(name="test", instructions="Test agent", client=build_chat_client())

    # Use default_state to trigger the code path that can raise an exception
    add_agent_framework_fastapi_endpoint(app, agent, path="/error-test", default_state={"key": "value"})

    client = TestClient(app)

    # Mock copy.deepcopy to raise an exception during default_state processing
    with patch("agent_framework_ag_ui._endpoint.copy.deepcopy") as mock_deepcopy:
        mock_deepcopy.side_effect = Exception("Simulated internal error")
        response = client.post("/error-test", json={"messages": [{"role": "user", "content": "Hello"}]})

    assert response.status_code == 500
    assert response.json() == {"detail": "An internal error has occurred."}


async def test_endpoint_streaming_error_emits_run_error_event():
    """Streaming exceptions should emit RUN_ERROR instead of terminating silently."""

    class FailingStreamWorkflow(AgentFrameworkWorkflow):
        async def run(self, input_data: dict[str, Any]):
            del input_data
            yield RunStartedEvent(run_id="run-1", thread_id="thread-1")
            raise RuntimeError("stream exploded")

    app = FastAPI()
    add_agent_framework_fastapi_endpoint(app, FailingStreamWorkflow(), path="/stream-error")
    client = TestClient(app)

    response = client.post("/stream-error", json={"messages": [{"role": "user", "content": "Hello"}]})
    assert response.status_code == 200

    content = response.content.decode("utf-8")
    lines = [line for line in content.split("\n") if line.startswith("data: ")]
    event_types = [json.loads(line[6:]).get("type") for line in lines]

    assert "RUN_STARTED" in event_types
    assert "RUN_ERROR" in event_types


async def test_endpoint_with_dependencies_blocks_unauthorized(build_chat_client):
    """Test that endpoint blocks requests when authentication dependency fails."""
    app = FastAPI()
    agent = Agent(name="test", instructions="Test agent", client=build_chat_client())

    async def require_api_key(x_api_key: str | None = Header(None)):
        if x_api_key != "secret-key":
            raise HTTPException(status_code=401, detail="Unauthorized")

    add_agent_framework_fastapi_endpoint(app, agent, path="/protected", dependencies=[Depends(require_api_key)])

    client = TestClient(app)

    # Request without API key should be rejected
    response = client.post("/protected", json={"messages": [{"role": "user", "content": "Hello"}]})
    assert response.status_code == 401
    assert response.json()["detail"] == "Unauthorized"


async def test_endpoint_with_dependencies_allows_authorized(build_chat_client):
    """Test that endpoint allows requests when authentication dependency passes."""
    app = FastAPI()
    agent = Agent(name="test", instructions="Test agent", client=build_chat_client())

    async def require_api_key(x_api_key: str | None = Header(None)):
        if x_api_key != "secret-key":
            raise HTTPException(status_code=401, detail="Unauthorized")

    add_agent_framework_fastapi_endpoint(app, agent, path="/protected", dependencies=[Depends(require_api_key)])

    client = TestClient(app)

    # Request with valid API key should succeed
    response = client.post(
        "/protected",
        json={"messages": [{"role": "user", "content": "Hello"}]},
        headers={"x-api-key": "secret-key"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"


async def test_endpoint_with_multiple_dependencies(build_chat_client):
    """Test that endpoint supports multiple dependencies."""
    app = FastAPI()
    agent = Agent(name="test", instructions="Test agent", client=build_chat_client())

    execution_order: list[str] = []

    async def first_dependency():
        execution_order.append("first")

    async def second_dependency():
        execution_order.append("second")

    add_agent_framework_fastapi_endpoint(
        app,
        agent,
        path="/multi-deps",
        dependencies=[Depends(first_dependency), Depends(second_dependency)],
    )

    client = TestClient(app)
    response = client.post("/multi-deps", json={"messages": [{"role": "user", "content": "Hello"}]})

    assert response.status_code == 200
    assert "first" in execution_order
    assert "second" in execution_order


async def test_endpoint_without_dependencies_is_accessible(build_chat_client):
    """Test that endpoint without dependencies remains accessible (backward compatibility)."""
    app = FastAPI()
    agent = Agent(name="test", instructions="Test agent", client=build_chat_client())

    # No dependencies parameter - should be accessible without auth
    add_agent_framework_fastapi_endpoint(app, agent, path="/open")

    client = TestClient(app)
    response = client.post("/open", json={"messages": [{"role": "user", "content": "Hello"}]})

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"


async def test_endpoint_invalid_agent_type_raises_typeerror():
    """Passing an invalid agent type raises TypeError."""
    app = FastAPI()

    with pytest.raises(TypeError, match="must be SupportsAgentRun"):
        add_agent_framework_fastapi_endpoint(app, agent="not_an_agent")  # type: ignore[arg-type]


async def test_endpoint_requires_snapshot_scope_resolver_when_store_configured(build_chat_client):
    """Snapshot persistence setup must require an explicit Snapshot Scope resolver."""
    app = FastAPI()
    agent = Agent(name="test", instructions="Test agent", client=build_chat_client())
    store = InMemoryAGUIThreadSnapshotStore()

    with pytest.raises(ValueError, match="snapshot_scope_resolver is required"):
        add_agent_framework_fastapi_endpoint(app, agent, path="/snapshots", snapshot_store=store)


async def test_endpoint_requires_snapshot_scope_resolver_when_wrapped_runner_has_store(build_chat_client):
    """Pre-wrapped runners with snapshot stores must also provide a Snapshot Scope resolver."""
    app = FastAPI()
    agent = Agent(name="test", instructions="Test agent", client=build_chat_client())
    wrapped_agent = AgentFrameworkAgent(agent=agent, snapshot_store=InMemoryAGUIThreadSnapshotStore())

    with pytest.raises(ValueError, match="snapshot_scope_resolver is required"):
        add_agent_framework_fastapi_endpoint(app, wrapped_agent, path="/snapshots")


async def test_endpoint_accepts_snapshot_store_with_scope_resolver(build_chat_client):
    """Endpoint behavior remains the normal event stream when snapshot persistence is explicitly configured."""
    app = FastAPI()
    agent = Agent(name="test", instructions="Test agent", client=build_chat_client())
    store = InMemoryAGUIThreadSnapshotStore()

    add_agent_framework_fastapi_endpoint(
        app,
        agent,
        path="/snapshots",
        snapshot_store=store,
        snapshot_scope_resolver=lambda _request: "tenant-a",
    )

    client = TestClient(app)
    response = client.post(
        "/snapshots",
        json={"messages": [{"role": "user", "content": "Hello"}], "thread_id": "thread-1"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"


async def test_agent_endpoint_hydrates_stored_thread_snapshot_without_invoking_agent(streaming_chat_client_stub):
    """A Hydrate Request replays stored agent messages and state without invoking the wrapped agent."""
    app = FastAPI()
    call_count = 0

    async def stream_fn(messages: Any, options: Any, **kwargs: Any):
        nonlocal call_count
        del messages, options, kwargs
        call_count += 1
        yield ChatResponseUpdate(contents=[Content.from_text(text="Stored reply")])

    agent = Agent(name="test", instructions="Test agent", client=streaming_chat_client_stub(stream_fn))
    store = InMemoryAGUIThreadSnapshotStore()
    add_agent_framework_fastapi_endpoint(
        app,
        agent,
        path="/snapshots",
        state_schema={"recipe": {"type": "string"}},
        snapshot_store=store,
        snapshot_scope_resolver=lambda _request: "tenant-a",
    )
    client = TestClient(app)

    first_response = client.post(
        "/snapshots",
        json={
            "thread_id": "thread-1",
            "messages": [{"role": "user", "content": "Hello"}],
            "state": {"recipe": "pasta"},
        },
    )
    assert first_response.status_code == 200
    assert call_count == 1

    hydrate_response = client.post("/snapshots", json={"thread_id": "thread-1", "messages": []})

    assert hydrate_response.status_code == 200
    assert call_count == 1
    events = _decode_sse_events(hydrate_response)
    event_types = [event.get("type") for event in events]
    assert event_types == ["RUN_STARTED", "STATE_SNAPSHOT", "MESSAGES_SNAPSHOT", "RUN_FINISHED"]
    assert events[1]["snapshot"] == {"recipe": "pasta"}
    assert any(message.get("role") == "user" and message.get("content") == "Hello" for message in events[2]["messages"])
    assert any(
        message.get("role") == "assistant" and message.get("content") == "Stored reply"
        for message in events[2]["messages"]
    )


async def test_agent_endpoint_hydrates_snapshots_by_scope_and_thread(streaming_chat_client_stub):
    """Hydration uses Snapshot Scope and AG-UI Thread id together when reading stored snapshots."""
    app = FastAPI()
    call_count = 0

    async def stream_fn(messages: Any, options: Any, **kwargs: Any):
        nonlocal call_count
        del messages, options, kwargs
        call_count += 1
        yield ChatResponseUpdate(contents=[Content.from_text(text="Tenant A reply")])

    agent = Agent(name="test", instructions="Test agent", client=streaming_chat_client_stub(stream_fn))
    store = InMemoryAGUIThreadSnapshotStore()
    add_agent_framework_fastapi_endpoint(
        app,
        agent,
        path="/snapshots",
        state_schema={"tenant": {"type": "string"}},
        snapshot_store=store,
        snapshot_scope_resolver=lambda request: request.forwarded_props["tenant"],
    )
    client = TestClient(app)

    first_response = client.post(
        "/snapshots",
        json={
            "thread_id": "thread-1",
            "messages": [{"role": "user", "content": "Hello tenant A"}],
            "state": {"tenant": "tenant-a"},
            "forwardedProps": {"tenant": "tenant-a"},
        },
    )
    assert first_response.status_code == 200
    assert call_count == 1

    tenant_b_response = client.post(
        "/snapshots",
        json={"thread_id": "thread-1", "messages": [], "forwardedProps": {"tenant": "tenant-b"}},
    )
    assert tenant_b_response.status_code == 200
    assert call_count == 1
    assert [event.get("type") for event in _decode_sse_events(tenant_b_response)] == [
        "RUN_STARTED",
        "RUN_FINISHED",
    ]

    tenant_a_response = client.post(
        "/snapshots",
        json={"thread_id": "thread-1", "messages": [], "forwardedProps": {"tenant": "tenant-a"}},
    )
    assert tenant_a_response.status_code == 200
    assert call_count == 1
    tenant_a_events = _decode_sse_events(tenant_a_response)
    assert [event.get("type") for event in tenant_a_events] == [
        "RUN_STARTED",
        "STATE_SNAPSHOT",
        "MESSAGES_SNAPSHOT",
        "RUN_FINISHED",
    ]
    assert tenant_a_events[1]["snapshot"] == {"tenant": "tenant-a"}
    assert any(message.get("content") == "Tenant A reply" for message in tenant_a_events[2]["messages"])


async def test_agent_endpoint_prepends_stored_snapshot_for_new_user_turn(streaming_chat_client_stub):
    """A normal agent turn with a known thread id prepends stored history and keeps the new user input."""
    app = FastAPI()
    captured_messages: list[list[tuple[str, str]]] = []

    async def stream_fn(messages: Any, options: Any, **kwargs: Any):
        del options, kwargs
        captured_messages.append([(message.role, message.text) for message in messages])
        yield ChatResponseUpdate(contents=[Content.from_text(text=f"Reply {len(captured_messages)}")])

    agent = Agent(name="test", instructions="Test agent", client=streaming_chat_client_stub(stream_fn))
    store = InMemoryAGUIThreadSnapshotStore()
    add_agent_framework_fastapi_endpoint(
        app,
        agent,
        path="/snapshots",
        state_schema={"recipe": {"type": "string"}},
        snapshot_store=store,
        snapshot_scope_resolver=lambda _request: "tenant-a",
    )
    client = TestClient(app)

    first_response = client.post(
        "/snapshots",
        json={
            "thread_id": "thread-1",
            "messages": [{"id": "user-1", "role": "user", "content": "Plan dinner"}],
            "state": {"recipe": "pasta"},
        },
    )
    assert first_response.status_code == 200

    second_response = client.post(
        "/snapshots",
        json={
            "thread_id": "thread-1",
            "messages": [{"id": "user-2", "role": "user", "content": "Add dessert"}],
        },
    )

    assert second_response.status_code == 200
    assert len(captured_messages) == 2
    assert captured_messages[1] == [
        ("user", "Plan dinner"),
        ("assistant", "Reply 1"),
        (
            "system",
            (
                "Current state of the application:\n"
                '{\n  "recipe": "pasta"\n}\n\n'
                "When modifying state, you MUST include ALL existing data plus your changes.\n"
                "For example, if adding one new item to a list, include ALL existing items PLUS the new item.\n"
                "Never replace existing data - always preserve and append or merge."
            ),
        ),
        ("user", "Add dessert"),
    ]
    events = _decode_sse_events(second_response)
    state_snapshots = [event for event in events if event.get("type") == "STATE_SNAPSHOT"]
    assert state_snapshots[0]["snapshot"] == {"recipe": "pasta"}


async def test_agent_endpoint_deduplicates_full_history_and_merges_fresh_state(streaming_chat_client_stub):
    """Stored prior history is authoritative while incoming full history and fresh state remain supported."""
    app = FastAPI()
    captured_messages: list[list[tuple[str, str]]] = []

    async def stream_fn(messages: Any, options: Any, **kwargs: Any):
        del options, kwargs
        captured_messages.append([(message.role, message.text) for message in messages])
        yield ChatResponseUpdate(contents=[Content.from_text(text=f"Reply {len(captured_messages)}")])

    agent = Agent(name="test", instructions="Test agent", client=streaming_chat_client_stub(stream_fn))
    store = InMemoryAGUIThreadSnapshotStore()
    add_agent_framework_fastapi_endpoint(
        app,
        agent,
        path="/snapshots",
        state_schema={"recipe": {"type": "string"}, "theme": {"type": "string"}},
        snapshot_store=store,
        snapshot_scope_resolver=lambda _request: "tenant-a",
    )
    client = TestClient(app)

    first_response = client.post(
        "/snapshots",
        json={
            "thread_id": "thread-1",
            "messages": [{"id": "user-1", "role": "user", "content": "Plan dinner"}],
            "state": {"recipe": "pasta", "theme": "dark"},
        },
    )
    assert first_response.status_code == 200
    first_snapshot = _latest_messages_snapshot(first_response)

    second_response = client.post(
        "/snapshots",
        json={
            "thread_id": "thread-1",
            "messages": [*first_snapshot, {"id": "user-2", "role": "user", "content": "Add dessert"}],
            "state": {"recipe": "salad"},
        },
    )
    assert second_response.status_code == 200

    second_non_system_messages = [message for message in captured_messages[1] if message[0] != "system"]
    assert second_non_system_messages == [
        ("user", "Plan dinner"),
        ("assistant", "Reply 1"),
        ("user", "Add dessert"),
    ]
    second_events = _decode_sse_events(second_response)
    second_state_snapshots = [event for event in second_events if event.get("type") == "STATE_SNAPSHOT"]
    assert second_state_snapshots[0]["snapshot"] == {"recipe": "salad", "theme": "dark"}

    second_snapshot = _latest_messages_snapshot(second_response)
    conflicting_history = [message.copy() for message in second_snapshot]
    conflicting_history[0]["content"] = "Tampered dinner plan"
    conflicting_history[1]["content"] = "Tampered reply"
    third_response = client.post(
        "/snapshots",
        json={
            "thread_id": "thread-1",
            "messages": [*conflicting_history, {"id": "user-3", "role": "user", "content": "Pick wine"}],
        },
    )
    assert third_response.status_code == 200

    third_texts = [text for role, text in captured_messages[2] if role != "system"]
    assert third_texts == ["Plan dinner", "Reply 1", "Add dessert", "Reply 2", "Pick wine"]
    assert "Tampered dinner plan" not in third_texts
    assert "Tampered reply" not in third_texts
    third_state_snapshots = [
        event for event in _decode_sse_events(third_response) if event.get("type") == "STATE_SNAPSHOT"
    ]
    assert third_state_snapshots[0]["snapshot"] == {"recipe": "salad", "theme": "dark"}


async def test_endpoint_encoding_failure_emits_run_error():
    """Event encoding failure emits RUN_ERROR event in the SSE stream."""
    from unittest.mock import patch

    class SimpleWorkflow(AgentFrameworkWorkflow):
        async def run(self, input_data: dict[str, Any]):
            del input_data
            yield RunStartedEvent(run_id="run-1", thread_id="thread-1")

    app = FastAPI()
    add_agent_framework_fastapi_endpoint(app, SimpleWorkflow(), path="/encode-fail")
    client = TestClient(app)

    with patch("ag_ui.encoder.EventEncoder.encode") as mock_encode:
        # First call fails (the RUN_STARTED event), second call succeeds (the error event)
        mock_encode.side_effect = [ValueError("encode boom"), 'data: {"type":"RUN_ERROR"}\n\n']
        response = client.post("/encode-fail", json={"messages": [{"role": "user", "content": "go"}]})

    assert response.status_code == 200
    content = response.content.decode("utf-8")
    assert "RUN_ERROR" in content


async def test_endpoint_double_encoding_failure_terminates():
    """When both event and error encoding fail, stream terminates gracefully."""
    from unittest.mock import patch

    class SimpleWorkflow(AgentFrameworkWorkflow):
        async def run(self, input_data: dict[str, Any]):
            del input_data
            yield RunStartedEvent(run_id="run-1", thread_id="thread-1")

    app = FastAPI()
    add_agent_framework_fastapi_endpoint(app, SimpleWorkflow(), path="/double-fail")
    client = TestClient(app)

    with patch("ag_ui.encoder.EventEncoder.encode") as mock_encode:
        # Both calls fail - event encode and error event encode
        mock_encode.side_effect = ValueError("always fails")
        response = client.post("/double-fail", json={"messages": [{"role": "user", "content": "go"}]})

    # Should still get 200 (SSE stream), just with no events
    assert response.status_code == 200
