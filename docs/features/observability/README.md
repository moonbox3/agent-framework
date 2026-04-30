# Observability

## Overview

Agent Framework emits comprehensive OpenTelemetry **traces**, **logs**, and **metrics** for every agent run, chat-client call, and tool execution. These signals follow the [OpenTelemetry Semantic Conventions for GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/) and integrate with any OpenTelemetry-compatible backend — Application Insights, Aspire Dashboard, Jaeger, Grafana, etc.

## Enabling telemetry

Call `configure_otel_providers()` once at application startup:

```python
from agent_framework.observability import configure_otel_providers

configure_otel_providers()
```

The function reads standard OpenTelemetry environment variables (`OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_SERVICE_NAME`, etc.) to configure exporters automatically. Pass custom exporters for more control:

```python
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from agent_framework.observability import configure_otel_providers

configure_otel_providers(exporters=[OTLPSpanExporter(endpoint="http://localhost:4317")])
```

For Azure Foundry projects, use `client.configure_azure_monitor()` which retrieves the Application Insights connection string from the project automatically.

> [!NOTE]
> If you configure providers manually (e.g., Azure Monitor SDK), call `enable_instrumentation()` instead of `configure_otel_providers()` to activate the Agent Framework telemetry code paths without replacing your existing exporters.

## MCP trace propagation

Whenever there is an active OpenTelemetry span context, Agent Framework automatically propagates trace context to MCP servers via the `params._meta` field of `tools/call` requests. It uses the globally-configured OpenTelemetry propagator(s) (W3C Trace Context by default, producing `traceparent` and `tracestate`), so custom propagators (B3, Jaeger, etc.) are also supported. This enables distributed tracing across agent-to-MCP-server boundaries, compliant with the [MCP `_meta` specification](https://modelcontextprotocol.io/specification/2025-11-25/basic#_meta).

**Scope:** automatic `_meta` injection applies only to MCP sessions that the agent process itself opens — `MCPStreamableHTTPTool`, `MCPStdioTool`, and `MCPWebsocketTool` (or any other client-opened `MCPTool` subclass). It does **not** apply to hosted/provider-managed MCP tool configurations such as `FoundryChatClient.get_mcp_tool(...)`, `OpenAIChatClient.get_mcp_tool(...)`, `AnthropicClient.get_mcp_tool(...)`, `GeminiChatClient.get_mcp_tool(...)`, or toolbox-fetched tools (for example, `toolbox = await client.get_toolbox(...)`, then passing `toolbox.tools` into `Agent(tools=...)`), because in those cases the `tools/call` message is issued by the provider service runtime rather than by the agent process. As a result, the framework has no opportunity to inject trace context into those requests, and propagating `traceparent`/`tracestate` across that hosted-service boundary is the responsibility of the service runtime, not Agent Framework. If end-to-end distributed tracing to the downstream MCP server is required, use a client-opened MCP transport instead of a hosted connector.

## Samples

See [python/samples/02-agents/observability](../../../python/samples/02-agents/observability/) for runnable examples covering all configuration patterns.
