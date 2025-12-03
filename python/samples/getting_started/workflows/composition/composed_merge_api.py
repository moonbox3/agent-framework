# Copyright (c) Microsoft. All rights reserved.

import asyncio
from dataclasses import dataclass

from agent_framework import (
    Executor,
    WorkflowBuilder,
    WorkflowContext,
    WorkflowOutputEvent,
    handler,
)

"""Demonstrates the merge() API for low-level workflow composition.

This sample shows the difference between add_workflow() and merge():
- add_workflow(): Returns a ConnectionHandle for wiring via .start/.outputs
- merge(): Merges graphs directly, caller uses connect() with known IDs

Use merge() when you:
1. Know the internal executor IDs of the fragment
2. Want direct control over edge creation
3. Are building composition utilities

Use add_workflow() when you:
1. Want encapsulated composition via handles
2. Don't need to know internal IDs
3. Prefer the simpler API
"""


@dataclass
class Document:
    id: str
    content: str


@dataclass
class ProcessedDocument:
    id: str
    content: str
    word_count: int


@dataclass
class EnrichedDocument:
    id: str
    content: str
    word_count: int
    summary: str


class Ingest(Executor):
    @handler
    async def ingest(self, text: str, ctx: WorkflowContext[Document]) -> None:
        await ctx.send_message(Document(id="doc-1", content=text))


class Process(Executor):
    @handler
    async def process(self, doc: Document, ctx: WorkflowContext[ProcessedDocument]) -> None:
        await ctx.send_message(
            ProcessedDocument(
                id=doc.id,
                content=doc.content,
                word_count=len(doc.content.split()),
            )
        )


class Enrich(Executor):
    @handler
    async def enrich(self, doc: ProcessedDocument, ctx: WorkflowContext[EnrichedDocument]) -> None:
        summary = doc.content[:50] + "..." if len(doc.content) > 50 else doc.content
        await ctx.send_message(
            EnrichedDocument(
                id=doc.id,
                content=doc.content,
                word_count=doc.word_count,
                summary=summary,
            )
        )


class Publish(Executor):
    @handler
    async def publish(
        self,
        doc: EnrichedDocument,
        ctx: WorkflowContext[EnrichedDocument, EnrichedDocument],
    ) -> None:
        print(f"  Publishing: {doc.id} ({doc.word_count} words)")
        await ctx.yield_output(doc)


async def main() -> None:
    """Demonstrate merge() vs add_workflow() composition."""

    # ==========================================================================
    # Fragment definitions (reusable workflow pieces)
    # ==========================================================================
    ingest_fragment = WorkflowBuilder(name="ingest").set_start_executor(Ingest(id="ingest"))

    process_fragment = (
        WorkflowBuilder(name="process")
        .add_edge(Process(id="process"), Enrich(id="enrich"))
        .set_start_executor("process")
    )

    publish_fragment = WorkflowBuilder(name="publish").set_start_executor(Publish(id="publish"))

    # ==========================================================================
    # Option A: Using add_workflow() with ConnectionHandles
    # ==========================================================================
    print("Option A: add_workflow() with ConnectionHandles")
    print("-" * 50)

    builder_a = WorkflowBuilder()

    # add_workflow returns handles - we use .start and .outputs[0]
    ingest_handle = builder_a.add_workflow(ingest_fragment, prefix="a")
    process_handle = builder_a.add_workflow(process_fragment, prefix="b")
    publish_handle = builder_a.add_workflow(publish_fragment, prefix="c")

    # Wire using handles (encapsulated - no need to know internal IDs)
    builder_a.connect(ingest_handle.outputs[0], process_handle.start)
    builder_a.connect(process_handle.outputs[0], publish_handle.start)
    builder_a.set_start_executor(ingest_handle.start)

    print("  Executor IDs (encapsulated via handles):")
    print(f"    Ingest start: {ingest_handle.start}")
    print(f"    Process start: {process_handle.start}")
    print(f"    Publish start: {publish_handle.start}")

    workflow_a = builder_a.build()
    async for event in workflow_a.run_stream("Sample document content for add_workflow demo"):
        if isinstance(event, WorkflowOutputEvent):
            print(f"  Output: {event.data}")
    print()

    # ==========================================================================
    # Option B: Using merge() with direct ID access
    # ==========================================================================
    print("Option B: merge() with direct ID access")
    print("-" * 50)

    builder_b = WorkflowBuilder()

    # merge() returns MergeResult - maps original IDs to prefixed IDs
    ingest_ids = builder_b.merge(ingest_fragment, prefix="in")
    proc_ids = builder_b.merge(process_fragment, prefix="proc")
    pub_ids = builder_b.merge(publish_fragment, prefix="out")

    # Print available executor IDs after merge
    print("  Available executor IDs after merge:")
    for eid in sorted(builder_b.executor_ids):
        print(f"    - {eid}")

    # Wire using MergeResult - no need to know the "/" delimiter!
    # Attribute access: ids.executor_name -> "prefix/executor_name"
    builder_b.add_edge(ingest_ids.ingest, proc_ids.process)
    builder_b.add_edge(proc_ids.enrich, pub_ids.publish)
    builder_b.set_start_executor(ingest_ids.ingest)

    workflow_b = builder_b.build()
    async for event in workflow_b.run_stream("Sample document content for merge demo"):
        if isinstance(event, WorkflowOutputEvent):
            print(f"  Output: {event.data}")
    print()

    # ==========================================================================
    # Option C: Using merge() with get_executor() for type safety
    # ==========================================================================
    print("Option C: merge() with get_executor() for type safety")
    print("-" * 50)

    builder_c = WorkflowBuilder()

    # Merge fragments - MergeResult also supports dictionary access
    i = builder_c.merge(ingest_fragment, prefix="i")
    p = builder_c.merge(process_fragment, prefix="p")
    o = builder_c.merge(publish_fragment, prefix="o")

    # Use get_executor() with MergeResult IDs for full Executor access
    # This provides type information and better error messages
    ingest_exec = builder_c.get_executor(i["ingest"])  # dict access works too
    process_exec = builder_c.get_executor(p.process)
    enrich_exec = builder_c.get_executor(p.enrich)
    publish_exec = builder_c.get_executor(o.publish)

    print("  Retrieved executors:")
    print(f"    ingest: {ingest_exec.id} (input types: {ingest_exec.input_types})")
    print(f"    process: {process_exec.id} (input types: {process_exec.input_types})")
    print(f"    enrich: {enrich_exec.id} (output types: {enrich_exec.output_types})")
    print(f"    publish: {publish_exec.id} (input types: {publish_exec.input_types})")

    # Wire using executor objects
    builder_c.add_edge(ingest_exec, process_exec)
    builder_c.add_edge(enrich_exec, publish_exec)
    builder_c.set_start_executor(ingest_exec)

    workflow_c = builder_c.build()
    async for event in workflow_c.run_stream("Sample document content for get_executor demo"):
        if isinstance(event, WorkflowOutputEvent):
            print(f"  Output: {event.data}")


if __name__ == "__main__":
    asyncio.run(main())
