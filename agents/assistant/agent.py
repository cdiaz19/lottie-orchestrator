"""AssistantMesh — supervisor routing between research, critic and publisher workers."""

from __future__ import annotations

from pathlib import Path

from agents.critic.agent import CriticAgent
from agents.critic.schema import CriticInput
from agents.publisher.agent import PublisherAgent
from agents.publisher.schema import PublisherInput
from agents.research.agent import ResearchAgent
from agents.research.schema import ResearchInput
from lottie.llm import LLMProvider
from lottie.mesh import MeshAgent, MeshEngine, MeshNode, MeshState, StepResult
from lottie.mesh.errors import MeshError
from lottie.project.config import AgentConfig

_DESCRIPTIONS = {
    "research": "Retrieves and synthesizes knowledge to answer the task.",
    "critic": "Reviews the latest draft and suggests one concrete improvement.",
    "publisher": "Publishes/finalizes the answer for release.",
}


class AssistantMesh(MeshAgent):
    """Reference mesh: research → critic → publisher, supervised by the injected LLM."""

    @classmethod
    def from_project(
        cls,
        *,
        llm: LLMProvider,
        root: Path,
        config: AgentConfig,
        enable_benchmarks: bool | None = None,
    ) -> AssistantMesh:
        declared = set(config.workers)
        if declared and declared != set(_DESCRIPTIONS):
            raise ValueError(
                f"assistant config.yaml workers {sorted(declared)} "
                f"do not match the mesh's worker adapters {sorted(_DESCRIPTIONS)}"
            )
        if not set(config.interrupt_before) <= set(_DESCRIPTIONS):
            raise ValueError(
                f"assistant config.yaml interrupt_before {sorted(config.interrupt_before)} "
                f"contains workers not in the mesh's adapters {sorted(_DESCRIPTIONS)}"
            )
        research = ResearchAgent.from_project(
            llm=llm, root=root, config=config, enable_benchmarks=enable_benchmarks
        )
        critic = CriticAgent(llm, enable_benchmarks=enable_benchmarks)
        publisher = PublisherAgent(llm, enable_benchmarks=enable_benchmarks)

        # interrupt_before workers require the LangGraph engine (checkpointed pause/
        # resume). Keep the langgraph import local so `import agents.assistant.agent`
        # works without the [mesh] extra installed.
        engine: MeshEngine | None = None
        if config.interrupt_before:
            try:
                from lottie.mesh.langgraph_engine import LangGraphEngine
            except ImportError as exc:
                raise MeshError(
                    "assistant interrupt_before requires the [mesh] extra: "
                    "pip install lottie-orchestrator[mesh]"
                ) from exc
            engine = LangGraphEngine(interrupt_before=config.interrupt_before)

        mesh = cls(
            llm,
            nodes={},
            descriptions=_DESCRIPTIONS,
            engine=engine,
            enable_benchmarks=enable_benchmarks,
        )
        mesh._nodes = {
            "research": mesh._research_node(research),
            "critic": mesh._critic_node(critic),
            "publisher": mesh._publisher_node(publisher),
        }
        return mesh

    def _research_node(self, research: ResearchAgent) -> MeshNode:
        def _run(state: MeshState) -> MeshState:
            out = research.run(ResearchInput(query=state.task))
            self._accumulate(research.last_metrics)
            return state.with_step(StepResult(worker="research", result=out.digest))

        return _run

    def _critic_node(self, critic: CriticAgent) -> MeshNode:
        def _run(state: MeshState) -> MeshState:
            draft = state.history[-1].result if state.history else state.task
            out = critic.run(CriticInput(text=draft))
            self._accumulate(critic.last_metrics)
            return state.with_step(StepResult(worker="critic", result=out.review))

        return _run

    def _publisher_node(self, publisher: PublisherAgent) -> MeshNode:
        def _run(state: MeshState) -> MeshState:
            draft = state.history[-1].result if state.history else state.task
            out = publisher.run(PublisherInput(text=draft))
            self._accumulate(publisher.last_metrics)
            return state.with_step(StepResult(worker="publisher", result=out.published))

        return _run
