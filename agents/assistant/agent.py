"""AssistantMesh — supervisor routing between the research and critic workers."""

from __future__ import annotations

from pathlib import Path

from agents.critic.agent import CriticAgent
from agents.critic.schema import CriticInput
from agents.research.agent import ResearchAgent
from agents.research.schema import ResearchInput
from lottie.llm import LLMProvider
from lottie.mesh import MeshAgent, MeshNode, MeshState, StepResult
from lottie.project.config import AgentConfig

_DESCRIPTIONS = {
    "research": "Retrieves and synthesizes knowledge to answer the task.",
    "critic": "Reviews the latest draft and suggests one concrete improvement.",
}


class AssistantMesh(MeshAgent):
    """Reference mesh: research → critic, supervised by the injected LLM."""

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
        research = ResearchAgent.from_project(
            llm=llm, root=root, config=config, enable_benchmarks=enable_benchmarks
        )
        critic = CriticAgent(llm, enable_benchmarks=enable_benchmarks)
        mesh = cls(
            llm,
            nodes={},
            descriptions=_DESCRIPTIONS,
            enable_benchmarks=enable_benchmarks,
        )
        mesh._nodes = {
            "research": mesh._research_node(research),
            "critic": mesh._critic_node(critic),
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
