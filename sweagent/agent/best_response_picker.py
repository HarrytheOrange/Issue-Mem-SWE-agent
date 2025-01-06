from abc import abstractmethod
from typing import Literal

from jinja2 import Template
from pydantic import BaseModel

from sweagent.agent.models import AbstractModel
from sweagent.agent.problem_statement import ProblemStatement
from sweagent.types import Trajectory
from sweagent.utils.log import get_logger

logger = get_logger(__name__)


class AbstractBestActionPicker(BaseModel):
    @abstractmethod
    def pick_best(self, problem_statement: ProblemStatement, trajectory: Trajectory, actions: list[str]) -> int:
        pass


class BinaryTrajectoryComparison(AbstractBestActionPicker):
    system_template: str
    instance_template: str

    def __init__(self, model: AbstractModel):
        self.model = model

    def _format_trajectory(self, trajectory: Trajectory) -> str:
        steps = []
        for i, step in enumerate(trajectory):
            steps.append(f"Action {i}: {step['action']}\n Observation {i}: {step['observation']}")
        return "\n".join(steps)

    def format_messages(
        self,
        *,
        problem_statement: ProblemStatement,
        trajectory: Trajectory,
        thought1: str,
        action1: str,
        thought2: str,
        action2: str,
    ) -> list[dict]:
        system_message = self.system_template
        logger.debug(f"MODEL INPUT (system)\n{system_message}")
        ps_format_dict = {
            "problem_statement": problem_statement.get_problem_statement(),
            **problem_statement.get_extra_fields(),
        }
        user_message = Template(self.instance_template).render(
            **ps_format_dict,
            traj=self._format_trajectory(trajectory),
            action1=action1,
            action2=action2,
        )
        logger.debug(f"MODEL INPUT (user)\n{user_message}")
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    def pick_best(
        self, *, problem_statement: ProblemStatement, trajectory: Trajectory, thoughts: list[str], actions: list[str]
    ) -> int:
        assert len(thoughts) == len(actions)
        best_idx = 0
        for i in range(len(actions)):
            messages = self.format_messages(
                problem_statement=problem_statement,
                trajectory=trajectory,
                thought1=thoughts[best_idx],
                action1=actions[best_idx],
                thought2=thoughts[i],
                action2=actions[i],
            )
            response = self.model.query(messages)["message"]  # type: ignore
            idx = self.interpret(response)
            best_idx = i if idx == 1 else best_idx

        return best_idx

    def interpret(self, response: str) -> Literal[0, 1]:
        """Interpret response from LM. Note: 1-based indexing"""
        last_line = response.strip().split("\n")[-1].strip()
        if "first" in last_line.lower():
            return 0
        elif "second" in last_line.lower():
            return 1
        logger.warning("Could not interpret response: %s, will choose first submission.", response)
        return 0


BestResponsePicker = BinaryTrajectoryComparison
