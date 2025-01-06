from abc import abstractmethod
from textwrap import dedent
from typing import Any, Literal

from jinja2 import Template
from pydantic import BaseModel

from sweagent.agent.models import AbstractModel
from sweagent.agent.problem_statement import ProblemStatement
from sweagent.exceptions import FormatError
from sweagent.tools.tools import ToolHandler
from sweagent.types import Trajectory
from sweagent.utils.log import get_logger

logger = get_logger(__name__)


class GetActionOutput(BaseModel):
    completion: dict[str, Any]
    messages: list[dict[str, Any]] = []
    trajectory_items: list[dict[str, Any]] = []
    extra_info: dict[str, Any] = {}


class AbstractBestActionPicker(BaseModel):
    def setup(self, model: AbstractModel, tools: ToolHandler):
        self._model = model
        self._tools = tools

    @abstractmethod
    def get_action(
        self,
        problem_statement: ProblemStatement,
        trajectory: Trajectory,
        history: list[dict[str, Any]],
        completions: list[dict[str, Any]],
    ) -> GetActionOutput:
        """Returns action with tool calls"""
        pass


class AskColleagues(AbstractBestActionPicker):
    type: Literal["ask_colleagues"] = "ask_colleagues"

    def get_colleague_discussion(self, completions: list[dict[str, Any]]) -> str:
        """Concat all completions into a single string"""
        out = "Your colleagues had the following ideas: \n\n"
        n_parsed_ok = 0
        for i, completion in enumerate(completions):
            try:
                thought, action = self._tools.parse_actions(completion)
            except FormatError:
                logger.warning("Could not parse completion %s, skipping.", completion)
                continue
            n_parsed_ok += 1
            out += f"Thought (colleague {i}): {thought}\nProposed Action (colleague {i}): {action}\n\n"
        if n_parsed_ok == 0:
            msg = "No completions could be parsed."
            raise FormatError(msg)
        out += (
            "Please summarize and compare the ideas and propose and action to take. "
            "Finally choose one action to perform and explain it in detail and include it as a tool call. "
            "<important>You must include a thought and action (as a tool/function call). Do not try to invoke commands with triple backticks, use function calls instead.</important>"
        )
        return out

    def get_action(
        self,
        problem_statement: ProblemStatement,
        trajectory: Trajectory,
        history: list[dict[str, Any]],
        completions: list[dict[str, Any]],
    ) -> GetActionOutput:
        """Returns action with tool calls"""
        discussion = self.get_colleague_discussion(completions)
        logger.info(f"COLLEAGUE DISCUSSION:\n{discussion}")
        new_messages = [
            {"role": "user", "content": discussion},
        ]
        final_completion = self._model.query(history + new_messages)  # type: ignore
        return GetActionOutput(
            completion=final_completion,
            extra_info={"colleagues": discussion},
        )


class BinaryTrajectoryComparison(AbstractBestActionPicker):
    type: Literal["binary_trajectory_comparison"] = "binary_trajectory_comparison"

    system_template: str = """<context>You are an expert software engineer overseeing junior developers. They suggest actions to take to solve a problem. You must choose the best action to take. </context>"""
    instance_template: str = dedent("""
    We're solving the following problem

    <problem_statement>
    {{problem_statement}}
    </problem_statement>

    So far, we've performed the following actions:

    <trajectory>
    {{traj}}
    </trajectory>

    Two junior developers suggested the following actions:

    <thought1>
    {{thought1}}
    </thought1>

    <action1>
    {{action1}}
    </action1>

    <thought2>
    {{thought2}}
    </thought2>

    <action2>
    {{action2}}
    </action2>

    Please compare the two actions in detail.

    Which action should we take?

    If you think the first action is better, respond with "first".
    If you think the second action is better, respond with "second".

    The last line of your response MUST be "first" or "second".
    """)

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
            thought1=thought1,
            thought2=thought2,
        )
        logger.debug(f"MODEL INPUT (user)\n{user_message}")
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    def get_action(
        self,
        *,
        problem_statement: ProblemStatement,
        trajectory: Trajectory,
        history: list[dict[str, Any]],
        completions: list[dict[str, Any]],
    ) -> GetActionOutput:
        parsed_completions = [self._tools.parse_actions(a) for a in completions]
        thoughts: list[str] = [h[0] for h in parsed_completions]
        actions: list[str] = [h[1] for h in parsed_completions]
        assert len(thoughts) == len(actions)
        best_idx = 0
        for i in range(1, len(actions)):
            messages = self.format_messages(
                problem_statement=problem_statement,
                trajectory=trajectory,
                thought1=thoughts[best_idx],
                action1=actions[best_idx],
                thought2=thoughts[i],
                action2=actions[i],
            )
            response = self._model.query(messages)["message"]  # type: ignore
            logger.info(f"RESPONSE: {response}")
            idx = self.interpret(response)
            best_idx = i if idx == 1 else best_idx

        return GetActionOutput(
            completion=completions[best_idx],
        )

    def interpret(self, response: str) -> Literal[0, 1]:
        """Interpret response from LM. Note: 1-based indexing"""
        last_line = response.strip().split("\n")[-1].strip()
        if "first" in last_line.lower():
            return 0
        elif "second" in last_line.lower():
            return 1
        logger.warning("Could not interpret response: %s, will choose first submission.", response)
        return 0


BestResponsePicker = BinaryTrajectoryComparison | AskColleagues
