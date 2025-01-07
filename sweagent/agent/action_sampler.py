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


class ActionSamplerOutput(BaseModel):
    completion: dict[str, Any]
    messages: list[dict[str, Any]] = []
    trajectory_items: list[dict[str, Any]] = []
    extra_info: dict[str, Any] = {}


class AbstractActionSampler(BaseModel):
    def setup(self, model: AbstractModel, tools: ToolHandler):
        self._model = model
        self._tools = tools

    @abstractmethod
    def get_action(
        self,
        problem_statement: ProblemStatement,
        trajectory: Trajectory,
        history: list[dict[str, Any]],
    ) -> ActionSamplerOutput:
        """Returns action with tool calls"""
        pass


class AskColleagues(AbstractActionSampler):
    type: Literal["ask_colleagues"] = "ask_colleagues"

    n_samples: int = 2

    def model_post_init(self, __context: Any) -> None:
        self._logger = get_logger("action_sampler", emoji="👥")

    def get_colleague_discussion(self, completions: list[dict[str, Any]]) -> str:
        """Concat all completions into a single string"""
        out = "Your colleagues had the following ideas: \n\n"
        n_parsed_ok = 0
        for i, completion in enumerate(completions):
            try:
                thought, action = self._tools.parse_actions(completion)
            except FormatError:
                self._logger.warning("Could not parse completion %s, skipping.", completion)
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
    ) -> ActionSamplerOutput:
        """Returns action with tool calls"""
        completions = self._model.query(history, n=self.n_samples)  # type: ignore
        discussion = self.get_colleague_discussion(completions)
        self._logger.info(f"COLLEAGUE DISCUSSION:\n{discussion}")
        new_messages = [
            {"role": "user", "content": discussion},
        ]
        final_completion = self._model.query(history + new_messages)  # type: ignore
        return ActionSamplerOutput(
            completion=final_completion,
            extra_info={"colleagues": discussion},
        )


class BinaryTrajectoryComparison(AbstractActionSampler):
    type: Literal["binary_trajectory_comparison"] = "binary_trajectory_comparison"

    n_samples: int = 2

    comparison_temperature: float | None = None
    """Override the model's temperature. If None, take the temperature configured for the model."""

    system_template: str = """<setting>You are an expert software engineer overseeing junior developers. They suggest actions to take to solve a problem. You must choose the best action to take. </setting>"""
    instance_template: str = dedent("""
    We're solving the following problem

    <problem_statement>
    {{problem_statement}}
    </problem_statement>

    So far, we've performed the following actions:

    <trajectory>
    {{traj}}
    </trajectory>
    """)

    comparison_template: str = dedent("""
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

    def model_post_init(self, __context: Any) -> None:
        self._logger = get_logger("action_sampler", emoji="👥")

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
        counts1: int,
        counts2: int,
        use_cache_control: bool = False,
    ) -> list[dict]:
        system_message = self.system_template
        self._logger.debug(f"MODEL INPUT (system)\n{system_message}")
        ps_format_dict = {
            "problem_statement": problem_statement.get_problem_statement(),
            **problem_statement.get_extra_fields(),
        }
        user_message = Template(self.instance_template).render(
            **ps_format_dict,
            traj=self._format_trajectory(trajectory),
        )
        self._logger.debug(f"MODEL INPUT (instance)\n{user_message}")
        comparison_message = Template(self.comparison_template).render(
            thought1=thought1,
            action1=action1,
            thought2=thought2,
            action2=action2,
            counts1=counts1,
            counts2=counts2,
        )
        self._logger.debug(f"MODEL INPUT (comparison)\n{comparison_message}")
        cache_control_kwargs = {"cache_control": {"type": "ephemeral"}} if use_cache_control else {}
        return [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": [{"type": "text", "text": user_message, **cache_control_kwargs}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": comparison_message,
                    }
                ],
            },
        ]

    def filter_duplicates(self, completions: list[tuple[str, str]]) -> list[tuple[str, str, int]]:
        """Filter out duplicate actions, keeping the longest thought"""
        thoughts: list[str] = []
        actions: list[str] = []
        counts: list[int] = []
        for pc in completions:
            if pc[1] not in actions:
                thoughts.append(pc[0])
                actions.append(pc[1])
                counts.append(1)
            else:
                self._logger.debug(f"Found duplicate action of {pc[1]}")
                found = actions.index(pc[1])
                counts[found] += 1
                if len(thoughts[found]) < len(pc[0]):
                    # New thought is longer, update
                    thoughts[found] = pc[0]
        return list(zip(thoughts, actions, counts))

    def parse_completions(self, completions: list[dict[str, Any]]) -> list[tuple[str, str]]:
        parsed_completions = []
        for completion in completions:
            try:
                thought, action = self._tools.parse_actions(completion)
            except FormatError:
                self._logger.warning("Could not parse completion %s, skipping.", completion)
                continue
            parsed_completions.append((thought, action))
        if len(parsed_completions) == 0:
            msg = "No completions could be parsed."
            raise FormatError(msg)
        return parsed_completions

    def get_action(
        self,
        *,
        problem_statement: ProblemStatement,
        trajectory: Trajectory,
        history: list[dict[str, Any]],
    ) -> ActionSamplerOutput:
        completions = self._model.query(history, n=self.n_samples)  # type: ignore
        parsed_completions = self.parse_completions(completions)
        parsed_completions = self.filter_duplicates(parsed_completions)
        if len(parsed_completions) == 1:
            self._logger.warning("Only identical actions were proposed.")
        best_idx = 0
        comparison_log = []
        for i in range(1, len(parsed_completions)):
            messages = self.format_messages(
                problem_statement=problem_statement,
                trajectory=trajectory,
                thought1=parsed_completions[best_idx][0],
                action1=parsed_completions[best_idx][1],
                thought2=parsed_completions[i][0],
                action2=parsed_completions[i][1],
                counts1=parsed_completions[best_idx][2],
                counts2=parsed_completions[i][2],
                use_cache_control=len(parsed_completions) >= 3,
            )
            response = self._model.query(messages, temperature=self.comparison_temperature)["message"]  # type: ignore
            self._logger.info(f"RESPONSE: {response}")
            idx = self.interpret(response)
            comparison_log.append(
                {
                    "comparison_between": (best_idx, i),
                    "messages": messages,
                    "response": response,
                    "idx": idx,
                }
            )
            best_idx = i if idx == 1 else best_idx

        return ActionSamplerOutput(
            completion=completions[best_idx],
            extra_info={"comparison_log": comparison_log},
        )

    def interpret(self, response: str) -> Literal[0, 1]:
        """Interpret response from LM. Note: 1-based indexing"""
        last_line = response.strip().split("\n")[-1].strip()
        if "first" in last_line.lower():
            return 0
        elif "second" in last_line.lower():
            return 1
        self._logger.warning("Could not interpret response: %s, will choose first submission.", response)
        return 0


ActionSampler = BinaryTrajectoryComparison | AskColleagues
