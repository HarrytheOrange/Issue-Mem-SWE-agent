"""The reviewer implements a retry loop for the agent to retry
solving the issue and to select the best solution.
"""

from __future__ import annotations

import copy
import re
from abc import ABC, abstractmethod
from typing import Any, Literal

from jinja2 import Template
from pydantic import BaseModel, ConfigDict

from sweagent.agent.history_processors import _set_cache_control
from sweagent.agent.models import (
    AbstractModel,
    InstanceStats,
    LiteLLMModel,
    ModelConfig,
    get_model,
)
from sweagent.agent.problem_statement import ProblemStatement
from sweagent.tools.tools import ToolConfig
from sweagent.types import ReviewerResult, ReviewSubmission, Trajectory, TrajectoryStep
from sweagent.utils.log import get_logger

# --- INTERFACES ---


class AbstractReviewer(ABC):
    """The reviewer checks a single solution and tries to predict
    if it successfully solves the issue.
    """

    @abstractmethod
    def review(self, instance: ProblemStatement, submission: ReviewSubmission) -> ReviewerResult:
        """Returns True if the submission is believed to be correct"""


class AbstractRetryLoop(ABC):
    """The review loop controls how often the agent tries to solve
    the issue and how it selects the best solution.
    """

    def retry(self) -> bool:
        """Returns True if the agent should retry solving the issue"""
        return False

    def on_submit(self, submission: ReviewSubmission) -> None:
        """Called when the agent submits a solution"""

    def on_model_query(self, attempt_stats: InstanceStats):
        """Called before the model is queried. Can be used to implement
        stop conditions based on attempt cost etc.
        """

    def on_attempt_started(self, i_attempt: int, agent):
        """Called when a new attempt is started"""
        pass

    @abstractmethod
    def get_best(self) -> int:
        """Returns the best solution"""

    def get_forwarded_vars(self) -> dict[str, Any]:
        """Get the variables that should be forwarded to the next iteration.

        Returns:
            A dictionary of variables that should be forwarded to the next iteration.
        """
        return {}


# --- CONFIGS ---


class TrajFormatterConfig(BaseModel):
    #: Filter the following actions from the trajectory
    filter: list[str] = []
    #: Filter outputs from the following actions from the trajectory
    output_filter: list[str] = []
    #: Format of the trajectory item
    item_template: str = "Model: {{response}}\n\nObservation: {{observation}}"
    only_show_last_n_output: int = 0

    model_config = ConfigDict(extra="forbid")


class ReviewerConfig(BaseModel):
    """The configuration for the reviewer"""

    output_type: Literal["bool", "float"] = "float"
    system_template: str
    instance_template: str
    #: If a submission autosubmits because of total cost or a similar exit status,
    #: it will be desk rejected
    reject_exit_status: bool = True
    traj_formatter: TrajFormatterConfig
    n_sample: int = 5

    type: Literal["reviewer"] = "reviewer"

    model_config = ConfigDict(extra="forbid")

    def get_reviewer(self, model: AbstractModel) -> AbstractReviewer:
        return Reviewer(self, model)


class ScoreRetryLoopConfig(BaseModel):
    """The configuration for the review loop"""

    type: Literal["score"] = "score"

    reviewer_config: ReviewerConfig

    max_attempts_for_score: dict[float, int] = {}
    #: Given a maximum score, the maximum number of attempts
    #: to try. This is a very general way of configuring when to stop.

    #: If set > 0 and there are more than this number of consecutive attempts
    #: with an 'exit cost' exit stats, the review loop will quit.
    max_n_consec_exit_cost: int = 0
    #: Minimal $ that need to be left in order for us to start a new attempt
    min_budget_for_new_attempt: float = 0.0
    #: Override model temperature for first len(list) attempts
    temperature_override: list[float] = [0.0]

    model: ModelConfig

    model_config = ConfigDict(extra="forbid")

    def validate(self):
        """Checks config. Raises `ValueError` in case of misconfiguration"""
        ...

    def __post_init__(self):
        self.validate()

    def get_retry_loop(self, problem_statement: ProblemStatement) -> AbstractRetryLoop:
        return ScoreRetryLoop(self, problem_statement)


RetryLoopConfig = ScoreRetryLoopConfig

# --- IMPLEMENTATIONS ---


class Reviewer(AbstractReviewer):
    def __init__(self, config: ReviewerConfig, model):
        self._config = config
        self._model = model
        self._traj_formatter = TrajectoryFormatter(config=config.traj_formatter)
        self.logger = get_logger("reviewer", emoji="🧑‍⚖️")

    def format_messages(self, instance: ProblemStatement, submission: ReviewSubmission):
        system_message = self._config.system_template
        self.logger.debug(f"MODEL INPUT (system)\n{system_message}")
        ps_format_dict = {
            "problem_statement": instance.get_problem_statement(),
            **instance.get_extra_fields(),
        }
        user_message = Template(self._config.instance_template).render(
            **ps_format_dict,
            **submission.to_format_dict(),
            traj=self._traj_formatter.format_trajectory(submission.trajectory),
        )
        self.logger.debug(f"MODEL INPUT (user)\n{user_message}")
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    def interpret(self, response: str) -> bool | float:
        last_line = response.strip().split("\n")[-1].strip()
        if self._config.output_type == "bool":
            if "success" in last_line.lower():
                return True
            elif "fail" in last_line.lower():
                return False
            self.logger.warning("Could not interpret response: %s, will reject submission.", response)
            return False
        elif self._config.output_type == "float":
            # Find all numbers in the last line and take the last one
            numbers = re.findall(r"\d+\.?\d*", last_line)
            if numbers:
                return float(numbers[-1])
            else:
                self.logger.warning(
                    "Could not interpret response: %s, will reject submission.",
                    response,
                )
                return 0.0
        raise ValueError

    def review(self, instance: ProblemStatement, submission: ReviewSubmission) -> ReviewerResult:
        exit_status = submission.info.get("exit_status")
        messages = []
        if not exit_status:
            answers = ["No exit status in submission, will reject."]
            accept = False
        elif self._config.reject_exit_status and exit_status.strip() != "submitted":
            answers = [f"Submission desk-rejected because of exit status {exit_status!r}."]
            accept = False
        else:
            messages = self.format_messages(instance, submission)
            if self._config.n_sample > 1:
                _set_cache_control(messages[-1])  # type: ignore
            answers = []
            accepts = []
            for _ in range(self._config.n_sample):
                answers.append(self._model.query(messages, temperature=0.0)["message"])
                accepts.append(self.interpret(answers[-1]))
            accept = sum(accepts) / len(accepts)
        self.logger.info(f"First answer: {answers[0]}")
        self.logger.info(f"Final score: {accept}")
        return ReviewerResult(accept=accept, outputs=answers, messages=messages)


# todo: Couldn't I just replace the whole thing with Jinja templates?


class TrajectoryFormatter:
    def __init__(
        self,
        config: TrajFormatterConfig,
    ):
        """Formats trajectories for the use in prompts"""
        self._config = config

    def _include_step(self, item: TrajectoryStep) -> bool:
        action = item["action"].strip()
        for f in self._config.filter:
            if action.startswith(f):
                return False
        return True

    def _include_step_output(self, item: TrajectoryStep, i_step: int, n_steps: int) -> bool:
        if self._config.only_show_last_n_output > 0 and i_step < n_steps - self._config.only_show_last_n_output:
            return False
        action = item["action"].strip()
        for f in self._config.output_filter:
            if action.startswith(f):
                return False
        return True

    def _format_trajectory_step(self, step: TrajectoryStep, i_step: int, *, n_steps: int, i_traj: int = 1) -> str:
        step = copy.deepcopy(step)
        if not self._include_step_output(step, i_step, n_steps=n_steps):
            step["observation"] = "[Output omitted]"
        return Template(self._config.item_template).render(
            **step,
            i_step=i_step,
            i_traj=i_traj,
        )

    def format_trajectory(self, trajectory: Trajectory, i_traj: int = 1) -> str:
        traj_messages = [step for step in trajectory if self._include_step(step)]
        return "\n\n".join(
            [
                self._format_trajectory_step(step, i_step, i_traj=i_traj, n_steps=len(traj_messages))
                for i_step, step in enumerate(traj_messages)
            ]
        )


class ScoreRetryLoop(AbstractRetryLoop):
    def __init__(
        self,
        loop_config: ScoreRetryLoopConfig,
        problem_statement: ProblemStatement,
    ):
        self._model = get_model(loop_config.model, tools=ToolConfig())
        self._problem_statement = problem_statement
        self._reviewer: AbstractReviewer = loop_config.reviewer_config.get_reviewer(self._model)
        self._loop_config = loop_config
        # Note: These are "cumulative" submissions, i.e., they include all retries
        # up to that point.
        self._submissions: list[ReviewSubmission] = []
        self._reviews: list[ReviewerResult] = []
        #: Number of consecutive exit cost submissions
        self._n_consec_exit_cost: int = 0
        #: Original temperature
        self._terminal_temperature: float | None = None
        self.logger = get_logger("review_loop", emoji="🔄")

    # Properties
    # ----------

    @property
    def reviews(self) -> list[ReviewerResult]:
        return self._reviews

    @property
    def _n_attempts(self) -> int:
        return len(self._submissions)

    # -------

    def _override_temperature(self, i_attempt: int) -> None:
        if not isinstance(self._model, LiteLLMModel):
            return
        # Attempts are 1-indexed
        self.logger.debug(f"Setting temperature for attempt {i_attempt}")
        if i_attempt == 0:
            self._terminal_temperature = self._model.config.temperature
            self.logger.debug(f"Set terminal temperature to {self._terminal_temperature}")
        if i_attempt < len(self._loop_config.temperature_override):
            self._model.config.temperature = self._loop_config.temperature_override[i_attempt]
        else:
            assert self._terminal_temperature is not None
            self._model.config.temperature = self._terminal_temperature
        self.logger.debug(f"Set temperature to {self._model.config.temperature}")

    def on_attempt_started(self, i_attempt: int, agent: str) -> None:
        self._override_temperature(i_attempt)

    def on_submit(self, submission: ReviewSubmission) -> None:
        self._submissions.append(submission)
        self._review()

    def _review(self) -> float:
        review = self._reviewer.review(self._problem_statement, self._submissions[-1])
        self._reviews.append(review)
        exit_status = self._submissions[-1].info.get("exit_status", "")
        if exit_status and "exit_cost" in exit_status.lower():
            self._n_consec_exit_cost += 1
        else:
            self._n_consec_exit_cost = 0
        return review.accept

    def retry(self) -> bool:
        max_score = max([r.accept for r in self._reviews])
        stat_str = f"n_samples={self._n_attempts}, max_score={max_score}"

        # Given a maximum score, look up what the minimum and maximum number of attempts are
        for _score, _max_attempts in sorted(
            self._loop_config.max_attempts_for_score.items(), reverse=True, key=lambda x: x[0]
        ):
            if max_score >= _score:
                max_attempts = _max_attempts
                break
        else:
            max_attempts = 0

        if self._n_attempts >= max_attempts > 0:
            self.logger.info(
                f"Exiting retry loop ({stat_str}): `max_attempts`={max_attempts} for highscore {max_score} reached"
            )
            return False

        max_n_exit_cost = self._loop_config.max_n_consec_exit_cost
        if self._n_consec_exit_cost >= max_n_exit_cost > 0:
            self.logger.info(f"Exiting retry loop ({stat_str}): {max_n_exit_cost} exit cost attempts reached")
            return False

        # Todo: Check if there's enough budget left for a new reasonable attempt
        remaining_budget = self._model.instance_cost_limit - self._model.stats.instance_cost
        if (
            self._loop_config.min_budget_for_new_attempt > 0
            and remaining_budget < self._loop_config.min_budget_for_new_attempt
        ):
            self.logger.info(f"Exiting retry loop ({stat_str}): Not enough budget left for a new attempt")
            return False

        return True

    def get_best(self) -> int | None:
        if len(self._reviews) == 0:
            return None
        best_score = max([r.accept for r in self._reviews])
        best_indices = [i for i, r in enumerate(self._reviews) if abs(r.accept - best_score) <= 1e-10]
        return best_indices[0]


def get_retry_loop_from_config(
    config: RetryLoopConfig | None, problem_statement: ProblemStatement
) -> AbstractRetryLoop | None:
    if config is None:
        return None
    return config.get_retry_loop(problem_statement=problem_statement)
