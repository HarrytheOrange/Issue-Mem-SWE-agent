"""The reviewer implements a retry loop for the agent to retry
solving the issue and to select the best solution.
"""

from __future__ import annotations

import copy
import re
from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np
from jinja2 import Template
from pydantic import BaseModel, ConfigDict

from sweagent.agent.history_processors import _set_cache_control
from sweagent.agent.models import (
    AbstractModel,
    InstanceStats,
    ModelConfig,
    get_model,
)
from sweagent.agent.problem_statement import ProblemStatement
from sweagent.tools.tools import ToolConfig
from sweagent.types import AgentInfo, Trajectory, TrajectoryStep
from sweagent.utils.log import get_logger


class ReviewSubmission(BaseModel):
    """Information that's passed to the reviewer"""

    #: Total trajectory (including several retries)
    trajectory: Trajectory
    #: Aggregate info dict (including several retries)
    info: AgentInfo
    #: Model stats for this attempt
    model_stats: InstanceStats

    def to_format_dict(self, *, suffix="") -> dict[str, Any]:
        """Return all the data that is used to format the
        messages. Trajectory is excluded because it needs special treatment.
        """
        out = {}
        info = copy.deepcopy(self.info)
        if not info.get("submission"):
            # Observed that not all exit_cost lead to autosubmission
            # so sometimes this might be missing.
            info["submission"] = ""
        for k, v in info.items():
            if isinstance(v, str):
                out[f"{k}{suffix}"] = v
            elif isinstance(v, dict):
                for k2, v2 in v.items():
                    out[f"{k}_{k2}{suffix}"] = v2
        return out


class ReviewerResult(BaseModel):
    accept: bool | float
    outputs: list[str]
    messages: list[dict[str, Any]]


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

    system_template: str
    instance_template: str
    #: If a submission autosubmits because of total cost or a similar exit status,
    #: it will get this malus to its score
    failure_score_penalty: float = 0.0
    traj_formatter: TrajFormatterConfig
    n_sample: int = 5
    reduce_by_std: float = 0.0
    score_range: tuple[float | None, float | None] = (None, None)
    #: If set, we assume that the score is in the range [score_range[0], score_range[1]]
    #: Reviews that are outside this range will be ignored

    type: Literal["reviewer"] = "reviewer"

    model_config = ConfigDict(extra="forbid")

    def get_reviewer(self, model: AbstractModel) -> AbstractReviewer:
        return Reviewer(self, model)


class ScoreRetryLoopConfig(BaseModel):
    """The configuration for the review loop"""

    type: Literal["score"] = "score"

    reviewer_config: ReviewerConfig

    accept_score: float
    max_accepts: int = 1
    max_attempts: int

    #: Minimal $ that need to be left in order for us to start a new attempt
    min_budget_for_new_attempt: float = 0.0
    #: Override model temperature for first len(list) attempts

    cost_limit: float

    model: ModelConfig

    model_config = ConfigDict(extra="forbid")

    def validate(self):
        """Checks config. Raises `ValueError` in case of misconfiguration"""
        ...

    def __post_init__(self):
        self.validate()

    def get_retry_loop(self, problem_statement: ProblemStatement) -> ScoreRetryLoop:
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
        # Find all numbers in the last line and take the last one
        numbers = re.findall(r"\d+\.?\d*", last_line)
        if not numbers:
            msg = f"Could not interpret response: {last_line!r}"
            raise ValueError(msg)
        number = float(numbers[-1])
        if self._config.score_range[0] is not None and number < self._config.score_range[0]:
            msg = f"Score {number} is below the minimum score {self._config.score_range[0]}"
            raise ValueError(msg)
        if self._config.score_range[1] is not None and number > self._config.score_range[1]:
            msg = f"Score {number} is above the maximum score {self._config.score_range[1]}"
            raise ValueError(msg)
        return number

    def review(self, instance: ProblemStatement, submission: ReviewSubmission) -> ReviewerResult:
        exit_status = submission.info.get("exit_status")
        messages = []
        penalty = 0.0
        if not exit_status or exit_status.strip() != "submitted":
            penalty = self._config.failure_score_penalty
        messages = self.format_messages(instance, submission)
        if self._config.n_sample > 1:
            _set_cache_control(messages[-1])  # type: ignore
        answers = []
        accepts = []
        for _ in range(self._config.n_sample):
            try:
                answer = self._model.query(messages)["message"]
            except Exception as e:
                self.logger.warning(f"Query failed: {e}", exc_info=True)
                continue
            try:
                score = self.interpret(answer)
            except ValueError as e:
                self.logger.warning(f"Could not interpret response: {answer!r}, got {e}")
                continue
            answers.append(answer)
            accepts.append(score)
        if not accepts:
            answers = ["No valid scores found, failing submission"]
            accepts = [-100.0]
        accept = sum(accepts) / len(accepts) - penalty
        std = np.std(accepts).item()
        if self._config.reduce_by_std > 0:
            accept -= std * self._config.reduce_by_std
        self.logger.info(f"First answer: {answers[0]}")
        self.logger.info(f"Final score: {accept} (penalty: {penalty}, std: {std}), individual: {accepts}")
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
        config: ScoreRetryLoopConfig,
        problem_statement: ProblemStatement,
    ):
        # This model will not share instance cost with the parent agent
        self._model = get_model(config.model, tools=ToolConfig())
        self._problem_statement = problem_statement
        self._reviewer: AbstractReviewer = config.reviewer_config.get_reviewer(self._model)
        self._config = config
        # Note: These are "cumulative" submissions, i.e., they include all retries
        # up to that point.
        self._submissions: list[ReviewSubmission] = []
        self._reviews: list[ReviewerResult] = []
        #: Number of consecutive exit cost submissions
        self._n_consec_exit_cost: int = 0
        self.logger = get_logger("review_loop", emoji="🔄")

    # Properties
    # ----------

    @property
    def reviews(self) -> list[ReviewerResult]:
        return self._reviews

    @property
    def _n_attempts(self) -> int:
        return len(self._submissions)

    @property
    def _n_accepted(self) -> int:
        return sum(r.accept >= self._config.accept_score for r in self._reviews)

    @property
    def model_stats(self) -> InstanceStats:
        return self._model.stats

    @property
    def _total_attempt_stats(self) -> InstanceStats:
        return sum((s.model_stats for s in self._submissions), start=InstanceStats())

    # -------

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
        stat_str = f"n_samples={self._n_attempts}, max_score={max_score}, n_accepted={self._n_accepted}"

        if self._total_attempt_stats.instance_cost > self._config.cost_limit > 0:
            self.logger.info(
                f"Exiting retry loop ({stat_str}): Total attempt cost ({self._total_attempt_stats.instance_cost}) "
                f"exceeds cost limit ({self._config.cost_limit})"
            )
            return False

        if self._n_attempts >= self._config.max_attempts > 0:
            self.logger.info(f"Exiting retry loop ({stat_str}): max_attempts={self._config.max_attempts} reached")
            return False

        if self._n_accepted >= self._config.max_accepts > 0:
            self.logger.info(f"Exiting retry loop ({stat_str}): max_accepts={self._config.max_accepts} reached")
            return False

        remaining_budget = self._config.cost_limit - self._total_attempt_stats.instance_cost
        if self._config.min_budget_for_new_attempt > 0 and remaining_budget < self._config.min_budget_for_new_attempt:
            msg = (
                f"Exiting retry loop ({stat_str}): Not enough budget left for a new attempt "
                f"({remaining_budget} remaining, {self._config.min_budget_for_new_attempt} required)"
            )
            self.logger.info(msg)
            return False

        return True

    def get_best(self) -> int | None:
        if len(self._reviews) == 0:
            return None
        scores = [r.accept for r in self._reviews]
        self.logger.debug(f"Scores: {scores}")
        chosen_idx = max(range(len(scores)), key=scores.__getitem__)
        self.logger.info(f"Best submission: {chosen_idx}")
        return chosen_idx

    # def get_best(self) -> int | None:
    #     if len(self._reviews) == 0:
    #         return None
    #     scores = [r.accept for r in self._reviews]
    #     # IMPORTANT: Do not take s.info.model_stats.api_calls, because this is the cumulative cost over all attempts
    #     steps = [s.model_stats.api_calls for s in self._submissions]
    #     self.logger.debug(f"Scores: {scores}, n_steps: {steps}")
    #     good_submissions = [i for i, s in enumerate(scores) if s >= self._config.accept_score]
    #     self.logger.debug(f"Good submissions: {good_submissions} with scores: {[scores[i] for i in good_submissions]}")
    #     if not good_submissions:
    #         good_submissions = list(range(len(self._reviews)))
    #         self.logger.debug("No good submissions found.")
    #     good_submissions = sorted(good_submissions, key=lambda i: scores[i])[: self._config.keep_top_scores]
    #     if self._config.choice_variable == "n_steps":
    #         # Lowest number of steps
    #         chosen_idx = min(good_submissions, key=lambda i: steps[i])
    #     elif self._config.choice_variable == "score":
    #         # Highest score
    #         chosen_idx = max(good_submissions, key=lambda i: scores[i])
    #     self.logger.info(f"Best submission: {chosen_idx}")
    #     return chosen_idx


def get_retry_loop_from_config(config: RetryLoopConfig, problem_statement: ProblemStatement) -> ScoreRetryLoop:
    return config.get_retry_loop(problem_statement=problem_statement)
