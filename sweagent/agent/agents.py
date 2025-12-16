from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

import yaml
from jinja2 import Template
from pydantic import BaseModel, ConfigDict, Field, model_validator
from simple_parsing.helpers.fields import field
from swerex.exceptions import BashIncorrectSyntaxError, CommandTimeoutError, SwerexException
from tenacity import RetryError
from typing_extensions import Self
from unidiff import UnidiffParseError

from sweagent import __version__, get_agent_commit_hash, get_rex_commit_hash, get_rex_version
from sweagent.agent.action_sampler import AbstractActionSampler, ActionSamplerConfig
from sweagent.agent.history_processors import DefaultHistoryProcessor, HistoryProcessor
from sweagent.agent.hooks.abstract import AbstractAgentHook, CombinedAgentHook
from sweagent.agent.models import (
    AbstractModel,
    HumanModel,
    HumanThoughtModel,
    InstanceStats,
    ModelConfig,
    get_model,
)
from sweagent.agent.problem_statement import ProblemStatement, ProblemStatementConfig
from sweagent.agent.reviewer import (
    ChooserRetryLoop,
    RetryLoopConfig,
    ReviewSubmission,
    ScoreRetryLoop,
    get_retry_loop_from_config,
)
from sweagent.environment.swe_env import SWEEnv
from sweagent.exceptions import (
    ContentPolicyViolationError,
    ContextWindowExceededError,
    CostLimitExceededError,
    FormatError,
    TotalCostLimitExceededError,
)
from sweagent.tools.parsing import (
    ActionOnlyParser,
    ThoughtActionParser,
)
from sweagent.tools.tools import ToolConfig, ToolHandler
from sweagent.types import AgentInfo, AgentRunResult, StepOutput, Trajectory, TrajectoryStep
from sweagent.utils.config import _convert_paths_to_abspath, _strip_abspath_from_dict
from sweagent.utils.jinja_warnings import _warn_probably_wrong_jinja_syntax
from sweagent.utils.log import get_logger
from sweagent.utils.patch_formatter import PatchFormatter


class IssueSearchRAGContextConfig(BaseModel):
    """Configuration for injecting issue_search_rag results into templates."""

    enabled: bool = False
    field_name: str = "issue_rag_context"
    topk: int = Field(default=1, ge=1, le=10)
    service_url: str | None = None
    service_url_env: str = "ISSUE_SEARCH_RAG_URL"
    default_service_url: str = "http://host.docker.internal:9012/search"
    timeout: float = Field(default=20.0, gt=0)
    max_patch_chars: int = Field(default=4000, ge=256)
    failure_message: str = "No retrieved issue/PR example available."

    def resolve_service_url(self) -> str:
        return self.service_url or os.environ.get(self.service_url_env, self.default_service_url)


class IssueMemoryRAGContextConfig(BaseModel):
    """Configuration for injecting issue_memory_rag results into templates."""

    enabled: bool = False
    field_name: str = "issue_memory_context"
    topk: int = Field(default=1, ge=1, le=10)
    service_url: str | None = None
    service_url_env: str = "ISSUE_MEMORY_RAG_URL"
    default_service_url: str = "http://host.docker.internal:9013/search"
    timeout: float = Field(default=20.0, gt=0)
    max_section_chars: int = Field(default=1200, ge=128)
    failure_message: str = "No retrieved experience memory available."

    def resolve_service_url(self) -> str:
        return self.service_url or os.environ.get(self.service_url_env, self.default_service_url)


class ExperienceSubagentContextConfig(BaseModel):
    """Configuration for running an LLM-driven experience search subagent (exp_search/exp_read)."""

    enabled: bool = False
    """If True, run the subagent at setup time and expose its output as a template variable."""

    field_name: str = "experience_subagent_context"
    """Template variable name to inject the subagent output into."""

    inject_as_message: bool = False
    """If True, also inject the retrieved context as an extra message at the end of setup."""

    message_template: str = (
        "Retrieved relevant historical fix experiences (use as guidance, not as ground truth):\n\n"
        "{{experience_context}}"
    )
    """Template for the injected startup message. Available variables: `experience_context`."""

    # exp_search/exp_read service configuration
    search_url: str | None = None
    search_url_env: str = "GRAPH_EXP_SEARCH_URL"
    default_search_url: str = "http://172.30.182.85:9030/search"

    read_url: str | None = None
    read_url_env: str = "GRAPH_EXP_READ_URL"
    default_read_url: str = "http://172.30.182.85:9030/get_experience"

    timeout: float = Field(default=10.0, gt=0)

    # retrieval / loop controls
    top_k: int = Field(default=10, ge=1, le=50)
    max_rounds: int = Field(default=3, ge=0, le=1000)
    """Maximum number of search rounds. If set to 0, search until the model decides to stop (with safety guards)."""
    read_k_per_round: int = Field(default=2, ge=0, le=10)
    """Maximum number of experiences to read and include in the returned context (cap is applied)."""

    output_mode: Literal["summary", "raw"] = "summary"
    """If 'raw', return concatenated exp_read contents (up to read_k_per_round). If 'summary', return an LLM summary."""

    debug: bool = False
    """If True, log the retrieval loop (queries, decisions, selections) for debugging."""

    # output sizing
    max_chars_per_experience: int = Field(default=2000, ge=256)
    max_total_chars: int = Field(default=8000, ge=1024)

    # LLM controls
    decision_temperature: float | None = Field(default=0.0, ge=0.0, le=2.0)
    summary_temperature: float | None = Field(default=0.2, ge=0.0, le=2.0)

    failure_message: str = "No retrieved experience available."
    """Returned when the subagent believes there are no relevant experiences for the query."""

    error_message: str = "Experience retrieval failed; continue without it."
    """Returned when retrieval fails due to service/model/config errors."""

    def resolve_search_url(self, *, tool_env_vars: dict[str, Any] | None = None) -> str:
        if self.search_url:
            return self.search_url
        if tool_env_vars and self.search_url_env in tool_env_vars:
            return str(tool_env_vars[self.search_url_env])
        return os.environ.get(self.search_url_env, self.default_search_url)

    def resolve_read_url(self, *, tool_env_vars: dict[str, Any] | None = None) -> str:
        if self.read_url:
            return self.read_url
        if tool_env_vars and self.read_url_env in tool_env_vars:
            return str(tool_env_vars[self.read_url_env])
        return os.environ.get(self.read_url_env, self.default_read_url)


class TemplateConfig(BaseModel):
    """This configuration is used to define almost all message templates that are
    formatted by the agent and sent to the LM.
    """

    system_template: str = ""
    instance_template: str = ""
    next_step_template: str = "Observation: {{observation}}"

    next_step_truncated_observation_template: str = (
        "Observation: {{observation[:max_observation_length]}}<response clipped>"
        "<NOTE>Observations should not exceeded {{max_observation_length}} characters. "
        "{{elided_chars}} characters were elided. Please try a different command that produces less output "
        "or use head/tail/grep/redirect the output to a file. Do not use interactive pagers.</NOTE>"
    )
    """Message template for when the agent's observation was truncated.
    Available variables: `observation`, `max_observation_length`, `elided_chars`
    """

    max_observation_length: int = 100_000
    """Truncate observation to this length if it exceeds it.
    This in measured in characters, i.e., as `len(observation)`.
    """

    next_step_no_output_template: str = None  # type: ignore
    """Template for the next step when the last output was empty. Defaults to next_step_template."""

    strategy_template: str | None = None
    demonstration_template: str | None = None

    demonstrations: list[Path] = field(default_factory=list)
    """Paths to demonstrations. If path is not absolute, it is assumed to be
    relative to the SWE_AGENT_CONFIG_ROOT (if set) or the SWE-agent repository root
    """

    issue_search_rag_context: IssueSearchRAGContextConfig | None = None
    issue_memory_rag_context: IssueMemoryRAGContextConfig | None = None
    experience_subagent_context: ExperienceSubagentContextConfig | None = None

    put_demos_in_history: bool = False
    """If True, add demonstration to history instead of as a single message"""

    disable_image_processing: bool = False
    """If True, disable image processing for multimodal problem statements (i.e. SWEBenchMultimodalProblemStatement).
    """

    shell_check_error_template: str = (
        "Your bash command contained syntax errors and was NOT executed. "
        "Please fix the syntax errors and try again. This can be the result "
        "of not adhering to the syntax for multi-line commands. Here is the output of `bash -n`:\n"
        "{{bash_stdout}}\n{{bash_stderr}}"
    )
    """Message template for when the agent's bash command contains syntax errors.
    Available variables: `bash_stdout`, `bash_stderr`
    """

    command_cancelled_timeout_template: str = (
        "The command '{{command}}' was cancelled because it took more than {{timeout}} seconds. "
        "Please try a different command that completes more quickly. "
        "Note: A common source of this error is if the command is interactive or requires user input "
        "(it is impossible to receive user input in the current environment, so the command will never complete)."
    )
    """Message template for when the agent's command was cancelled because it took too long.
    Available variables: `timeout`, `command`
    """

    def model_post_init(self, __context):
        self.demonstrations = _convert_paths_to_abspath(self.demonstrations)
        if self.next_step_no_output_template is None:
            self.next_step_no_output_template = self.next_step_template

    @model_validator(mode="after")
    def validate_template_jinja_syntax(self) -> Self:
        template_fields = [field for field in self.model_fields.keys() if field.endswith("_template")]
        for field in template_fields:
            value = getattr(self, field)
            _warn_probably_wrong_jinja_syntax(value)
        return self

    @model_validator(mode="after")
    def warnings(self) -> Self:
        logger = get_logger("swea-config", emoji="🔧")
        if self.put_demos_in_history and self.demonstration_template is not None:
            logger.warning("demonstration_template is ignored when put_demos_in_history is True")
        if not self.system_template or not self.instance_template:
            logger.warning(
                "system_template/instance_template is not set, using empty string. Perhaps you were"
                " overwriting the default config? See https://swe-agent.com/latest/usage/cl_tutorial/"
                " for more information. Note: You can ignore this warning in human mode."
            )
        return self


class DefaultAgentConfig(BaseModel):
    """This configuration object specifies the behavior of an agent."""

    name: str = "main"
    templates: TemplateConfig = Field(default_factory=TemplateConfig)
    tools: ToolConfig = Field(default_factory=ToolConfig)
    history_processors: list[HistoryProcessor] = Field(default_factory=lambda: [DefaultHistoryProcessor()])
    model: ModelConfig = Field(description="Model options.")

    max_requeries: int = 3
    """Maximum number of times to requery the model after an error, such as a
    formatting error, a blocked action, or a bash syntax error.
    """
    action_sampler: ActionSamplerConfig | None = None

    type: Literal["default"] = "default"

    # pydantic config
    model_config = ConfigDict(extra="forbid")


class ShellAgentConfig(BaseModel):
    name: str = "main"
    templates: TemplateConfig = Field(default_factory=TemplateConfig)
    tools: ToolConfig = Field(default_factory=ToolConfig)
    history_processors: list[HistoryProcessor] = Field(default_factory=lambda: [DefaultHistoryProcessor()])
    model: ModelConfig = Field(description="Model options.")

    max_requeries: int = 3
    """Maximum number of times to requery the model after an error, such as a
    formatting error, a blocked action, or a bash syntax error.
    """

    type: Literal["shell"] = "shell"

    # pydantic config
    model_config = ConfigDict(extra="forbid")


class RetryAgentConfig(BaseModel):
    name: str = "retry_main"
    agent_configs: list[DefaultAgentConfig]
    retry_loop: RetryLoopConfig
    type: Literal["retry"] = "retry"
    model_config = ConfigDict(extra="forbid")


AgentConfig = Annotated[DefaultAgentConfig | RetryAgentConfig | ShellAgentConfig, Field(union_mode="left_to_right")]


class _BlockedActionError(Exception):
    """Raised when the agent's action is blocked"""


class _RetryWithOutput(Exception):
    """Used for internal control flow"""


class _RetryWithoutOutput(Exception):
    """Used for internal control flow"""


class _ExitForfeit(Exception):
    """Used for internal control flow"""


class _TotalExecutionTimeExceeded(Exception):
    """Used for internal control flow"""


RETRY_WITH_OUTPUT_TOKEN = "###SWE-AGENT-RETRY-WITH-OUTPUT###"
RETRY_WITHOUT_OUTPUT_TOKEN = "###SWE-AGENT-RETRY-WITHOUT-OUTPUT###"
EXIT_FORFEIT_TOKEN = "###SWE-AGENT-EXIT-FORFEIT###"


class AbstractAgent:
    def __init__(self, *args, **kwargs):
        model: AbstractModel
        replay_config: BaseModel | None
        logger: logging.Logger

    @classmethod
    def from_config(cls, config: AgentConfig) -> Self: ...

    def add_hook(self, hook: AbstractAgentHook) -> None: ...

    def get_trajectory_data(self) -> dict[str, Any]: ...

    def step(self) -> StepOutput: ...

    def run(self, *args, **kwargs) -> AgentRunResult: ...


def get_agent_from_config(config: AgentConfig) -> AbstractAgent:
    if config.type == "default":
        return DefaultAgent.from_config(config)
    elif config.type == "retry":
        return RetryAgent.from_config(config)
    elif config.type == "shell":
        # Need to defer import to avoid circular dependency
        from sweagent.agent.extra.shell_agent import ShellAgent

        return ShellAgent.from_config(config)
    else:
        msg = f"Unknown agent type: {config.type}"
        raise ValueError(msg)


class RetryAgent(AbstractAgent):
    def __init__(self, config: RetryAgentConfig):
        # Always copy config to avoid shared state between different instances
        self.config = config.model_copy(deep=True)
        self._hooks = []
        self._i_attempt = 0
        self.logger = get_logger("swea-agent", emoji="🤠")
        self._agent: DefaultAgent | None = None
        self._attempt_data: list[dict[str, Any]] = []
        self._total_instance_attempt_stats = InstanceStats()
        """Note that total_instance_attempt_stats only accumulates the states of the sub-agent,
        not the reviewer. Use self._total_instance_stats for the total stats.
        """
        self._chook = CombinedAgentHook()
        self._traj_path: Path | None = None
        self._problem_statement: ProblemStatement | None = None
        self._env: SWEEnv | None = None
        self._output_dir: Path | None = None
        self._rloop: ScoreRetryLoop | ChooserRetryLoop | None = None

    @property
    def _total_instance_stats(self) -> InstanceStats:
        assert self._rloop is not None
        return self._total_instance_attempt_stats + self._rloop.review_model_stats

    @classmethod
    def from_config(cls, config: RetryAgentConfig) -> Self:
        return cls(config)

    def add_hook(self, hook: AbstractAgentHook) -> None:
        self._chook.add_hook(hook)
        self._hooks.append(hook)

    def setup(
        self, env: SWEEnv, problem_statement: ProblemStatement | ProblemStatementConfig, output_dir: Path = Path(".")
    ) -> None:
        """Setup the retry agent for a new problem instance.
        This is mostly a bookkeeping step.
        """
        self._total_instance_attempt_stats = InstanceStats()
        self._problem_statement = problem_statement
        self._traj_path = output_dir / (self._problem_statement.id + ".traj")
        self._env = env
        self._output_dir = output_dir
        self._rloop = get_retry_loop_from_config(self.config.retry_loop, problem_statement=problem_statement)

    def _setup_agent(self) -> AbstractAgent:
        """Setup the agent for the current attempt."""
        # todo: Could select "best" agent config based on previous attempts if I run > number of set up configs
        agent_config = self.config.agent_configs[self._i_attempt % len(self.config.agent_configs)].model_copy(deep=True)
        remaining_budget = self.config.retry_loop.cost_limit - self._total_instance_stats.instance_cost
        if remaining_budget < agent_config.model.per_instance_cost_limit:
            self.logger.debug("Setting agent per-attempt cost limit to remaining budget: %s", remaining_budget)
            agent_config.model.per_instance_cost_limit = remaining_budget
        self._agent = DefaultAgent.from_config(agent_config)
        for hook in self._hooks:
            self._agent.add_hook(hook)
        assert self._output_dir is not None
        sub_agent_output_dir = self._output_dir / f"attempt_{self._i_attempt}"
        assert self._problem_statement is not None
        assert self._env is not None
        self._agent.setup(env=self._env, problem_statement=self._problem_statement, output_dir=sub_agent_output_dir)
        return self._agent

    def _next_attempt(self) -> None:
        """Prepare for the next attempt: Reset the environment and setup the next agent."""
        assert self._env is not None
        self._i_attempt += 1
        self._env.hard_reset()
        self._setup_agent()

    def step(self) -> StepOutput:
        """Step the agent of the current attempt.
        Attempt autosubmit if an error occurs (though all errors should already be handled by the attempt agent).
        """
        assert self._agent is not None
        # Failsafe cost check, this should not actually happen, because the sub-agent should have already been
        # initialized with the correct cost limit to not exceed the total cost limit. Using factor of 1.1, because
        # sub-agent might only catch the cost limit after attempting.
        if self._total_instance_stats.instance_cost > 1.1 * self.config.retry_loop.cost_limit > 0:
            msg = "Total instance cost exceeded cost limit. This should not happen, please report this. Triggering autosubmit."
            self.logger.critical(msg)
            return self._agent.attempt_autosubmission_after_error(step=StepOutput())
        try:
            step = self._agent.step()
        except TotalCostLimitExceededError:
            # Need to make sure that this error causes everything to stop
            raise
        except Exception as e:
            msg = "Error in agent step: %s. This really shouldn't happen, please report this. Triggering autosubmit."
            self.logger.critical(msg, e, exc_info=True)
            step = self._agent.attempt_autosubmission_after_error(step=StepOutput())
        return step

    def _finalize_agent_run(self) -> None:
        """Add the agent results to our list of results"""
        assert self._agent is not None
        self._agent.save_trajectory()
        self._attempt_data.append(self._agent.get_trajectory_data())
        self._total_instance_attempt_stats += self._agent.model.stats

    def get_trajectory_data(self, choose: bool) -> dict[str, Any]:
        """Get all data that we save in .traj files."""
        assert self._rloop is not None

        data = {
            "attempts": self._attempt_data,
        }

        if choose:
            try:
                best_attempt_idx = self._rloop.get_best()
            except TotalCostLimitExceededError:
                raise
            except Exception as e:
                self.logger.critical(f"Error getting best attempt index: {e}. Setting to 0.", exc_info=True)
                best_attempt_idx = 0
            data |= copy.deepcopy(self._attempt_data[best_attempt_idx])  # type: ignore
            data["info"]["best_attempt_idx"] = best_attempt_idx
            data["info"]["rloop_model_stats"] = self._rloop.review_model_stats.model_dump()
            # Overwrite model stats with total stats
            data["info"]["model_stats"] = self._total_instance_stats.model_dump()
            if isinstance(self._rloop, ChooserRetryLoop):
                data["info"]["chooser"] = (
                    self._rloop._chooser_output.model_dump() if self._rloop._chooser_output else {}
                )
        return data

    def save_trajectory(self, choose: bool) -> None:
        data = self.get_trajectory_data(choose=choose)
        assert self._traj_path is not None
        self._traj_path.write_text(json.dumps(data, indent=2))

    def run(
        self,
        env: SWEEnv,
        problem_statement: ProblemStatement | ProblemStatementConfig,
        output_dir: Path = Path("."),
    ) -> AgentRunResult:
        """Run the agent on a problem instance. This method contains the
        main loop that repeatedly calls `self._step` until the problem is solved.

        Args:
            env: The environment to run the agent on.
            problem_statement: The problem statement to run the agent on.
            output_dir: Directory to save the trajectory to
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        self.setup(env=env, problem_statement=problem_statement, output_dir=output_dir)
        assert self._rloop is not None

        # Run action/observation loop
        self._chook.on_run_start()
        step_output = StepOutput()
        self._setup_agent()
        assert self._agent is not None
        while not step_output.done:
            step_output = self.step()
            self.save_trajectory(choose=False)
            if step_output.done:
                self._rloop.on_submit(
                    ReviewSubmission(
                        trajectory=self._agent.trajectory,
                        info=self._agent.info,
                        model_stats=self._agent.model.stats,
                    )
                )
                if isinstance(self._rloop, ScoreRetryLoop):
                    self._agent.info["review"] = self._rloop.reviews[-1].model_dump()  # type: ignore
                self._finalize_agent_run()
                self.save_trajectory(choose=False)
                if self._rloop.retry():
                    assert self._env is not None
                    self._next_attempt()
                    step_output.done = False
        self.save_trajectory(choose=True)  # call again after we finalized
        self._chook.on_run_done(trajectory=self._agent.trajectory, info=self._agent.info)

        self.logger.info("Trajectory saved to %s", self._traj_path)

        # Here we want to return the "global" information (e.g., submission should
        # be the best submission instead of the last one, etc.), so we get it from the traj file
        data = self.get_trajectory_data(choose=True)
        return AgentRunResult(info=data["info"], trajectory=data["trajectory"])


class DefaultAgent(AbstractAgent):
    def __init__(
        self,
        *,
        templates: TemplateConfig,
        tools: ToolHandler,
        history_processors: list[HistoryProcessor],
        model: AbstractModel,
        max_requeries: int = 3,
        name: str = "main",
        _catch_errors: bool = True,
        _always_require_zero_exit_code: bool = False,
        action_sampler_config: ActionSamplerConfig | None = None,
    ):
        """The agent handles the behaviour of the model and how it interacts with the environment.

        To run the agent, either call `self.run` or `self.setup` and then `self.step` in a loop.
        """
        self._catch_errors = _catch_errors
        self._always_require_zero_exit_code = _always_require_zero_exit_code
        self.name = name
        self.model = model
        self.templates = templates
        self.tools = tools
        if isinstance(self.model, HumanThoughtModel):
            self.tools.config.parse_function = ThoughtActionParser()
        elif isinstance(self.model, HumanModel):
            self.tools.config.parse_function = ActionOnlyParser()
        self.history_processors = history_processors
        self.max_requeries = max_requeries
        self.logger = get_logger("swea-agent", emoji="🤠")
        # Set in run method
        self._env: SWEEnv | None = None
        self._problem_statement: ProblemStatement | ProblemStatementConfig | None = None
        self.traj_path: Path | None = None

        #: The following three attributes collect the information about how the agent
        #: solved the problem.
        self.history = []
        self._trajectory = []
        self.info = AgentInfo()
        self._template_extra_fields: dict[str, Any] = {}

        self._chook = CombinedAgentHook()

        self._replay_config: BaseModel | None = None
        """This can be set to a RunSingleConfig from the Run instance whenever possible.
        It can be used to replay the agent's trajectory in an environment.
        """

        self._action_sampler: AbstractActionSampler | None = None
        if action_sampler_config is not None:
            self._action_sampler = action_sampler_config.get(self.model, self.tools)

        #: Count how many timeout errors have occurred consecutively. Kills agent
        #: after 5 of them.
        self._n_consecutive_timeouts = 0
        self._total_execution_time = 0.0

    @classmethod
    def from_config(cls, config: DefaultAgentConfig) -> Self:
        # To ensure that all models stay completely independent, we deepcopy the
        # model config, because it lives on as a property in the model, tools, etc.
        config = config.model_copy(deep=True)
        model = get_model(config.model, config.tools)
        return cls(
            templates=config.templates,
            tools=ToolHandler(config.tools),
            history_processors=config.history_processors,
            model=model,
            max_requeries=config.max_requeries,
            action_sampler_config=config.action_sampler,
        )

    def add_hook(self, hook: AbstractAgentHook) -> None:
        """Add hook to agent"""
        hook.on_init(agent=self)
        self._chook.add_hook(hook)

    # Properties
    # ----------

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    @property
    def replay_config(self) -> BaseModel | None:
        return self._replay_config

    @replay_config.setter
    def replay_config(self, value: BaseModel):
        # Do import here to avoid circular dependency
        from sweagent.run.run_single import RunSingleConfig

        self._replay_config = RunSingleConfig.model_validate(_strip_abspath_from_dict(value.model_dump()))

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Return the history of the agent for this attempt since the last reset,
        processed through all history processors.
        """
        filtered_history = [entry for entry in self.history if entry["agent"] == self.name]  # type: ignore

        # Chain the history processors
        messages = filtered_history
        for processor in self.history_processors:
            messages = processor(messages)

        return messages  # type: ignore

    # Methods
    # -------

    def _append_history(self, item: dict[str, Any]) -> None:
        """Adds an item to the history."""
        self._chook.on_query_message_added(**item)
        self.history.append(item)  # type: ignore

    def setup(
        self,
        env: SWEEnv,
        problem_statement: ProblemStatement | ProblemStatementConfig,
        output_dir: Path = Path("."),
    ) -> None:
        """Setup the agent for a new instance. This includes
        formatting the system message and adding demonstrations to the history.

        This method is called by `self.run`.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # apply template configuration to multimodal problem statements
        if hasattr(problem_statement, "type") and problem_statement.type == "swe_bench_multimodal":
            from sweagent.agent.problem_statement import SWEBenchMultimodalProblemStatement

            if isinstance(problem_statement, SWEBenchMultimodalProblemStatement):
                # apply the global disable_image_processing setting if it's not explicitly set
                if not problem_statement.disable_image_processing and self.templates.disable_image_processing:
                    problem_statement.disable_image_processing = True

        self._problem_statement = problem_statement
        self._env = env
        iid = self._problem_statement.id
        self.logger.info("Setting up agent for instance %s", iid)

        # Save/reset some attributes
        self.traj_path = output_dir / (self._problem_statement.id + ".traj")
        self.logger.info("Trajectory will be saved to %s", self.traj_path)

        self._chook.on_tools_installation_started()
        self.tools.install(self._env)
        self._chook.on_setup_attempt()
        self.info = AgentInfo()
        self.info["swe_agent_hash"] = get_agent_commit_hash()
        self.info["swe_agent_version"] = __version__
        self.info["swe_rex_version"] = get_rex_version()
        self.info["swe_rex_hash"] = get_rex_commit_hash()
        assert self._env is not None
        assert self._problem_statement is not None
        self._env.set_env_variables({"PROBLEM_STATEMENT": self._problem_statement.get_problem_statement_for_env()})
        self._prepare_template_extra_fields()
        self.add_system_message_to_history()
        self.add_demonstrations_to_history()
        self.add_instance_template_to_history(state=self.tools.get_state(self._env))

        exp_config = self.templates.experience_subagent_context
        if exp_config and exp_config.enabled and exp_config.inject_as_message:
            exp_context = self._template_extra_fields.get(exp_config.field_name, "")
            if exp_context and exp_context != exp_config.failure_message:
                injected = Template(exp_config.message_template).render(experience_context=exp_context)
                self._append_history(
                    {
                        "role": "user",
                        "content": injected,
                        "agent": self.name,
                        "message_type": "experience_subagent_context",
                    }
                )
        self._chook.on_setup_done()

    def add_system_message_to_history(self) -> None:
        """Add system message to history"""
        assert self._problem_statement is not None
        system_msg = Template(self.templates.system_template).render(**self._get_format_dict())
        self.logger.info(f"SYSTEM ({self.name})\n{system_msg}")
        self._append_history(
            {"role": "system", "content": system_msg, "agent": self.name, "message_type": "system_prompt"}
        )

    def add_demonstrations_to_history(self) -> None:
        """Add demonstrations to history"""
        for demonstration_path in self.templates.demonstrations:
            self._add_demonstration_to_history(demonstration_path)

    def _add_demonstration_to_history(self, demonstration_path: Path) -> None:
        """Load demonstration from disk and add to history"""
        if self.templates.demonstration_template is None and not self.templates.put_demos_in_history:
            msg = "Cannot use demonstrations without a demonstration template or put_demos_in_history=True"
            raise ValueError(msg)

        # Load history
        self.logger.info(f"DEMONSTRATION: {demonstration_path}")
        _demo_text = Path(demonstration_path).read_text()
        if demonstration_path.suffix == ".yaml":
            demo_history = yaml.safe_load(_demo_text)["history"]
        else:
            demo_history = json.loads(_demo_text)["history"]

        if self.templates.put_demos_in_history:
            # Add demonstrations to history step-by-step
            for entry in demo_history:
                if entry["role"] != "system":
                    entry["is_demo"] = True
                    self._append_history(entry)
        else:
            # Add demonstration as single message to history
            demo_history = [entry for entry in demo_history if entry["role"] != "system"]
            demo_message = "\n".join([entry["content"] for entry in demo_history])
            assert self.templates.demonstration_template is not None
            demonstration = Template(self.templates.demonstration_template).render(demonstration=demo_message)
            self._append_history(
                {
                    "agent": self.name,
                    "content": demonstration,
                    "is_demo": True,
                    "role": "user",
                    "message_type": "demonstration",
                },
            )

    def _prepare_template_extra_fields(self) -> None:
        """Populate extra template variables such as issue_search_rag context."""
        self._template_extra_fields = {}
        memory_config = self.templates.issue_memory_rag_context
        if memory_config and memory_config.enabled:
            memory_context = self._get_issue_memory_rag_context(memory_config)
            self._template_extra_fields[memory_config.field_name] = memory_context

        search_config = self.templates.issue_search_rag_context
        if search_config and search_config.enabled:
            context = self._get_issue_search_rag_context(search_config)
            self._template_extra_fields[search_config.field_name] = context

        exp_config = self.templates.experience_subagent_context
        if exp_config and exp_config.enabled:
            exp_context = self._get_experience_subagent_context(exp_config)
            self._template_extra_fields[exp_config.field_name] = exp_context

    def _exp_post_json(self, url: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any] | None:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except Exception as exc:  # noqa: BLE001 - single fallback path
            self.logger.warning("experience_subagent POST failed (%s): %s", url, exc)
            return None

    def _exp_get_json(self, url: str, *, timeout: float, params: dict[str, str]) -> dict[str, Any] | None:
        query = urllib.parse.urlencode(params)
        full_url = f"{url}?{query}"
        try:
            with urllib.request.urlopen(full_url, timeout=timeout) as resp:
                return json.load(resp)
        except Exception as exc:  # noqa: BLE001 - single fallback path
            self.logger.warning("experience_subagent GET failed (%s): %s", full_url, exc)
            return None

    def _query_model_text(self, messages: list[dict[str, Any]], *, temperature: float | None) -> str | None:
        """Query the model and return assistant text; return None on failure.

        Requeries (a few times) if tool calls are returned or the reply is empty.
        """
        # Many models will not tool-call unless prompted, but SWE-agent may always send tools.
        # We requery with a stronger instruction if tool calls are emitted.
        for attempt in range(3):
            try:
                out = self.model.query(messages, temperature=temperature)  # type: ignore[arg-type]
            except TypeError:
                try:
                    out = self.model.query(messages)  # type: ignore[arg-type]
                except Exception as exc:  # noqa: BLE001 - best-effort subagent
                    self.logger.warning("experience_subagent model query failed: %s", exc)
                    return None
            except Exception as exc:  # noqa: BLE001 - best-effort subagent
                self.logger.warning("experience_subagent model query failed: %s", exc)
                return None
            if isinstance(out, list):
                out = out[0]
            tool_calls = out.get("tool_calls") if isinstance(out, dict) else None
            text = (out.get("message") if isinstance(out, dict) else "") or ""
            if tool_calls:
                # Requery, explicitly forbidding tool calls.
                messages = messages + [
                    {
                        "role": "user",
                        "content": "Do NOT call any tools/functions. Reply with plain text only.",
                    }
                ]
                continue
            if text.strip():
                return text
            messages = messages + [
                {
                    "role": "user",
                    "content": "Your last reply was empty. Reply with plain text only.",
                }
            ]
        return None

    def _safe_json_loads(self, text: str) -> dict[str, Any] | None:
        text = text.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Common failure: model wraps in ```json ... ```
            fence_prefix = "```"
            if fence_prefix in text:
                stripped = text
                stripped = stripped.replace("```json", "```").replace("```JSON", "```")
                parts = stripped.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("{") and part.endswith("}"):
                        try:
                            return json.loads(part)
                        except json.JSONDecodeError:
                            continue
            return None

    def _clip(self, text: str, *, max_chars: int) -> str:
        text = (text or "").strip()
        if max_chars > 0 and len(text) > max_chars:
            return text[:max_chars] + "\n... [truncated] ..."
        return text

    def _parse_tool_call_arguments(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        fn = tool_call.get("function") or {}
        args = fn.get("arguments", {})
        if isinstance(args, dict):
            return args
        if isinstance(args, str):
            try:
                parsed = json.loads(args)
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    def _try_run_experience_subagent_tool(self, step: StepOutput) -> str | None:
        """Handle the host-side `experience_subagent` tool call if present."""
        tool_calls = step.tool_calls or []
        if not tool_calls or len(tool_calls) != 1:
            return None
        tool_call = tool_calls[0]
        fn = tool_call.get("function") or {}
        if fn.get("name") != "experience_subagent":
            return None

        args = self._parse_tool_call_arguments(tool_call)
        query = args.get("query")
        if not isinstance(query, str):
            query = ""

        base_cfg = (
            self.templates.experience_subagent_context.model_copy(deep=True)
            if self.templates.experience_subagent_context is not None
            else ExperienceSubagentContextConfig()
        )

        def _override_int(field_name: str) -> None:
            value = args.get(field_name)
            if isinstance(value, int):
                setattr(base_cfg, field_name, value)

        _override_int("max_rounds")
        _override_int("top_k")
        _override_int("read_k_per_round")

        return self._get_experience_subagent_context(base_cfg, query=query)

    def _get_experience_subagent_context(self, config: ExperienceSubagentContextConfig, *, query: str | None = None) -> str:
        """Run an LLM-driven loop over exp_search/exp_read.

        - In `summary` mode: return an LLM summary of retrieved experiences.
        - In `raw` mode: return concatenated exp_read contents (up to `read_k_per_round`).
        """
        assert self._problem_statement is not None
        problem = self._problem_statement.get_problem_statement()
        search_url = config.resolve_search_url(tool_env_vars=self.tools.config.env_variables)
        read_url = config.resolve_read_url(tool_env_vars=self.tools.config.env_variables)

        read_limit = min(max(0, int(config.read_k_per_round)), 3)  # hard cap for prompt size and determinism
        if read_limit <= 0:
            self.info["experience_subagent"] = {  # type: ignore
                "debug": {"read_limit": read_limit},
                "outcome": "error",
                "exit_reason": "read_limit_zero",
            }
            return config.error_message

        debug_enabled = bool(config.debug) or bool(os.environ.get("SWE_AGENT_EXPERIENCE_SUBAGENT_DEBUG"))
        debug: dict[str, Any] = {
            "rounds": [],
            "search_url": search_url,
            "read_url": read_url,
            "top_k": config.top_k,
            "max_rounds": config.max_rounds,
            "read_limit": min(max(0, int(config.read_k_per_round)), 3),
            "output_mode": config.output_mode,
        }
        current_query = query.strip() if isinstance(query, str) and query.strip() else problem
        seen_queries: set[str] = {current_query}
        max_rounds = config.max_rounds if config.max_rounds > 0 else 1000
        selected_ids_final: list[str] = []
        exit_reason: str = "unknown"
        outcome: Literal["ok", "not_found", "error"] = "error"

        for i_round in range(max_rounds):
            if debug_enabled:
                self.logger.info("[experience_subagent] round=%d query=%s", i_round, current_query)
            search_payload = {"query": current_query, "top_k": config.top_k}
            search_resp = self._exp_post_json(search_url, search_payload, timeout=config.timeout)
            if not search_resp or not search_resp.get("success"):
                exit_reason = "search_failed"
                outcome = "error"
                break

            results = search_resp.get("results") or []
            candidates: list[dict[str, Any]] = []
            for item in results:
                if not isinstance(item, dict):
                    continue
                uid = str(item.get("id") or "").strip()
                if not uid:
                    continue
                preview = item.get("bug_description") or item.get("content_preview") or ""
                if not isinstance(preview, str):
                    preview = str(preview)
                score = item.get("score", item.get("similarity_score", None))
                candidates.append(
                    {
                        "id": uid,
                        "score": score,
                        "preview": self._clip(preview, max_chars=400),
                    }
                )

            if not candidates:
                exit_reason = "no_candidates"
                outcome = "not_found"
                break

            decision_messages: list[dict[str, Any]] = [
                {
                    "role": "system",
                    "content": (
                        "You are a retrieval controller for software bug-fix experiences. "
                        "You can ONLY select candidate IDs from the provided list. "
                        "Decide whether the current candidates are relevant enough to read. "
                        "If yes, set stop=true and select up to max_ids_to_read. "
                        "If not, set stop=false and propose a refined next_query. "
                        "If you believe no relevant experience can be found, set stop=true and return an empty selected_ids. "
                        "Return STRICT JSON only; do not call any tools/functions."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "problem_statement": problem,
                            "current_query": current_query,
                            "previous_queries": sorted(seen_queries),
                            "candidates": candidates,
                            "constraints": {
                                "max_ids_to_read": read_limit,
                                "must_return_json": True,
                            },
                            "output_schema": {
                                "selected_ids": ["string"],
                                "next_query": "string",
                                "stop": "boolean",
                                "rationale": "string",
                            },
                        },
                        ensure_ascii=False,
                    ),
                },
            ]

            decision_text = self._query_model_text(decision_messages, temperature=config.decision_temperature)
            if decision_text is None:
                exit_reason = "model_query_failed"
                outcome = "error"
                break
            decision = self._safe_json_loads(decision_text)
            if decision is None:
                # One correction attempt: ask for strict JSON only.
                correction_messages = decision_messages + [
                    {
                        "role": "user",
                        "content": "Your previous output was not valid JSON. Reply with STRICT JSON only (no prose, no markdown).",
                    }
                ]
                decision_text = self._query_model_text(correction_messages, temperature=config.decision_temperature)
                if decision_text is None:
                    exit_reason = "model_query_failed"
                    outcome = "error"
                    break
                decision = self._safe_json_loads(decision_text)
            decision = decision or {}
            if not decision:
                exit_reason = "decision_invalid_json"
                outcome = "error"
                break

            selected_ids_raw = decision.get("selected_ids") or []
            if isinstance(selected_ids_raw, str):
                selected_ids = [selected_ids_raw]
            elif isinstance(selected_ids_raw, list):
                selected_ids = [str(x) for x in selected_ids_raw]
            else:
                selected_ids = []
            selected_ids = [s.strip() for s in selected_ids if s and s.strip()]
            selected_ids = selected_ids[:read_limit]

            next_query = decision.get("next_query")
            if not isinstance(next_query, str):
                next_query = ""
            stop = bool(decision.get("stop", False))
            rationale = decision.get("rationale")
            if not isinstance(rationale, str):
                rationale = ""

            debug["rounds"].append(
                {
                    "round": i_round,
                    "query": current_query,
                    "selected_ids": selected_ids,
                    "next_query": next_query,
                    "stop": stop,
                    "rationale": rationale,
                    "n_candidates": len(candidates),
                }
            )
            if debug_enabled:
                self.logger.info(
                    "[experience_subagent] decision stop=%s selected_ids=%s next_query=%s",
                    stop,
                    selected_ids,
                    next_query,
                )

            if stop:
                selected_ids_final = selected_ids
                exit_reason = "stop"
                outcome = "ok" if selected_ids_final else "not_found"
                break

            next_query = next_query.strip()
            if not next_query:
                exit_reason = "no_next_query"
                outcome = "not_found"
                break
            if next_query in seen_queries:
                exit_reason = "repeated_query"
                outcome = "not_found"
                break
            seen_queries.add(next_query)
            current_query = next_query

        if not selected_ids_final:
            self.info["experience_subagent"] = {  # type: ignore
                "debug": debug,
                "outcome": outcome,
                "exit_reason": exit_reason,
                "selected_ids_final": selected_ids_final,
            }
            return config.failure_message if outcome == "not_found" else config.error_message

        collected: list[dict[str, str]] = []
        total_chars = 0
        for uid in selected_ids_final:
            read_resp = self._exp_get_json(read_url, timeout=config.timeout, params={"id": uid})
            if not read_resp or not read_resp.get("success"):
                continue
            data = read_resp.get("data") or {}
            if not isinstance(data, dict):
                continue
            fix_exp = data.get("fix_experience") or ""
            if not isinstance(fix_exp, str):
                fix_exp = str(fix_exp)
            repo = data.get("repo") or ""
            if not isinstance(repo, str):
                repo = str(repo)
            clipped = self._clip(fix_exp, max_chars=config.max_chars_per_experience)
            remaining = max(config.max_total_chars - total_chars, 0)
            if remaining <= 0:
                break
            clipped = clipped[:remaining]
            total_chars += len(clipped)
            collected.append({"id": uid, "repo": repo, "fix_experience": clipped})

        if not collected:
            self.info["experience_subagent"] = {  # type: ignore
                "debug": debug,
                "outcome": "error",
                "exit_reason": "read_failed_or_empty",
                "selected_ids_final": selected_ids_final,
            }
            return config.error_message

        # Store debug info for the trajectory.
        self.info["experience_subagent"] = {  # type: ignore
            "debug": debug,
            "n_collected": len(collected),
            "outcome": "ok",
            "exit_reason": exit_reason,
            "selected_ids_final": selected_ids_final,
        }

        if config.output_mode == "raw":
            blocks: list[str] = []
            for idx, item in enumerate(collected, start=1):
                header = f"Experience #{idx} (id={item.get('id','')}, repo={item.get('repo','')})"
                blocks.append(header + "\n" + (item.get("fix_experience") or "").strip())
            return "\n\n---\n\n".join(blocks).strip()

        payload_items = [
            {"id": item.get("id", ""), "repo": item.get("repo", ""), "fix_experience": item.get("fix_experience", "")}
            for item in collected
        ]

        summary_messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are a senior software engineer. Summarize the retrieved historical fix experiences into "
                    "actionable guidance for solving the given problem. Focus on transferable patterns, checks, "
                    "and pitfalls. Keep it concise but concrete."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "problem_statement": problem,
                        "retrieved_experiences": payload_items,
                        "requested_output": {
                            "format": "plain_text",
                            "include": [
                                "Which experiences seem most relevant and why",
                                "Concrete steps / heuristics to apply",
                                "Potential pitfalls / edge cases",
                            ],
                        },
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        summary_text = self._query_model_text(summary_messages, temperature=config.summary_temperature)
        if summary_text is None or not summary_text.strip():
            self.info["experience_subagent"] = {  # type: ignore
                "debug": debug,
                "n_collected": len(collected),
                "outcome": "error",
                "exit_reason": "summary_model_query_failed",
                "selected_ids_final": selected_ids_final,
            }
            return config.error_message
        summary = summary_text.strip()

        return summary

    def _get_issue_memory_rag_context(self, config: IssueMemoryRAGContextConfig) -> str:
        """Fetch structured memory snippets via the local memory RAG service."""
        assert self._problem_statement is not None
        query = self._problem_statement.get_problem_statement()
        request_body = json.dumps({"query": query, "topk": config.topk}).encode("utf-8")
        req = urllib.request.Request(
            config.resolve_service_url(),
            data=request_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=config.timeout) as resp:
                payload = json.load(resp)
        except Exception as exc:  # noqa: BLE001 - want a single fallback path
            self.logger.warning("issue_memory_rag request failed: %s", exc)
            return config.failure_message

        if not payload.get("success"):
            self.logger.warning("issue_memory_rag returned error: %s", payload.get("error"))
            return config.failure_message

        results = payload.get("results") or []
        if not results:
            return config.failure_message

        def _clip(text: str) -> str:
            text = text.strip()
            max_chars = config.max_section_chars
            if max_chars and len(text) > max_chars:
                return text[: max_chars] + "\n... [truncated] ..."
            return text

        summaries: list[str] = []
        for idx, item in enumerate(results[: config.topk], start=1):
            score = item.get("similarity_score")
            try:
                score_str = f"{float(score):.4f}"
            except (TypeError, ValueError):
                score_str = "N/A"
            description_raw = item.get("description") or item.get("document") or ""
            if not isinstance(description_raw, str):
                description_raw = str(description_raw)
            description = description_raw.strip()
            procedural_raw = item.get("procedural_memory") or ""
            if not isinstance(procedural_raw, str):
                procedural_raw = str(procedural_raw)
            procedural = procedural_raw.strip()
            parts = [
                f"Memory #{idx} (similarity {score_str})",
                f"- Source: {item.get('source_file', 'N/A')}",
            ]
            if description:
                parts.append("Summary:\n" + _clip(description))
            if procedural:
                parts.append("Procedural Notes:\n" + _clip(procedural))
            summaries.append("\n".join(parts))

        return "\n\n".join(summaries)

    def _get_issue_search_rag_context(self, config: IssueSearchRAGContextConfig) -> str:
        """Fetch the closest issue/PR pair via the local RAG service."""
        assert self._problem_statement is not None
        query = self._problem_statement.get_problem_statement()
        request_body = json.dumps({"query": query, "topk": config.topk}).encode("utf-8")
        req = urllib.request.Request(
            config.resolve_service_url(),
            data=request_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=config.timeout) as resp:
                payload = json.load(resp)
        except Exception as exc:  # noqa: BLE001 - want a single fallback path
            self.logger.warning("issue_search_rag request failed: %s", exc)
            return config.failure_message

        if not payload.get("success"):
            self.logger.warning("issue_search_rag returned error: %s", payload.get("error"))
            return config.failure_message

        results = payload.get("results") or []
        if not results:
            return config.failure_message

        best = results[0]
        patch = (best.get("patch") or "").strip()
        if config.max_patch_chars and len(patch) > config.max_patch_chars:
            patch = patch[: config.max_patch_chars] + "\n... [truncated] ..."

        score = best.get("similarity_score")
        try:
            score_str = f"{float(score):.4f}"
        except (TypeError, ValueError):
            score_str = "N/A"

        lines = [
            "Closest retrieved issue/PR example:",
            f"- Repo: {best.get('repo', 'N/A')}",
            f"- File: {best.get('file', 'N/A')}",
            f"- PR: {best.get('pr_number', 'N/A')}",
            f"- Similarity: {score_str}",
            "Patch:",
            patch or "(empty patch)",
        ]
        return "\n".join(lines)

    def _get_format_dict(self, **kwargs) -> dict[str, Any]:
        """Get the dictionary of key value pairs used to format the templates

        Args:
            **kwargs: additional keyword arguments to be added to the format dictionary
        """
        assert self._problem_statement is not None
        assert self._env is not None
        return dict(
            command_docs=self.tools.config.command_docs,
            **self.tools.config.env_variables,
            **kwargs,
            problem_statement=self._problem_statement.get_problem_statement(),
            repo=self._env.repo.repo_name if self._env.repo is not None else "",
            **self._problem_statement.get_extra_fields(),
            **self._template_extra_fields,
        )

    def _add_templated_messages_to_history(
        self, templates: list[str], tool_call_ids: list[str] | None = None, **kwargs: str | int | None
    ) -> None:
        """Populate selected template(s) with information (e.g., issue, arguments, state)
        and add to history.

        Args:
            templates: templates to populate and add to history
            tool_call_ids: tool call ids to be added to the history
            **kwargs: keyword arguments to be passed to the templates (in addition to the
                ones in `self._get_format_dict`)
        """
        messages = []

        format_dict = self._get_format_dict(**kwargs)
        for template in templates:
            try:
                messages.append(Template(template).render(**format_dict))
            except KeyError:
                self.logger.debug("The following keys are available: %s", format_dict.keys())
                raise

        message = "\n".join(messages)

        # We disable syntax highlighting here, because some inputs can lead to a complete cross-thread
        # freeze in the agent. See https://github.com/SWE-agent/SWE-agent/issues/901 .
        self.logger.info(f"🤖 MODEL INPUT\n{message}", extra={"highlighter": None})
        history_item: dict[str, Any] = {
            "role": "user",
            "content": message,
            "agent": self.name,
            "message_type": "observation",
        }
        if tool_call_ids:
            assert len(tool_call_ids) == 1, "This should be ensured by the FunctionCalling parse method"
            history_item["role"] = "tool"
            history_item["tool_call_ids"] = tool_call_ids
        self._append_history(history_item)

    def add_step_to_history(self, step: StepOutput) -> None:
        """Adds a step (command that was run and output) to the model history"""
        self._append_history(
            {
                "role": "assistant",
                "content": step.output,
                "thought": step.thought,
                "action": step.action,
                "agent": self.name,
                "tool_calls": step.tool_calls,
                "message_type": "action",
                "thinking_blocks": step.thinking_blocks,
            },
        )

        elided_chars = 0
        if step.observation.strip() == "":
            # Show no output template if observation content was empty
            templates = [self.templates.next_step_no_output_template]
        elif len(step.observation) > self.templates.max_observation_length:
            templates = [self.templates.next_step_truncated_observation_template]
            elided_chars = len(step.observation) - self.templates.max_observation_length
        else:
            # Show standard output template if there is observation content
            templates = [self.templates.next_step_template]
        self._add_templated_messages_to_history(
            templates,
            observation=step.observation,
            elided_chars=elided_chars,
            max_observation_length=self.templates.max_observation_length,
            tool_call_ids=step.tool_call_ids,
            **step.state,
        )

    def add_instance_template_to_history(self, state: dict[str, str]) -> None:
        """Add observation to history, as well as the instance template or demonstrations if we're
        at the start of a new attempt.
        """
        templates: list[str] = []
        # Determine observation template based on what prior observation was
        assert self.history[-1]["role"] == "system" or self.history[-1].get("is_demo", False)
        # Show instance template if prev. obs. was initial system message
        templates = [self.templates.instance_template]
        if self.templates.strategy_template is not None:
            templates.append(self.templates.strategy_template)

        self._add_templated_messages_to_history(templates, **state)  # type: ignore

    def get_trajectory_data(self) -> dict[str, Any]:
        """Get all data that we save in .traj files."""

        assert self._env is not None
        # The deepcopy here is important because else the
        # data["info"]["model_stats"] update will create havoc!
        attempt_data = copy.deepcopy(
            {
                "trajectory": self.trajectory,
                "history": self.history,
                "info": self.info,
            }
        )
        attempt_data["replay_config"] = self.replay_config.model_dump_json() if self.replay_config is not None else None
        attempt_data["environment"] = self._env.name
        return attempt_data

    def save_trajectory(
        self,
    ) -> None:
        """Save the trajectory to disk.
        This includes the history, the environment state, and the model stats.
        """
        data = self.get_trajectory_data()
        assert self.traj_path is not None
        self.traj_path.write_text(json.dumps(data, indent=2))

    def get_model_requery_history(
        self, error_template: str, *, output: str, **kwargs: str | int | float | bool | None
    ) -> list[dict[str, str]]:
        """Ask the model to correct after a hitting one of the following errors:

        1. Malformatted output (could not parse action)
        2. Blocked action (command is on the blocklist)
        3. Bash command syntax error

        At the time this function is called, the proposed action and observation are not part of the history
        yet.

        This function adds temporary history based on the error template and queries the model.
        If the model is able to correct itself, the records of the mistakes will not be part of the history
        (but they are saved in the trajectory).

        Args:
            error_template: error template
            output: model output
            **kwargs: keyword arguments to be passed to the error template

        Returns:
            model output after requery
        """
        format_dict = {**kwargs, **self._get_format_dict()}
        error_template = Template(error_template).render(**format_dict)

        self.logger.warning(f"{error_template}")

        return self.messages + [
            {"role": "assistant", "content": output, "agent": self.name, "message_type": "assistant"},
            {"role": "user", "content": error_template, "agent": self.name, "message_type": "user"},
        ]

    def attempt_autosubmission_after_error(self, step: StepOutput) -> StepOutput:
        """For most exceptions, we attempt to still extract the patch and submit that.
        This means we send the `submit` command to the runtime and parse the output.
        """
        self.logger.warning("Attempting autosubmission after error")
        step = step.model_copy(deep=True)
        step.done = True
        assert self._env is not None
        if not asyncio.run(self._env.deployment.is_alive(timeout=10)):
            # The agent is dead. This is very bad. Maybe we can take a 'diff' that was saved
            # for a previous step? (if running with diff in tools)
            self.logger.error("Runtime is no longer alive")
            try:
                last_trajectory_step = self.trajectory[-1]
            except IndexError:
                self.logger.info("No last trajectory step to extract patch from")
                return step
            if "diff" not in last_trajectory_step["state"]:
                self.logger.info("No diff in last trajectory step state, cannot autosubmit")
                return step
            diff = last_trajectory_step["state"]["diff"]
            self.logger.info("Using diff from last trajectory step to autosubmit")
            step.submission = diff
            if step.submission:
                step.observation = "Environment died unexpectedly. Exited (autosubmitted)"
                step.exit_status = f"submitted ({step.exit_status})"
            else:
                self.logger.info("Diff from last traj step empty.")
            return step
        # Let us manually run the submission command and collect the output
        repo_name = "/"
        if self._env.repo is not None:
            repo_name = f"/{self._env.repo.repo_name}"
        submission_command = "git add -A && git diff --cached > /root/model.patch"
        self.logger.info("Executing submission command %s in %s", submission_command, repo_name)
        try:
            self._env.execute_command(submission_command, check=True, cwd=repo_name)
        except Exception as e:
            self.logger.error("Failed to execute submission command, got %s", e)
        # There's still hope for the submission, because the `/root/model.patch` file might have been
        # generated by the state command
        step = self.handle_submission(step, observation="", force_submission=True)
        if step.submission:
            self.logger.info("Exiting with autosubmission")
            step.observation = "Exited (autosubmitted)"
        return step

    def handle_submission(self, step: StepOutput, *, observation="", force_submission: bool = False) -> StepOutput:
        """Check if there was a submission in the observation and handle it.

        Args:
            step:
            observation: If specified, will use this rather than stepobservation
            force_submission: If True, will always submit even if no submission is found

        Returns:
            step: step with submission and observation updated (if submission was found)
        """
        step = step.model_copy(deep=True)
        assert self.tools is not None
        is_submission = self.tools.check_for_submission_cmd(observation or step.observation)
        if is_submission or force_submission:
            assert self._env is not None
            try:
                submission = self._env.read_file("/root/model.patch", encoding="utf-8", errors="backslashreplace")
            except FileNotFoundError:
                self.logger.warning("Submission file not found, no submission was made")
                return step
            except Exception as e:
                self.logger.exception("Failed to read submission file, got %s", e)
                return step
            if submission.strip() != "":
                step.submission = submission
            else:
                step.submission = None
            step.observation = submission
            if not step.exit_status:
                step.exit_status = "submitted"
            elif step.submission:
                step.exit_status = f"submitted ({step.exit_status})"
            step.done = True
            self.logger.info(f"Found submission: {submission}")
        return step

    def _get_edited_files_with_context(self, patch: str) -> dict[str, str]:
        """Get the edited files with context from the patch"""
        assert self._env is not None
        try:
            if self._env.repo is None:
                pf = None
            else:
                pf = (
                    PatchFormatter(
                        patch,
                        read_method=lambda path: self._env.read_file(  # type: ignore[attr-defined]
                            PurePosixPath("/") / self._env.repo.repo_name / path  # type: ignore[attr-defined]
                        ),
                    )
                    if patch
                    else None
                )
        except UnidiffParseError:
            self.logger.error("Failed to parse patch with unidiff. Some variables will be empty.")
            pf = None
            # We still need to populate the variables
        out = {}
        for context_length in [30, 50, 70]:
            value = "Empty. No edited files found."
            if pf is not None:
                value = pf.get_files_str(original=False, context_length=context_length)
            out[f"edited_files{context_length}"] = value
        return out

    def handle_action(self, step: StepOutput) -> StepOutput:
        """Runs an action proposed by the agent in the environment and returns the corresponding output.

        Args:
            action: command to run in bash shell
            output: output from model (only used for error handling)

        Returns:
            action_execution_output: action execution output
        """
        if self.tools.should_block_action(step.action):
            raise _BlockedActionError()

        if step.action.strip() == "exit":
            self.logger.info("Exiting agent")
            step.done = True
            step.observation = "Exited"
            step.exit_status = "exit_command"
            assert self._env is not None
            step.state = self.tools.get_state(env=self._env)  # for history
            return step

        assert self._env is not None
        tool_calls = step.tool_calls or []
        is_exp_subagent_call = (
            len(tool_calls) == 1 and ((tool_calls[0].get("function") or {}).get("name") == "experience_subagent")
        )
        if is_exp_subagent_call:
            self._chook.on_action_started(step=step)
            execution_t0 = time.perf_counter()
            step.observation = self._try_run_experience_subagent_tool(step) or "experience_subagent failed."
            step.execution_time = time.perf_counter() - execution_t0
            self._total_execution_time += step.execution_time
            self._chook.on_action_executed(step=step)
            step.state = self.tools.get_state(env=self._env)
            return self.handle_submission(step)

        self._chook.on_action_started(step=step)
        execution_t0 = time.perf_counter()
        run_action: str = self.tools.guard_multiline_input(step.action).strip()
        try:
            step.observation = self._env.communicate(
                input=run_action,
                timeout=self.tools.config.execution_timeout,
                check="raise" if self._always_require_zero_exit_code else "ignore",
            )
        except CommandTimeoutError:
            self._n_consecutive_timeouts += 1
            if self._n_consecutive_timeouts >= self.tools.config.max_consecutive_execution_timeouts:
                msg = "Exiting agent due to too many consecutive execution timeouts"
                self.logger.critical(msg)
                step.execution_time = time.perf_counter() - execution_t0
                self._total_execution_time += step.execution_time
                raise
            try:
                self._env.interrupt_session()
            except Exception as f:
                self.logger.exception("Failed to interrupt session after command timeout: %s", f, exc_info=True)
                step.execution_time = time.perf_counter() - execution_t0
                self._total_execution_time += step.execution_time
                raise
            step.observation = Template(self.templates.command_cancelled_timeout_template).render(
                **self._get_format_dict(),
                timeout=self.tools.config.execution_timeout,
                command=run_action,
            )
        else:
            self._n_consecutive_timeouts = 0
        step.execution_time = time.perf_counter() - execution_t0
        self._total_execution_time += step.execution_time
        self._chook.on_action_executed(step=step)
        step.state = self.tools.get_state(env=self._env)

        if RETRY_WITH_OUTPUT_TOKEN in step.observation:
            step.observation = step.observation.replace(RETRY_WITH_OUTPUT_TOKEN, "")
            raise _RetryWithOutput()
        elif RETRY_WITHOUT_OUTPUT_TOKEN in step.observation:
            step.observation = step.observation.replace(RETRY_WITHOUT_OUTPUT_TOKEN, "")
            raise _RetryWithoutOutput()
        elif EXIT_FORFEIT_TOKEN in step.observation:
            raise _ExitForfeit()

        return self.handle_submission(step)

    def forward(self, history: list[dict[str, str]]) -> StepOutput:
        """Forward the model without handling errors.

        All exceptions raised will contain the `StepOutput` object
        with some of the attributes set.

        Args:
            history: history to query the model with

        Returns:
            step_output: step output
        """
        if self._total_execution_time > self.tools.config.total_execution_timeout:
            raise _TotalExecutionTimeExceeded()

        # we continuously add actions, output etc. to the step object
        # because some of the specific exception handling requires some of these
        # attributes (e.g., if we want to requery the model for a bash syntax error, we
        # need to have the previous model output to format the requery template)
        step = StepOutput()
        step.query = copy.deepcopy(history)
        try:
            # Forward model and get actions
            self._chook.on_model_query(messages=history, agent=self.name)
            # todo: Add all options to the extra info
            if self._action_sampler is not None:
                assert self._problem_statement is not None
                best = self._action_sampler.get_action(
                    problem_statement=self._problem_statement,
                    trajectory=self.trajectory,
                    history=history,
                )
                output = best.completion
                # todo: Handle history and trajectory
                step.extra_info.update(best.extra_info)
            else:
                output = self.model.query(history)  # type: ignore
            step.output = output["message"]
            # todo: Can't I override the parser in __init__?
            step.thought, step.action = self.tools.parse_actions(output)
            step.thinking_blocks = output.get("thinking_blocks", [])
            if output.get("tool_calls") is not None:
                step.tool_call_ids = [call["id"] for call in output["tool_calls"]]
                step.tool_calls = output["tool_calls"]
            self.logger.info(f"💭 THOUGHT\n{step.thought}\n\n🎬 ACTION\n{step.action.strip()}")
            self._chook.on_actions_generated(step=step)
            return self.handle_action(step)
        except Exception as e:
            if step.action == step.thought == "":
                # Probably the parsing failed/no action included. Let's still fill in thought
                # so that trajectory viewers have something to show us for this step.
                step.thought = step.output
            # Attach the step object to the exception
            e.step = step  # type: ignore
            raise

    def forward_with_handling(self, history: list[dict[str, str]]) -> StepOutput:
        """Forward the model and handle errors, requerying the model if we can.
        For example, if the model outputs a bash command that has syntax errors,
        we will not execute it but requery the model for a corrected command.

        Note: This will update the trajectory, but not the history.

        Args:
            history: history to forward

        Returns:
            step_output: step output
        """

        def handle_error_with_autosubmission(exit_status: str, message: str) -> StepOutput:
            """Attempts to autosubmit (extract patch from the environment) and stops the loop."""
            self.logger.warning(message)
            return self.attempt_autosubmission_after_error(
                StepOutput(
                    thought=message,
                    exit_status=exit_status,
                    output=message,
                    done=True,
                )
            )

        def handle_error_with_retry(exception: Exception, template: str, n_requeries: int) -> list[dict[str, str]]:
            """Requeries the model if the error is a format/blocklist/bash syntax error."""
            self.logger.warning("Requerying model after %s (%dth requery)", type(exception).__name__, n_requeries)
            step: StepOutput = getattr(exception, "step", StepOutput())
            self.add_step_to_trajectory(step)
            exception_message = getattr(exception, "message", "")
            if not exception_message:
                try:
                    exception_message = exception.args[0]
                except (IndexError, AttributeError):
                    pass
            return self.get_model_requery_history(
                error_template=template,
                **step.to_template_format_dict(),
                **getattr(exception, "extra_info", {}),
                exception_message=exception_message,
            )

        n_format_fails = 0
        while n_format_fails < self.max_requeries:
            try:
                return self.forward(history)

            # Errors that are raised

            except KeyboardInterrupt:
                raise
            except EOFError:
                raise

            # Errors that cause requery

            except FormatError as e:
                n_format_fails += 1
                history = handle_error_with_retry(
                    exception=e, template=self.tools.config.format_error_template, n_requeries=n_format_fails
                )
            except _BlockedActionError as e:
                n_format_fails += 1
                history = handle_error_with_retry(
                    exception=e, template=self.tools.config.filter.blocklist_error_template, n_requeries=n_format_fails
                )
            except ContentPolicyViolationError:
                self.logger.warning("Content policy violation, trying to resample")
                n_format_fails += 1
                # Try if simply resampling helps here
                pass
            except BashIncorrectSyntaxError as e:
                n_format_fails += 1
                history = handle_error_with_retry(
                    exception=e,
                    template=self.templates.shell_check_error_template,
                    n_requeries=n_format_fails,
                )
            except _RetryWithOutput as e:
                history = handle_error_with_retry(
                    exception=e,
                    template=self.templates.next_step_template,
                    n_requeries=n_format_fails,
                )
            except _RetryWithoutOutput:
                pass
                # Requery with the same template as the last step

            # Errors that cause exit

            except _ExitForfeit:
                self.logger.info("Exiting due to forfeit")
                return handle_error_with_autosubmission(
                    "exit_forfeit",
                    "Exiting due to forfeit",
                )

            except _TotalExecutionTimeExceeded:
                self.logger.exception("Exiting due to total execution time exceeded", exc_info=True)
                return handle_error_with_autosubmission(
                    "exit_total_execution_time",
                    "Exit due to total execution time exceeded",
                )

            except CommandTimeoutError:
                self.logger.exception("Exiting due to multiple consecutive command timeouts", exc_info=True)
                return handle_error_with_autosubmission(
                    "exit_command_timeout",
                    "Exit due to multiple consecutive command timeouts",
                )

            except ContextWindowExceededError:
                return handle_error_with_autosubmission(
                    "exit_context",
                    "Exit due to context window",
                )
            except TotalCostLimitExceededError:
                raise
            except CostLimitExceededError:
                return handle_error_with_autosubmission(
                    "exit_cost",
                    "Exit due to cost limit",
                )
            except RetryError as e:
                self.logger.exception(f"Exiting due to retry error: {e}", exc_info=True)
                return handle_error_with_autosubmission(
                    "exit_api",
                    f"Exit due to retry error: {e}",
                )
            except SwerexException as e:
                self.logger.exception(f"Exiting due to environment error: {e}", exc_info=True)
                return handle_error_with_autosubmission(
                    "exit_environment_error",
                    f"Exit due to environment error: {e}",
                )
            except RuntimeError as e:
                self.logger.exception(f"Exiting due to runtime error: {e}", exc_info=True)
                return handle_error_with_autosubmission(
                    "exit_error",
                    f"Exit due to runtime error: {e}",
                )
            except Exception as e:
                self.logger.exception(f"Exiting due to unknown error: {e}", exc_info=True)
                return handle_error_with_autosubmission(
                    "exit_error",
                    f"Exit due to unknown error: {e}",
                )
        self.logger.exception(
            "Exit due to repeated format/blocklist/bash syntax errors",
            exc_info=True,
        )
        return handle_error_with_autosubmission(
            "exit_format",
            "Exit due to repeated format/blocklist/bash syntax errors",
        )

    def add_step_to_trajectory(self, step: StepOutput) -> None:
        trajectory_step = TrajectoryStep(
            {
                "action": step.action,
                "observation": step.observation,
                "response": step.output,
                "thought": step.thought,
                "execution_time": step.execution_time,
                "state": step.state,
                "query": step.query,
                "extra_info": step.extra_info,
            },
        )
        self.trajectory.append(trajectory_step)

    def step(self) -> StepOutput:
        """Run a step of the agent. This is a wrapper around `self.forward_with_handling`
        with additional bookkeeping:

        1. Update message history with performed action and observation
        2. Update trajectory with the final executed result
        3. Update the info dictionary

        Returns:
            step_output: step output (same as the output of `self.forward_with_handling`)
        """

        assert self._env is not None
        self._chook.on_step_start()

        n_step = len(self.trajectory) + 1
        self.logger.info("=" * 25 + f" STEP {n_step} " + "=" * 25)
        step_output = self.forward_with_handling(self.messages)
        self.add_step_to_history(step_output)

        self.info["submission"] = step_output.submission
        self.info["exit_status"] = step_output.exit_status  # type: ignore
        self.info.update(self._get_edited_files_with_context(patch=step_output.submission or ""))  # type: ignore
        self.info["model_stats"] = self.model.stats.model_dump()

        self.add_step_to_trajectory(step_output)

        self._chook.on_step_done(step=step_output, info=self.info)
        return step_output

    def run(
        self,
        env: SWEEnv,
        problem_statement: ProblemStatement | ProblemStatementConfig,
        output_dir: Path = Path("."),
    ) -> AgentRunResult:
        """Run the agent on a problem instance. This method contains the
        main loop that repeatedly calls `self._step` until the problem is solved.

        Args:
            setup_args: Arguments to pass to the agent's setup method.
            env: The environment to run the agent on.
            traj_dir: Directory to save the trajectory to
        """
        self.setup(env=env, problem_statement=problem_statement, output_dir=output_dir)

        # Run action/observation loop
        self._chook.on_run_start()
        step_output = StepOutput()
        while not step_output.done:
            step_output = self.step()
            self.save_trajectory()
        self._chook.on_run_done(trajectory=self.trajectory, info=self.info)

        self.logger.info("Trajectory saved to %s", self.traj_path)

        # Here we want to return the "global" information (e.g., submission should
        # be the best submission instead of the last one, etc.), so we get it from the traj file
        data = self.get_trajectory_data()
        return AgentRunResult(info=data["info"], trajectory=data["trajectory"])
