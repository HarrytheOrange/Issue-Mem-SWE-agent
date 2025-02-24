# Environment variables

This page details all environment variables that are currently in use by SWE-agent.

* All API keys (for LMs and GitHub) can be set as an environment variable. See [here](../installation/keys.md) for more information.
* `SWE_AGENT_CONFIG_ROOT`: Used to resolve relative paths in the [config](config.md)

The following three variables can only be set as environment variables, not in the config file

* `SWE_AGENT_LOG_TIME`: Add timestamps to log
* `SWE_AGENT_LOG_STREAM_LEVEL`: Level of logging that is shown on the command line interface (`TRACE` being a custom level below `DEBUG`). Will have no effect for `run-batch`.

!!! hint "Persisting environment variables"
    Most environment variables can also be added to `.env` instead.