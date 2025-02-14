# Models and API keys

## Setting API keys

In order to access the LM of your choice (and to access private GitHub repositories), you need to supply the corresponding keys.

There are two options to do this:

1. Set the corresponding [environment variables](https://www.cherryservers.com/blog/how-to-set-list-and-manage-linux-environment-variables).
2. Create a `.env` file at the root of this repository. All of the variables defined there will take the place of environment variables.


Here's an example

```
# Remove the comment '#' in front of the line for all keys that you have set
# GITHUB_TOKEN='GitHub Token for access to private repos'
# OPENAI_API_KEY='OpenAI API Key Here if using OpenAI Model'
# ANTHROPIC_API_KEY='Anthropic API Key Here if using Anthropic Model'
# TOGETHER_API_KEY='Together API Key Here if using Together Model'
```

See the following links for tutorials on obtaining [Anthropic](https://docs.anthropic.com/en/api/getting-started), [OpenAI](https://platform.openai.com/docs/quickstart/step-2-set-up-your-api-key), and [Github](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens) tokens.

## Supported API models

We support all models supported by [litellm](https://github.com/BerriAI/litellm), see their list [here](https://docs.litellm.ai/docs/providers).

Here are a few options for `--agent.model.name`:

| Model | API key | Comment |
| ----- | ------- | ------- |
| `claude-3-5-sonnet-20241022` | `ANTHROPIC_API_KEY` | Our recommended model |
| `gpt-4o` | `OPENAI_API_KEY` | |
| `o1-preview` | `OPENAI_API_KEY` | You might need to set temperature and sampling to the supported values. |

!!! warning "Function calling and more: Setting the correct parser"

    The default config uses function calling to retrieve actions from the model response, i.e.,
    the model directly provides the action as a JSON object.
    If your model doesn't support function calling, you can use the `thought_action` parser by setting
    `agent.tools.parse_function` to `thought_action`.
    Then, we extract the last triple-backticks block from the model response as the action.
    See [our API docs](../reference/parsers.md) for more details on parsers.
    Remember to document the tools in your prompt as the model will not be able to see the function signature
    like with function calling.

## Using local models

We currently support all models that serve to an endpoint with an OpenAI-compatible API.

For example, to use llama, you can folloow the [litellm instructions](https://docs.litellm.ai/docs/providers/ollama) and set

```
agent:
  model:
    name: ollama/llama2
    api_base: http://localhost:11434
    per_instance_cost_limit: 0
    total_cost_limit: 0
    per_instance_call_limit: 100
```

If you do not disable the default cost limits, you will see an error because the cost calculator will not be able to find the model in the `litellm` model cost dictionary.
Please use the `per_instance_call_limit` instead to limit the runtime per issue.

Please see the above note about using a config that uses the `thought_action` parser instead of the function calling parser.

## Complete model options

!!! hint "Complete model options"

    See [our API docs](../reference/model_config.md) for all available options.

## Models for testing

We also provide models for testing SWE-agent without spending any credits

* `HumanModel` and `HumanThoughtModel` will prompt for input from the user that stands in for the output of the LM. This can be used to create new [demonstrations](../config/demonstrations.md#manual).
* `ReplayModel` takes a trajectory as input and "replays it"
* `InstantEmptySubmitTestModel` will create an empty `reproduce.py` and then submit

### Debugging

* If you get `Error code: 404`, please check your configured keys, in particular
  whether you set `OPENAI_API_BASE_URL` correctly (if you're not using it, the
  line should be deleted or commented out).
  Also see [this issue](https://github.com/SWE-agent/SWE-agent/issues/467)
  for reference.

% include-markdown "../_footer.md" %}