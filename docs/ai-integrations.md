# AI Integrations

Large language models can drive vita49io through the helper schemas in the `ai_tools/` directory. Each schema mirrors a high-level task such as parsing a data packet or building a context packet with `IQStreamWriter`.

A typical workflow is:

1. Load the relevant JSON schema and register it as a callable function with your LLM runtime (OpenAI, Anthropic, etc.).
2. When the model returns a structured call, forward the arguments to the vita49io helper or class method described in the schema.
3. Translate the result (`bytes`, dictionaries, or NumPy arrays) into the format your agent expects.

Because vita49io exposes deterministic, side-effect-free helpers, it is well suited for autonomous tooling scenarios such as RF test automation or synthetic capture generation.
