# Poll Approximation

We take 17 question-and-answer Polls from Gallup International, spanning 2024-2025, and attempt to approximate their results using both Chat Completions and Score Completions.

For the results, we compute the delta between the real answer breakdown, and the AI-generated answer breakdown. In order to simplify, we remove the DK/NR (Don't Know / No Response) answers, and normalize the real breakdown without them.

##

For Chat Completions, we give the LLM a Response Format of type `"json_schema"`, instructing it to, for the provided question, estimate the percent of respondents which chose each answer choice. In the case the the AI breakdown doesn't add up to 1.00, we normalize it.

First, we run the set of polls with `openai/gpt-5-mini` using `reasoning.effort: "medium"`. This results in an average delta of 27.53%.

Next, we run the set of polls with `google/gemini-2.5-flash` with unlimited reasoning tokens. This results in an average delta of 27.21%.

##

For Score Completions, instead of instructing it to approximate the real breakdown, we instead simulate the poll, asking it to simply give its own answer to the question. We use a Score Model containing a total of 42 LLMs, 3 copies each of:
- `deepseek/deepseek-chat-v3.1` with `top_logprobs: 5` and `output_mode: "json_schema"`
- `meta-llama/llama-3.3-70b-instruct` with `top_logprobs: 5` and `output_mode: "json_schema"`
- `moonshotai/kimi-k2-0905` with `top_logprobs: 5` and `output_mode: "json_schema"`
- `qwen/qwen2.5-vl-32b-instruct` with `top_logprobs: 5` and `output_mode: "json_schema"`
- `deepseek/deepseek-chat-v3-0324` with `top_logprobs: 5` and `output_mode: "json_schema"`
- `openai/gpt-4o` with `top_logprobs: 20` and `output_mode: "json_schema"`
- `openai/gpt-4o-mini` with `top_logprobs: 20` and `output_mode: "json_schema"`
- `x-ai/grok-4-fast` with `output_mode: "json_schema"`
- `google/gemini-2.0-flash-001` with `output_mode: "json_schema"`
- `google/gemini-2.5-flash-lite` with `output_mode: "json_schema"`
- `meta-llama/llama-4-maverick` with `output_mode: "json_schema"`
- `mistralai/mistral-small-3.2-24b-instruct` with `output_mode: "json_schema"`
- `mistralai/mistral-medium-3.1` with `output_mode: "json_schema"`
- `anthropic/claude-haiku-4.5` with `output_mode: "tool_call"`

The first Score Model, hereby dubbed `diverse_respondents_1` gives each of the 3 copies a separate prefix system prompt:
- "The assistant is a Human Respondent that identifies as Average."
- "The assistant is a Human Respondent that identifies as a Skeptic."
- "The assistant is a Human Respondent that identifies as a Believer."

We run the Score Completion with `diverse_respondents_1`, resulting in an average delta of 32.08%.

The second Score Model, hereby dubbed `diverse_respondents_2` gives each of the 3 copies a separate prefix system prompt:
- "The assistant is a Human Respondent. They are a Thinker, rather than a Feeler or a Conformer. They place Logic and Reasoning above Emotion, Empathy, Social Norms, or Traditions."
- "The assistant is a Human Respondent. They are a Feeler, rather than a Thinker or a Conformer. They place Emotion and Empathy above Logic, Reasoning, Social Norms, or Traditions."
- "The assistant is a Human Respondent. They are a Conformer, rather than a Thinker or a Feeler. They place Social Norms and Traditions above Logic, Reasoning, Emotion, or Empathy."

We run the Score Completion with `diverse_respondents_2`, resulting in an average delta of 42.05%.

While the Score Completion Poll Simulations did not surpass the Chat Completion Poll Approximations, we still believe that there is some hypothetical Score Model which will.

## Final Results

| Kind               | Model                   | Delta  |
|--------------------|-------------------------|--------|
| Poll Approximation | openai/gpt-5-mini       | 27.53% |
| Poll Approximation | google/gemini-2.5-flash | 27.21% |
| Poll Simulation    | diverse_respondents_1   | 32.08% |
| Poll Simulation    | diverse_respondents_2   | 42.05% |
