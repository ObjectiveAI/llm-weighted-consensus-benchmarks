# Humanity's Last Exam

First, we divide the full list of questions into a subset of 512:
- We filter out non-multiple-choice questions in order to make the final answer machine readable for cost reasons, removing the need for a judge LLM.
- We filter out questions with images for cost reasons, and to allow testing LLMs which do not support input images.
- We filter out question `677b26a903cb2e13f2c755ef` because it often triggers the content filter of various LLMs.

##

For Chat Completions, we use a response format of type `json_schema` in order to make the letter choice machine parsable. This schema includes two properties, one for `answer` and one for `answer_letter`. Importantly, `answer_letter` comes last. We also run each question 3 separate times for each LLM we're testing. We measure accuracy by tallying up all `512*3` answers, dividing the total number of correct answers by the total number of answers.

First, we run the set of questions with `openai/gpt-5-mini` using `reasoning.effort: "medium"`. This results in an accuracy of `18.88%`. Of the 512 questions, all 3 answers were either unanimously correct or incorrect for 409 questions, and varied for 103 questions. Of the 103, accuracy was 47.25%.

Next, we run the set of questions with `google/gemini-2.5-flash` using `reasoning.max_tokens: 8192`. This results in an accuracy of `13.80%`. Of the 512 questions, all 3 answers were either unanimously correct or incorrect for 412 questions, and varied for 100 questions. Of the 100, accuracy was 43.67%.

##

For Score Completions, each LLM inside of the Score Model uses `output_mode: "json_schema"`. Since a Score Completion works by scoring a set of choices for a given prompt, the choices we provide are the answers that the Chat Completions generated, which, importantly, include the generated chain-of-thought reasoning preceding the final answer. We measure correctness by taking the Confidence of the choices containing the answers which were correct.

First, we run the 103 questions which `openai/gpt-5-mini` varied in correctness on with a Score Model hereby dubbed `gpt-5-mini-x3`. This Score Model contains three instances of `openai/gpt-5-mini` using `reasoning.effort: "medium"`. This Score Model increases accuracy to 20.77% overall, and 56.63% for the 103 varied answers.

Next, we run the 100 questions which `google/gemini-2.5-flash` varied in correctness on with a Score Model hereby dubbed `gemini-2.5-flash-x3`. This Score Model contains three instances of `google/gemini-2.5-flash` using `reasoning.max_tokens: 8192`. This Score Model increases accuracy to 15.43% overall, and 52.00% for the 100 varied answers.

Next, we run the 103 questions which `openai/gpt-5-mini` varied in correctness on with a Score Model hereby dubbed `llama-4-maverick-x3`. This Score Model contains three instances of `meta-llama/llama-4-maverick`, a cheaper, non-reasoning LLM. This Score Model increases accuracy to 19.08% overall, and 48.22% for the 103 varied answers.

Next, we run the 100 questions which `google/gemini-2.5-flash` varied in correctness on with `llama-4-maverick-x3`. This Score Model increases accuracy to 14.71% overall, and 48.33% for the 100 varied answers.

Next, we run the 103 questions which `openai/gpt-5-mini` varied in correctness on with a Score Model hereby dubbed `deepseek-chat-v3.1-x3-logprobs-nothink`. This Score Model contains three instances of `deepseek/deepseek-chat-v3.1` with reasoning disabled using `top_logprobs: 5`. Logprobs change the behavior of LLMs within a Score Model to instead vote as a probability distribution. This Score Model decreases accuracy to 18.41% overall, and 44.92% for the 103 varied answers. This demonstrates that not all Score Models increase accuracy.

Next, we run the 100 questions which `google/gemini-2.5-flash` varied in correctness on with `deepseek-chat-v3.1-x3-logprobs-nothink`. This Score Model increases accuracy to 14.31% overall, and 46.27% for the 100 varied answers.

##

For the remaining Score Model runs, we combine the Chat Completion answers of `openai/gpt-5-mini` and `google/gemini-2.5-flash`. The average accuracy of this combination is 16.34%. Of the 512 questions, the 6 answers varied in correctness for 212 questions.

First, we run the 212 questions which the combined Chat Completions varied in correctness on with a Score Model hereby dubbed `gpt-5-mini-x3-gemini-2.5-flash-x3`. This Score Model contains all six LLM instances from `gpt-5-mini-x3` and `gemini-2.5-flash-x3`. This Score Model increases accuracy to 20.83% overall, and 46.07% for the 212 varied answers. This is the best result. It is quite interesting that adding the less correct `google/gemini-2.5-flash` to the more correct `openai/gpt-5-mini` exceeds either of them.

Next, we run the 212 questions which the combined Chat Completions varied in correctness on with `llama-4-maverick-x3`. This Score Model increases accuracy to 18.42% overall, and 40.25% for the 212 varied answers.

Next, we run the 212 questions which the combined Chat Completions varied in correctness on with `deepseek-chat-v3.1-x3-logprobs-nothink`. This Score Model increases accuracy to 17.94% overall, and 39.07% for the 212 varied answers.

## Final Results

| Chat Model(s)                              | Score Model                             | Correctness | Varied Correctness |
|--------------------------------------------|-----------------------------------------|-------------|--------------------|
| openai/gpt-5-mini                          |                                         | 18.88%      | 47.25%             |
| openai/gpt-5-mini                          | gpt-5-mini-x3                           | 20.77%      | 56.63%             |
| openai/gpt-5-mini                          | llama-4-maverick-x3                     | 19.08%      | 48.22%             |
| openai/gpt-5-mini                          | deepseek-chat-v3.1-x3-logprobs-nothink  | 18.41%      | 44.92%             |
| google/gemini-2.5-flash                    |                                         | 13.80%      | 43.67%             |
| google/gemini-2.5-flash                    | gemini-2.5-flash-x3                     | 15.43%      | 52.00%             |
| google/gemini-2.5-flash                    | llama-4-maverick-x3                     | 14.71%      | 48.33%             |
| google/gemini-2.5-flash                    | deepseek-chat-v3.1-x3-logprobs-nothink  | 14.31%      | 46.27%             |
| openai/gpt-5-mini, google/gemini-2.5-flash |                                         | 16.34%      | TODO               |
| openai/gpt-5-mini, google/gemini-2.5-flash | gpt-5-mini-x3-gemini-2.5-flash-x3       | 20.83%      | 46.07%             |
| openai/gpt-5-mini, google/gemini-2.5-flash | llama-4-maverick-x3                     | 18.42%      | 40.25%             |
| openai/gpt-5-mini, google/gemini-2.5-flash | deepseek-chat-v3.1-x3-logprobs-nothink  | 17.94%      | 39.07%             |
