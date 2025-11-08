# IT Ticket Classification

We take an IT Ticket Classification dataset from https://www.opendatabay.com, containing 47,837 tickets and labelled categories. The tickets are lowercased and stripped of punctuation. Each of them is of one of the following categories:
- Hardware
- HR Support
- Access
- Miscellaneous
- Storage
- Purchase
- Internal Project
- Administrative rights

To start, we only run categorization of the first 1,000 tickets.

##

For Chat Completions, we use a response format of type `json_schema` in order to make the AI categorization machine parsable.

First, we run the first 1,000 tickets with `openai/gpt-5-nano` with `reasoning.effort: "medium"`. This yields an accuracy of 39.80% and a cost of $0.221583715000.

Next, we run the first 1,000 tickets with `google/gemini-2.5-flash-lite` with `reasoning.max_tokens: 8192`. This yields an accuracy of 42.80% and a cost of $0.284396900000.

##

For Score Completions, we distinguish between accuracy and winner_accuracy. Since each category is provided with a Confidence Score, accuracy is the average Confidence in the correct category, and winner accuracy is the percent of the time that the correct category has the highest Confidence.

First, we run the first 1,000 tickets with a Score Model hereby dubbed `llama_3_3_70b_instruct`, containing `meta-llama/llama-3.3-70b-instruct` with `top_logprobs: 20`. This yields an accuracy of 45.13%, a winner accuracy of 44.80%, and a cost of $0.026013408600.

Next, we run the first 1,000 tickets with a Score Model hereby dubbed `gpt_4o_mini`, containing `openai/gpt-4o-mini` with `top_logprobs: 20`. This yields an accuracy of 44.02%, a winner accuracy of 44.00%, and a cost of $0.027850884000.

Next, we run the first 1,000 tickets with a Score Model hereby dubbed `mistral_small_3_1_24b_instruct`, containing `mistralai/mistral-small-3.1-24b-instruct`. This yields an accuracy of 37.10%, a winner accuracy of 37.10%, and a cost of $0.009675350000.

Next, we run the first 1,000 tickets with a Score Model hereby dubbed `gemini_2_5_flash_lite`, containing `google/gemini-2.5-flash-lite` with reasoning disabled. This yields an accuracy of 37.00%, a winner accuracy of 37.00%, and a cost of $0.019510700000.

##

We run four more Score Completions on the same Score Models as the previous four, but this time, each is given a suffix system prompt.

First, we run the first 1,000 tickets with a Score Model hereby dubbed `llama_3_3_70b_instruct_concrete` with a suffix system prompt "You HEAVILY AVOID ever categorizing ANYTHING as 'Miscellaneous', 'Access', or 'Administrative rights'.". This yields an accuracy of 40.78%, a winner accuracy of 40.40%, and a cost of $0.029435196600.

Next, we run the first 1,000 tickets with a Score Model hereby dubbed `gpt_4o_mini_concrete` with a suffix system prompt "You HEAVILY AVOID ever categorizing ANYTHING as 'Miscellaneous', 'Access', or 'Internal Project'.". This yields an accuracy of 35.68%, a winner accuracy of 35.10%, and a cost of $0.031450930500.

Next, we run the first 1,000 tickets with a Score Model hereby dubbed `mistral_small_3_1_24b_instruct_concrete` with a suffix system prompt "You HEAVILY AVOID ever categorizing ANYTHING as 'Miscellaneous', 'Access', or 'Administrative rights'.". This yields an accuracy of 33.00%, a winner accuracy of 33.00%, and a cost of $0.011030835500.

Next, we run the first 1,000 tickets with a Score Model hereby dubbed `gemini_2_5_flash_lite_concrete` with a suffix system prompt "You HEAVILY AVOID ever categorizing ANYTHING as 'Miscellaneous' or 'Access'.". This yields an accuracy of 32.20%, a winner accuracy of 32.20%, and a cost of $0.021675500000.

##

We run a weight computation function against the eight score model results. This function computes weights for each of the eight score models optimizing Winner Accuracy, such that, a combination of their results, when weighted, will yield the highest possible winner accuracy. We prioritize Winner Accuracy because, for real world application, the winner is what matters. Taking these computed weights and applying them to the 8 Score Model results yields an accuracy of 41.20% and a winner accuracy of 47.10%.

## First 1,000 Row Results

| Model                                   | Accuracy | Winner Accuracy | Cost            |
|-----------------------------------------|----------|-----------------|-----------------|
| openai/gpt-5-nano                       | 39.80%   | 39.80%          | $0.221583715000 |
| google/gemini-2.5-flash-lite            | 42.80%   | 42.80%          | $0.284396900000 |
| llama_3_3_70b_instruct                  | 45.13%   | 44.80%          | $0.026013408600 |
| llama_3_3_70b_instruct_concrete         | 40.78%   | 40.40%          | $0.029435196600 |
| gpt_4o_mini                             | 44.02%   | 44.00%          | $0.027850884000 |
| gpt_4o_mini_concrete                    | 35.68%   | 35.10%          | $0.031450930500 |
| mistral_small_3_1_24b_instruct          | 37.10%   | 37.10%          | $0.011030835500 |
| mistral_small_3_1_24b_instruct_concrete | 33.00%   | 33.00%          | $0.011030835500 |
| gemini_2_5_flash_lite                   | 37.00%   | 37.00%          | $0.019510700000 |
| gemini_2_5_flash_lite_concrete          | 32.20%   | 32.20%          | $0.021675500000 |
| Score Models with Optimized Weights     | 41.20%   | 47.10%          |                 |

##

We run Chat Completions again, against the full 47,837 tickets.

`openai/gpt-5-nano` yields an accuracy of 36.47% and a cost of $10.602041536500.

`google/gemini-2.5-flash-lite` yields an accuracy of 40.87% and a cost of $14.728574800000.

##

We run another Score Completion using a new composite Score Model containing each Score LLM in the previously defined eight Score Models, hereby dubbed `combined_weight_optimized`. For each LLM within the Score Model, we assign it the static weight that was computed by the weight computation function.

This yields an accuracy of 40.03%, a winner accuracy of 43.38%, and a cost of $8.211427694700.

This Score Model has achieved a real-world-use greater accuracy than either of the Chat Completion LLMs at a lower cost.

## Final Results

| Model                        | Accuracy | Winner Accuracy | Cost             |
|------------------------------|----------|-----------------|------------------|
| openai/gpt-5-nano            | 36.47%   | 36.47%          | $10.602041536500 |
| google/gemini-2.5-flash-lite | 40.87%   | 40.87%          | $14.728574800000 |
| combined_weight_optimized    | 40.03%   | 43.38%          | $8.2114276947000 |
