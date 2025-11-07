import asyncio
import csv
import json
from typing import Tuple
from dotenv import load_dotenv
import os
import httpx
from openai import AsyncOpenAI
from pathlib import Path
import numpy as np
from scipy.optimize import minimize


dataset_file = "all_tickets_processed_improved_v3.csv"
if not Path(dataset_file).exists():
    raise FileNotFoundError(
        "Download IT Support Ticket Topic Classifier dataset from https://www.opendatabay.com/data/dataset/5e817530-63a1-43be-a7a7-8be1473afdbf"
    )

load_dotenv()

chat_completions_base_url = os.environ["CHAT_COMPLETIONS_BASE_URL"]
chat_completions_api_key = os.getenv("CHAT_COMPLETIONS_API_KEY", "")
score_completions_base_url = os.environ["SCORE_COMPLETIONS_BASE_URL"]
score_completions_api_key = os.getenv("SCORE_COMPLETIONS_API_KEY", "")

chat_completions_openai = AsyncOpenAI(
    base_url=chat_completions_base_url,
    api_key=chat_completions_api_key,
)
score_completions_openai = AsyncOpenAI(
    base_url=score_completions_base_url,
    api_key=score_completions_api_key,
)

row_chunk_size = 100

ticket_categories = [
    "Hardware",
    "HR Support",
    "Access",
    "Miscellaneous",
    "Storage",
    "Purchase",
    "Internal Project",
    "Administrative rights",
]


async def chat_completion(
    model: str,
    reasoning_max_tokens: int | None,
    reasoning_effort: str | None,
    provider_require_parameters: bool | None,
    provider_ignore: list[str] | None,
    prompt: str,
):
    request = {
        "base_url": chat_completions_base_url,
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": f"Categorize this IT Ticket:\n\n{prompt}",
            }
        ],
        **(
            {"reasoning": {"max_tokens": reasoning_max_tokens}}
            if reasoning_max_tokens
            else {}
        ),
        "reasoning_effort": reasoning_effort,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "category",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": ticket_categories,
                        }
                    },
                    "required": ["category"],
                    "additionalProperties": False,
                },
            },
        },
        "usage": {
            "include": True,
        },
        **(
            {
                "provider": {
                    **(
                        {"require_parameters": provider_require_parameters}
                        if provider_require_parameters is not None
                        else {}
                    ),
                    **(
                        {"ignore": provider_ignore}
                        if provider_ignore is not None
                        else {}
                    ),
                }
            }
            if provider_require_parameters is not None or provider_ignore is not None
            else {}
        ),
    }
    response = await chat_completions_openai.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": f"Categorize this IT Ticket:\n\n{prompt}",
            }
        ],
        reasoning_effort=reasoning_effort,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "category",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": ticket_categories,
                        }
                    },
                    "required": ["category"],
                    "additionalProperties": False,
                },
            },
        },
        extra_body={
            **(
                {"reasoning": {"max_tokens": reasoning_max_tokens}}
                if reasoning_max_tokens
                else {}
            ),
            "usage": {
                "include": True,
            },
            **(
                {
                    "provider": {
                        **(
                            {"require_parameters": provider_require_parameters}
                            if provider_require_parameters is not None
                            else {}
                        ),
                        **(
                            {"ignore": provider_ignore}
                            if provider_ignore is not None
                            else {}
                        ),
                    }
                }
                if provider_require_parameters is not None
                or provider_ignore is not None
                else {}
            ),
        },
    )
    response = json.loads(response.model_dump_json())
    try:
        category = json.loads(response["choices"][0]["message"]["content"])["category"]
        if category not in ticket_categories:
            raise ValueError(
                f"Invalid category: {category}, content: {response['choices'][0]['message']['content']}"
            )
    except Exception as e:
        raise ValueError(f"Failed to parse response: {response}") from e
    return {
        "request": request,
        "response": response,
    }


async def score_completion(
    model,
    prompt: str,
):
    request = {
        "base_url": score_completions_base_url,
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": f"Categorize this IT Ticket:\n\n{prompt}",
            }
        ],
        "choices": ticket_categories,
        "usage": {
            "include": True,
        },
    }
    response = await score_completions_openai.post(
        "/score/completions",
        cast_to=httpx.Response,
        body={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": f"Categorize this IT Ticket:\n\n{prompt}",
                }
            ],
            "choices": ticket_categories,
            "usage": {
                "include": True,
            },
        },
    )
    response = response.json()
    return {
        "request": request,
        "response": response,
    }


async def process_chat_completion_ticket(
    model: str,
    reasoning_max_tokens: int | None,
    reasoning_effort: str | None,
    provider_require_parameters: bool | None,
    provider_ignore: list[str] | None,
    row,
    index: int,
) -> Tuple[float, float]:  # correctness, cost
    prompt = row["Document"]
    answer = row["Topic_group"]
    if answer not in ticket_categories:
        raise ValueError(f"Invalid answer category: {answer}")

    # generate completion if not exists
    path = Path(f"chat/{model}/{index}.json")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        result = await chat_completion(
            model=model,
            reasoning_max_tokens=reasoning_max_tokens,
            reasoning_effort=reasoning_effort,
            provider_require_parameters=provider_require_parameters,
            provider_ignore=provider_ignore,
            prompt=prompt,
        )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)
        answer = json.loads(result["response"]["choices"][0]["message"]["content"])[
            "category"
        ]
        if answer == row["Topic_group"]:
            print(f"{index}/{model}: 100.00%")
        else:
            print(f"{index}/{model}: 0.00%")

    # check correctness
    with open(path, "r") as f:
        result = json.load(f)
        answer = json.loads(result["response"]["choices"][0]["message"]["content"])[
            "category"
        ]
        usage = result["response"]["usage"]
        cost = usage.get(
            "total_cost",
            usage.get("cost", 0.0)
            + usage.get("cost_details", {}).get("upstream_inference_cost", 0.0),
        )
        if answer == row["Topic_group"]:
            return 1.0, cost
        else:
            return 0.0, cost


async def process_score_completion_ticket(
    model_nickname: str, model, row, index: int
) -> Tuple[float, bool, float, list[float]]:  # correctness, is_winner, cost, vote
    prompt = row["Document"]
    answer = row["Topic_group"]
    if answer not in ticket_categories:
        raise ValueError(f"Invalid answer category: {answer}")
    answer_choice_index = ticket_categories.index(answer)

    # generate completion if not exists
    path = Path(f"score/{model_nickname}/{index}.json")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        result = await score_completion(
            model=model,
            prompt=prompt,
        )
        if (
            len(result["response"]["choices"]) == 9
            and result["response"]["choices"][8]["error"] is not None
        ):
            raise ValueError(
                f"{model_nickname} - {index}: Scoring error: {result['response']['choices'][8]['error']}"
            )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)
        print(
            f"{index}/{model_nickname}: {result['response']['choices'][answer_choice_index]['confidence'] * 100:.2f}%"
        )

    # check correctness
    with open(path, "r") as f:
        result = json.load(f)
        choices = result["response"]["choices"]
        winner_index = max(
            range(len(choices)),
            key=lambda i: (
                choices[i]["confidence"]
                if choices[i]["confidence"] is not None
                else 0.0
            ),
        )
        # check ties. there's a more efficient way but I ain't got time
        for i in range(len(ticket_categories)):
            if (
                i != answer_choice_index
                and choices[i]["confidence"]
                == choices[answer_choice_index]["confidence"]
            ):
                winner_index = None
        if len(choices) == 9 and choices[8]["error"] is not None:
            raise ValueError(
                f"{model_nickname} - {index}: Scoring error: {choices[8]['error']}"
            )
        return (
            choices[answer_choice_index]["confidence"],
            winner_index == answer_choice_index,
            result["response"]["usage"].get("total_cost", 0.0),
            choices[8]["message"]["vote"],
        )


async def process_chat_completion_tickets_chunk(
    model: str,
    reasoning_max_tokens: int | None,
    reasoning_effort: str | None,
    provider_require_parameters: bool | None,
    provider_ignore: list[str] | None,
    rows: list,
    start_index: int,
) -> Tuple[float, int, float]:  # score, score_count, cost
    index = start_index
    futures = []
    for row in rows:
        futures.append(
            process_chat_completion_ticket(
                model=model,
                reasoning_max_tokens=reasoning_max_tokens,
                reasoning_effort=reasoning_effort,
                provider_require_parameters=provider_require_parameters,
                provider_ignore=provider_ignore,
                row=row,
                index=index,
            )
        )
        index += 1
    results = await asyncio.gather(*futures, return_exceptions=True)
    score = 0.0
    score_count = 0
    cost = 0.0
    for result in results:
        if isinstance(result, Exception):
            raise result
        else:
            score += result[0]
            score_count += 1
            cost += result[1]
    return score, score_count, cost


async def process_score_completion_tickets_chunk(
    model_nickname: str,
    model,
    rows: list,
    start_index: int,
) -> Tuple[
    float, int, float, int, float, list[float]
]:  # score, score_count, winner_score, winner_score_count, cost, vote
    index = start_index
    futures = []
    for row in rows:
        futures.append(
            process_score_completion_ticket(
                model_nickname=model_nickname,
                model=model,
                row=row,
                index=index,
            )
        )
        index += 1
    results = await asyncio.gather(*futures, return_exceptions=True)
    score = 0.0
    score_count = 0
    winner_score = 0.0
    winner_score_count = 0
    cost = 0.0
    votes = []
    for result in results:
        if isinstance(result, Exception):
            raise result
        else:
            score += result[0]
            score_count += 1
            if result[1]:
                winner_score += 1.0
            else:
                winner_score += 0.0
            winner_score_count += 1
            cost += result[2]
            votes.append(result[3])
    return score, score_count, winner_score, winner_score_count, cost, votes


async def process_chat_completion_tickets(
    model: str,
    reasoning_max_tokens: int | None,
    reasoning_effort: str | None,
    provider_require_parameters: bool | None,
    provider_ignore: list[str] | None,
    rows: list,
):
    score = 0.0
    score_count = 0
    cost = 0.0
    for i in range(0, len(rows), row_chunk_size):
        chunk = rows[i : i + row_chunk_size]
        chunk_score, chunk_score_count, chunk_cost = (
            await process_chat_completion_tickets_chunk(
                model=model,
                reasoning_max_tokens=reasoning_max_tokens,
                reasoning_effort=reasoning_effort,
                provider_require_parameters=provider_require_parameters,
                provider_ignore=provider_ignore,
                rows=chunk,
                start_index=i,
            )
        )
        score += chunk_score
        score_count += chunk_score_count
        cost += chunk_cost
    print(f"{model}: {(score/score_count)*100:.2f}% - ${cost:.12f}")


async def process_score_completion_tickets(
    model_nickname: str,
    model,
    rows: list,
) -> list[list[list[float]]]:
    score = 0.0
    score_count = 0
    winner_score = 0.0
    winner_score_count = 0
    cost = 0.0
    votes = []
    for i in range(0, len(rows), row_chunk_size):
        chunk = rows[i : i + row_chunk_size]
        (
            chunk_score,
            chunk_score_count,
            chunk_winner_score,
            chunk_winner_score_count,
            chunk_cost,
            v,
        ) = await process_score_completion_tickets_chunk(
            model_nickname=model_nickname,
            model=model,
            rows=chunk,
            start_index=i,
        )
        score += chunk_score
        score_count += chunk_score_count
        winner_score += chunk_winner_score
        winner_score_count += chunk_winner_score_count
        cost += chunk_cost
        votes.extend(v)
    print(
        f"{model_nickname}: {(score/score_count)*100:.2f}% - {(winner_score/winner_score_count)*100:.2f}% - ${cost:.12f}"
    )
    return votes


def compute_voter_weights(
    voters_answers: list[list[list[float]]],  # V x Q x C
    correct_answers: list[int],
    starts=800,  # number of tries from the top
    steps=400,  # refinement per direction
    rounds=40,  # coordinate ascent passes
):
    voters = np.array(voters_answers, float)  # (V,Q,C)
    correct = np.array(correct_answers, int)  # (Q,)
    V, Q, C = voters.shape

    def accuracy(w):
        w = np.asarray(w, float)
        w = w / np.sum(w)
        weighted = np.tensordot(w, voters, axes=(0, 0))  # (Q, C)
        preds = np.argmax(weighted, axis=1)
        return float(np.mean(preds == correct))

    # Global best across restarts
    global_best_w = None
    global_best_acc = -1

    for _ in range(starts):

        # Random baseline per start
        w = np.random.rand(V)
        w /= np.sum(w)
        best_w = w.copy()
        best_acc = accuracy(best_w)

        for _ in range(rounds):
            improved = False
            for v in range(V):
                for amt in np.linspace(0, 1, steps):
                    w_test = best_w * (1 - amt)
                    w_test[v] += amt
                    w_test /= np.sum(w_test)

                    acc = accuracy(w_test)
                    if acc > best_acc:
                        best_acc = acc
                        best_w = w_test
                        improved = True
            if not improved:
                break

        if best_acc > global_best_acc:
            global_best_acc = best_acc
            global_best_w = best_w.copy()

    return global_best_w.tolist()


def compute_accuracy(
    weights: list[float],
    voters_answers: list[list[list[float]]],
    correct_answers: list[int],
) -> float:
    w = np.array(weights, dtype=float)
    w = w / np.sum(w)  # ensure normalized

    voters = np.array(voters_answers)  # (V, Q, C)
    correct = np.array(correct_answers)  # (Q,)

    # Weighted probabilities across voters:
    weighted = np.tensordot(w, voters, axes=(0, 0))  # (Q, C)

    # Winner accuracy (argmax match)
    preds = np.argmax(weighted, axis=1)
    winner_accuracy = np.mean(preds == correct)

    # Score accuracy: average of probability assigned to correct answer
    correct_probs = weighted[np.arange(len(correct)), correct]
    score_accuracy = float(np.mean(correct_probs))

    print(
        f"Weighted Score Accuracy (correct class probability): {score_accuracy*100:.2f}%"
    )
    print(f"Weighted Winner Accuracy: {winner_accuracy*100:.2f}%")


async def main():
    rows = None
    with open(dataset_file, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # run first 1000 rows with chat completions
    await process_chat_completion_tickets(
        model="openai/gpt-5-nano",
        reasoning_max_tokens=None,
        reasoning_effort="medium",
        provider_require_parameters=None,
        provider_ignore=None,
        rows=rows[:1000],
    )
    await process_chat_completion_tickets(
        model="google/gemini-2.5-flash-lite",
        reasoning_max_tokens=8192,
        reasoning_effort=None,
        provider_require_parameters=None,
        provider_ignore=None,
        rows=rows[:1000],
    )

    # run first 1000 rows with score completions, 1x LLM score models
    llama_3_3_70b_instruct_votes = await process_score_completion_tickets(
        model_nickname="llama_3_3_70b_instruct",
        model={
            "llms": [
                {
                    "model": "meta-llama/llama-3.3-70b-instruct",
                    "top_logprobs": 20,
                    "provider": {"only": ["Crusoe"]},
                },
            ],
        },
        rows=rows[:1000],
    )
    llama_3_3_70b_instruct_concrete_votes = await process_score_completion_tickets(
        model_nickname="llama_3_3_70b_instruct_concrete",
        model={
            "llms": [
                {
                    "model": "meta-llama/llama-3.3-70b-instruct",
                    "top_logprobs": 20,
                    "provider": {"only": ["Crusoe"]},
                    "suffix_messages": [
                        {
                            "role": "system",
                            "content": "You HEAVILY AVOID ever categorizing ANYTHING as 'Miscellaneous', 'Access', or 'Administrative rights'.",
                        }
                    ],
                },
            ],
        },
        rows=rows[:1000],
    )
    gpt_4o_mini_votes = await process_score_completion_tickets(
        model_nickname="gpt_4o_mini",
        model={
            "llms": [
                {
                    "model": "openai/gpt-4o-mini",
                    "top_logprobs": 20,
                },
            ],
        },
        rows=rows[:1000],
    )
    gpt_4o_mini_concrete_votes = await process_score_completion_tickets(
        model_nickname="gpt_4o_mini_concrete",
        model={
            "llms": [
                {
                    "model": "openai/gpt-4o-mini",
                    "top_logprobs": 20,
                    "suffix_messages": [
                        {
                            "role": "system",
                            "content": "You HEAVILY AVOID ever categorizing ANYTHING as 'Miscellaneous', 'Access', or 'Internal Project'.",
                        }
                    ],
                },
            ],
        },
        rows=rows[:1000],
    )
    mistral_small_3_1_24b_instruct_votes = await process_score_completion_tickets(
        model_nickname="mistral_small_3_1_24b_instruct",
        model={
            "llms": [
                {
                    "model": "mistralai/mistral-small-3.1-24b-instruct",
                    "provider": {"only": ["DeepInfra"]},
                },
            ],
        },
        rows=rows[:1000],
    )
    mistral_small_3_1_24b_instruct_concrete_votes = await process_score_completion_tickets(
        model_nickname="mistral_small_3_1_24b_instruct_concrete",
        model={
            "llms": [
                {
                    "model": "mistralai/mistral-small-3.1-24b-instruct",
                    "provider": {"only": ["DeepInfra"]},
                    "suffix_messages": [
                        {
                            "role": "system",
                            "content": "You HEAVILY AVOID ever categorizing ANYTHING as 'Miscellaneous', 'Access', or 'Administrative rights'.",
                        }
                    ],
                },
            ],
        },
        rows=rows[:1000],
    )
    gemini_2_5_flash_lite_votes = await process_score_completion_tickets(
        model_nickname="gemini_2_5_flash_lite",
        model={
            "llms": [
                {
                    "model": "google/gemini-2.5-flash-lite",
                },
            ],
        },
        rows=rows[:1000],
    )
    gemini_2_5_flash_lite_concrete_votes = await process_score_completion_tickets(
        model_nickname="gemini_2_5_flash_lite_concrete",
        model={
            "llms": [
                {
                    "model": "google/gemini-2.5-flash-lite",
                    "suffix_messages": [
                        {
                            "role": "system",
                            "content": "You HEAVILY AVOID ever categorizing ANYTHING as 'Miscellaneous' or 'Access'.",
                        }
                    ],
                },
            ],
        },
        rows=rows[:1000],
    )

    # compute optimized per-LLM weights for an aggregate Score Model
    answers = [ticket_categories.index(row["Topic_group"]) for row in rows[:1000]]
    weights = compute_voter_weights(
        voters_answers=[
            llama_3_3_70b_instruct_votes,
            llama_3_3_70b_instruct_concrete_votes,
            gpt_4o_mini_votes,
            gpt_4o_mini_concrete_votes,
            mistral_small_3_1_24b_instruct_votes,
            mistral_small_3_1_24b_instruct_concrete_votes,
            gemini_2_5_flash_lite_votes,
            gemini_2_5_flash_lite_concrete_votes,
        ],
        correct_answers=answers,
    )
    print(weights)
    compute_accuracy(
        weights=weights,
        voters_answers=[
            llama_3_3_70b_instruct_votes,
            llama_3_3_70b_instruct_concrete_votes,
            gpt_4o_mini_votes,
            gpt_4o_mini_concrete_votes,
            mistral_small_3_1_24b_instruct_votes,
            mistral_small_3_1_24b_instruct_concrete_votes,
            gemini_2_5_flash_lite_votes,
            gemini_2_5_flash_lite_concrete_votes,
        ],
        correct_answers=answers,
    )

    # run all rows with chat completions
    await process_chat_completion_tickets(
        model="openai/gpt-5-nano",
        reasoning_max_tokens=None,
        reasoning_effort="medium",
        provider_require_parameters=None,
        provider_ignore=None,
        rows=rows,
    )
    await process_chat_completion_tickets(
        model="google/gemini-2.5-flash-lite",
        reasoning_max_tokens=8192,
        reasoning_effort=None,
        provider_require_parameters=None,
        provider_ignore=None,
        rows=rows,
    )

    # run all rows with score completions, combined optimized weight Score Model
    await process_score_completion_tickets(
        model_nickname="combined_weight_optimized",
        model={
            "llms": [
                {
                    "model": "meta-llama/llama-3.3-70b-instruct",
                    "top_logprobs": 20,
                    "provider": {"only": ["Crusoe"]},
                    "weight": {"type": "static", "weight": 0.3504270413935248},
                },
                {
                    "model": "meta-llama/llama-3.3-70b-instruct",
                    "top_logprobs": 20,
                    "provider": {"only": ["Crusoe"]},
                    "suffix_messages": [
                        {
                            "role": "system",
                            "content": "You HEAVILY AVOID ever categorizing ANYTHING as 'Miscellaneous', 'Access', or 'Administrative rights'.",
                        }
                    ],
                    "weight": {"type": "static", "weight": 0.061971234377302946},
                },
                {
                    "model": "openai/gpt-4o-mini",
                    "top_logprobs": 20,
                    "weight": {"type": "static", "weight": 0.41338181851550476},
                },
                {
                    "model": "openai/gpt-4o-mini",
                    "top_logprobs": 20,
                    "suffix_messages": [
                        {
                            "role": "system",
                            "content": "You HEAVILY AVOID ever categorizing ANYTHING as 'Miscellaneous', 'Access', or 'Internal Project'.",
                        }
                    ],
                    "weight": {"type": "static", "weight": 0.058969288787965354},
                },
                {
                    "model": "mistralai/mistral-small-3.1-24b-instruct",
                    "provider": {"only": ["DeepInfra"]},
                    "weight": {"type": "static", "weight": 0.025693100504557108},
                },
                {
                    "model": "mistralai/mistral-small-3.1-24b-instruct",
                    "provider": {"only": ["DeepInfra"]},
                    "suffix_messages": [
                        {
                            "role": "system",
                            "content": "You HEAVILY AVOID ever categorizing ANYTHING as 'Miscellaneous', 'Access', or 'Administrative rights'.",
                        }
                    ],
                    "weight": {"type": "static", "weight": 0.05644770737622784},
                },
                {
                    "model": "google/gemini-2.5-flash-lite",
                    "weight": {"type": "static", "weight": 0.00871669023365444},
                },
                {
                    "model": "google/gemini-2.5-flash-lite",
                    "suffix_messages": [
                        {
                            "role": "system",
                            "content": "You HEAVILY AVOID ever categorizing ANYTHING as 'Miscellaneous' or 'Access'.",
                        }
                    ],
                    "weight": {"type": "static", "weight": 0.024393118811262805},
                },
            ],
        },
        rows=rows,
    )


if __name__ == "__main__":
    i = 0
    while True:
        try:
            asyncio.run(main())
            break
        except Exception as e:
            i += 1
            if i >= 10:
                print("Exceeded maximum retries, exiting.")
                raise e
            else:
                print(f"Error occurred: {e}. Retrying {i}/10...")
