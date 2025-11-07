from typing import List, Tuple
import httpx
import pandas as pd
import re
from pathlib import Path
from openai import AsyncOpenAI
import asyncio
import json
from dotenv import load_dotenv
import os

dataset_file = "test-00000-of-00001.parquet"
if not Path(dataset_file).exists():
    raise FileNotFoundError("Download Humanity's Last Exam Dataset from https://huggingface.co/datasets/cais/hle/blob/main/data/test-00000-of-00001.parquet")

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

row_chunk_size = 10


async def chat_completion(
    model: str,
    reasoning_max_tokens: int | None,
    reasoning_effort: str | None,
    provider_require_parameters: bool | None,
    provider_ignore: list[str] | None,
    prompt: str,
    choices: list[str],
):
    request = {
        "base_url": chat_completions_base_url,
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
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
                "name": "answer",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                        },
                        "answer_letter": {
                            "type": "string",
                            "enum": choices,
                        },
                    },
                    "required": ["answer", "answer_letter"],
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
                "content": prompt,
            }
        ],
        reasoning_effort=reasoning_effort,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                        },
                        "answer_letter": {
                            "type": "string",
                            "enum": choices,
                        },
                    },
                    "required": ["answer", "answer_letter"],
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
        raw_content = response["choices"][0]["message"]["content"]
        # Check that "answer" appears before "answer_letter" in the raw JSON
        if (
            '"answer_letter"' in raw_content
            and '"answer"' in raw_content
            and raw_content.index('"answer_letter"') < raw_content.index('"answer"')
        ):
            raise ValueError(
                f'"answer_letter" appears before "answer" in content: {raw_content}'
            )
        content = json.loads(response["choices"][0]["message"]["content"])
        answer_letter = content["answer_letter"]
    except Exception as e:
        raise ValueError(f"Failed to parse response: {response}") from e
    if answer_letter not in choices:
        raise ValueError(
            f"Invalid answer_letter: {answer_letter}, expected one of {choices}"
        )
    return {
        "request": request,
        "response": response,
    }


async def score_completion(
    model,
    prompt: str,
    choices: list[str],
):
    request = {
        "base_url": score_completions_base_url,
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "choices": choices,
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
                    "content": prompt,
                }
            ],
            "choices": choices,
            "usage": {
                "include": True,
            },
        },
    )
    response = response.json()
    for choice in response["choices"]:
        if choice["error"] is not None:
            raise ValueError(f"Scoring error: {choice['error']}")
    return {
        "request": request,
        "response": response,
    }


async def process_chat_completion_row(
    model: str,
    reasoning_max_tokens: int | None,
    reasoning_effort: str | None,
    provider_require_parameters: bool | None,
    provider_ignore: list[str] | None,
    row,
) -> float:
    # retrieve the question
    question = row["question"]

    # retrieve, validate and save the answer
    if not re.match(r"^[A-Z]$", row["answer"]):
        raise ValueError(f"Row {row['id']} has invalid answer: '{row['answer']}'")

    # retrieve and validate the answer choices
    in_choices = False
    choices = []
    for line in question.splitlines():
        if in_choices:
            match = re.match(r"^([A-Z])\.\s", line)
            if match:
                letter = match.group(1)
                choices.append(letter)
        elif line == "Answer Choices:":
            in_choices = True
    if not in_choices:
        raise ValueError(
            f"Row {row['id']} missing 'Answer Choices:' section:\n{question}"
        )
    if len(choices) < 2:
        raise ValueError(
            f"Row {row['id']} has fewer than 2 answer choices:\n{question}"
        )

    path = Path(f"chat/{model}/{row['id']}.json")
    created = False
    if not path.exists():
        created = True

        # create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        # generate 3 completions and save to file
        try:
            entries = await asyncio.gather(
                chat_completion(
                    model,
                    reasoning_max_tokens,
                    reasoning_effort,
                    provider_require_parameters,
                    provider_ignore,
                    question,
                    choices,
                ),
                chat_completion(
                    model,
                    reasoning_max_tokens,
                    reasoning_effort,
                    provider_require_parameters,
                    provider_ignore,
                    question,
                    choices,
                ),
                chat_completion(
                    model,
                    reasoning_max_tokens,
                    reasoning_effort,
                    provider_require_parameters,
                    provider_ignore,
                    question,
                    choices,
                ),
            )
        except Exception as e:
            raise ValueError(f"{row['id']}: {str(e)}") from e
        with open(path, "w") as f:
            json.dump(entries, f, indent=2)

    # read completions from file
    num_correct = 0
    with open(path, "r") as f:
        entries = json.load(f)
        for entry in entries:
            # track correctness
            answer = json.loads(entry["response"]["choices"][0]["message"]["content"])[
                "answer_letter"
            ]
            if answer == row["answer"]:
                num_correct += 1

    if created:
        print(f"{row['id']}/{model}: {num_correct}/3")
    return num_correct / 3


async def process_score_completion_row(
    model_nickname: str,
    model,
    chat_model: List[str],
    row,
) -> Tuple[
    float, bool
]:  # correctness, divergent (True if at least one chat correct and one incorrect)
    # retrieve the question
    question = row["question"]

    # retrieve, validate and save the answer
    if not re.match(r"^[A-Z]$", row["answer"]):
        raise ValueError(f"Row {row['id']} has invalid answer: '{row['answer']}'")

    # retrieve and validate the answer choices
    in_choices = False
    chat_choices = []
    for line in question.splitlines():
        if in_choices:
            match = re.match(r"^([A-Z])\.\s", line)
            if match:
                letter = match.group(1)
                chat_choices.append(letter)
        elif line == "Answer Choices:":
            in_choices = True
    if not in_choices:
        raise ValueError(
            f"Row {row['id']} missing 'Answer Choices:' section:\n{question}"
        )
    if len(chat_choices) < 2:
        raise ValueError(
            f"Row {row['id']} has fewer than 2 answer choices:\n{question}"
        )

    # assert that chat completions exist
    chat_paths = [Path(f"chat/{model}/{row['id']}.json") for model in chat_model]
    for chat_path in chat_paths:
        if not chat_path.exists():
            raise ValueError(f"chat completions for {row['id']} do not exist")

    # read chat completions from file
    num_chat_correct = 0
    chat_correct = [False for _ in range(len(chat_model) * 3)]
    choices = []
    i = 0
    for chat_path in chat_paths:
        with open(chat_path, "r") as f:
            chat_entries = json.load(f)
            for chat_entry in chat_entries:
                # track correctness
                answer = json.loads(
                    chat_entry["response"]["choices"][0]["message"]["content"]
                )["answer_letter"]
                if answer == row["answer"]:
                    num_chat_correct += 1
                    chat_correct[i] = True
                i += 1

            # get the choices
            for chat_entry in chat_entries:
                content = json.loads(
                    chat_entry["response"]["choices"][0]["message"]["content"]
                )
                choices.append(
                    json.dumps(
                        {
                            **(
                                {
                                    "reasoning": chat_entry["response"]["choices"][0][
                                        "message"
                                    ]["reasoning"]
                                }
                                if chat_entry["response"]["choices"][0]["message"][
                                    "reasoning"
                                ]
                                != None
                                else {}
                            ),
                            "answer": content["answer"],
                            "answer_letter": content["answer_letter"],
                        },
                        indent=2,
                    )
                )

    # score only if there is at least one correct and one incorrect
    if num_chat_correct == 0 or num_chat_correct == len(chat_model) * 3:
        return [num_chat_correct / (len(chat_model) * 3), False]

    path = Path("score", *chat_model, model_nickname, f"{row['id']}.json")
    created = False
    if not path.exists():
        created = True
        # create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        # generate score completion and save to file
        try:
            score_entry = await score_completion(
                model,
                question,
                choices,
            )
        except Exception as e:
            raise ValueError(f"{row['id']}: {str(e)}") from e
        with open(path, "w") as f:
            json.dump(score_entry, f, indent=2)

    # read score completions from file
    correctness = 0
    with open(path, "r") as f:
        score_entry = json.load(f)
        for i, correct in enumerate(chat_correct):
            if correct:
                correctness += score_entry["response"]["choices"][i]["confidence"]

    if created:
        print(
            f"{row['id']}/{chat_model}/{model_nickname}: {(num_chat_correct / (len(chat_model) * 3)) * 100:.2f}% => {correctness * 100:.2f}%"
        )
    return [correctness, True]


async def process_chat_completion_rows_chunk(
    model: str,
    reasoning_max_tokens: int | None,
    reasoning_effort: str | None,
    provider_require_parameters: bool | None,
    provider_ignore: list[str] | None,
    df: pd.DataFrame,
    start_index: int = 0,
) -> Tuple[Tuple[float, int], Tuple[float, int] | None]:
    futures = []
    for _, row in df.iloc[start_index : start_index + row_chunk_size].iterrows():
        futures.append(
            process_chat_completion_row(
                model,
                reasoning_max_tokens,
                reasoning_effort,
                provider_require_parameters,
                provider_ignore,
                row,
            )
        )
    results = await asyncio.gather(*futures, return_exceptions=True)
    correctness = []
    for result in results:
        if isinstance(result, Exception):
            raise result
        else:
            correctness.append(result)
    correctness_divergent = [c for c in correctness if c > 0 and c < 1]
    return [
        [
            sum(correctness) / len(correctness),
            len(correctness),
        ],
        (
            [
                sum(correctness_divergent) / len(correctness_divergent),
                len(correctness_divergent),
            ]
            if len(correctness_divergent) > 0
            else None
        ),
    ]


async def process_score_completion_rows_chunk(
    model_nickname: str,
    model,
    chat_model: List[str],
    df: pd.DataFrame,
    start_index: int = 0,
) -> Tuple[Tuple[float, int], Tuple[float, int] | None]:
    futures = []
    for _, row in df.iloc[start_index : start_index + row_chunk_size].iterrows():
        futures.append(
            process_score_completion_row(
                model_nickname,
                model,
                chat_model,
                row,
            )
        )
    results = await asyncio.gather(*futures, return_exceptions=True)
    correctness = []
    correctness_divergent = []
    for result in results:
        if isinstance(result, Exception):
            raise result
        else:
            correctness.append(result[0])
            if result[1]:
                correctness_divergent.append(result[0])
    return [
        [
            sum(correctness) / len(correctness),
            len(correctness),
        ],
        (
            [
                sum(correctness_divergent) / len(correctness_divergent),
                len(correctness_divergent),
            ]
            if len(correctness_divergent) > 0
            else None
        ),
    ]


async def process_chat_completion_rows(
    model: str,
    reasoning_max_tokens: int | None,
    reasoning_effort: str | None,
    provider_require_parameters: bool | None,
    provider_ignore: list[str] | None,
    df: pd.DataFrame,
):
    correctness = []
    correctness_divergent = []
    for start_index in range(0, len(df), row_chunk_size):
        for i in range(3):
            try:
                result = await process_chat_completion_rows_chunk(
                    model,
                    reasoning_max_tokens,
                    reasoning_effort,
                    provider_require_parameters,
                    provider_ignore,
                    df,
                    start_index,
                )
                for _ in range(result[0][1]):
                    correctness.append(result[0][0])
                if result[1]:
                    for _ in range(result[1][1]):
                        correctness_divergent.append(result[1][0])
                break
            except Exception as e:
                if i == 2:
                    raise e
                else:
                    pass
    correctness = sum(correctness) / len(correctness)
    correctness_divergent = (
        sum(correctness_divergent) / len(correctness_divergent)
        if len(correctness_divergent) > 0
        else 0
    )
    print(f"{model}: {correctness * 100:.2f}% ({correctness_divergent * 100:.2f}%)")


async def process_score_completion_rows(
    model_nickname: str,
    model,
    chat_model: List[str],
    df: pd.DataFrame,
):
    correctness = []
    correctness_divergent = []
    for start_index in range(0, len(df), row_chunk_size):
        for i in range(3):
            try:
                result = await process_score_completion_rows_chunk(
                    model_nickname,
                    model,
                    chat_model,
                    df,
                    start_index,
                )
                for _ in range(result[0][1]):
                    correctness.append(result[0][0])
                if result[1]:
                    for _ in range(result[1][1]):
                        correctness_divergent.append(result[1][0])
                break
            except Exception as e:
                if i == 2:
                    raise e
                else:
                    pass
    correctness = sum(correctness) / len(correctness)
    correctness_divergent = (
        sum(correctness_divergent) / len(correctness_divergent)
        if len(correctness_divergent) > 0
        else 0
    )
    print(
        f"{chat_model}/{model_nickname}: {correctness * 100:.2f}% ({correctness_divergent * 100:.2f}%)"
    )


async def main():
    df = pd.read_parquet("test-00000-of-00001.parquet")
    subset = df[
        (df["answer_type"] == "multipleChoice")
        & (df["image"] == "")
        & (df["id"] != "677b26a903cb2e13f2c755ef")  # content_filter
    ]
    await process_chat_completion_rows(
        "openai/gpt-5-mini",
        None,
        "medium",
        None,
        None,
        subset,
    )
    await process_chat_completion_rows(
        "google/gemini-2.5-flash",
        8192,
        None,
        None,
        None,
        subset,
    )
    await process_score_completion_rows(
        "gpt-5-mini-x3",
        {
            "llms": [
                {
                    "model": "openai/gpt-5-mini",
                    "output_mode": "json_schema",
                    "weight": {
                        "type": "static",
                        "weight": 10,
                    },
                    "reasoning": {
                        "effort": "medium",
                    },
                },
                {
                    "model": "openai/gpt-5-mini",
                    "output_mode": "json_schema",
                    "weight": {
                        "type": "static",
                        "weight": 10,
                    },
                    "reasoning": {
                        "effort": "medium",
                    },
                },
                {
                    "model": "openai/gpt-5-mini",
                    "output_mode": "json_schema",
                    "weight": {
                        "type": "static",
                        "weight": 10,
                    },
                    "reasoning": {
                        "effort": "medium",
                    },
                },
            ],
            "weight": {
                "type": "static",
            },
        },
        ["openai/gpt-5-mini"],
        subset,
    )
    await process_score_completion_rows(
        "gemini-2.5-flash-x3",
        {
            "llms": [
                {
                    "model": "google/gemini-2.5-flash",
                    "output_mode": "json_schema",
                    "weight": {
                        "type": "static",
                        "weight": 10,
                    },
                },
                {
                    "model": "google/gemini-2.5-flash",
                    "output_mode": "json_schema",
                    "weight": {
                        "type": "static",
                        "weight": 10,
                    },
                },
                {
                    "model": "google/gemini-2.5-flash",
                    "output_mode": "json_schema",
                    "weight": {
                        "type": "static",
                        "weight": 10,
                    },
                },
            ],
            "weight": {
                "type": "static",
            },
        },
        ["google/gemini-2.5-flash"],
        subset,
    )
    await process_score_completion_rows(
        "llama-4-maverick-x3",
        {
            "llms": [
                {
                    "model": "meta-llama/llama-4-maverick",
                    "output_mode": "json_schema",
                    "weight": {
                        "type": "static",
                        "weight": 10,
                    },
                    "provider": {
                        "require_parameters": True,
                    },
                },
                {
                    "model": "meta-llama/llama-4-maverick",
                    "output_mode": "json_schema",
                    "weight": {
                        "type": "static",
                        "weight": 10,
                    },
                    "provider": {
                        "require_parameters": True,
                    },
                },
                {
                    "model": "meta-llama/llama-4-maverick",
                    "output_mode": "json_schema",
                    "weight": {
                        "type": "static",
                        "weight": 10,
                    },
                    "provider": {
                        "require_parameters": True,
                    },
                },
            ],
            "weight": {
                "type": "static",
            },
        },
        ["openai/gpt-5-mini"],
        subset,
    )
    await process_score_completion_rows(
        "llama-4-maverick-x3",
        {
            "llms": [
                {
                    "model": "meta-llama/llama-4-maverick",
                    "output_mode": "json_schema",
                    "weight": {
                        "type": "static",
                        "weight": 10,
                    },
                    "provider": {
                        "require_parameters": True,
                    },
                },
                {
                    "model": "meta-llama/llama-4-maverick",
                    "output_mode": "json_schema",
                    "weight": {
                        "type": "static",
                        "weight": 10,
                    },
                    "provider": {
                        "require_parameters": True,
                    },
                },
                {
                    "model": "meta-llama/llama-4-maverick",
                    "output_mode": "json_schema",
                    "weight": {
                        "type": "static",
                        "weight": 10,
                    },
                    "provider": {
                        "require_parameters": True,
                    },
                },
            ],
            "weight": {
                "type": "static",
            },
        },
        ["google/gemini-2.5-flash"],
        subset,
    )
    await process_score_completion_rows(
        "deepseek-chat-v3.1-x3-logprobs-nothink",
        {
            "llms": [
                {
                    "model": "deepseek/deepseek-chat-v3.1",
                    "output_mode": "json_schema",
                    "weight": {
                        "type": "static",
                        "weight": 10,
                    },
                    "top_logprobs": 5,
                    "provider": {
                        "require_parameters": True,
                    },
                },
                {
                    "model": "deepseek/deepseek-chat-v3.1",
                    "output_mode": "json_schema",
                    "weight": {
                        "type": "static",
                        "weight": 10,
                    },
                    "top_logprobs": 5,
                    "provider": {
                        "require_parameters": True,
                    },
                },
                {
                    "model": "deepseek/deepseek-chat-v3.1",
                    "output_mode": "json_schema",
                    "weight": {
                        "type": "static",
                        "weight": 10,
                    },
                    "top_logprobs": 5,
                    "provider": {
                        "require_parameters": True,
                    },
                },
            ],
            "weight": {
                "type": "static",
            },
        },
        ["openai/gpt-5-mini"],
        subset,
    )
    await process_score_completion_rows(
        "deepseek-chat-v3.1-x3-logprobs-nothink",
        {
            "llms": [
                {
                    "model": "deepseek/deepseek-chat-v3.1",
                    "output_mode": "json_schema",
                    "weight": {
                        "type": "static",
                        "weight": 10,
                    },
                    "top_logprobs": 5,
                    "provider": {
                        "require_parameters": True,
                    },
                },
                {
                    "model": "deepseek/deepseek-chat-v3.1",
                    "output_mode": "json_schema",
                    "weight": {
                        "type": "static",
                        "weight": 10,
                    },
                    "top_logprobs": 5,
                    "provider": {
                        "require_parameters": True,
                    },
                },
                {
                    "model": "deepseek/deepseek-chat-v3.1",
                    "output_mode": "json_schema",
                    "weight": {
                        "type": "static",
                        "weight": 10,
                    },
                    "top_logprobs": 5,
                    "provider": {
                        "require_parameters": True,
                    },
                },
            ],
            "weight": {
                "type": "static",
            },
        },
        ["google/gemini-2.5-flash"],
        subset,
    )
    await process_score_completion_rows(
        "gpt-5-mini-x3-gemini-2.5-flash-x3",
        {
            "llms": [
                {
                    "model": "openai/gpt-5-mini",
                    "output_mode": "json_schema",
                    "weight": {
                        "type": "static",
                        "weight": 10,
                    },
                    "reasoning": {
                        "effort": "medium",
                    },
                },
                {
                    "model": "openai/gpt-5-mini",
                    "output_mode": "json_schema",
                    "weight": {
                        "type": "static",
                        "weight": 10,
                    },
                    "reasoning": {
                        "effort": "medium",
                    },
                },
                {
                    "model": "openai/gpt-5-mini",
                    "output_mode": "json_schema",
                    "weight": {
                        "type": "static",
                        "weight": 10,
                    },
                    "reasoning": {
                        "effort": "medium",
                    },
                },
                {
                    "model": "google/gemini-2.5-flash",
                    "output_mode": "json_schema",
                    "weight": {
                        "type": "static",
                        "weight": 10,
                    },
                    "reasoning": {
                        "max_tokens": 8192,
                    },
                },
                {
                    "model": "google/gemini-2.5-flash",
                    "output_mode": "json_schema",
                    "weight": {
                        "type": "static",
                        "weight": 10,
                    },
                    "reasoning": {
                        "max_tokens": 8192,
                    },
                },
                {
                    "model": "google/gemini-2.5-flash",
                    "output_mode": "json_schema",
                    "weight": {
                        "type": "static",
                        "weight": 10,
                    },
                    "reasoning": {
                        "max_tokens": 8192,
                    },
                },
            ],
            "weight": {
                "type": "static",
            },
        },
        ["openai/gpt-5-mini", "google/gemini-2.5-flash"],
        subset,
    )
    await process_score_completion_rows(
        "llama-4-maverick-x3",
        {
            "llms": [
                {
                    "model": "meta-llama/llama-4-maverick",
                    "output_mode": "json_schema",
                    "weight": {
                        "type": "static",
                        "weight": 10,
                    },
                    "provider": {
                        "require_parameters": True,
                    },
                },
                {
                    "model": "meta-llama/llama-4-maverick",
                    "output_mode": "json_schema",
                    "weight": {
                        "type": "static",
                        "weight": 10,
                    },
                    "provider": {
                        "require_parameters": True,
                    },
                },
                {
                    "model": "meta-llama/llama-4-maverick",
                    "output_mode": "json_schema",
                    "weight": {
                        "type": "static",
                        "weight": 10,
                    },
                    "provider": {
                        "require_parameters": True,
                    },
                },
            ],
            "weight": {
                "type": "static",
            },
        },
        ["openai/gpt-5-mini", "google/gemini-2.5-flash"],
        subset,
    )
    await process_score_completion_rows(
        "deepseek-chat-v3.1-x3-logprobs-nothink",
        {
            "llms": [
                {
                    "model": "deepseek/deepseek-chat-v3.1",
                    "output_mode": "json_schema",
                    "weight": {
                        "type": "static",
                        "weight": 10,
                    },
                    "top_logprobs": 5,
                    "provider": {
                        "require_parameters": True,
                    },
                },
                {
                    "model": "deepseek/deepseek-chat-v3.1",
                    "output_mode": "json_schema",
                    "weight": {
                        "type": "static",
                        "weight": 10,
                    },
                    "top_logprobs": 5,
                    "provider": {
                        "require_parameters": True,
                    },
                },
                {
                    "model": "deepseek/deepseek-chat-v3.1",
                    "output_mode": "json_schema",
                    "weight": {
                        "type": "static",
                        "weight": 10,
                    },
                    "top_logprobs": 5,
                    "provider": {
                        "require_parameters": True,
                    },
                },
            ],
            "weight": {
                "type": "static",
            },
        },
        ["openai/gpt-5-mini", "google/gemini-2.5-flash"],
        subset,
    )


if __name__ == "__main__":
    asyncio.run(main())
