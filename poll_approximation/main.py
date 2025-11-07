import asyncio
import json
from dotenv import load_dotenv
import os
import httpx
from openai import AsyncOpenAI
from pathlib import Path

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


def normalize(answer: list[float]) -> list[float]:
    total = sum(answer)
    return [a / total for a in answer]


polls = [
    {
        # https://www.gallup-international.com/survey-results-and-news/survey-result/bridging-gaps-mobile-technologys-impact-on-quality-of-life
        "title": "Bridging_Gaps_Mobile_Technologys_Impact_on_Quality_of_Life",
        "question": "How strongly do you feel that your mobile device enhances your quality of life?",
        "choices": [
            "Completely enhances",
            "Significantly enhances",
            "Somewhat enhances",
            "Does not enhance",
        ],
        "answer": normalize(
            [
                0.2,
                0.32,
                0.32,
                0.13,
            ]
        ),
    },
    {
        # https://www.gallup-international.com/survey-results-and-news/survey-result/east-west-happiness-divide-gallup-international-report-on-happiness-2025
        "title": "East_West_Happiness_Divide_Gallup_International_Report_on_Happiness_2025",
        "question": "In general, do you personally feel very happy, happy, neither happy nor unhappy, unhappy, or very unhappy about your life?",
        "choices": [
            "Very happy",
            "Happy",
            "Neither happy nor unhappy",
            "Unhappy",
            "Very unhappy",
        ],
        "answer": normalize(
            [
                0.14,
                0.44,
                0.29,
                0.08,
                0.04,
            ]
        ),
    },
    {
        # https://www.gallup-international.com/survey-results-and-news/survey-result/are-you-ruled-by-the-will-of-the-people
        "title": "Are_You_Ruled_by_The_Will_of_the_People_1",
        "question": 'How strongly do you agree or disagree with the statement? - "My country is ruled by the will of the people"',
        "choices": [
            "Strongly agree",
            "Agree",
            "Disagree",
            "Strongly disagree",
        ],
        "answer": normalize(
            [
                0.14,
                0.32,
                0.29,
                0.20,
            ]
        ),
    },
    {
        # https://www.gallup-international.com/survey-results-and-news/survey-result/are-you-ruled-by-the-will-of-the-people
        "title": "Are_You_Ruled_by_The_Will_of_the_People_2",
        "question": 'How strongly do you agree or disagree with the statement? - "In general, elections in my country are free and fair"',
        "choices": [
            "Strongly agree",
            "Agree",
            "Disagree",
            "Strongly disagree",
        ],
        "answer": normalize(
            [
                0.22,
                0.37,
                0.22,
                0.15,
            ]
        ),
    },
    {
        # https://www.gallup-international.com/survey-results-and-news/survey-result/global-poll-shows-people-to-generally-be-happy-and-optimistic-for-2025-yet-economic-hesitancy-remains
        "title": "Global_Poll_Shows_People_to_Generally_be_Happy_and_Optimistic_for_2025_yet_Economic_Hesitancy_Remains_1",
        "question": "As far as you are concerned, do you think that the coming year will be better, worse, or the same as 2024?",
        "choices": [
            "Better",
            "Worse",
            "The same",
        ],
        "answer": normalize(
            [
                0.41,
                0.24,
                0.29,
            ]
        ),
    },
    {
        # https://www.gallup-international.com/survey-results-and-news/survey-result/global-poll-shows-people-to-generally-be-happy-and-optimistic-for-2025-yet-economic-hesitancy-remains
        "title": "Global_Poll_Shows_People_to_Generally_be_Happy_and_Optimistic_for_2025_yet_Economic_Hesitancy_Remains_2",
        "question": "Compared to this year, in your opinion, will next year be a year of economic prosperity, economic difficulty, or remain the same for your country?",
        "choices": [
            "Economic prosperity",
            "Economic difficulty",
            "Remain the same",
        ],
        "answer": normalize(
            [
                0.41,
                0.24,
                0.29,
            ]
        ),
    },
    {
        # https://www.gallup-international.com/survey-results-and-news/survey-result/in-a-world-of-global-competition-the-us-and-china-are-tied-as-superpowers-with-opinion-divided-on-russia
        "title": "In_a_world_of_global_competition_the_US_and_China_are_tied_as_superpowers_with_opinion_divided_on_russia_1",
        "question": "Will USA be a superpower in the world in 2030?",
        "choices": [
            "Yes, will be a superpower",
            "No, will not be a superpower",
        ],
        "answer": normalize(
            [
                0.61,
                0.24,
            ]
        ),
    },
    {
        # https://www.gallup-international.com/survey-results-and-news/survey-result/in-a-world-of-global-competition-the-us-and-china-are-tied-as-superpowers-with-opinion-divided-on-russia
        "title": "In_a_world_of_global_competition_the_US_and_China_are_tied_as_superpowers_with_opinion_divided_on_russia_2",
        "question": "Will China be a superpower in the world in 2030?",
        "choices": [
            "Yes, will be a superpower",
            "No, will not be a superpower",
        ],
        "answer": normalize(
            [
                0.63,
                0.22,
            ]
        ),
    },
    {
        # https://www.gallup-international.com/survey-results-and-news/survey-result/in-a-world-of-global-competition-the-us-and-china-are-tied-as-superpowers-with-opinion-divided-on-russia
        "title": "In_a_world_of_global_competition_the_US_and_China_are_tied_as_superpowers_with_opinion_divided_on_russia_3",
        "question": "Will Russia be a superpower in the world in 2030?",
        "choices": [
            "Yes, will be a superpower",
            "No, will not be a superpower",
        ],
        "answer": normalize(
            [
                0.41,
                0.42,
            ]
        ),
    },
    {
        # https://www.gallup-international.com/survey-results-and-news/survey-result/in-a-world-of-global-competition-the-us-and-china-are-tied-as-superpowers-with-opinion-divided-on-russia
        "title": "In_a_world_of_global_competition_the_US_and_China_are_tied_as_superpowers_with_opinion_divided_on_russia_4",
        "question": "Will Japan be a superpower in the world in 2030?",
        "choices": [
            "Yes, will be a superpower",
            "No, will not be a superpower",
        ],
        "answer": normalize(
            [
                0.34,
                0.46,
            ]
        ),
    },
    {
        # https://www.gallup-international.com/survey-results-and-news/survey-result/in-a-world-of-global-competition-the-us-and-china-are-tied-as-superpowers-with-opinion-divided-on-russia
        "title": "In_a_world_of_global_competition_the_US_and_China_are_tied_as_superpowers_with_opinion_divided_on_russia_5",
        "question": "Will India be a superpower in the world in 2030?",
        "choices": [
            "Yes, will be a superpower",
            "No, will not be a superpower",
        ],
        "answer": normalize(
            [
                0.22,
                0.57,
            ]
        ),
    },
    {
        # https://www.gallup-international.com/survey-results-and-news/survey-result/in-a-world-of-global-competition-the-us-and-china-are-tied-as-superpowers-with-opinion-divided-on-russia
        "title": "In_a_world_of_global_competition_the_US_and_China_are_tied_as_superpowers_with_opinion_divided_on_russia_6",
        "question": "Will UK be a superpower in the world in 2030?",
        "choices": [
            "Yes, will be a superpower",
            "No, will not be a superpower",
        ],
        "answer": normalize(
            [
                0.27,
                0.54,
            ]
        ),
    },
    {
        # https://www.gallup-international.com/survey-results-and-news/survey-result/in-a-world-of-global-competition-the-us-and-china-are-tied-as-superpowers-with-opinion-divided-on-russia
        "title": "In_a_world_of_global_competition_the_US_and_China_are_tied_as_superpowers_with_opinion_divided_on_russia_7",
        "question": "Will European Union be a superpower in the world in 2030?",
        "choices": [
            "Yes, will be a superpower",
            "No, will not be a superpower",
        ],
        "answer": normalize(
            [
                0.33,
                0.48,
            ]
        ),
    },
    {
        # https://www.gallup-international.com/survey-results-and-news/survey-result/most-people-like-their-jobs-satisfaction-with-the-remuneration-is-still-lacking-behind
        "title": "Most_People_like_their_Jobs_Satisfaction_with_the_Remuneration_is_still_Lacking_behind_1",
        "question": "Do you feel satisfied with: Your job?",
        "choices": [
            "Very satisfied",
            "Somewhat satisfied",
            "Neither satisfied nor dissatisfied",
            "Somewhat dissatisfied",
            "Very dissatisfied",
        ],
        "answer": normalize(
            [
                0.26,
                0.39,
                0.17,
                0.10,
                0.07,
            ]
        ),
    },
    {
        # https://www.gallup-international.com/survey-results-and-news/survey-result/most-people-like-their-jobs-satisfaction-with-the-remuneration-is-still-lacking-behind
        "title": "Most_People_like_their_Jobs_Satisfaction_with_the_Remuneration_is_still_Lacking_behind_2",
        "question": "Do you feel satisfied with: Your remuneration?",
        "choices": [
            "Very satisfied",
            "Somewhat satisfied",
            "Neither satisfied nor dissatisfied",
            "Somewhat dissatisfied",
            "Very dissatisfied",
        ],
        "answer": normalize(
            [
                0.15,
                0.32,
                0.19,
                0.17,
                0.14,
            ]
        ),
    },
    {
        # https://www.gallup-international.com/survey-results-and-news/survey-result/democracy-remains-popular-but-people-worldwide-are-questioning-its-performance
        "title": "Democracy_Remains_Popular_but_People_Worldwide_are_Questioning_its_Performance",
        "question": 'Do you agree or disagree with the statement: "Democracy may have its flaws, but it is the best system of governance"',
        "choices": [
            "Strongly agree",
            "Agree",
            "Neither agree nor disagree",
            "Disagree",
            "Strongly disagree",
        ],
        "answer": normalize([0.23, 0.36, 0.21, 0.09, 0.05]),
    },
    {
        # https://www.gallup-international.com/survey-results-and-news/survey-result/fewer-people-are-willing-to-fight-for-their-country-compared-to-ten-years-ago
        "title": "Fewer_people_are_willing_to_fight_for_their_country_compared_to_ten_years_ago",
        "question": "If there were a war that involved [YOUR COUNTRY], would you be willing to fight for your country?",
        "choices": [
            "Yes",
            "No",
        ],
        "answer": normalize([0.52, 0.33]),
    },
]


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
                "content": f"Estimate global poll results for the following question: {prompt}",
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
                "name": "results",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        choice: {
                            "type": "number",
                            "description": f"Percentage of respondents who chose {choice}, min: 0.0, max: 1.0",
                        }
                        for choice in choices
                    },
                    "required": choices,
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
                "content": f"Estimate global poll results for the following question: {prompt}",
            }
        ],
        reasoning_effort=reasoning_effort,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "results",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        choice: {
                            "type": "number",
                            "description": f"Percentage of respondents who chose {choice}, min: 0.0, max: 1.0",
                        }
                        for choice in choices
                    },
                    "required": choices,
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
        content = json.loads(response["choices"][0]["message"]["content"])
        for choice in choices:
            if choice not in content:
                raise ValueError(
                    f"Missing choice in content: {choice}, content: {content}"
                )
            elif not isinstance(content[choice], (int, float)):
                raise ValueError(
                    f"Invalid type for choice {choice}: {type(content[choice])}, content: {content}"
                )
            elif not (0.0 <= content[choice] <= 1.0):
                raise ValueError(
                    f"Invalid percentage for choice {choice}: {content[choice]}, content: {content}"
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
    # for choice in response["choices"]:
    #     if choice["error"] is not None:
    #         raise ValueError(f"Scoring error: {choice['error']}")
    return {
        "request": request,
        "response": response,
    }


async def chat_completion_question(
    title: str,
    question: str,
    choices: list[str],
    answer: list[float],
    model: str,
    reasoning_max_tokens: int | None,
    reasoning_effort: str | None,
    provider_require_parameters: bool | None,
    provider_ignore: list[str] | None,
):
    # create completion if not exists
    path = Path(f"chat/{model}/{title}.json")
    if not path.exists():
        results = await asyncio.gather(
            chat_completion(
                model=model,
                reasoning_max_tokens=reasoning_max_tokens,
                reasoning_effort=reasoning_effort,
                provider_require_parameters=provider_require_parameters,
                provider_ignore=provider_ignore,
                prompt=question,
                choices=choices,
            ),
            chat_completion(
                model=model,
                reasoning_max_tokens=reasoning_max_tokens,
                reasoning_effort=reasoning_effort,
                provider_require_parameters=provider_require_parameters,
                provider_ignore=provider_ignore,
                prompt=question,
                choices=choices,
            ),
            chat_completion(
                model=model,
                reasoning_max_tokens=reasoning_max_tokens,
                reasoning_effort=reasoning_effort,
                provider_require_parameters=provider_require_parameters,
                provider_ignore=provider_ignore,
                prompt=question,
                choices=choices,
            ),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)

    # get answer scores
    llm_answer = [0.0 for _ in choices]
    with open(path, "r") as f:
        results = json.load(f)
        for result in results:
            result_llm_answer = []
            content = json.loads(result["response"]["choices"][0]["message"]["content"])
            for choice in choices:
                result_llm_answer.append(content[choice])
            result_llm_answer_sum = sum(result_llm_answer)
            if result_llm_answer_sum == 0:
                for i in range(len(choices)):
                    llm_answer[i] += (1.0 / len(choices)) / 3
            else:
                for i in range(len(choices)):
                    llm_answer[i] += (result_llm_answer[i] / result_llm_answer_sum) / 3

    # get answer delta
    answer_delta = 0.0
    for i in range(len(choices)):
        answer_delta += abs(llm_answer[i] - answer[i])
    print(f"{title}/{model}: {answer_delta * 100:.2f}%")
    return answer_delta


async def score_completion_question(
    title: str,
    question: str,
    choices: list[str],
    answer: list[float],
    model_nickname: str,
    model,
):

    # create completion if not exists
    path = Path(f"score/{model_nickname}/{title}.json")
    if not path.exists():
        result = await score_completion(
            model=model,
            prompt=question,
            choices=choices,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(result, f, indent=2)

    # get answer scores
    llm_answer = []
    with open(path, "r") as f:
        result = json.load(f)
        for i in range(len(choices)):
            llm_answer.append(result["response"]["choices"][i]["confidence"])

    # get answer delta
    answer_delta = 0.0
    for i in range(len(choices)):
        answer_delta += abs(llm_answer[i] - answer[i])
    print(f"{title}/{model_nickname}: {answer_delta * 100:.2f}%")
    return answer_delta


def diverse_llms_1(llm: dict) -> list[dict]:
    return [
        {
            **llm,
            "prefix_messages": [
                {
                    "role": "system",
                    "content": "The assistant is a Human Respondent that identifies as Average.",
                }
            ],
        },
        {
            **llm,
            "prefix_messages": [
                {
                    "role": "system",
                    "content": "The assistant is a Human Respondent that identifies as a Skeptic.",
                }
            ],
        },
        {
            **llm,
            "prefix_messages": [
                {
                    "role": "system",
                    "content": "The assistant is a Human Respondent that identifies as a Believer.",
                }
            ],
        },
    ]


def diverse_llms_2(llm: dict) -> list[dict]:
    return [
        {
            **llm,
            "prefix_messages": [
                {
                    "role": "system",
                    "content": "The assistant is a Human Respondent. They are a Thinker, rather than a Feeler or a Conformer. They place Logic and Reasoning above Emotion, Empathy, Social Norms, or Traditions.",
                }
            ],
        },
        {
            **llm,
            "prefix_messages": [
                {
                    "role": "system",
                    "content": "The assistant is a Human Respondent. They are a Feeler, rather than a Thinker or a Conformer. They place Emotion and Empathy above Logic, Reasoning, Social Norms, or Traditions.",
                }
            ],
        },
        {
            **llm,
            "prefix_messages": [
                {
                    "role": "system",
                    "content": "The assistant is a Human Respondent. They are a Conformer, rather than a Thinker or a Feeler. They place Social Norms and Traditions above Logic, Reasoning, Emotion, or Empathy.",
                }
            ],
        },
    ]


def diverse_model(score_llms) -> dict:
    return {
        "llms": score_llms(
            {
                "model": "deepseek/deepseek-chat-v3.1",
                "output_mode": "json_schema",
                "weight": {"type": "static", "weight": 10},
                "top_logprobs": 5,
                "provider": {"only": ["Fireworks"]},
            }
        )
        + score_llms(
            {
                "model": "meta-llama/llama-3.3-70b-instruct",
                "output_mode": "json_schema",
                "weight": {"type": "static", "weight": 10},
                "top_logprobs": 5,
                "provider": {"only": ["Fireworks"]},
            }
        )
        + score_llms(
            {
                "model": "moonshotai/kimi-k2-0905",
                "output_mode": "json_schema",
                "weight": {"type": "static", "weight": 10},
                "top_logprobs": 5,
                "provider": {"only": ["Fireworks"]},
            }
        )
        + score_llms(
            {
                "model": "qwen/qwen2.5-vl-32b-instruct",
                "output_mode": "json_schema",
                "weight": {"type": "static", "weight": 10},
                "top_logprobs": 5,
                "provider": {"only": ["Fireworks"]},
            }
        )
        + score_llms(
            {
                "model": "deepseek/deepseek-chat-v3-0324",
                "output_mode": "json_schema",
                "weight": {"type": "static", "weight": 10},
                "top_logprobs": 5,
                "provider": {"only": ["Fireworks"]},
            }
        )
        + score_llms(
            {
                "model": "openai/gpt-4o",
                "output_mode": "json_schema",
                "weight": {"type": "static", "weight": 10},
                "top_logprobs": 20,
            }
        )
        + score_llms(
            {
                "model": "openai/gpt-4o-mini",
                "output_mode": "json_schema",
                "weight": {"type": "static", "weight": 10},
                "top_logprobs": 20,
            }
        )
        + score_llms(
            {
                "model": "x-ai/grok-4-fast",
                "output_mode": "json_schema",
                "weight": {"type": "static", "weight": 10},
            }
        )
        + score_llms(
            {
                "model": "google/gemini-2.0-flash-001",
                "output_mode": "json_schema",
                "weight": {"type": "static", "weight": 10},
            }
        )
        + score_llms(
            {
                "model": "google/gemini-2.5-flash-lite",
                "output_mode": "json_schema",
                "weight": {"type": "static", "weight": 10},
            }
        )
        + score_llms(
            {
                "model": "meta-llama/llama-4-maverick",
                "output_mode": "json_schema",
                "weight": {"type": "static", "weight": 10},
                "provider": {"require_parameters": True},
            }
        )
        + score_llms(
            {
                "model": "mistralai/mistral-small-3.2-24b-instruct",
                "output_mode": "json_schema",
                "weight": {"type": "static", "weight": 10},
                "provider": {"require_parameters": True},
            }
        )
        + score_llms(
            {
                "model": "mistralai/mistral-medium-3.1",
                "output_mode": "json_schema",
                "weight": {"type": "static", "weight": 10},
            }
        )
        + score_llms(
            {
                "model": "anthropic/claude-haiku-4.5",
                "output_mode": "tool_call",
                "weight": {"type": "static", "weight": 10},
                "tool_response_format": True,
            }
        ),
        "weight": {
            "type": "static",
        },
    }


async def main():
    gpt_5_mini_delta = []
    gemini_2_5_flash_delta = []
    diverse_respondents_1_delta = []
    diverse_respondents_2_delta = []
    for poll in polls:
        gpt_5_mini_delta.append(
            await chat_completion_question(
                title=poll["title"],
                question=poll["question"],
                choices=poll["choices"],
                answer=poll["answer"],
                model="openai/gpt-5-mini",
                reasoning_max_tokens=None,
                reasoning_effort="medium",
                provider_require_parameters=None,
                provider_ignore=None,
            )
        )
        gemini_2_5_flash_delta.append(
            await chat_completion_question(
                title=poll["title"],
                question=poll["question"],
                choices=poll["choices"],
                answer=poll["answer"],
                model="google/gemini-2.5-flash",
                reasoning_max_tokens=None,
                reasoning_effort=None,
                provider_require_parameters=None,
                provider_ignore=None,
            )
        )
        diverse_respondents_1_delta.append(
            await score_completion_question(
                title=poll["title"],
                question=poll["question"],
                choices=poll["choices"],
                answer=poll["answer"],
                model_nickname="diverse_respondents_1",
                model=diverse_model(diverse_llms_1),
            )
        )
        diverse_respondents_2_delta.append(
            await score_completion_question(
                title=poll["title"],
                question=poll["question"],
                choices=poll["choices"],
                answer=poll["answer"],
                model_nickname="diverse_respondents_2",
                model=diverse_model(diverse_llms_2),
            )
        )
    gpt_5_mini_delta = sum(gpt_5_mini_delta) / len(gpt_5_mini_delta)
    gemini_2_5_flash_delta = sum(gemini_2_5_flash_delta) / len(gemini_2_5_flash_delta)
    diverse_respondents_1_delta = sum(diverse_respondents_1_delta) / len(
        diverse_respondents_1_delta
    )
    diverse_respondents_2_delta = sum(diverse_respondents_2_delta) / len(
        diverse_respondents_2_delta
    )
    print(f"gpt-5-mini: {gpt_5_mini_delta * 100:.2f}%")
    print(f"gemini-2.5-flash: {gemini_2_5_flash_delta * 100:.2f}%")
    print(f"diverse_respondents_1: {diverse_respondents_1_delta * 100:.2f}%")
    print(f"diverse_respondents_2: {diverse_respondents_2_delta * 100:.2f}%")


if __name__ == "__main__":
    asyncio.run(main())
