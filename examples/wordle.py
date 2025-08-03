from typing import Annotated
from pydantic import BaseModel, Field
from agentia import magic
import asyncio
import rich
import requests
import datetime
import dotenv

dotenv.load_dotenv()


class Char(BaseModel):
    letter: str = Field(min_length=1, max_length=1, description="A single character")
    correct: bool = Field(
        default=False, description="Is the letter in the correct position?"
    )
    present: bool = Field(
        default=False, description="Is the letter present in the word?"
    )
    absent: bool = Field(
        default=False, description="Is the letter not present in the word?"
    )


Word = Annotated[
    list[Char], Field(min_length=5, max_length=5, description="A 5-letter word")
]


class Inputs(BaseModel):
    tries: list[Word] = Field(
        default_factory=list, max_length=6, description="List of tries made by the user"
    )


async def predict(inputs: Inputs):
    all_chars = set()
    absent_chars = set()
    for word in inputs.tries:
        for char in word:
            all_chars.add(char.letter)
            if char.absent:
                absent_chars.add(char.letter)

    used = "".join(sorted(all_chars))
    unused = "".join(sorted(set("abcdefghijklmnopqrstuvwxyz") - set(used)))
    absent = "".join(sorted(absent_chars))
    allowed_chars = set("abcdefghijklmnopqrstuvwxyz") - set(absent)

    _Char = Annotated[
        str,
        Field(
            min_length=1,
            max_length=1,
            description="A single character",
            json_schema_extra={"choices": list(allowed_chars) if allowed_chars else []},
        ),
    ]

    class NextTry(BaseModel):
        word: list[_Char] = Field(  # type: ignore
            min_length=5, max_length=5, description="The next 5-letter word to try"
        )

    # @magic(model="openai/o3-pro")
    @magic(model="openai/o4-mini")
    async def _predict(
        tries: Inputs,
        used_chars: list[str],
        unused_chars: list[str],
        absent_chars: list[str],
    ) -> NextTry:
        """
        You are an AI assistant that predicts the next word to try in a Wordle game.
        The rule of the game:
        1. You have 6 tries to guess a 5-letter word.
        2. After each try, you receive feedback on each letter:
           - If the letter is in the correct position, it is marked as correct.
           - If the letter is present but not in the correct position, it is marked as present.
           - If the letter is not present at all, it is marked as absent.
        3. You must use the feedback to predict the next word to try.
        4. The word must be a valid 5-letter word and should not contain letters that are marked as absent.
        5. The word should ideally contain letters that are marked as present or correct.
        6. Don't repeat the exact same word that has been tried before.

        You will receive the following inputs:
        - `tries`: A list of tries made by the user, each containing the letter and its status (correct, present, absent).
        - `used_chars`: A list of characters that have been used in previous tries.
        - `unused_chars`: A list of characters that have not been used yet.
        - `absent_chars`: A list of characters that are guaranteed to not present in the word at all.
        """
        ...

    next_try = await _predict(
        tries=inputs,
        used_chars=list(used),
        unused_chars=list(unused),
        absent_chars=list(absent),
    )
    return next_try.word


async def run_game(solution: str, max_tries: int = 6):
    inputs = Inputs(tries=[])
    for i in range(max_tries):
        rich.print(f"[dim][{i+1}][/dim] ", end="", flush=True)
        result = []
        word = await predict(inputs)
        w = ""

        for i, c in enumerate(word):
            result_char = Char(letter=c)
            if word[i] == solution[i]:
                w += f"[green]{c}[/green]"
                result_char.correct = True
            elif c in solution:
                w += f"[yellow]{c}[/yellow]"
                result_char.present = True
            else:
                w += f"[red]{c}[/red]"
                result_char.absent = True
            result.append(result_char)
        rich.print(f"{w}", flush=True)
        inputs.tries.append(result)

        if "".join(word) == solution:
            stars = "⭐" * (max_tries - i)
            rich.print(f"\n[bold green]SOLVED![/bold green] {stars}")
            break
    else:
        rich.print(f"\n[bold red]FAILED! Answer: {solution}[/bold red]")


def get_today_solution():
    yyyy_mm_dd = datetime.datetime.now().strftime("%Y-%m-%d")
    response = requests.get(f"https://www.nytimes.com/svc/wordle/v2/{yyyy_mm_dd}.json")
    if response.status_code == 200:
        data = response.json()
        return data["solution"].lower()
    else:
        raise Exception("Failed to fetch today's Wordle solution.")


if __name__ == "__main__":
    rich.print("[bold blue]Starting Wordle game...[/bold blue]\n")
    solution = get_today_solution()
    asyncio.run(run_game(solution))
