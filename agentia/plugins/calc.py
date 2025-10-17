from . import Plugin
from .. import tool
from typing import Annotated


class CalculatorPlugin(Plugin):
    @tool
    def evaluate(
        self,
        expression: Annotated[
            str, "The math expression to evaluate. Must be an valid python expression."
        ],
    ):
        """Execute a math expression and return the result. The expression must be an valid python expression that can be execuated by `eval()`."""
        result = eval(expression)
        return result
