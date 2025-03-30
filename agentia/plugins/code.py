from ..decorators import *
from . import Plugin
from typing import Annotated
import traceback


class CodePlugin(Plugin):
    @tool
    def execute(self, python_code: Annotated[str, "The python code to run."]):
        """Execute python code and return the result. The expression must be an valid python expression that can be execuated by `eval()`."""
        from contextlib import redirect_stdout
        import io

        f = io.StringIO()
        with redirect_stdout(f):
            try:
                exec(python_code)
                return f.getvalue()
            except BaseException as e:
                return {"error": str(e), "traceback": repr(traceback.format_exc())}
