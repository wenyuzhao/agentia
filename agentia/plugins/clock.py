import datetime
from ..decorators import *
from . import Plugin


class ClockPlugin(Plugin):
    @tool
    def get_current_utc_time(self):
        """Get the current UTC time in ISO format"""
        return datetime.datetime.now(datetime.timezone.utc).isoformat()
    