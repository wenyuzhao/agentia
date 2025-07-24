from agentia.agent import Agent


class RealtimeSession:
    def __init__(self, agent: Agent):
        self.agent = agent

    async def __aenter__(self): ...

    async def __aexit__(self, exc_type, exc_value, traceback): ...
