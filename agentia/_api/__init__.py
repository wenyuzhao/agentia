import dataclasses
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from agentia.agent import Agent, Event
from agentia import LOGGER

app = FastAPI(root_path="/api")


@app.get("/v1/agents")
async def get_agents():
    agents = Agent.get_all_agents()
    return {"agents": [agent.to_dict() for agent in agents]}


@app.get("/v1/agents/{agent_id}")
async def get_agent(agent_id: str):
    agents = Agent.get_all_agents()
    for agent in agents:
        if agent.id == agent_id:
            return {"agent": agent.to_dict()}
    raise HTTPException(status_code=404, detail="Agent not found")


@app.put("/v1/agents/{agent_id}")
async def create_agent(agent_id: str):
    raise HTTPException(status_code=500, detail="Not implemented")


@app.patch("/v1/agents/{agent_id}")
async def update_agent(agent_id: str):
    raise HTTPException(status_code=500, detail="Not implemented")


@app.delete("/v1/agents/{agent_id}")
async def delete_agent(agent_id: str):
    raise HTTPException(status_code=500, detail="Not implemented")


@app.get("/v1/agents/{agent_id}/sessions")
async def get_sessions(agent_id: str):
    sessions = Agent.get_all_sessions(agent_id)
    return {"sessions": [session.to_dict() for session in sessions]}


@app.get("/v1/agents/{agent_id}/sessions/{session_id}")
async def get_session(agent_id: str, session_id: str):
    agent_info = None
    for agent in Agent.get_all_agents():
        if agent.id == agent_id:
            agent_info = agent.to_dict()
            break
    if agent_info is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    session = Agent.load_session_info(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.agent != agent_id:
        raise HTTPException(status_code=404, detail="Session not found")
    agent = Agent.load_from_config(agent_id, persist=True, session_id=session_id)
    history = []
    for e in agent.history.get():
        if isinstance(e, Event):
            history.append(dataclasses.asdict(e))
        else:
            x = dict(**e.to_json())
            x["type"] = "message"
            history.append(x)
    return {"session": session.to_dict(), "history": history}


@app.put("/v1/agents/{agent_id}/sessions")
async def create_session(agent_id: str):
    try:
        agent = Agent.load_from_config(agent_id, persist=True)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=404, detail="Agent not found")
    await agent.save()
    session_id = agent.session_id
    session = Agent.load_session_info(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session": session.to_dict()}


async def run_chat_completion(websocket: WebSocket, agent: Agent, prompt: str):
    await websocket.send_json({"type": "response.start"})
    async for e in agent.chat_completion(prompt, stream=True, events=True):
        if isinstance(e, Event):
            await websocket.send_json({"type": "event", "event": dataclasses.asdict(e)})
        else:
            await websocket.send_json({"type": "message.start"})
            async for s in e:
                await websocket.send_json({"type": "message.delta", "delta": s})
            await websocket.send_json({"type": "message.end"})
    await websocket.send_json({"type": "response.end"})


SERVER_LOGGER = LOGGER.getChild("server")


@app.websocket("/v1/agents/{agent_id}/sessions/{session_id}/chat")
async def chat(websocket: WebSocket, agent_id: str, session_id: str):
    await websocket.accept()
    agent = Agent.load_from_config(agent_id, persist=True, session_id=session_id)
    SERVER_LOGGER.info(f"WebSocket connected: agent={agent_id} session={session_id}")

    try:
        while True:
            req = await websocket.receive_json()
            if "type" not in req:
                await websocket.send_json({"type": "error", "error": "Invalid request"})
                continue
            try:
                match req["type"]:
                    case "ping":
                        await websocket.send_json({"type": "pong"})
                    case "prompt":
                        prompt = req["prompt"]
                        await run_chat_completion(websocket, agent, prompt)
                    case _:
                        await websocket.send_json(
                            {"type": "error", "error": "Invalid request"}
                        )
            except BaseException as e:
                agent.log.error(e)
                await websocket.send_json({"error": str(e)})
    except WebSocketDisconnect as e:
        SERVER_LOGGER.info(
            f"WebSocket disconnected: agent={agent_id} session={session_id}"
        )
        raise e
