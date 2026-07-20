"""Agent handoff helpers for multi-agent LiveKit sessions."""

from __future__ import annotations

from livekit.agents.voice import Agent


async def transfer_with_context(from_agent: Agent, to_agent: Agent) -> None:
    """Copy full conversation history from ``from_agent`` to ``to_agent``, then transfer.

    Sets ``_continue_after_transfer`` so the new agent's ``on_enter``
    triggers ``generate_reply`` once the switch is fully complete.
    """
    to_agent._chat_ctx = from_agent._chat_ctx.copy(tools=to_agent._tools)
    to_agent._continue_after_transfer = True
    from_agent.session.update_agent(to_agent)
