"""Honcho memory tool for querying user context."""

from typing import Any, TYPE_CHECKING

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.honcho.client import HonchoClient


class HonchoTool(Tool):
    """
    Query Honcho to retrieve context about the user.

    Use this tool to understand the user's background, preferences,
    past interactions, goals, or communication style.
    """

    name = "honcho_query"
    description = (
        "Query Honcho's memory to retrieve relevant context about the user. "
        "Use this when you need to understand the user's background, preferences, "
        "past interactions, goals, or communication style. "
        "Returns insights based on Honcho's learned model of the user."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "A natural language question about the user, e.g. "
                    "'What are this user's main goals?' or "
                    "'What communication style does this user prefer?' or "
                    "'What technical background does this user have?'"
                )
            }
        },
        "required": ["query"]
    }

    def __init__(self, client: "HonchoClient"):
        self._client = client
        self._current_user_id: str | None = None

    def set_context(self, user_id: str) -> None:
        """Set the current user context for queries."""
        self._current_user_id = user_id

    async def execute(self, query: str, **kwargs: Any) -> str:
        if not self._current_user_id:
            return "Error: No user context set. Unable to query Honcho."

        try:
            result = self._client.chat(self._current_user_id, query)
            return result
        except Exception as e:
            return f"Error querying Honcho: {str(e)}"
