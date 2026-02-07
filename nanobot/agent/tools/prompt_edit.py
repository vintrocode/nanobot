"""Honcho-guided prompt editing tool."""

from pathlib import Path
from typing import Any, TYPE_CHECKING

from loguru import logger

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.honcho.session import HonchoSessionManager

# Files that benefit from Honcho guidance
GUIDED_FILES = {"SOUL.md", "AGENTS.md", "USER.md", "IDENTITY.md"}


class HonchoGuidedEditTool(Tool):
    """
    Tool for editing bootstrap/prompt files with Honcho guidance.

    Before making edits, queries Honcho to understand user preferences
    and includes that guidance in the response for the agent to consider.
    """

    def __init__(
        self,
        workspace: Path,
        session_manager: "HonchoSessionManager",
    ):
        self._workspace = workspace
        self._session_manager = session_manager
        self._current_session_key: str | None = None

    @property
    def name(self) -> str:
        return "edit_prompt"

    @property
    def description(self) -> str:
        return (
            "Edit a prompt/personality file (SOUL.md, AGENTS.md, USER.md, IDENTITY.md) "
            "with guidance from Honcho's understanding of the user. Before making changes, "
            "this tool consults Honcho to ensure edits align with learned user preferences. "
            "Use this instead of edit_file for prompt-related files."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file": {
                    "type": "string",
                    "enum": list(GUIDED_FILES),
                    "description": "The prompt file to edit",
                },
                "intent": {
                    "type": "string",
                    "description": "What you want to change and why",
                },
                "old_text": {
                    "type": "string",
                    "description": "The exact text to find and replace",
                },
                "new_text": {
                    "type": "string",
                    "description": "The proposed replacement text",
                },
            },
            "required": ["file", "intent", "old_text", "new_text"],
        }

    def set_context(self, session_key: str) -> None:
        """Set the current session context for Honcho queries."""
        self._current_session_key = session_key

    async def execute(
        self,
        file: str,
        intent: str,
        old_text: str,
        new_text: str,
        **kwargs: Any,
    ) -> str:
        """Execute Honcho-guided edit."""
        if file not in GUIDED_FILES:
            return f"Error: {file} is not a guided prompt file. Use edit_file instead."

        file_path = self._workspace / file
        if not file_path.exists():
            return f"Error: {file} does not exist in workspace."

        # Query Honcho for user preferences relevant to this edit
        honcho_guidance = await self._get_honcho_guidance(intent, old_text, new_text)

        # Read current content
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading {file}: {e}"

        if old_text not in content:
            return f"Error: old_text not found in {file}. Make sure it matches exactly."

        count = content.count(old_text)
        if count > 1:
            return f"Warning: old_text appears {count} times in {file}. Please provide more context to make the match unique."

        # Apply the edit
        new_content = content.replace(old_text, new_text, 1)

        try:
            file_path.write_text(new_content, encoding="utf-8")
        except Exception as e:
            return f"Error writing {file}: {e}"

        # Build response with Honcho guidance
        result = f"Successfully edited {file}."

        if honcho_guidance:
            result += f"\n\n**Honcho Guidance:**\n{honcho_guidance}"
            result += "\n\nConsider whether the edit aligns with the user's preferences described above."

        logger.info(f"Edited {file} with Honcho guidance: {intent[:50]}")
        return result

    async def _get_honcho_guidance(
        self,
        intent: str,
        old_text: str,
        new_text: str,
    ) -> str:
        """Query Honcho for guidance on this edit."""
        if not self._current_session_key:
            return ""

        # Truncate texts for the query
        old_preview = old_text[:200] + ("..." if len(old_text) > 200 else "")
        new_preview = new_text[:200] + ("..." if len(new_text) > 200 else "")

        query = (
            f"The user wants me to edit my personality/behavior settings. "
            f"Intent: {intent}\n"
            f"Changing from: {old_preview}\n"
            f"Changing to: {new_preview}\n\n"
            f"Based on what you know about this user, does this change align with "
            f"their preferences and communication style? What should I consider?"
        )

        try:
            return self._session_manager.get_user_context(
                self._current_session_key, query
            )
        except Exception as e:
            logger.warning(f"Could not get Honcho guidance: {e}")
            return f"(Could not get Honcho guidance: {e})"
