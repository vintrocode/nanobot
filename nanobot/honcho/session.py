"""Honcho-based session management for conversation history."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from honcho import Honcho
from honcho.peer import PeerConfig
from honcho.session import SessionPeerConfig
from loguru import logger

from nanobot.honcho.client import get_honcho_client


@dataclass
class HonchoSession:
    """
    A conversation session backed by Honcho.

    Provides the same interface as the original Session class
    but stores messages in Honcho for AI-native memory.
    """

    key: str  # channel:chat_id
    user_peer_id: str  # Honcho peer ID for the user
    assistant_peer_id: str  # Honcho peer ID for the assistant
    honcho_session_id: str  # Honcho session ID
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Add a message to the local cache."""
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs,
        }
        self.messages.append(msg)
        self.updated_at = datetime.now()

    def get_history(self, max_messages: int = 50) -> list[dict[str, Any]]:
        """
        Get message history for LLM context.

        Args:
            max_messages: Maximum messages to return.

        Returns:
            List of messages in LLM format.
        """
        recent = (
            self.messages[-max_messages:]
            if len(self.messages) > max_messages
            else self.messages
        )
        return [{"role": m["role"], "content": m["content"]} for m in recent]

    def clear(self) -> None:
        """Clear all messages in the session."""
        self.messages = []
        self.updated_at = datetime.now()


class HonchoSessionManager:
    """
    Manages conversation sessions using Honcho.

    Replaces the file-based SessionManager with Honcho's
    AI-native memory system for user modeling.
    """

    def __init__(self, honcho: Honcho | None = None):
        """
        Initialize the session manager.

        Args:
            honcho: Optional Honcho client. If not provided, uses the singleton.
        """
        self._honcho = honcho
        self._cache: dict[str, HonchoSession] = {}
        self._peers_cache: dict[str, Any] = {}
        self._sessions_cache: dict[str, Any] = {}

    @property
    def honcho(self) -> Honcho:
        """Get the Honcho client, initializing if needed."""
        if self._honcho is None:
            self._honcho = get_honcho_client()
        return self._honcho

    def _get_or_create_peer(self, peer_id: str, is_assistant: bool = False) -> Any:
        """
        Get or create a Honcho peer.

        Args:
            peer_id: The peer identifier.
            is_assistant: If True, set observe_me=False for the AI peer.

        Returns:
            The Honcho peer object.
        """
        cache_key = f"{peer_id}:{is_assistant}"
        if cache_key in self._peers_cache:
            return self._peers_cache[cache_key]

        if is_assistant:
            peer = self.honcho.peer(peer_id, configuration=PeerConfig(observe_me=False))
        else:
            # Pass metadata={} to trigger get-or-create behavior
            peer = self.honcho.peer(peer_id, metadata={})

        self._peers_cache[cache_key] = peer
        return peer

    def _get_or_create_honcho_session(
        self, session_id: str, user_peer: Any, assistant_peer: Any
    ) -> Any:
        """
        Get or create a Honcho session with peers configured.

        Args:
            session_id: The session identifier.
            user_peer: The user peer object.
            assistant_peer: The assistant peer object.

        Returns:
            The Honcho session object.
        """
        if session_id in self._sessions_cache:
            return self._sessions_cache[session_id]

        # Pass metadata={} to trigger get-or-create behavior
        session = self.honcho.session(session_id, metadata={})

        # Configure peer observation settings
        user_config = SessionPeerConfig(observe_me=True, observe_others=True)
        ai_config = SessionPeerConfig(observe_me=False, observe_others=True)

        session.add_peers([(user_peer, user_config), (assistant_peer, ai_config)])

        self._sessions_cache[session_id] = session
        return session

    def get_or_create(self, key: str) -> HonchoSession:
        """
        Get an existing session or create a new one.

        Args:
            key: Session key (usually channel:chat_id).

        Returns:
            The session.
        """
        if key in self._cache:
            return self._cache[key]

        # Parse key to extract user identifier
        # Format: channel:chat_id (e.g., "telegram:123456789")
        parts = key.split(":", 1)
        channel = parts[0] if len(parts) > 1 else "default"
        chat_id = parts[1] if len(parts) > 1 else key

        # Create peer IDs
        user_peer_id = f"user-{channel}-{chat_id}"
        assistant_peer_id = "nanobot-assistant"

        # Get or create peers
        user_peer = self._get_or_create_peer(user_peer_id, is_assistant=False)
        assistant_peer = self._get_or_create_peer(assistant_peer_id, is_assistant=True)

        # Get or create Honcho session
        honcho_session = self._get_or_create_honcho_session(
            key, user_peer, assistant_peer
        )

        # Create local session wrapper
        session = HonchoSession(
            key=key,
            user_peer_id=user_peer_id,
            assistant_peer_id=assistant_peer_id,
            honcho_session_id=key,
        )

        self._cache[key] = session
        logger.debug(f"Created Honcho session: {key}")
        return session

    def save(self, session: HonchoSession) -> None:
        """
        Save messages to Honcho.

        This syncs the local message cache to Honcho's storage.

        Args:
            session: The session to save.
        """
        if not session.messages:
            return

        # Get the Honcho session and peers
        user_peer = self._get_or_create_peer(session.user_peer_id, is_assistant=False)
        assistant_peer = self._get_or_create_peer(
            session.assistant_peer_id, is_assistant=True
        )
        honcho_session = self._sessions_cache.get(session.key)

        if not honcho_session:
            honcho_session = self._get_or_create_honcho_session(
                session.key, user_peer, assistant_peer
            )

        # Convert messages to Honcho format and send
        # Only send new messages (those without a 'synced' flag)
        new_messages = [m for m in session.messages if not m.get("_synced")]

        if not new_messages:
            return

        honcho_messages = []
        for msg in new_messages:
            peer = user_peer if msg["role"] == "user" else assistant_peer
            honcho_messages.append(peer.message(msg["content"]))
            msg["_synced"] = True

        try:
            honcho_session.add_messages(honcho_messages)
            logger.debug(f"Synced {len(honcho_messages)} messages to Honcho for {session.key}")
        except Exception as e:
            # Mark messages as not synced on failure
            for msg in new_messages:
                msg["_synced"] = False
            logger.error(f"Failed to sync messages to Honcho: {e}")

        # Update cache
        self._cache[session.key] = session

    def delete(self, key: str) -> bool:
        """
        Delete a session from cache.

        Note: This only removes from local cache. Honcho sessions
        are retained for user modeling purposes.

        Args:
            key: Session key.

        Returns:
            True if deleted from cache, False if not found.
        """
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def get_user_context(self, session_key: str, query: str) -> str:
        """
        Query Honcho's dialectic chat for user context.

        Args:
            session_key: The session key to get context for.
            query: Natural language question about the user.

        Returns:
            Honcho's response about the user.
        """
        session = self._cache.get(session_key)
        if not session:
            return "No session found for this context."

        user_peer = self._get_or_create_peer(session.user_peer_id, is_assistant=False)

        try:
            return user_peer.chat(query)
        except Exception as e:
            logger.error(f"Failed to get user context from Honcho: {e}")
            return f"Unable to retrieve user context: {e}"

    def get_prefetch_context(self, session_key: str) -> dict[str, str]:
        """
        Pre-fetch common user context attributes.

        Args:
            session_key: The session key to get context for.

        Returns:
            Dictionary of user context attributes.
        """
        session = self._cache.get(session_key)
        if not session:
            return {}

        user_peer = self._get_or_create_peer(session.user_peer_id, is_assistant=False)

        context = {}
        queries = {
            "communication_style": "What communication style does this user prefer? Be concise.",
            "expertise_level": "What is this user's technical expertise level? Be concise.",
            "goals": "What are this user's current goals or priorities? Be concise.",
            "preferences": "What key preferences should I know about this user? Be concise.",
        }

        for key, query in queries.items():
            try:
                context[key] = user_peer.chat(query)
            except Exception as e:
                logger.warning(f"Failed to fetch {key} from Honcho: {e}")
                context[key] = ""

        return context

    def list_sessions(self) -> list[dict[str, Any]]:
        """
        List all cached sessions.

        Returns:
            List of session info dicts.
        """
        return [
            {
                "key": s.key,
                "created_at": s.created_at.isoformat(),
                "updated_at": s.updated_at.isoformat(),
                "message_count": len(s.messages),
            }
            for s in self._cache.values()
        ]
