"""Honcho client wrapper for nanobot integration."""

from typing import Any

from loguru import logger


class HonchoClient:
    """
    Wrapper around Honcho SDK for nanobot memory integration.

    Handles peer creation, session management, and message storage.
    """

    def __init__(
        self,
        workspace_id: str,
        api_key: str,
        environment: str = "production",
    ):
        from honcho import Honcho

        self._honcho = Honcho(
            workspace_id=workspace_id,
            api_key=api_key,
            environment=environment,
        )
        self._workspace_id = workspace_id

        # Cache for peers (peer_id -> Peer)
        self._peers: dict[str, Any] = {}

        # Cache for sessions (session_key -> Session)
        self._sessions: dict[str, Any] = {}

        # Create the nanobot AI peer (observed)
        self._assistant_peer = None

        logger.info(f"Honcho client initialized for workspace: {workspace_id}")

    def _get_or_create_peer(self, peer_id: str, observe_me: bool = True) -> Any:
        """Get or create a peer by ID."""
        from honcho.peer import PeerConfig

        if peer_id not in self._peers:
            config = PeerConfig(observe_me=observe_me)
            self._peers[peer_id] = self._honcho.peer(peer_id, configuration=config)
            logger.debug(f"Created Honcho peer: {peer_id} (observe_me={observe_me})")
        return self._peers[peer_id]

    def get_user_peer(self, user_id: str) -> Any:
        """Get or create a user peer (observed by Honcho)."""
        return self._get_or_create_peer(user_id, observe_me=True)

    def get_assistant_peer(self) -> Any:
        """Get or create the nanobot assistant peer (observed)."""
        if self._assistant_peer is None:
            self._assistant_peer = self._get_or_create_peer("nanobot", observe_me=True)
        return self._assistant_peer

    def get_or_create_session(self, session_key: str, user_id: str) -> Any:
        """
        Get or create a Honcho session.

        Args:
            session_key: The session key (channel:chat_id format)
            user_id: The user's peer ID

        Returns:
            Honcho session object
        """
        from honcho.session import SessionPeerConfig

        if session_key not in self._sessions:
            session = self._honcho.session(session_key)

            # Add peers to session
            user_peer = self.get_user_peer(user_id)
            assistant_peer = self.get_assistant_peer()

            # User is observed (Honcho builds a model of them)
            user_config = SessionPeerConfig(observe_me=True, observe_others=True)
            # AI is also observed (Honcho builds a model of the assistant too)
            ai_config = SessionPeerConfig(observe_me=True, observe_others=True)

            session.add_peers([
                (user_peer, user_config),
                (assistant_peer, ai_config)
            ])

            self._sessions[session_key] = session
            logger.debug(f"Created Honcho session: {session_key}")

        return self._sessions[session_key]

    def add_messages(
        self,
        session_key: str,
        user_id: str,
        user_content: str,
        assistant_content: str,
    ) -> None:
        """
        Add a user-assistant message exchange to Honcho.

        Args:
            session_key: The session key
            user_id: The user's peer ID
            user_content: The user's message content
            assistant_content: The assistant's response content
        """
        session = self.get_or_create_session(session_key, user_id)
        user_peer = self.get_user_peer(user_id)
        assistant_peer = self.get_assistant_peer()

        session.add_messages([
            user_peer.message(user_content),
            assistant_peer.message(assistant_content),
        ])
        logger.debug(f"Added messages to Honcho session: {session_key}")

    def chat(self, user_id: str, query: str, stream: bool = False) -> str:
        """
        Query Honcho's dialectic chat about a user.

        Args:
            user_id: The user peer ID to query about
            query: Natural language question about the user
            stream: Whether to stream the response

        Returns:
            Honcho's response about the user
        """
        peer = self.get_user_peer(user_id)

        if stream:
            response_stream = peer.chat(query, stream=True)
            chunks = []
            for chunk in response_stream.iter_text():
                chunks.append(chunk)
            return "".join(chunks)
        else:
            return peer.chat(query)

    def get_context(
        self,
        session_key: str,
        user_id: str,
        tokens: int = 2000,
        summary: bool = True,
    ) -> Any:
        """
        Get conversation context from Honcho.

        Args:
            session_key: The session key
            user_id: The user peer ID
            tokens: Max tokens for context
            summary: Include conversation summaries

        Returns:
            Honcho context object
        """
        session = self.get_or_create_session(session_key, user_id)
        user_peer = self.get_user_peer(user_id)

        return session.context(
            tokens=tokens,
            peer_target=user_peer.id,
            summary=summary,
        )
