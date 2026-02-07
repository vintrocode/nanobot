"""Agent loop: the core processing engine."""

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.agent.context import ContextBuilder
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebSearchTool, WebFetchTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.honcho import HonchoTool
from nanobot.agent.tools.prompt_edit import HonchoGuidedEditTool
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.budget import SpendBudget


class AgentLoop:
    """
    The agent loop is the core processing engine.
    
    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """
    
    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 20,
        max_spend_dollars: float | None = None,
        input_price_per_million: float = 15.0,
        output_price_per_million: float = 75.0,
        brave_api_key: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        cron_service: "CronService | None" = None,
        restrict_to_workspace: bool = False,
        honcho_enabled: bool = True,
        honcho_prefetch: bool = True,
        honcho_context_tokens: int | None = None,
    ):
        from nanobot.config.schema import ExecToolConfig
        from nanobot.cron.service import CronService
        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.max_spend_dollars = max_spend_dollars
        self.input_price_per_million = input_price_per_million
        self.output_price_per_million = output_price_per_million
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        self.honcho_enabled = honcho_enabled and bool(os.environ.get("HONCHO_API_KEY"))
        self.honcho_prefetch = honcho_prefetch
        self.honcho_context_tokens = honcho_context_tokens

        # Initialize session manager (Honcho or fallback)
        self.sessions: Any = None
        self._init_session_manager()

        self.context = ContextBuilder(workspace, honcho_session_manager=self.sessions if self.honcho_enabled else None)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            brave_api_key=brave_api_key,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )

        self._running = False
        self._register_default_tools()

    def _init_session_manager(self) -> None:
        """Initialize the appropriate session manager."""
        if self.honcho_enabled:
            try:
                from nanobot.honcho.session import HonchoSessionManager
                self.sessions = HonchoSessionManager(context_tokens=self.honcho_context_tokens)
                logger.info("Using Honcho for session management")
            except Exception as e:
                logger.warning(f"Failed to initialize Honcho, falling back to local sessions: {e}")
                from nanobot.session.manager import SessionManager
                self.sessions = SessionManager(self.workspace)
                self.honcho_enabled = False
        else:
            from nanobot.session.manager import SessionManager
            self.sessions = SessionManager(self.workspace)
            logger.info("Using local file-based session management")
    
    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        # File tools (restrict to workspace if configured)
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        self.tools.register(ReadFileTool(allowed_dir=allowed_dir))
        self.tools.register(WriteFileTool(allowed_dir=allowed_dir))
        self.tools.register(EditFileTool(allowed_dir=allowed_dir))
        self.tools.register(ListDirTool(allowed_dir=allowed_dir))
        
        # Shell tool
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
        ))
        
        # Web tools
        self.tools.register(WebSearchTool(api_key=self.brave_api_key))
        self.tools.register(WebFetchTool())
        
        # Message tool
        message_tool = MessageTool(send_callback=self.bus.publish_outbound)
        self.tools.register(message_tool)
        
        # Spawn tool (for subagents)
        spawn_tool = SpawnTool(manager=self.subagents)
        self.tools.register(spawn_tool)
        
        # Cron tool (for scheduling)
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))

        # Honcho tools (for querying user context and guided prompt editing)
        if self.honcho_enabled:
            from nanobot.honcho.session import HonchoSessionManager
            if isinstance(self.sessions, HonchoSessionManager):
                honcho_tool = HonchoTool(session_manager=self.sessions)
                self.tools.register(honcho_tool)

                # Honcho-guided prompt editing tool
                prompt_edit_tool = HonchoGuidedEditTool(
                    workspace=self.workspace,
                    session_manager=self.sessions,
                )
                self.tools.register(prompt_edit_tool)
    
    async def run(self) -> None:
        """Run the agent loop, processing messages from the bus."""
        self._running = True
        logger.info("Agent loop started")
        
        while self._running:
            try:
                # Wait for next message
                msg = await asyncio.wait_for(
                    self.bus.consume_inbound(),
                    timeout=1.0
                )
                
                # Process it
                try:
                    response = await self._process_message(msg)
                    if response:
                        await self.bus.publish_outbound(response)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    # Send error response
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=f"Sorry, I encountered an error: {str(e)}"
                    ))
            except asyncio.TimeoutError:
                continue
    
    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")
    
    async def _process_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a single inbound message.

        Args:
            msg: The inbound message to process.

        Returns:
            The response message, or None if no response needed.
        """
        # Handle system messages (subagent announces)
        # The chat_id contains the original "channel:chat_id" to route back to
        if msg.channel == "system":
            return await self._process_system_message(msg)

        # Handle user commands
        command_response = await self._handle_command(msg)
        if command_response is not None:
            return command_response

        logger.info(f"Processing message from {msg.channel}:{msg.sender_id}")
        
        # Get or create session
        session = self.sessions.get_or_create(msg.session_key)
        
        # Update tool contexts
        message_tool = self.tools.get("message")
        if isinstance(message_tool, MessageTool):
            message_tool.set_context(msg.channel, msg.chat_id)
        
        spawn_tool = self.tools.get("spawn")
        if isinstance(spawn_tool, SpawnTool):
            spawn_tool.set_context(msg.channel, msg.chat_id)
        
        cron_tool = self.tools.get("cron")
        if isinstance(cron_tool, CronTool):
            cron_tool.set_context(msg.channel, msg.chat_id)

        # Update Honcho tool context
        honcho_tool = self.tools.get("query_user_context")
        if isinstance(honcho_tool, HonchoTool):
            honcho_tool.set_context(msg.session_key)

        # Update prompt edit tool context
        prompt_edit_tool = self.tools.get("edit_prompt")
        if isinstance(prompt_edit_tool, HonchoGuidedEditTool):
            prompt_edit_tool.set_context(msg.session_key)

        # Pre-fetch user context from Honcho if enabled
        user_context = None
        if self.honcho_enabled and self.honcho_prefetch:
            from nanobot.honcho.session import HonchoSessionManager
            if isinstance(self.sessions, HonchoSessionManager):
                try:
                    user_context = self.sessions.get_prefetch_context(msg.session_key, msg.content)
                except Exception as e:
                    logger.warning(f"Failed to pre-fetch Honcho context: {e}")

        # Pass user context to spawn tool for subagents
        spawn_tool = self.tools.get("spawn")
        if isinstance(spawn_tool, SpawnTool):
            spawn_tool.set_user_context(user_context)

        # Build initial messages (use get_history for LLM-formatted messages)
        messages = self.context.build_messages(
            history=session.get_history(),
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=msg.chat_id,
            user_context=user_context,
        )

        # Create budget tracker for this request
        budget = None
        if self.max_spend_dollars:
            budget = SpendBudget(
                max_spend_dollars=self.max_spend_dollars,
                input_price_per_million=self.input_price_per_million,
                output_price_per_million=self.output_price_per_million,
            )
            # Pass budget to subagent manager
            self.subagents.set_shared_budget(budget)

        # Agent loop
        iteration = 0
        final_content = None
        budget_exhausted = False

        while iteration < self.max_iterations:
            iteration += 1

            # Check budget before LLM call
            if budget and budget.is_exhausted:
                budget_exhausted = True
                break

            # Call LLM
            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model
            )

            # Track spend
            if budget and response.usage:
                budget.add_usage(
                    response.usage.get("prompt_tokens", 0),
                    response.usage.get("completion_tokens", 0),
                    source="agent",
                )
                logger.debug(f"Budget: {budget.get_summary()}")

            # Handle tool calls
            if response.has_tool_calls:
                # Add assistant message with tool calls
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)  # Must be JSON string
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts
                )

                # Execute tools
                for tool_call in response.tool_calls:
                    args_str = json.dumps(tool_call.arguments)
                    logger.debug(f"Executing tool: {tool_call.name} with arguments: {args_str}")
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                # No tool calls, we're done
                final_content = response.content
                break

        # Handle budget exhaustion
        if budget_exhausted and budget:
            final_content = (
                f"I've reached my budget limit for this request. "
                f"{budget.get_summary()}"
            )
        elif final_content is None:
            final_content = "I've completed processing but have no response to give."
        
        # Save to session
        session.add_message("user", msg.content)
        session.add_message("assistant", final_content)
        self.sessions.save(session)
        
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content
        )
    
    async def _process_system_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a system message (e.g., subagent announce).
        
        The chat_id field contains "original_channel:original_chat_id" to route
        the response back to the correct destination.
        """
        logger.info(f"Processing system message from {msg.sender_id}")
        
        # Parse origin from chat_id (format: "channel:chat_id")
        if ":" in msg.chat_id:
            parts = msg.chat_id.split(":", 1)
            origin_channel = parts[0]
            origin_chat_id = parts[1]
        else:
            # Fallback
            origin_channel = "cli"
            origin_chat_id = msg.chat_id
        
        # Use the origin session for context
        session_key = f"{origin_channel}:{origin_chat_id}"
        session = self.sessions.get_or_create(session_key)
        
        # Update tool contexts
        message_tool = self.tools.get("message")
        if isinstance(message_tool, MessageTool):
            message_tool.set_context(origin_channel, origin_chat_id)
        
        spawn_tool = self.tools.get("spawn")
        if isinstance(spawn_tool, SpawnTool):
            spawn_tool.set_context(origin_channel, origin_chat_id)
        
        cron_tool = self.tools.get("cron")
        if isinstance(cron_tool, CronTool):
            cron_tool.set_context(origin_channel, origin_chat_id)

        # Update Honcho tool context
        honcho_tool = self.tools.get("query_user_context")
        if isinstance(honcho_tool, HonchoTool):
            honcho_tool.set_context(session_key)

        # Update prompt edit tool context
        prompt_edit_tool = self.tools.get("edit_prompt")
        if isinstance(prompt_edit_tool, HonchoGuidedEditTool):
            prompt_edit_tool.set_context(session_key)

        # Pre-fetch user context from Honcho if enabled
        user_context = None
        if self.honcho_enabled and self.honcho_prefetch:
            from nanobot.honcho.session import HonchoSessionManager
            if isinstance(self.sessions, HonchoSessionManager):
                try:
                    user_context = self.sessions.get_prefetch_context(session_key, msg.content)
                except Exception as e:
                    logger.warning(f"Failed to pre-fetch Honcho context: {e}")

        # Build messages with the announce content
        messages = self.context.build_messages(
            history=session.get_history(),
            current_message=msg.content,
            channel=origin_channel,
            chat_id=origin_chat_id,
            user_context=user_context,
        )
        
        # Agent loop (limited for announce handling)
        iteration = 0
        final_content = None
        
        while iteration < self.max_iterations:
            iteration += 1
            
            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model
            )
            
            if response.has_tool_calls:
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts
                )
                
                for tool_call in response.tool_calls:
                    args_str = json.dumps(tool_call.arguments)
                    logger.debug(f"Executing tool: {tool_call.name} with arguments: {args_str}")
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                final_content = response.content
                break
        
        if final_content is None:
            final_content = "Background task completed."
        
        # Save to session (mark as system message in history)
        session.add_message("user", f"[System: {msg.sender_id}] {msg.content}")
        session.add_message("assistant", final_content)
        self.sessions.save(session)
        
        return OutboundMessage(
            channel=origin_channel,
            chat_id=origin_chat_id,
            content=final_content
        )

    async def _handle_command(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Handle user commands like /clear, /help.

        Args:
            msg: The inbound message to check for commands.

        Returns:
            Response if command was handled, None otherwise.
        """
        content = msg.content.strip().lower()

        # /clear - Clear conversation history
        if content in ("/clear", "/reset", "/new"):
            session = self.sessions.get_or_create(msg.session_key)
            session.clear()
            self.sessions.save(session)
            logger.info(f"Cleared session: {msg.session_key}")
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="Session cleared. Starting fresh conversation."
            )

        # /status - Show session info
        if content == "/status":
            session = self.sessions.get_or_create(msg.session_key)
            msg_count = len(session.messages)
            honcho_status = "enabled" if self.honcho_enabled else "disabled"
            prefetch_status = "on" if self.honcho_prefetch else "off"
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=f"Session: {msg.session_key}\nMessages: {msg_count}\nHoncho: {honcho_status} (prefetch: {prefetch_status})"
            )

        # /help - Show available commands
        if content == "/help":
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="Commands:\n/clear - Clear conversation history\n/status - Show session info\n/help - Show this help"
            )

        return None

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
    ) -> str:
        """
        Process a message directly (for CLI or cron usage).
        
        Args:
            content: The message content.
            session_key: Session identifier.
            channel: Source channel (for context).
            chat_id: Source chat ID (for context).
        
        Returns:
            The agent's response.
        """
        msg = InboundMessage(
            channel=channel,
            sender_id="user",
            chat_id=chat_id,
            content=content
        )
        
        response = await self._process_message(msg)
        return response.content if response else ""
