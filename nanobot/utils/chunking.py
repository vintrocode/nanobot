"""Message chunking utilities for platform-specific message limits."""

import re
from typing import Iterator

DISCORD_MAX_LENGTH = 2000
CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```")


def chunk_message(content: str, max_length: int = DISCORD_MAX_LENGTH) -> Iterator[str]:
    """
    Split content into chunks that respect Discord's message limit.

    Rules:
    1. Never split in the middle of a code block
    2. Prefer splitting at paragraph boundaries
    3. Fall back to sentence/word boundaries

    Args:
        content: The message content to split
        max_length: Maximum characters per chunk (default: 2000 for Discord)

    Yields:
        Message chunks, each within max_length
    """
    if len(content) <= max_length:
        yield content
        return

    # Find all code blocks and their positions
    code_blocks = [(m.start(), m.end()) for m in CODE_BLOCK_PATTERN.finditer(content)]

    current_pos = 0

    while current_pos < len(content):
        # Calculate end position
        end_pos = min(current_pos + max_length, len(content))

        if end_pos >= len(content):
            # Last chunk
            yield content[current_pos:]
            break

        # Check if we're cutting through a code block
        in_code_block = None
        for start, end in code_blocks:
            if start < end_pos <= end:
                in_code_block = (start, end)
                break

        if in_code_block:
            block_start, block_end = in_code_block
            if block_start > current_pos:
                # Cut before the code block starts
                end_pos = block_start
            elif block_end - current_pos <= max_length:
                # Include the whole code block if it fits
                end_pos = block_end
            else:
                # Very long code block - must split it
                # Find a good newline position within the code
                chunk = content[current_pos : current_pos + max_length]
                last_newline = chunk.rfind("\n")
                if last_newline > max_length // 2:
                    end_pos = current_pos + last_newline + 1
        else:
            # Not in code block - find good break point
            chunk = content[current_pos:end_pos]

            # Prefer paragraph break
            para_break = chunk.rfind("\n\n")
            if para_break > max_length // 2:
                end_pos = current_pos + para_break + 2
            else:
                # Try newline
                newline = chunk.rfind("\n")
                if newline > max_length // 2:
                    end_pos = current_pos + newline + 1
                else:
                    # Try space
                    space = chunk.rfind(" ")
                    if space > max_length // 2:
                        end_pos = current_pos + space + 1

        yield content[current_pos:end_pos]
        current_pos = end_pos
