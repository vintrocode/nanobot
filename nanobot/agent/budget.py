"""Spend budget tracking for API cost management."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SpendBudget:
    """
    Tracks API spend across agent and subagent calls.

    Shared between parent agent and subagents to enforce a total spend limit
    for a single request.
    """

    max_spend_dollars: float
    spent_dollars: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    _calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def remaining_dollars(self) -> float:
        """Remaining budget in dollars."""
        return max(0, self.max_spend_dollars - self.spent_dollars)

    @property
    def is_exhausted(self) -> bool:
        """Whether the budget has been fully spent."""
        return self.spent_dollars >= self.max_spend_dollars

    @property
    def utilization(self) -> float:
        """Percentage of budget used (0.0 to 1.0+)."""
        if self.max_spend_dollars <= 0:
            return 0.0
        return self.spent_dollars / self.max_spend_dollars

    def add_cost(
        self,
        cost: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        source: str = "agent",
    ) -> None:
        """
        Add cost from an LLM response.

        Args:
            cost: Cost in dollars (from LLMResponse.cost)
            input_tokens: Number of input/prompt tokens
            output_tokens: Number of output/completion tokens
            source: Label for tracking (e.g., "agent", "subagent:task_id")
        """
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.spent_dollars += cost

        self._calls.append({
            "source": source,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
        })

    def get_summary(self) -> str:
        """Get a human-readable summary of spend."""
        return (
            f"${self.spent_dollars:.4f} of ${self.max_spend_dollars:.2f} "
            f"({self.input_tokens:,} in, {self.output_tokens:,} out)"
        )

    def get_detailed_summary(self) -> str:
        """Get a detailed breakdown of spend by source."""
        lines = [f"Total: {self.get_summary()}"]
        if self._calls:
            lines.append("Calls:")
            for call in self._calls:
                lines.append(
                    f"  - {call['source']}: ${call['cost']:.4f} "
                    f"({call['input_tokens']:,} in, {call['output_tokens']:,} out)"
                )
        return "\n".join(lines)
