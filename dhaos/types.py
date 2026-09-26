"""Types partagés entre backends, outils et agent.

Représentation *neutre* d'une conversation : chaque backend (Ollama, Claude)
convertit depuis/vers son propre format. Le prompt système est passé à part,
jamais dans la liste de messages.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

Role = Literal["user", "assistant", "tool"]

TextCallback = Callable[[str], None]


@dataclass
class ToolSpec:
    """Description d'un outil exposé au modèle (schéma JSON standard)."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema de type object


@dataclass
class ToolCall:
    """Appel d'outil demandé par le modèle."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    """Un tour de conversation.

    - ``user``      : ``content`` = texte de l'utilisateur.
    - ``assistant`` : ``content`` = texte, ``tool_calls`` = appels demandés,
      ``raw`` = charge utile propre au backend à rejouer telle quelle (par ex.
      les blocs de contenu Claude, *y compris* les blocs de réflexion, qui
      doivent être renvoyés inchangés lors d'un enchaînement d'outils).
    - ``tool``      : résultat d'un outil ; ``tool_call_id`` et ``name``
      identifient l'appel, ``is_error`` signale un échec.
    """

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    is_error: bool = False
    raw: Any = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [
                {"id": c.id, "name": c.name, "arguments": c.arguments} for c in self.tool_calls
            ]
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            d["name"] = self.name
        if self.is_error:
            d["is_error"] = True
        if self.raw is not None:
            d["raw"] = self.raw
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Message":
        return cls(
            role=d["role"],
            content=d.get("content", "") or "",
            tool_calls=[
                ToolCall(id=c["id"], name=c["name"], arguments=c.get("arguments") or {})
                for c in d.get("tool_calls", [])
            ],
            tool_call_id=d.get("tool_call_id"),
            name=d.get("name"),
            is_error=bool(d.get("is_error", False)),
            raw=d.get("raw"),
        )


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens


StopReason = Literal["end_turn", "tool_use", "max_tokens", "refusal", "error"]


@dataclass
class ChatResponse:
    """Réponse complète d'un backend pour un tour (le texte a déjà été
    diffusé via ``on_text`` pendant la génération)."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: StopReason = "end_turn"
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    raw: Any = None  # charge utile à rejouer (voir Message.raw)
    thinking: str = ""  # résumé de réflexion si le backend en fournit un

    def to_assistant_message(self) -> Message:
        return Message(
            role="assistant",
            content=self.text,
            tool_calls=list(self.tool_calls),
            raw=self.raw,
        )
