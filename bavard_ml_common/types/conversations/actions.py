import typing as t
from enum import Enum

from pydantic import BaseModel


class TagValue(BaseModel):
    """Represents a named entity's type and value.
    """
    tagType: str
    value: str


class Actor(Enum):
    USER = 'USER'
    AGENT = 'AGENT'
    HUMAN_AGENT = 'HUMAN_AGENT'


class Sentiment(Enum):
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    POSITIVE = "POSITIVE"


class UserAction(BaseModel):
    """Represents any type of UserAction (email, utterance, option, etc).
    """
    type: str
    utterance: t.Optional[str]
    translatedUtterance: t.Optional[str]
    intent: t.Optional[str]
    sentiment: t.Optional[Sentiment]
    confidence: t.Optional[float]
    ood: t.Optional[bool]
    oodConfidence: t.Optional[float]
    tags: t.Optional[t.List[TagValue]]


class AgentAction(BaseModel):
    """Represents any type of agent action (form, utterance, email, etc.)
    """
    type: str
    name: str  # the action's name
    utterance: t.Optional[str]


class HumanAgentAction(BaseModel):
    type: str
    utterance: t.Optional[str]
