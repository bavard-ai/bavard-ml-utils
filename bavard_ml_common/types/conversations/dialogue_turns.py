import typing as t

from pydantic import BaseModel

from bavard_ml_common.types.conversations.actions import Actor, AgentAction, UserAction, HumanAgentAction


class SlotValue(BaseModel):
    name: str
    value: str


class DialogueState(BaseModel):
    slotValues: t.List[SlotValue]


class BaseDialogueTurn(BaseModel):
    state: t.Optional[DialogueState]
    actor: Actor


class UserDialogueTurn(BaseDialogueTurn):
    userAction: UserAction
    actor = Actor.USER


class AgentDialogueTurn(BaseDialogueTurn):
    agentAction: AgentAction
    actor = Actor.AGENT


class HumanAgentDialogueTurn(BaseDialogueTurn):
    humanAgentAction: HumanAgentAction
    actor = Actor.HUMAN_AGENT


DialogueTurn = t.Union[AgentDialogueTurn, HumanAgentDialogueTurn, UserDialogueTurn]