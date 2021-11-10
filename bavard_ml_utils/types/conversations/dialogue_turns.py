import typing as t

from pydantic import BaseModel, Field

from bavard_ml_utils.types.conversations.actions import Actor, AgentAction, HumanAgentAction, UserAction


class DialogueState(BaseModel):
    slotValues: t.Dict[str, t.Any]


class BaseDialogueTurn(BaseModel):
    state: t.Optional[DialogueState]
    actor: Actor


class UserDialogueTurn(BaseDialogueTurn):
    userAction: UserAction
    actor = Field(Actor.USER, const=True)


class AgentDialogueTurn(BaseDialogueTurn):
    agentAction: AgentAction
    actor = Field(Actor.AGENT, const=True)


class HumanAgentDialogueTurn(BaseDialogueTurn):
    humanAgentAction: HumanAgentAction
    actor = Field(Actor.HUMAN_AGENT, const=True)


DialogueTurn = t.Union[AgentDialogueTurn, HumanAgentDialogueTurn, UserDialogueTurn]
