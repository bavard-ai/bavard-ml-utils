from pydantic import BaseModel


class StrPredWithConf(BaseModel):
    value: str
    confidence: float
