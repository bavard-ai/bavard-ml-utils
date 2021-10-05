from pydantic import BaseModel


class StrPredWithConf(BaseModel):
    """
    Represents a string prediction that was made (:attr:`value`), along with the :attr:`confidence` of that prediction.
    """

    value: str
    confidence: float
