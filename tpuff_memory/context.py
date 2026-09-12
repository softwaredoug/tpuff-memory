from pydantic import BaseModel


class RequestContext(BaseModel):
    question: str
    question_id: int
