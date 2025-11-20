from typing import list
from pydantic import BaseModel, Field

class source(BaseModel):
    """Schema for a source used by the agent"""
    source: str = Field(description="Source of the answer")

class AgentResponse(BaseModel):
    """Schema for the agent response with answers and sources"""
    output: str = Field(description="Final answer to the question")
    source: List[str] = Field(description="List of sources used to answer the question")