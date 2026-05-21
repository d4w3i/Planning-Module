from pydantic import BaseModel, Field
from typing import Literal


class Change(BaseModel):
    type: Literal["add", "delete"] = Field(
        ..., description="The type of change (add or delete)."
    )
    content: str = Field(
        ...,
        description="The perfect prompt for the change, without any additional text.",
    )
    line: int = Field(..., description="The line number of the change.")


class Hunk(BaseModel):
    start_line: int = Field(..., description="The starting line number of the hunk.")
    changes: list[Change] = Field(..., description="A list of changes in the hunk.")


class FilePlan(BaseModel):
    file: str = Field(..., description="The path of the file being changed.")
    hunks: list[Hunk] = Field(..., description="A list of hunks in the file.")


class FinalPlan(BaseModel):
    files: list[FilePlan] = Field(
        ..., description="A list of files with their respective hunks and changes."
    )
