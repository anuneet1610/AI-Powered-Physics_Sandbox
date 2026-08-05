from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Literal
from langchain_core.messages import BaseMessage
from langchain_core.messages import SystemMessage, HumanMessage
from sandbox_objects.bodies import Ball
from sandbox_objects.bodies import Rectangle
from sandbox_objects.spring import Spring
from sandbox_objects.wall import Wall
from dotenv import load_dotenv

load_dotenv()

llm = init_chat_model("google_genai:gemini-3.1-flash-lite")

from typing import Optional

# ------------------------
# Edit operation schemas
# ------------------------

class BallEdit(BaseModel):
    op: Literal["add", "update", "delete"]
    id: str = Field(description="Existing object id for update/delete. For add, a new unique snake_case id.")
    role_label: Optional[str] = None
    pos_label: Optional[str] = None
    x: Optional[float] = None
    y: Optional[float] = None
    vx: Optional[float] = None
    vy: Optional[float] = None
    radius: Optional[float] = None
    color: Optional[str] = None

class RectEdit(BaseModel):
    op: Literal["add", "update", "delete"]
    id: str
    role_label: Optional[str] = None
    pos_label: Optional[str] = None
    x: Optional[float] = None
    y: Optional[float] = None
    vx: Optional[float] = None
    vy: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None
    color: Optional[str] = None

class WallEdit(BaseModel):
    op: Literal["add", "update", "delete"]
    id: str
    role_label: Optional[str] = None
    pos_label: Optional[str] = None
    x1: Optional[float] = None
    y1: Optional[float] = None
    x2: Optional[float] = None
    y2: Optional[float] = None

class SpringEdit(BaseModel):
    op: Literal["add", "update", "delete"]
    id: str
    role_label: Optional[str] = None
    pos_label: Optional[str] = None
    obj1_id: Optional[str] = Field(default=None, description="id of first connected object, only for add")
    obj2_id: Optional[str] = Field(default=None, description="id of second connected object, only for add")

class EditPlanOutput(BaseModel):
    balls: list[BallEdit] = Field(default_factory=list)
    rects: list[RectEdit] = Field(default_factory=list)
    walls: list[WallEdit] = Field(default_factory=list)
    springs: list[SpringEdit] = Field(default_factory=list)

class EditState(TypedDict):
    message: str
    scene_description: str
    edit_plan: EditPlanOutput | None

def edit_planner(state: EditState) -> EditState:
    reasoning_response = llm.invoke([
        SystemMessage(content="""
            You are editing an existing 2D physics scene based on a user's request.
            You will be given the current scene (each object's id, role_label,
            pos_label, and physical parameters) and the user's requested change.

            World parameters:
            - World bounds: x in [0, 50], y in [0, 30], y increases UPWARD.

            For each change, reason step-by-step about:
            - which existing object(s) the user is referring to (use id, role_label,
              and pos_label together to resolve references like "the middle ball"
              or "the ball I launched")
            - whether this requires adding a new object, updating an existing one,
              or deleting one
            - if updating: which specific fields change, and to what new values
              (leave everything else untouched)
            - if adding: full parameters for the new object, plus a fresh unique id

            Only touch objects/fields the user's request actually implies.
            Do not modify anything not mentioned or clearly implied.
            Write this as prose reasoning. Do not output JSON yet.
        """),
        HumanMessage(content=f"""
            Current scene:
            {state['scene_description']}

            Requested change: {state['message']}
        """)
    ])

    print(reasoning_response.content)

    structured_llm = llm.with_structured_output(EditPlanOutput)
    plan = structured_llm.invoke([
        SystemMessage(content="""
            Convert the following reasoning into structured edit operations.
            For each operation:
            - op: "add", "update", or "delete"
            - id: existing object's id for update/delete; a new unique snake_case id for add
            - only set fields that actually change (leave others null) for "update"
            - set all relevant fields for "add"
            - for "delete", only id is required

            Do not invent operations not implied by the reasoning.
        """),
        HumanMessage(content=f"""
            Current scene:
            {state['scene_description']}

            Requested change: {state['message']}

            Reasoning:
            {reasoning_response.content}
        """)
    ])

    state["edit_plan"] = plan
    return state