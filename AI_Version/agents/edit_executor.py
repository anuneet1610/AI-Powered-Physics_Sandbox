from typing import Optional
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Literal
from langchain_core.messages import BaseMessage
from langchain_core.messages import SystemMessage, HumanMessage
from sandbox_objects.bodies import Ball, Rectangle
from sandbox_objects.wall import Wall
from sandbox_objects.spring import Spring
import re

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

_COLOR_NAMES = {
    "red": (255, 80, 80),
    "green": (80, 255, 80),
    "blue": (80, 80, 255),
    "yellow": (255, 255, 80),
    "orange": (255, 165, 60),
    "purple": (180, 80, 255),
    "pink": (255, 150, 200),
    "white": (255, 255, 255),
    "black": (20, 20, 20),
    "gray": (150, 150, 150),
    "grey": (150, 150, 150),
    "cyan": (80, 255, 255),
    "brown": (150, 100, 60),
}

def _parse_color(color, default=(200, 200, 200)):
    if not color:
        return default

    color = color.strip().lower()

    # Hex form: "#ff0000" or "ff0000"
    hex_match = re.fullmatch(r"#?([0-9a-f]{6})", color)
    if hex_match:
        hex_str = hex_match.group(1)
        r = int(hex_str[0:2], 16)
        g = int(hex_str[2:4], 16)
        b = int(hex_str[4:6], 16)
        return (r, g, b)

    # Named color
    if color in _COLOR_NAMES:
        return _COLOR_NAMES[color]

    return default

def apply_scene_edits(plan: EditPlanOutput, balls, rects, walls, springs):
    _apply_ball_edits(plan.balls, balls)
    _apply_rect_edits(plan.rects, rects)
    _apply_wall_edits(plan.walls, walls)
    _apply_spring_edits(plan.springs, springs, balls, rects)

def _apply_ball_edits(edits, balls):
    by_id = {b.id: b for b in balls}
    for e in edits:
        if e.op == "delete":
            balls[:] = [b for b in balls if b.id != e.id]
        elif e.op == "update" and e.id in by_id:
            obj = by_id[e.id]
            for field in ["role_label", "pos_label", "x", "y", "vx", "vy", "radius", "color"]:
                val = getattr(e, field)
                if val is not None:
                    setattr(obj, field, val)
        elif e.op == "add":
            new_ball = Ball(
                x=e.x, y=e.y, vx=e.vx or 0, vy=e.vy or 0,
                mass=5, radius=e.radius or 0.5,
                colour=_parse_color(e.color),
                id=e.id,
                role_label=e.role_label or e.id,
                pos_label=e.pos_label or "",
            )
            balls.append(new_ball)

def _apply_rect_edits(edits, rects):
    by_id = {rect.id: rect for rect in rects}
    for e in edits:
        if e.op == "delete":
            rects[:] = [rect for rect in rects if rect.id != e.id]
        elif e.op == "update" and e.id in by_id:
            obj = by_id[e.id]
            for field in ["role_label", "pos_label", "x", "y", "vx", "vy", "length", "width", "color"]:
                val = getattr(e, field)
                if val is not None:
                    setattr(obj, field, val)
        elif e.op == "add":
            new_rect = Rectangle(
                x=e.x, y=e.y, vx=e.vx or 0, vy=e.vy or 0,
                mass=5, length=e.length or 2,
                width=e.width or 1,
                colour=_parse_color(e.color),
                id=e.id,
                role_label=e.role_label or e.id,
                pos_label=e.pos_label or "",
            )
            new_rect.id = e.id
            new_rect.role_label = e.role_label or e.id
            rects.append(new_rect)

def _apply_wall_edits(edits, walls):
    by_id = {wall.id: wall for wall in walls}
    for e in edits:
        if e.op == "delete":
            walls[:] = [wall for wall in walls if wall.id != e.id]
        elif e.op == "update" and e.id in by_id:
            obj = by_id[e.id]
            for field in ["role_label", "pos_label", "x1", "x2", "y1", "y2"]:
                val = getattr(e, field)
                if val is not None:
                    setattr(obj, field, val)
        elif e.op == "add":
            new_wall = Wall(
                x1 = e.x1, x2 = e.x2, y1 = e.y1,
                y2 = e.y2,
                id=e.id,
                role_label=e.role_label or e.id,
                pos_label=e.pos_label or "",
            )
            new_wall.id = e.id
            new_wall.role_label = e.role_label or e.id
            walls.append(new_wall)

def _apply_spring_edits(edits, springs, balls, rects):
    lookup = {o.id: o for o in balls + rects}
    for e in edits:
        if e.op == "delete":
            springs[:] = [s for s in springs if s.id != e.id]
        elif e.op == "update":
            for sp in springs:
                if sp.id == e.id:
                    if e.role_label is not None:
                        sp.role_label = e.role_label
                    if e.pos_label is not None:
                        sp.pos_label = e.pos_label
        elif e.op == "add":
            a, b = lookup.get(e.obj1_id), lookup.get(e.obj2_id)
            if a and b:
                new_spring = Spring(
                    ball_a=a, ball_b=b,
                    id=e.id, role_label=e.role_label or e.id,
                    pos_label=e.pos_label or ""
                )
                springs.append(new_spring)