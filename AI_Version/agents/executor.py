from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from sandbox_objects.bodies import Rectangle
from sandbox_objects.bodies import Ball
from sandbox_objects.wall import Wall
from sandbox_objects.spring import Spring

class State(TypedDict):
    message: BaseMessage
    balls: list[list[int | str]] | None
    rects: list[list[int | str]] | None
    springs: list[dict] | None
    walls: list[list[int]] | None
    ball_obj: list[Ball] | None
    rect_obj: list[Rectangle] | None
    spring_obj: list[Spring] | None
    wall_obj: list[Wall] | None

def executor_bodies(state: State) -> State:
    # ball layout: [id, role_label, pos_label, x, y, vx, vy, radius, color]
    state["ball_obj"] = [
        Ball(
            x=ball[3], y=ball[4], vx=ball[5], vy=ball[6],
            radius=ball[7], colour=ball[8],
            id=ball[0], role_label=ball[1], pos_label=ball[2],
        )
        for ball in state["balls"]
    ]
    # rect layout: [id, role_label, pos_label, x, y, vx, vy, width, height, color]
    state["rect_obj"] = [
        Rectangle(
            x=rect[3], y=rect[4], vx=rect[5], vy=rect[6],
            length=rect[7], width=rect[8], colour=rect[9],
            id=rect[0], role_label=rect[1], pos_label=rect[2],
        )
        for rect in state["rects"]
    ]
    # wall layout: [id, role_label, pos_label, x1, y1, x2, y2]
    state["wall_obj"] = [
        Wall(
            x1=wall[3], y1=wall[4], x2=wall[5], y2=wall[6],
            id=wall[0], role_label=wall[1], pos_label=wall[2],
        )
        for wall in state["walls"]
    ]
    return state

def executor_springs(state: State) -> State:
    spring_objs = []
    for cfg in state["springs"]:
        obj1 = state["ball_obj"][cfg["obj1_index"]] if cfg["obj1_type"] == "ball" else state["rect_obj"][cfg["obj1_index"]]
        obj2 = state["ball_obj"][cfg["obj2_index"]] if cfg["obj2_type"] == "ball" else state["rect_obj"][cfg["obj2_index"]]
        spring_objs.append(
            Spring(
                ball_a=obj1, ball_b=obj2,
                id=cfg.get("id"),
                role_label=cfg.get("role_label"),
                pos_label=cfg.get("pos_label"),
            )
        )
    state["spring_obj"] = spring_objs
    return state
