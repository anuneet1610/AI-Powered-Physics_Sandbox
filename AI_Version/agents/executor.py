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
    state["ball_obj"] = [Ball(ball[0], ball[1], ball[2], ball[3], radius = ball[4], colour = ball[5]) for ball in state["balls"]]
    state["rect_obj"] = [Rectangle(rect[0], rect[1], rect[2], rect[3], length=rect[4], width=rect[5], colour=rect[6]) for rect in state["rects"]]
    state["wall_obj"] = [Wall(wall[0], wall[1], wall[2], wall[3]) for wall in state["walls"]]
    return state

def executor_springs(state: State) -> State:
    spring_objs = []
    for cfg in state["springs"]:
        obj1 = state["ball_obj"][cfg["obj1_index"]] if cfg["obj1_type"] == "ball" else state["rect_obj"][cfg["obj1_index"]]
        obj2 = state["ball_obj"][cfg["obj2_index"]] if cfg["obj2_type"] == "ball" else state["rect_obj"][cfg["obj2_index"]]
        spring_objs.append(Spring(obj1, obj2))
    state["spring_obj"] = spring_objs
    return state
