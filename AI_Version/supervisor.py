from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage
from sandbox_objects.bodies import Ball
from sandbox_objects.bodies import Rectangle
from sandbox_objects.spring import Spring
from sandbox_objects.wall import Wall
from sandbox_objects.main import run_pygame

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

from agents.planner import planner_bodies, planner_springs
from agents.executor import executor_bodies, executor_springs

builder = StateGraph(State)
builder.add_node("planner_bodies", planner_bodies)
builder.add_node("planner_springs", planner_springs)
builder.add_node("executor_bodies", executor_bodies)
builder.add_node("executor_springs", executor_springs)
# builder.add_node("run_headless", run_headless)

builder.add_edge(START, "planner_bodies")
builder.add_edge("planner_bodies", "executor_bodies")
builder.add_edge("executor_bodies", "planner_springs")
builder.add_edge("planner_springs", "executor_springs")
builder.add_edge("executor_springs", END)
# builder.add_edge("run_headless", END)

graph = builder.compile()

if __name__ == "__main__":
    final_state = graph.invoke({"message": "Design a mechanical system, where a ball rolls down a ramp, and then hits another ball"})
    run_pygame(final_state)