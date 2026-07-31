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

llm = init_chat_model("google_genai:gemini-3.1-flash-lite")

class BallConfig(BaseModel):
    x: int
    y: int
    vx: int
    vy: int
    radius: int
    color: str = Field(description="Color name or hex code, e.g. 'red' or '#ff0000'")

class RectConfig(BaseModel):
    x: int
    y: int
    vx: int
    vy: int
    width: int
    height: int
    color: str = Field(description="Color name or hex code, e.g. 'red' or '#ff0000'")

class SpringConfig(BaseModel):
    obj1_type: Literal["ball", "rect"]
    obj1_index: int
    obj2_type: Literal["ball", "rect"]
    obj2_index: int
    # stiffness: float
    # damping: float
    # rest_length: float | None = None  # None = compute from initial distance

# def planner_bodies(state: State) -> State:
#     structured_llm = llm.with_structured_output(PlanOutput)
#
#     plan = structured_llm.invoke([
#         SystemMessage(content="""
#             You specialize in designing classical mechanics systems. You are given the request of a user, and you need to help him
#             design a system for it. The system can only consist of balls and rectangles. You need to determine how many balls, rectangles and springs
#             the user wants to insert, and their configuration. For balls and rectangles, the configurations required are (position, velocity, size, color), whereas for springs
#             the requirements are the two objects
#
#             Guidelines:
#             - Coordinates should be reasonable for a simulation canvas (e.g. 0-800 for x, 0-600 for y).
#             - Balls need x, y (center), vx, vy, radius, color.
#             - Rectangles need x, y (top-left), vx, vy, width, height, color.
#             - Velocities should be reasonable. Keep it between -30 and +30.
#             - Infer sensible positions, velocities and sizes from the user's description even if they don't give exact numbers.
#             - Do not add any objects the user didn't ask for.
#             - Keep the radius of the balls under 2
#
#             World parameters:
#             - World bounds: x ∈ [0, 50], y ∈ [0, 30]
#             - y increases UPWARD. Ground is at y=0, ceiling at y=30.
#             - Gravity pulls objects toward y=0.
#             - A ball "resting/rolling on the ground" has y ≈ its radius, vy ≈ 0.
#             - A rectangle "resting on the ground" has y ≈ half its width, vy ≈ 0.
#             - Typical object sizes: ball radius 1-3, rectangle width/length 2-8 (this is a small 50x30 world, not pixels).
#             - Typical velocities: 1-15 units/sec. Do not exceed ~20 or objects will fly out of bounds almost immediately.
#             - "Opposite directions, same velocity" = equal magnitude vx, opposite sign.
#         """),
#         state["message"]
#     ])
#
#     state["balls"] = [[b.x, b.y, b.vx, b.vy, b.radius, b.color] for b in plan.balls]
#     state["rects"] = [[r.x, r.y, r.vx, r.vy, r.width, r.height, r.color] for r in plan.rects]
#
#     print(state["balls"])
#     print(state["rects"])
#
#     return state

class WallConfig(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

class PlanOutput(BaseModel):
    balls: list[BallConfig] = Field(default_factory=list)
    rects: list[RectConfig] = Field(default_factory=list)
    walls: list[WallConfig] = Field(default_factory=list)


def planner_bodies(state: State) -> State:
    # --- Step 1: reasoning, free text, no schema constraint ---
    reasoning_response = llm.invoke([
        SystemMessage(content="""
            You specialize in designing classical mechanics systems. Given the user's request,
            think step-by-step about the physical setup before producing any numbers.

            World parameters:
            - World bounds: x in [0, 50], y in [0, 30]
            - y increases UPWARD. Ground is at y=0, ceiling at y=30.
            - A ball "resting/rolling on the ground" has y = its radius, vy = 0.
            - A rectangle "resting on the ground" has y = half its width, vy = 0.
            - Typical sizes: ball radius 1-3, rect width/length 2-8.
            - Typical velocities: 1-15 units/sec.
            - Walls are static, purely collidable line segments defined by two endpoints
              (x1, y1) and (x2, y2). They have no velocity and never move.
            - Walls are used for ramps, barriers, funnels, enclosures, or any fixed
              boundary the user describes (e.g. "a ramp", "a wall behind the ball",
              "a V-shaped funnel"). Only add walls the user's request implies.

            For each object, reason explicitly about:
            - its role in the scene (e.g. "the ball being launched", "the target",
              "the ramp the ball rolls down")
            - its starting position and why (for walls: its two endpoints and why)
            - its velocity in x and y direction (magnitude and direction) and why
              (walls have no velocity — skip this for walls)
            - how it relates to the other objects (same line? perpendicular? timed to meet?
              does a wall block/redirect a specific object?)

            Write this as prose reasoning. Do not output JSON yet.
        """),
        state["message"]
    ])

    print(reasoning_response.content)

    # --- Step 2: extraction, structured output, grounded in step 1's reasoning ---
    structured_llm = llm.with_structured_output(PlanOutput)

    plan = structured_llm.invoke([
        SystemMessage(content="""
            Convert the following physical reasoning into the exact structured object list.
            Use the numbers and positions already decided in the reasoning — do not invent new ones.
        """),
        HumanMessage(content=f"""
            Original request: {state['message']}

            Reasoning:
            {reasoning_response.content}
        """)
    ])

    state["balls"] = [[b.x, b.y, b.vx, b.vy, b.radius, b.color] for b in plan.balls]
    state["rects"] = [[r.x, r.y, r.vx, r.vy, r.width, r.height, r.color] for r in plan.rects]
    state["walls"] = [[w.x1, w.y1, w.x2, w.y2] for w in plan.walls]

    print(state["balls"])
    print(state["rects"])
    print(state["walls"])

    return state


class SpringPlanOutput(BaseModel):
    springs: list[SpringConfig] = Field(default_factory=list)

def describe_objects(balls, rects):
    ans = []
    for i in range(len(balls)):
        ans.append(f"Ball {i}: x={balls[i][0]}, y={balls[i][1]}, vx={balls[i][2]}, vy={balls[i][3]}, radius={balls[i][4]}")

    for i in range(len(rects)):
        ans.append(f"Rectangle {i}, x={rects[i][0]}, y={rects[i][1]}, vx={rects[i][2]}, vy={rects[i][3]}, width={rects[i][4]}, height={rects[i][5]}")

def planner_springs(state: State) -> State:
    manifest = describe_objects(state["balls"], state["rects"])  # human-readable summary

    structured_llm = llm.with_structured_output(SpringPlanOutput)
    plan = structured_llm.invoke([
        SystemMessage(content=f"""
            You are adding springs to an existing mechanical scene. Springs connect
            two existing objects by index. Here are the objects currently in the scene:

            {manifest}

            Reference objects only by their type ("ball" or "rect") and index as listed.
            Understand the user query and only add springs the user's request implies.
            Don't add springs the user hasn't requested for.
        """),
        state["message"]
    ])

    return {"springs": [s.dict() for s in plan.springs]}

