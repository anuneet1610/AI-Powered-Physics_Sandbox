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
    id: str
    role_label: str
    pos_label: str
    x: int
    y: int
    vx: int
    vy: int
    radius: int
    color: str = Field(description="Color name or hex code, e.g. 'red' or '#ff0000'")

class RectConfig(BaseModel):
    id: str
    role_label: str
    pos_label: str
    x: int
    y: int
    vx: int
    vy: int
    width: int
    height: int
    color: str = Field(description="Color name or hex code, e.g. 'red' or '#ff0000'")

class SpringConfig(BaseModel):
    id: str
    role_label: str
    pos_label: str
    obj1_type: Literal["ball", "rect"]
    obj1_index: int
    obj2_type: Literal["ball", "rect"]
    obj2_index: int
    # stiffness: float
    # damping: float
    # rest_length: float | None = None  # None = compute from initial distance

class WallConfig(BaseModel):
    id: str
    role_label: str
    pos_label: str
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
            - Typical sizes: ball radius 1-2, rect width/length 2-8.
            - Typical velocities: 1-15 units/sec.
            - Walls are static, purely collidable line segments defined by two endpoints
              (x1, y1) and (x2, y2). They have no velocity and never move.
            - Walls are used for ramps, barriers, funnels, enclosures, or any fixed
              boundary the user describes (e.g. "a ramp", "a wall behind the ball",
              "a V-shaped funnel"). Only add walls the user's request implies.

            For each object, reason explicitly about:
            - a short stable identifier for it (e.g. "ball_launch", "rect_target", "wall_ramp")
            - its role in the scene (e.g. "the ball being launched", "the target",
              "the ramp the ball rolls down")
            - its starting position and why (for walls: its two endpoints and why),
              described in a short human-readable phrase (e.g. "bottom-left corner",
              "resting on the ground at x=10", "midair above the ramp")
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

            For every object (ball, rect, wall) also fill in:
            - id: a short, unique, stable snake_case identifier for this object
              (e.g. "ball_launch", "rect_target_1", "wall_ramp"). Reuse the identifier
              implied by the reasoning if one was given; otherwise invent a concise one.
              No two objects may share the same id.
            - role_label: a short human-readable label describing this object's role
              in the scene (e.g. "Launched ball", "Target block", "Ramp").
            - pos_label: a short human-readable phrase describing where the object
              starts (e.g. "Bottom-left corner", "Resting on the ground", "Midair, x=10").

            These three fields must be populated for every object, based on the reasoning above.
        """),
        HumanMessage(content=f"""
            Original request: {state['message']}

            Reasoning:
            {reasoning_response.content}
        """)
    ])

    state["balls"] = [
        [b.id, b.role_label, b.pos_label, b.x, b.y, b.vx, b.vy, b.radius, b.color]
        for b in plan.balls
    ]
    state["rects"] = [
        [r.id, r.role_label, r.pos_label, r.x, r.y, r.vx, r.vy, r.width, r.height, r.color]
        for r in plan.rects
    ]
    state["walls"] = [
        [w.id, w.role_label, w.pos_label, w.x1, w.y1, w.x2, w.y2]
        for w in plan.walls
    ]

    print(state["balls"])
    print(state["rects"])
    print(state["walls"])

    return state


class SpringPlanOutput(BaseModel):
    springs: list[SpringConfig] = Field(default_factory=list)

def describe_objects(balls, rects):
    ans = []
    for i in range(len(balls)):
        _id, role, pos, x, y, vx, vy, radius, color = balls[i]
        ans.append(f"Ball {i} (id={_id}, role={role}): x={x}, y={y}, vx={vx}, vy={vy}, radius={radius}")

    for i in range(len(rects)):
        _id, role, pos, x, y, vx, vy, width, height, color = rects[i]
        ans.append(f"Rectangle {i} (id={_id}, role={role}): x={x}, y={y}, vx={vx}, vy={vy}, width={width}, height={height}")

    return ans

def planner_springs(state: State) -> State:
    manifest = describe_objects(state["balls"], state["rects"])

    structured_llm = llm.with_structured_output(SpringPlanOutput)
    plan = structured_llm.invoke([
        SystemMessage(content=f"""
            You are adding springs to an existing mechanical scene. Springs connect
            two existing objects by index. Here are the objects currently in the scene:

            {manifest}

            IMPORTANT: Collisions (objects hitting, bouncing off, or colliding with
            each other) are handled automatically by the physics engine. Do NOT add
            a spring just because two objects are described as colliding, meeting,
            hitting, or interacting in mid-air. Only add a spring if the user
            EXPLICITLY asks for objects to be tethered, attached, connected, or
            linked by a spring (e.g. "connect the ball to the platform with a
            spring", "attach them with an elastic tether").

            Example: "two balls launched to collide mid-air" -> NO spring (they
            physically collide; nothing needs to be attached).
            Example: "a ball tethered to a wall by a spring" -> 1 spring.

            If in doubt, return an empty list. A missing spring is a much smaller
            mistake than a fabricated one.

            For every spring you DO add, also fill in:
            - id, role_label, pos_label as before.
        """),
        state["message"]
    ])
    return {"springs": [s.dict() for s in plan.springs]}

    print(plan)

    return {"springs": [s.dict() for s in plan.springs]}

