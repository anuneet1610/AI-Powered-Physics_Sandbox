from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from sandbox_objects.bodies import Rectangle
from sandbox_objects.bodies import Ball
import sandbox_objects.collisions as collisions
import matplotlib.pyplot as plt

class State(TypedDict):
    message: BaseMessage
    balls: list[list[int | str]] | None
    rects: list[list[int | str]] | None
    ball_obj: list[Ball] | None
    rect_obj: list[Rectangle] | None

def run_headless(state, ticks=2000, dt=1/60):
    balls = state["ball_obj"]
    rectangles = state["rect_obj"]

    print("Before")

    for i in range(len(balls)):
        print("Ball No.", i)
        print("x:", balls[i].x)
        print("y:", balls[i].y)
        print("vx:", balls[i].vx)
        print("vy:", balls[i].vy)
        print()

    for i in range(len(rectangles)):
        print("Rectangle No.", i)
        print("x:", rectangles[i].x)
        print("y:", rectangles[i].y)
        print("vx:", rectangles[i].vx)
        print("vy:", rectangles[i].vy)
        print()

    ball_y_coord = []
    ball_x_coord = []
    time = []
    for ball in balls:
        ball_y_coord.append([])
    for ball in balls:
        ball_x_coord.append([])
    # for _ in range(ticks):
    #     for ball in balls:
    #         ball.clear_forces()
    #     for rect in rectangles:
    #         rect.clear_forces()
    #     for ball in balls:
    #         ball.integrate(dt)
    #     for rect in rectangles:
    #         rect.integrate(dt)
    #     for ball in balls:
    #         for rect in rectangles:
    #             rect.check_collision_ball(ball)
    #     for i in range(len(balls)):
    #         for j in range(i + 1, len(balls)):
    #             balls[i].check_collision(balls[j])
    #     for i in range(len(rectangles)):
    #         for j in range(i + 1, len(rectangles)):
    #             rectangles[i].check_collision(rectangles[j])
    #     for ball in balls:
    #         collisions.clamp_ball_to_world(ball)
    #     for rect in rectangles:
    #         collisions.clamp_rect_to_world((rect))
    #
    #     time.append(_)
    #     for i in range(len(balls)):
    #         ball_y_coord[i].append(balls[i].y)
    #         ball_x_coord[i].append(balls[i].x)
    # plt.figure(figsize=(8, 8))
    #
    # for i, y_vals in enumerate(ball_y_coord):
    #     plt.plot(time, y_vals, label=f"Ball {i}")
    #
    # plt.xlabel("Time (s)")
    # plt.ylabel("y position")
    # plt.title("Object y-coordinate over time")
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.show()
    #
    # for i, x_vals in enumerate(ball_x_coord):
    #     plt.plot(time, x_vals, label=f"Ball {i}")
    #
    # plt.xlabel("Time (s)")
    # plt.ylabel("x position")
    # plt.title("Object x-coordinate over time")
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.show()

        # if _ % 100 == 0:
        #     print("Time:", _)
        #     for i in range(len(balls)):
        #         print("Ball No.", i)
        #         print("x:", balls[i].x)
        #         print("y:", balls[i].y)
        #         print("vx:", balls[i].vx)
        #         print("vy:", balls[i].vy)
        #         print()
        #
        #     for i in range(len(rectangles)):
        #         print("Rectangle No.", i)
        #         print("x:", rectangles[i].x)
        #         print("y:", rectangles[i].y)
        #         print("vx:", rectangles[i].vx)
        #         print("vy:", rectangles[i].vy)
        #         print()

    state["ball_obj"] = balls
    state["rect_obj"] = rectangles


    return state