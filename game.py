"""
game.py
-------
A small 3D puzzle game built with the Ursina engine, controllable either by
keyboard (for testing without a webcam) or by hand gestures (via
gesture_recognizer.py running MediaPipe in a background thread).

Puzzle: push the glowing crate onto the marked target tile to win.

Controls (keyboard fallback):
    W / Up Arrow     -> move forward
    A / Left Arrow   -> turn left
    D / Right Arrow  -> turn right
    Space            -> jump
    S                -> stop

Gesture mapping (when --gestures flag is used):
    OPEN_PALM   -> move forward
    POINT_LEFT  -> turn left
    POINT_RIGHT -> turn right
    THUMBS_UP   -> jump
    FIST        -> stop
"""

import argparse
import threading
import sys

from ursina import (
    Ursina, Entity, color, camera, Vec3, Text, held_keys,
    destroy, application, time as ursina_time, window, scene
)

from gesture_recognizer import GestureState, run_gesture_thread


# ---------------------------------------------------------------------------
# Game setup
# ---------------------------------------------------------------------------

MOVE_SPEED = 4
TURN_SPEED = 90  # degrees per second
JUMP_FORCE = 6
GRAVITY = 14
GROUND_SIZE = 12

gesture_state = GestureState()
use_gestures = False


def build_level():
    ground = Entity(
        model="plane",
        scale=(GROUND_SIZE, 1, GROUND_SIZE),
        color=color.rgb(40, 40, 55),
        texture="white_cube",
        texture_scale=(GROUND_SIZE, GROUND_SIZE),
        collider="box",
    )

    # Walls (simple boundary)
    wall_color = color.rgb(70, 70, 90)
    half = GROUND_SIZE / 2
    Entity(model="cube", color=wall_color, position=(0, 1, half), scale=(GROUND_SIZE, 2, 0.3), collider="box")
    Entity(model="cube", color=wall_color, position=(0, 1, -half), scale=(GROUND_SIZE, 2, 0.3), collider="box")
    Entity(model="cube", color=wall_color, position=(half, 1, 0), scale=(0.3, 2, GROUND_SIZE), collider="box")
    Entity(model="cube", color=wall_color, position=(-half, 1, 0), scale=(0.3, 2, GROUND_SIZE), collider="box")

    # Target tile (where the crate needs to go)
    target = Entity(
        model="cube",
        color=color.azure,
        position=(4, 0.05, 4),
        scale=(1.5, 0.05, 1.5),
    )

    # Pushable crate
    crate = Entity(
        model="cube",
        color=color.orange,
        position=(2, 0.5, 1),
        scale=(1, 1, 1),
        collider="box",
    )

    return target, crate


class Player(Entity):
    def __init__(self, **kwargs):
        super().__init__(
            model="cube",
            color=color.lime,
            scale=(0.7, 1.2, 0.7),
            position=(-4, 1, -4),
            collider="box",
            **kwargs,
        )
        self.velocity_y = 0
        self.grounded = True
        self.win_text = None

    def apply_command(self, command):
        dt = ursina_time.dt

        if command in ("OPEN_PALM", "w"):
            self.position += self.forward * MOVE_SPEED * dt
        elif command in ("POINT_LEFT", "a"):
            self.rotation_y -= TURN_SPEED * dt
        elif command in ("POINT_RIGHT", "d"):
            self.rotation_y += TURN_SPEED * dt
        elif command in ("THUMBS_UP", "space"):
            if self.grounded:
                self.velocity_y = JUMP_FORCE
                self.grounded = False
        elif command in ("FIST", "s"):
            pass  # explicit stop, no movement
        # "NONE" -> no input this frame

    def update(self):
        # Gravity / simple jump arc
        self.velocity_y -= GRAVITY * ursina_time.dt
        self.y += self.velocity_y * ursina_time.dt
        if self.y <= 1:
            self.y = 1
            self.velocity_y = 0
            self.grounded = True

        # Determine input source
        if use_gestures:
            command = gesture_state.get()
        else:
            command = None
            if held_keys["w"] or held_keys["up arrow"]:
                command = "w"
            elif held_keys["a"] or held_keys["left arrow"]:
                command = "a"
            elif held_keys["d"] or held_keys["right arrow"]:
                command = "d"
            elif held_keys["s"]:
                command = "s"
            if held_keys["space"]:
                self.apply_command("space")

        if command:
            self.apply_command(command)


def check_win(crate, target, status_text):
    dist = (Vec3(crate.x, 0, crate.z) - Vec3(target.x, 0, target.z)).length()
    if dist < 0.6:
        status_text.text = "PUZZLE SOLVED! Crate is on the target."
        status_text.color = color.lime
        return True
    else:
        status_text.text = "Push the orange crate onto the blue tile."
        status_text.color = color.white
        return False


def push_crate(player, crate):
    """Very simple push logic: if player is close behind crate, nudge it forward."""
    to_crate = Vec3(crate.x - player.x, 0, crate.z - player.z)
    if to_crate.length() < 1.0:
        push_dir = to_crate.normalized()
        crate.x += push_dir.x * 0.03
        crate.z += push_dir.z * 0.03


def main():
    global use_gestures

    parser = argparse.ArgumentParser(description="Gesture/Keyboard controlled 3D puzzle demo")
    parser.add_argument("--gestures", action="store_true", help="Enable webcam hand-gesture control")
    parser.add_argument("--no-preview", action="store_true", help="Hide the OpenCV webcam preview window")
    args = parser.parse_args()

    use_gestures = args.gestures

    app = Ursina()
    window.title = "Gesture-Controlled 3D Puzzle Demo"
    window.fps_counter.enabled = True

    target, crate = build_level()
    player = Player()

    camera.position = (0, 14, -16)
    camera.rotation_x = 35

    status_text = Text(
        text="Push the orange crate onto the blue tile.",
        position=(-0.5, 0.45),
        scale=1.3,
        background=True,
    )

    mode_text = Text(
        text=f"Control mode: {'GESTURES' if use_gestures else 'KEYBOARD'}",
        position=(-0.5, 0.4),
        scale=1,
        background=True,
    )

    if use_gestures:
        t = threading.Thread(
            target=run_gesture_thread,
            args=(gesture_state,),
            kwargs={"show_preview": not args.no_preview},
            daemon=True,
        )
        t.start()
        print("[game] Gesture control thread started. Show your hand to the webcam.")
    else:
        print("[game] Keyboard control mode. Use W/A/D + Space.")

    def update():
        push_crate(player, crate)
        check_win(crate, target, status_text)

    app.run()


if __name__ == "__main__":
    main()
