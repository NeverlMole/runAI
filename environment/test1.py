#!/usr/bin/env python3
"""
Example of how bodies interact with each other. For a body to be able to
move it needs to have joints. In this example, the "robot" is a red ball
with X and Y slide joints (and a Z slide joint that isn't controlled).
On the floor, there's a cylinder with X and Y slide joints, so it can
be pushed around with the robot. There's also a box without joints. Since
the box doesn't have joints, it's fixed and can't be pushed around.
"""
from mujoco_py import load_model_from_xml, MjSim, MjViewer, MjViewerBasic
import math
import os
import sys
import agent

model = load_model_from_xml(open(sys.argv[1], "r").read())
sim = MjSim(model)
viewer = MjViewer(sim)
t = 0
while True:
    t += 1
    action = agent.getAction()
    if action:
    	sim.data.ctrl[0] = 100
    else:
    	sim.data.ctrl[0] = 0
    sim.step()
    viewer.render()
    if t > 100 and os.getenv('TESTING') is not None:
        break
