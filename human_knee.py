
import math
from physics import *

def floor(points):
    for point in points:
        if point.position[1] < 0:
            point.position[1] = 0
            point.velocity[1] = 0
            point.position[0] -= point.velocity[0]*Map.timestep

def gravity(points):
    for point in points:
        point.force[1] -= point.mass*9.81

all = open("all.csv", "w")

for angle in range(60, 88, 2):
    L = 0.9
    theta = angle*math.pi/180.
    planted = Point([0.,0.,0.], 5.)
    knee = Point([-math.cos(theta)*L,math.sin(theta)*L,0.], 70.)
    Spring(planted, knee, 25000)
    knee.velocity = np.array([2.68,-0.25,0.])

    UniversalForce(floor)

    center = False
    crossover_time = 0
    fly_time = 0
    while Map.current_time < 0.20:
        Map.step()
        if not crossover_time:
            if knee.position[0] > 0:
                crossover_time = Map.current_time
                center = True
        if not fly_time:
            if planted.position[1] > 0.01:
                fly_time = Map.current_time
                all.write(str(angle) + "," + str(Map.current_time) + "," + str(knee.velocity[0]) + "," + str(knee.velocity[1]) + "\n")

    time = []
    position_v = []
    velocity_v = []
    position_h = []
    velocity_h = []
    force_h = []
    force_v = []
    for i in range(int(Map.current_time*200)):
        index = int(i*0.005/Map.timestep)
        time.append(i*0.005)
        position_h.append(knee.history.position[index][0])
        position_v.append(knee.history.position[index][1])
        velocity_h.append(knee.history.velocity[index][0])
        velocity_v.append(knee.history.velocity[index][1])
        force_h.append(knee.history.force[index][0])
        force_v.append(knee.history.force[index][1])

    f = open("data_" + str(angle) + ".csv", "w")

    f.write(str(angle) + "," + str(crossover_time) + "," + str(fly_time) + "\n")
    f.write(str(time[-1]) + "," + str(position_h[-1]) + "," + str(position_v[-1]) + "," + str(velocity_h[-1]) + "," + str(velocity_v[-1]) + "," + str(force_h[-1]) + "," + str(force_v[-1]) + "\n\n")

    for i in range(len(position_h)):
        f.write(str(time[i]) + "," + str(position_h[i]) + "," + str(position_v[i]) + "," + str(velocity_h[i]) + "," + str(velocity_v[i]) +  "," + str(force_h[i]) + "," + str(force_v[i]) + "\n")
    f.close()

    Map.reset()

all.close()