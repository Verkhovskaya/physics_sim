import numpy as np
import mpl_toolkits.mplot3d.axes3d as axes3d
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import math

def get_vector(start_point, end_point):
    return end_point.position - start_point.position

def get_vector_length(vector):
    norm = np.linalg.norm(vector)
    return norm

def get_distance(point_1, point_2):
    return get_vector_length(get_vector(point_1, point_2))

def to_unit_vector(vector):
    length = get_vector_length(vector)
    if abs(length) > 1e-16:
        return vector/length
    elif length > 0:
        return vector / (length + 1e-16)
    else:
        return vector / (length - 1e-16)

def get_unit_vector(start_point, end_point):
    return to_unit_vector(get_vector(start_point, end_point))

def get_angle(point_1, center, point_2):
    v1_u = get_unit_vector(center, point_1)
    v2_u = get_unit_vector(center, point_2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def get_torque_direction(point, center, other_point):
    v1 = get_vector(center, point)
    v2 = get_vector(center, other_point)
    cp1 = np.cross(v1, v2)
    cp2 = np.cross(v1, cp1)
    return to_unit_vector(cp2)

class Point:
    def __init__(self, position, mass, velocity=[0,0,0]):
        self.position = np.array([float(position[0]), float(position[1]), float(position[2])], dtype=np.double)
        self.velocity = np.array([float(velocity[0]), float(velocity[1]), float(velocity[2])], dtype=np.double)
        self.force = np.array([0.0,0.0,0.0], dtype=np.double)
        self.mass = float(mass)
        self.overrides = []
        Map.add_point(self)
        self.relationships = []
        self.history = PointHistory(position, velocity)
    
    def add_relationship(self, relationship):
        self.relationships.append(relationship)

    def apply_relationships(self):
        for relationship in self.relationships:
            self.force += relationship.calculate_force(self)

    def apply_overrides(self):
        for override in self.overrides:
            override(self)
    
    def update_velocity(self):
        self.velocity += self.force / self.mass * Map.timestep
    
    def update_position(self):
        self.position += self.velocity * Map.timestep
    
    def record(self):
        self.history.force.append(np.copy(self.force))
        self.history.velocity.append(np.copy(self.velocity))
        self.history.position.append(np.copy(self.position))


class PointHistory:
    def __init__(self, initial_position, initial_velocity):
        self.position = [initial_position]
        self.velocity = [initial_velocity]
        self.force = [np.array([0.0, 0.0, 0.0], dtype=np.double)]

class Map:
    points = []
    springs = []
    angular_springs = []
    planes = []

    current_time = 0.0
    timestep = 0.001
    num_steps = 0

    universal_overrides = []

    @staticmethod
    def reset():
        Map.points = []
        Map.springs = []
        Map.angular_springs = []
        Map.planes = []
        Map.current_time = 0.0
        Map.timestep = 0.001
        Map.num_steps = 0
        Map.universal_overrides = []

    @staticmethod
    def plot_3d(points):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        zline = np.array([0.0]*len(points))
        xline = np.array([0.0]*len(points))
        yline = np.array([0.0]*len(points))
        for i in range(len(points)):
            xline[i] = points[i][0]
            yline[i] = points[i][1]
            zline[i] = points[i][2]
        ax.plot3D(xline, yline, zline, 'red')
        plt.show()
    
    @staticmethod
    def plot_1d(points):
        plt.plot(points)
        plt.show()

    @staticmethod
    def animate_frame(iteration, data, timesteps_per_frame, points, lines):
        for i in range(len(Map.points)):
            position = Map.points[i].history.position[int(iteration*timesteps_per_frame)]
            points[i]._offsets3d = ([[position[0]], [position[1]], [position[2]]])
        for i in range(len(Map.springs)):
            p1 = Map.springs[i].point_1.history.position[int(iteration*timesteps_per_frame)]
            p2 = Map.springs[i].point_2.history.position[int(iteration*timesteps_per_frame)]
            lines[i][0]._verts3d = ([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]])
        return points + lines

    @staticmethod
    def add_point(new_point):
        Map.points.append(new_point)

    @staticmethod
    def step():
        for point in Map.points:
            point.force = np.array([0.0,0.0,0.0], dtype=np.double)
            point.apply_relationships()
            point.apply_overrides()
        for override in Map.universal_overrides:
            override(Map.points)
        for point in Map.points:
            point.update_velocity()
        for point in Map.points:
            point.update_position()
        for point in Map.points:
            point.record()
        Map.num_steps += 1
        Map.current_time += Map.timestep
    
    @staticmethod
    def run_simulation(length, increment=None):
        Map.gravity = gravity
        if increment:
            Map.timestep = increment
        num_steps = int(length/increment)
        print("Running simulation...")
        for i in range(num_steps):
            Map.step()
            if i % int(num_steps/10) == 0:
                print(str(int(i/(num_steps/10))*10)+"%")
        print("")
        print("Done!")

    
    @staticmethod
    def status():
        for point in Map.points:
            print(point.position)
            print(point.velocity)
            print()

    @staticmethod
    def close_animation():
        plt.close() #timer calls this function after 3 seconds and closes the window 

    @staticmethod
    def plot_path(FPS=100):
        points = [[[],[]] for point in Map.points]

        for i in range(int(FPS*Map.current_time)):
            t = i/FPS*1.
            index = int(t/Map.timestep)
            for j in range(len(points)):
                points[j][0].append(Map.points[j].history.position[index][0])
                points[j][1].append(Map.points[j].history.position[index][1])
        
        for point in points:
            plt.scatter(point[0], point[1], alpha=0.8, edgecolors='none', s=30)
        plt.show()

    @staticmethod
    def animate(save=None, timeout=True, FPS=20):
        num_frames = int(Map.current_time*FPS)
        interval = Map.num_steps/(num_frames)
        data = np.zeros((num_frames, len(Map.points), 3))
        for i in range(len(Map.points)):
            position = Map.points[i].history.position
            for j in range(num_frames):
                data[j,i,:] = position[int(j*interval)]
        fig = plt.figure()

        ax = axes3d.Axes3D(fig)

        # Initialize scatters
        dots = [ ax.scatter(data[0][i,0:1], data[0][i,1:2], data[0][i,2:]) for i in range(data[0].shape[0]) ]
        lines = []
        for spring in Map.springs:
            p1 = spring.point_1.history.position[0]
            p2 = spring.point_2.history.position[0]
            lines.append([ax.plot([p1[0],p2[0]], [p1[1],p2[1]], [p1[2],p2[2]])[0]])

        # Number of iterations
        iterations = len(data)

        # Setting the axes properties
        ax.set_xlim3d([-1, 2])
        ax.set_xlabel('X')

        ax.set_ylim3d([0, 3])
        ax.set_ylabel('Y')

        ax.set_zlim3d([0, 10])
        ax.set_zlabel('Z')

        ax.set_title('Nuclear Star Cluster')

        # Provide starting angle for the view.
        ax.view_init(90, 270) #(90, 270)

        ani = animation.FuncAnimation(fig, Map.animate_frame, iterations, fargs=(data, Map.num_steps/num_frames, dots, lines),
                                        interval=1000/FPS, blit=False, repeat=True)

        if save:
            ani.save(save, writer='imagemagick')
            print("Done saving")

        if timeout:
            timer = fig.canvas.new_timer(interval = Map.current_time*1000)
            timer.add_callback(Map.close_animation)
            timer.start()

        plt.show()

class Spring:
    def __init__(self, point_1, point_2, k, length=None, compression_only=False):
        self.k = k
        self.point_1 = point_1
        self.point_2 = point_2
        self.compression_only = compression_only
        if length:
            self.length = length
        else:
            self.length = get_distance(point_1, point_2)
        point_1.add_relationship(self)
        point_2.add_relationship(self)
        Map.springs.append(self)

    def calculate_force(self, point):
        if point == self.point_1:
            other_point = self.point_2
        elif point == self.point_2:
            other_point = self.point_1
        else:
            raise Exception("WTF are you doing?")

        length_delta = get_distance(point, other_point) - self.length
        if self.compression_only and length_delta > 0:
            return (0.,0.,0.)
        return get_unit_vector(point, other_point)*length_delta*self.k
    

class Damper:
    def __init__(self, point_1, point_2, k):
        self.k = k
        self.point_1 = point_1
        self.point_2 = point_2
        point_1.add_relationship(self)
        point_2.add_relationship(self)

    def calculate_force(self, point):
        if point == self.point_1:
            other_point = self.point_2
        elif point == self.point_2:
            other_point = self.point_1
        else:
            raise Exception("WTF are you doing?")
        
        unit_vector = get_unit_vector(point, other_point)
        velocity_point = np.dot(point.velocity, unit_vector)
        velocity_other_point = np.dot(other_point.velocity, unit_vector)
        return -unit_vector*(velocity_point-velocity_other_point)*self.k
    

class AngularSpring:
    def __init__(self, point_1, pivot, point_2, k, angle_deg=None):
        self.pivot = pivot
        self.point_1 = point_1
        self.point_2 = point_2
        point_1.add_relationship(self)
        point_2.add_relationship(self)
        self.k = k
        if angle_deg:
            self.angle = angle_deg*3.1415/180
        else:
            self.angle = get_angle(point_1, pivot, point_2)

    def calculate_force(self, point):
        if point == self.point_1:
            angle_delta = self.angle - get_angle(point, self.pivot, self.point_2)
            tangent_vector = get_torque_direction(point, self.pivot, self.point_2)
        else:
            angle_delta = self.angle - get_angle(point, self.pivot, self.point_1)
            tangent_vector = get_torque_direction(point, self.pivot, self.point_1)
        tangent_force = tangent_vector * angle_delta * self.k
        tangent_velocity = np.dot(tangent_vector, point.velocity)
        axial_vector = get_vector(point, self.pivot)
        axial_force = axial_vector * (tangent_velocity ** 2) / (get_vector_length(axial_vector) ** 2)
        return tangent_force + axial_force

class AngularDamper:
    def __init__(self, point_1, pivot, point_2, k):
        self.pivot = pivot
        self.point_1 = point_1
        self.point_2 = point_2
        point_1.add_relationship(self)
        point_2.add_relationship(self)
        self.k = k

    def calculate_force(self, point):
        if point == self.point_1:
            other_point = self.point_2
        else:
            other_point = self.point_1
        tangent_vector = get_torque_direction(point, self.pivot, other_point)
        tangent_vector_other = get_torque_direction(other_point, self.pivot, point)
        tangent_velocity_point = np.dot(tangent_vector, point.velocity)
        tangent_velocity_other = np.dot(tangent_vector_other, other_point.velocity)
        tangent_force = -tangent_vector * (tangent_velocity_point-tangent_velocity_other) * self.k
        axial_vector = get_vector(point, self.pivot)
        axial_force = 0 # axial_vector * (tangent_velocity_point ** 2) / (get_vector_length(axial_vector) ** 2)
        return tangent_force + axial_force

class PointForce:
    def __init__(self, point, func):
        point.overrides.append(func)

class UniversalForce:
    def __init__(self, func):
        Map.universal_overrides.append(func)

def gravity(points):
    for point in points:
        point.force[1] -= 9.81 * point.mass

def floor(points):
    for point in points:
        if point.position[1] < 0:
            point.position[1] = 0
            if point.force[1] < 0:
                point.force[1] = 0
            if point.velocity[1] < 0:
                point.velocity[1] = 0