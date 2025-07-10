import math
import time
from itertools import groupby
from operator import itemgetter

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from geometry_msgs.msg import Vector3Stamped

from sensor_msgs.msg import LaserScan
import rospy
import timeit
import tf

matplotlib.use('Qt5Agg')


class VectorFieldHistogram:
    def __init__(self,
                 num_readings: int,
                 max_range: float,
                 grid_size: float,
                 sector_angle: int,
                 obstacle_threshold: float,
                 s_max: int,
                 input_topic: str,
                 output_topic: str):
        self.num_readings = num_readings
        self.max_range = max_range
        self.grid_size = grid_size
        self.sector_angle = sector_angle
        self.obstacle_threshold = obstacle_threshold
        self.s_max = s_max
        self.output_topic = output_topic

        self.topic = input_topic  # "/hsrb/base_scan"
        self.robot_position = (0.0, 0.0)
        self.tf_listener = tf.TransformListener()
        self.distances = np.array([])
        self.angles = np.array([])
        self.polar_histogram = None
        self.theta_deg = None
        self.d_max = self.max_range
        self.obstacle_density = None

        rospy.Subscriber(self.topic, LaserScan, self.laser_callback)
        self.pub = rospy.Publisher(name=output_topic, data_class=Vector3Stamped, queue_size=10)
        self.rate = rospy.Rate(10)

    # calculate polar obstacle density, as indicator how many obstacles are within a sector - tbd
    # Unit testing by checking steering angles - tbd

    # Get LiDAR Data from /hsrb/base_scan topic
    def laser_callback(self, data: LaserScan):
        self.distances = np.array(data.ranges)
        self.angles = data.angle_min + np.arange(len(self.distances)) * data.angle_increment

        self.run()

    # Target point for cool stuff
    def target_sim(self):
        # start_time = time.perf_counter() - benchmark stuff
        target_point = (2, 3.5)

        try:
            (trans, rot) = self.tf_listener.lookupTransform("map", "base_footprint", rospy.Time(0))
            self.robot_position = (trans[0], trans[1])
            print(f'R_POS:{self.robot_position}')
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            self.robot_position = (0.0, 0.0)

        # calculating sector that target is in
        tx, ty = target_point
        target_angle = np.arctan2(ty - self.robot_position[1], tx - self.robot_position[0])
        print('target_angle:', target_angle)
        target_angle_deg = np.rad2deg(target_angle) % 240
        print('target_angle_deg:', target_angle_deg)
        target_sector = int(target_angle_deg // self.sector_angle)
        print(f"Target Sector:{target_sector}")
        # end_time = time.perf_counter()  # End timing
        # elapsed_time_ms = (end_time - start_time) * 1000
        # rospy.loginfo(f"target_sim took {elapsed_time_ms:.3f} ms") - benchmark stuff
        return target_sector, target_angle_deg, target_point

    def run(self):
        if len(self.distances) == 0:
            return
        target_sector, target_angle_deg, target_point = self.target_sim()
        self.update_histogram(self.distances, self.angles, target_sector, target_angle_deg)
        self.histogram_plot(target_point)

    def update_histogram(self, distances, angles, target_sector, target_angle_deg):
        # start_time = time.perf_counter()
        theta_deg = None
        x_points = distances * np.cos(angles)
        y_points = distances * np.sin(angles)

        grid_dim = int((2 * self.max_range) / self.grid_size)
        histogram_grid = np.zeros((grid_dim, grid_dim))
        # map converted laser scanner ranges onto histogram grid
        for x, y in zip(x_points, y_points):
            gx = int((x + self.max_range) / self.grid_size)
            gy = int((y + self.max_range) / self.grid_size)
            if 0 <= gx < grid_dim and 0 <= gy < grid_dim:
                histogram_grid[gx, gy] = 1

        num_sectors = int(240 / self.sector_angle)
        polar_histogram = np.zeros(num_sectors)
        angles = angles - angles[0]
        # calculate sector magnitudes
        for sector in range(num_sectors):
            start_angle = np.deg2rad(sector * self.sector_angle)
            end_angle = np.deg2rad((sector + 1) * self.sector_angle)
            mask = (angles >= start_angle) & (angles <= end_angle)
            sector_dists = distances[mask]
            # density = np.sum((self.max_range - sector_dists) / self.max_range)
            clipped_dists = np.clip(sector_dists, 0, self.d_max)
            magnitudes = 1 - (clipped_dists / self.d_max)  # magnitude stuff
            polar_histogram[sector] = np.sum(magnitudes)
        polar_histogram = polar_histogram[::-1]  # flips histogram, because we have a right hand coord system
        print(polar_histogram)

        # polar_histogram = self.smooth_polar_histogram(polar_histogram, l=5)
        # print(f'Magnitude Sum for Sector {sector}:{np.sum(magnitudes)}')
        free_sectors = np.where(polar_histogram < self.obstacle_threshold)[0]  # replace obstacle threshold with POD
        # valley calculation
        valleys = []
        for k, g in groupby(enumerate(free_sectors), lambda ix: ix[0] - ix[1]):
            valley = list(map(itemgetter(1), g))
            valleys.append(valley)
            print(f"Valley: {valley}")
            print(f"Valleys: {valleys}")
        # Find the valley containing the target sector
        selected_valley = None

        for valley in valleys:
            if min(valley) <= target_sector <= max(valley):
                selected_valley = valley
                print(f"Selected Valley: {selected_valley}")
                break

        if selected_valley:
            k_near = min(selected_valley)
            k_far = k_near + self.s_max
            if len(selected_valley) <= self.s_max:
                print("Valley is narrow")
                k_far = max(selected_valley)
            theta = (k_near + k_far) / 2
            best_sector = int(theta)
            theta_deg = best_sector * self.sector_angle
            print(f"Target angle: {target_angle_deg:.2f}° => Sector {target_sector}")
            print(f"Steering Valley: {selected_valley}")
            print(f"k_near: {k_near}, k_far: {k_far}, Steering direction sector: {theta}")
            print(f"Steering angle: {theta_deg:.2f}°")
        # Start searching for adjacent free valley
        else:
            total_sectors = len(polar_histogram)
            print("Target point is behind obstacle, choosing largest adjacent Valley")
            # check whether target sector is adjacent to any valleys, then check which of the two is larger
            nearby_sectors = set((target_sector + offset) % total_sectors for offset in range(-2, 2))
            print(f"Nearby sectors: {nearby_sectors}")
            candidate_valleys = []
            for valley in valleys:
                if not set(valley).isdisjoint(nearby_sectors):
                    candidate_valleys.append(valley)
            # candidate valley has been found in first iteration
            if candidate_valleys:
                selected_valley = max(candidate_valleys, key=len)
                print(f"Selected adjacent Valley: {selected_valley}")
                k_near = min(selected_valley)
                k_far = k_near + self.s_max
                if len(selected_valley) <= self.s_max:
                    print("Valley is narrow")
                    k_far = max(selected_valley)
                theta = (k_near + k_far) / 2
                best_sector = int(theta)
                theta_deg = best_sector * self.sector_angle
                print(f"Target angle: {target_angle_deg:.2f}° => Sector {target_sector}")
                print(f"Steering Valley: {selected_valley}")
                print(f"k_near: {k_near}, k_far: {k_far}, Steering direction sector: {theta}")
                print(f"Steering angle: {theta_deg:.2f}°")
            # Dynamically expand search radius if no candidate is found in first iteration
            else:
                print("No immediately adjacent Valleys found, expanding search radius")
                found_valley = None
                max_search_expansion = total_sectors // 2
                for radius in range(2, max_search_expansion + 1):
                    nearby_sectors = set()
                    for offset in range(-radius, radius + 1):
                        candidate_sector = target_sector + offset
                        if 0 <= candidate_sector < total_sectors:
                            nearby_sectors.add(candidate_sector)
                    print(f"Trying nearby sectors with radius {radius}: {sorted(nearby_sectors)}")

                    # check whether target sector is adjacent to any valleys, then check which of the two is larger
                    candidate_valleys = []
                    for valley in valleys:
                        if not set(valley).isdisjoint(nearby_sectors):
                            candidate_valleys.append(valley)
                    if candidate_valleys and len(candidate_valleys) >= 2:
                        found_valley = max(candidate_valleys, key=len)
                        print(f"Found valley at radius {radius}: {found_valley}")
                        break
                # calculate directional vector
                if found_valley:
                    selected_valley = found_valley
                    k_near = min(selected_valley)
                    k_far = k_near + self.s_max
                    if len(selected_valley) <= self.s_max:
                        print("Valley is narrow")
                        k_far = max(selected_valley)
                    theta = (k_near + k_far) / 2
                    best_sector = int(theta)
                    theta_deg = best_sector * self.sector_angle
                    print(f"Target angle: {target_angle_deg:.2f}° => Sector {target_sector}")
                    print(f"Steering Valley: {selected_valley}")
                    print(f"k_near: {k_near}, k_far: {k_far}, Steering direction sector: {theta}")
                    print(f"Steering angle: {theta_deg:.2f}°")

        self.polar_histogram = polar_histogram
        self.theta_deg = theta_deg
        self.x_points = x_points
        self.y_points = y_points

        # calculation of directional vector
        if theta_deg is not None:
            # TODO: Puts out slightly wrong directional vector, even though the math should be right...WHY U NO MATH?ß?!1
            theta_rad = math.radians(-(theta_deg - 120.0))

            direction_vector = Vector3Stamped()
            direction_vector.header.frame_id = "base_footprint"
            direction_vector.vector.x = math.cos(theta_rad)
            direction_vector.vector.y = math.sin(theta_rad)
            direction_vector.vector.z = 0.0

            self.pub.publish(direction_vector)
        else:
            direction_vector = (0.0, 0.0, 0.0)
        print(direction_vector.vector.x)
        print(direction_vector.vector.y)
        print(direction_vector.vector.z)
        print(f"Directional Vector: {direction_vector}")
        print("---------------------------------------")

        # benchmark stuff
        # end_time = time.perf_counter()  # End timing
        # elapsed_time_ms = (end_time - start_time) * 1000
        # rospy.loginfo(f"update_histogram took {elapsed_time_ms:.3f} ms")
        # log_file_path = "/home/yannis/Documents/histogram_benchmark.txt"
        # with open(log_file_path, "a") as f:
        #     f.write(f"update_histogram took {elapsed_time_ms:.4f}ms\n")

    # def smooth_polar_histogram(self, polar_histogram, l=5):
    #
    #     smoothed = np.zeros_like(polar_histogram)
    #     length = len(polar_histogram)
    #     print(f"[DEBUG] Smoothing histogram with {length} sectors")
    #     for k in range(length):
    #         value = 0
    #         weight_sum = 0
    #         for offset in range(-l, l+1):
    #             idx = k + offset % length # wrap around
    #             weight = 2 if offset != -l and offset !=l else 1 # this is dogwater
    #
    #             value += weight * polar_histogram[idx]
    #             weight_sum += weight
    #             smoothed[k] = value / weight_sum
    #             print(f'V:{value}')
    #             print(f'WS:{weight_sum}')
    #             print(f'IDX:{idx}')
    #
    #     return smoothed
    # ------------------- ONLY USE THE PLOT FOR TESTING ---------------------
    def histogram_plot(self, target_point):
        # Start of plot gen
        fig = plt.figure(figsize=(12, 6))
        ax0 = fig.add_subplot(1, 2, 1)
        ax1 = fig.add_subplot(1, 2, 2, polar=True)

        # LIDAR PointCloud
        ax0.set_title('LIDAR Map')
        ax0.scatter(-self.y_points, self.x_points, c='blue', s=1, label='LIDAR points')
        ax0.plot(-target_point[1], target_point[0], 'mx', markersize=10)
        ax0.plot(-self.robot_position[1], self.robot_position[0], 'go', markersize=10, label='Robot Position')
        ax0.set_xlim(-self.max_range, self.max_range)
        ax0.set_ylim(-self.max_range, self.max_range)
        ax0.set_aspect('equal')
        ax0.grid(True)
        ax0.legend()

        # Polar Histogram
        ax1.set_title('Polar Histogram')
        ax1.set_theta_direction(-1)
        ax1.set_theta_zero_location('W', 30)
        colors = ['green' if self.polar_histogram[i] < self.obstacle_threshold else 'red'
                  for i in range(len(self.polar_histogram))]
        angles_rad = np.deg2rad(np.arange(0, 240, self.sector_angle))
        ax1.bar(angles_rad, self.polar_histogram, width=np.deg2rad(self.sector_angle), bottom=0.0, align='edge',
                color=colors)
        ax1.plot('m--', linewidth=0.5, label='sectors')  # needed to portray legend for graph accurately
        for angles in angles_rad:
            ax1.plot([angles, angles], [0, np.max(self.polar_histogram)], 'm--', linewidth=0.5)
        if self.theta_deg is not None:
            best_rad = np.deg2rad(self.theta_deg)
            ax1.plot([best_rad, best_rad], [0, np.max(self.polar_histogram)], 'b--', linewidth=2,
                     label='best direction')
        ax1.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig('/home/yannis/Documents/vfh.png')


if __name__ == '__main__':
    rospy.init_node('vfh_node')
    vfh = VectorFieldHistogram(num_readings=240,
                               max_range=5.0,
                               grid_size=0.1,
                               sector_angle=5,
                               obstacle_threshold=8,
                               s_max=12,
                               input_topic="/hsrb/base_scan",
                               output_topic="/hsrb/VFH")

    rospy.spin()
