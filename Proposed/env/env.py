"""
This simulator follows the evaluation methodology in 3GPP TR 36.885, refer to Annex A and Urban case for details.

The settings  are consistent withthe paper "Learn to Compress CSI and Allocate Resources in Vehicular Networks" and "Spectrum Sharing in Vehicular Networks Based on Multi-Agent Reinforcement Learning"

Assume that each vehicle carries V2I link and V2V links simultaneously, and the number of V2I links is equal to total resource blocks. Each vehicle sets up V2V link with its nearest neighbor, in other words, we consider unicast only.
"""
import copy
import numpy as np
from .core import *


" ******************* some useful constants ************************* "
#! weights of each part of reward
WEIGHTS = [1., 1., 1., 1.]

# four directions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

# maybe the simulation area size is too large

# Vehicle drop model
LANE_WIDTH = 3.5
GRID_SIZE = (250.0, 433.0)     # !(width, height)
AREA_SIZE = (500.0, 866.0)     # 2*2 grid

# center position of horizontal (x) axis
UP_LANES = np.array([3.5/2, 3.5/2 + 3.5, 250+3.5/2, 250+3.5+3.5/2])
DOWN_LANES = np.array([250-3.5-3.5/2, 250-3.5/2, 500-3.5-3.5/2, 500-3.5/2])
# center position of vertical (y) axis
LEFT_LANES = np.array([3.5/2, 3.5/2 + 3.5, 433+3.5/2, 433+3.5+3.5/2])
RIGHT_LANES = np.array([433-3.5-3.5/2, 433-3.5/2, 866-3.5-3.5/2, 866-3.5/2])

assert len(UP_LANES) == len(DOWN_LANES) == len(LEFT_LANES) == len(RIGHT_LANES)

VEHICLE_VELOCITY = (10.0, 15.0)     # in m/s

# channel model
POWER_LEVEL = (23, 15, 5, -100)     # in dB

NOISE_POWER_DB = -114               # in dB
NOISE_POWER = np.power(10, NOISE_POWER_DB / 10)
ANTENNA_GAIN_BS = 8             # in dBi
ANTENNA_GAIN_VEHICLE = 3

NOISE_FIGURE_BS = 5             # in dB
NOISE_FIGURE_VEHICLE = 9

ANTENNA_HEIGHT_BS = 25          # in m
ANTENNA_HEIGHT_VEHICLE = 1.5

BANDWIDTH = 1e6                 # 1MHz per resource block
PAYLOAD_SIZE = 1060 * 8         # V2V payload size = 1060bytes every 100ms

# other configuration
# num_vehicle = 4                    # total resource blocks can be utilized
# NUM_NEIGHBOR = 1                # corresponds to unicast

FAST_UPDATE_PERIOD = 1e-3       # 1ms
SLOW_UPDATE_PERIOD = 1e-1       # 100ms
MESSAGE_GENERATION_PERIOD = 1e-1    # maximum delay tolerance = 100ms

# # large scale channel fading is fixed for a couple of episodes during training
# CHANNEL_FIXED_PERIOD = [100, 50, 20, 10, 1] 
"************************* end **********************************"


class Environment():
    """
    Simulator for vehicular networks. 
    """
    def __init__(self, num_vehicle = 4, num_packet = 2, reward_weights = WEIGHTS, training = True) -> None:
        """
        Initialize the simulator, when in training mode, the large scale channel fading updates in a slower frequency.
        """
        self.up_lanes = UP_LANES
        self.down_lanes = DOWN_LANES
        self.left_lanes = LEFT_LANES
        self.right_lanes = RIGHT_LANES
        self.width, self.height = np.array(AREA_SIZE)

        #! number of vehicles = number of RBs
        self.num_vehicle = num_vehicle
        self.num_rb = num_vehicle

        self.bandwidth = BANDWIDTH
        #! Observation and action space
        # self.obs_space = 3 * num_vehicle + 4               # observation space of a single agent
        self.obs_space = num_vehicle + 3
        self.act_space = len(POWER_LEVEL) * num_vehicle # action space of a single agent
        #! update period
        self.fast_period = FAST_UPDATE_PERIOD           # equals to minimum simulation step
        self.slow_period = SLOW_UPDATE_PERIOD           # equals to maximum delay tolerance
        #! channel fading
        # class for channel fading update
        self.v2v_channel = V2VChannel(num_vehicle, num_vehicle)
        self.v2i_channel = V2IChannel(num_vehicle, num_vehicle)
        #! vehicles
        self.vehicles = []
        #! real-time transmission rate of each V2I and V2V link
        self.v2i_rate = None
        self.v2v_rate = None
        #! remaining payload, selected sunchannel, transmision power and active link in fact for all vehicles' V2V Llinks
        self.num_packet = num_packet
        self.remaining_time = None
        self.remaining_load = None
        self.selected_rb = None
        self.transmision_power = None
        self.active_link = None
        #! interference sensed on each RB at the corresponding Rx of each Tx, used for transmission rate calculation
        self.v2v_interference = None
        self.v2i_interference = None
        #! interference sensed on each RB at the transmitter, used in observation space
        self.v2v_interference_tx = None


        #! time step
        self.t = 0

        #! weights for different part of reward
        self.reward_weights = reward_weights

        #! control the frequency of large scale channel fading update
        self.training = training
        if self.training:
            self.channel_fixed_period = 1
        else:
            self.channel_fixed_period = 1

        #! benchmark, Round-Robin method
        self.v2i_rate_benchmark = None
        self.v2v_rate_benchmark = None
        self.remaining_load_benchmark = None
        self.selected_rb_benchmark = None
        self.transmision_power_benchmark = None
        self.active_link_benchmark = None
        self.v2v_interference_benchmark = None
        self.v2i_interference_benchmark = None

    @property
    def payload_size(self):
        return PAYLOAD_SIZE * self.num_packet


    @property
    def observation_space(self, single = True):
        if single:
            return self.obs_space
        else:
            return (self.obs_space, self.num_vehicle)


    @property
    def action_space(self, single = True):
        if single:
            return self.act_space
        else:
            return (self.act_space, self.num_vehicle)


    def reset(self):
        """
        Reset (initialize) the environment and return the initial observation
        """

        " reset the time step "
        self.t = 0

        " reset (position, velocity, direction) of vehicles "
        self.init_vehicles()
        self.update_neighbor()

        " reset the channel "
        self.v2i_channel = V2IChannel(self.num_vehicle, self.num_rb)
        self.v2v_channel = V2VChannel(self.num_vehicle, self.num_rb)
          
        self.v2i_channel.update_gain(self.vehicles)
        self.v2v_channel.update_gain(self.vehicles)

        self.v2i_channel.update_fast_fading()
        self.v2v_channel.update_fast_fading()

        " sensed interference at the Rx "
        self.v2v_interference = np.zeros((self.num_vehicle, self.num_rb))
        self.v2i_interference = np.zeros((self.num_vehicle, self.num_rb)) 
        #! sensed interference at the Tx
        self.v2v_interference_tx = np.zeros((self.num_vehicle, self.num_rb))
        #! benchmark
        self.v2v_interference_benchmark = np.zeros((self.num_vehicle, self.num_rb))
        self.v2i_interference_benchmark = np.zeros((self.num_vehicle, self.num_rb)) 

        " reset real-time transmission rate "
        #! consider the occupied RB only
        self.v2i_rate = np.zeros(self.num_rb)
        self.v2v_rate = np.zeros(self.num_vehicle)
        #! benchmark
        self.v2i_rate_benchmark = np.zeros(self.num_rb)
        self.v2v_rate_benchmark = np.zeros(self.num_vehicle)


        " reset the remaining payload size, etc. "
        #! each TX-RX pair can only occupy one resource block
        #! all vehicles are synchronized
        self.remaining_time = self.slow_period * 1000   #! [unit: ms]
        self.remaining_load = np.ones(self.num_vehicle) * self.payload_size
        self.selected_rb = np.zeros(self.num_vehicle, dtype='int')
        self.transmision_power = np.zeros(self.num_vehicle, dtype='int')
        self.active_link = np.ones(self.num_vehicle,  dtype='bool')
        #! benchmark
        self.remaining_load_benchmark = np.ones(self.num_vehicle) * self.payload_size
        self.selected_rb_benchmark = np.zeros(self.num_vehicle, dtype='int')
        self.transmision_power_benchmark = np.zeros(self.num_vehicle, dtype='int')
        self.active_link_benchmark = np.ones(self.num_vehicle,  dtype='bool')


        return self.observation()



    #todo normalize the observations
    def observation(self):
        """
        Return the joint observation of all (V2V) vehicles, since we consider only V2V resource optimization.
        
        Observation of single vehicle includes: 
        1. V2V channel fast fading, shape = (num_vehicle, )
        2. V2I channel fast fading, shape = (num_vehicle, ) 
        3. V2V interference, shape = (num_vehicle, )
        4. V2V channel gain, shape = 1
        5. V2I channel gain, shape = 1
        6. remain payload size, shape = 1
        7. remain time, shape = 1 
        8. id, shape = 1
        """
        observations = np.zeros((self.num_vehicle, self.obs_space))
        v2v_fading = np.zeros(self.num_rb)
        v2i_fading = np.zeros(self.num_rb)
        v2v_interference = np.zeros(self.num_rb)  
        v2v_gain = 0.
        v2i_gain = 0. 
        remaining_load = 0.
        for i, tx in enumerate(self.vehicles):
            rx_id = tx.neighbor
            v2v_fading = self.v2v_channel.gain_with_fast_fading[i, rx_id, :] * 0.01
            v2i_fading = self.v2i_channel.gain_with_fast_fading[i, :] * 0.01
            #! sensed interference at corresponding Rx of each Tx
            # v2v_interference =  (self.v2v_interference[i, :] + 150) * 0.01
            #! sensed interference at each Tx
            v2v_interference =  (self.v2v_interference_tx[i, :] + 150) * 0.01
            v2v_gain = self.v2v_channel.gain[i, rx_id] * 0.01
            v2i_gain = self.v2i_channel.gain[i] * 0.01

            remaining_load = self.remaining_load[i] * 0.001
            # observation = np.concatenate(
            #     (v2v_fading, v2i_fading, v2v_interference,
            #         np.array([
            #             v2v_gain, v2i_gain, remaining_load, self.remaining_time
            #         ])))

            #! no CSI, append an ID
            observation = np.concatenate((v2v_interference, np.array([remaining_load, self.remaining_time, i])))

            observations[i] = observation
        
        return observations



    def step(self, actions, actions_benchmark):
        """
        1. Update position
        2. Update channel (slow or fast fading)
        3. Update interference
        4. Apply joint action
        5. Update remaining load, active link, etc
        6. Calculate reward
        7. feedback next joint observation
        """

        #! apply joint actions
        # assert actions.shape == (self.num_vehicle,)
        self.selected_rb = actions % self.num_rb
        self.transmision_power = actions // self.num_rb
        #! apply the benchmark actions
        self.selected_rb_benchmark = actions_benchmark % self.num_rb
        self.transmision_power_benchmark = actions_benchmark // self.num_rb

        #! update interference
        self.update_interference()
        self.update_interference_tx()
        self.calculate_rate()
        #! track the benchmark performance
        self.update_interference_benchmark()
        self.calculate_rate_benchmark()

        #! update remaining load and active link
        self.remaining_load -= self.v2v_rate * FAST_UPDATE_PERIOD * BANDWIDTH
        self.remaining_load[self.remaining_load <= 0] = 0
        self.active_link[np.where(self.remaining_load <= 0)] = False

        self.remaining_load_benchmark -= self.v2v_rate_benchmark * FAST_UPDATE_PERIOD * BANDWIDTH
        self.remaining_load_benchmark[self.remaining_load_benchmark <= 0] = 0
        self.active_link_benchmark[np.where(self.remaining_load_benchmark <= 0)] = False

        #! global reward
        reward = self.calculate_reward()

        self.t += 1
        self.remaining_time -= 1

        #! when meets SLOW_UPDATE_PERIOD, update the channel gain and 
        #! reset the remaining_load as a new message generated
        #? in other baselines, large-scale update every 100 * SLOW_UPDATE_PERIOD
        if self.t % (self.slow_period * 1e3) == 0:
            self.remaining_time = self.slow_period * 1000
            self.active_link = np.ones(self.num_vehicle, dtype='bool')
            self.remaining_load = np.ones(self.num_vehicle) * self.payload_size
            #! benchmark
            self.active_link_benchmark = np.ones(self.num_vehicle, dtype='bool')
            self.remaining_load_benchmark = np.ones(self.num_vehicle) * self.payload_size
            #! when in training mode, the large scale fading is fixed for a couple of episodes
            if self.training and self.t % (self.slow_period * 1e3 * self.channel_fixed_period) == 0:
                self.update_position()
                self.update_neighbor()  
                self.v2i_channel.update_gain(self.vehicles)
                self.v2v_channel.update_gain(self.vehicles)
            elif not self.training:
                self.update_position()
                self.update_neighbor()  
                self.v2i_channel.update_gain(self.vehicles)
                self.v2v_channel.update_gain(self.vehicles)

                    
        #! update fast fading every 2 steps
        if self.t % 2:
            self.v2i_channel.update_fast_fading()
            self.v2v_channel.update_fast_fading()

        #! joint observation
        joint_observation = self.observation()

        return joint_observation, reward



    def init_vehicles(self):
        """
        Initialize the (position, direction, velocity) for all vehicles. 
        """
        if len(self.vehicles) > 0:
            self.vehicles = []
        assert self.num_vehicle == 4 or self.num_vehicle == 8
        for i in range(self.num_vehicle // 4):
            # lane = np.random.randint(len(self.up_lanes))
            lane = np.random.randint(0, 2) + i * 2
            for direction in range(4):
                position = None
                velocity = np.random.randint(VEHICLE_VELOCITY[0], VEHICLE_VELOCITY[1])        # m/s
                if direction == UP:
                    velocity = np.array([0., velocity])
                    position = np.array([self.up_lanes[lane], np.random.randint(0.1 * self.height, 0.4 * self.height) + (2 * lane // 4) * 0.5 * self.height])
                elif direction == DOWN:
                    velocity = np.array([0., -velocity])
                    position = np.array([self.down_lanes[lane], np.random.randint(0.1 * self.height, 0.4 * self.height) + (2 * lane // 4) * 0.5 * self.height])
                elif direction == LEFT:
                    velocity = np.array([-velocity, 0.])
                    position = np.array([np.random.randint(0.1 * self.width, 0.4 * self.width) + (2 * lane // 4) * 0.5 * self.width, self.left_lanes[lane]])
                else:
                    velocity = np.array([velocity, 0.])
                    position = np.array([np.random.randint(0.1 * self.width, 0.4 * self.width) + (2 * lane // 4) * 0.5 * self.width, self.right_lanes[lane]])

                self.vehicles.append(Vehicle(i, position, direction, velocity))




    def update_position(self):
        """
        Update the position for all vehicles, every 100ms
        When comes to an intersecction, go straight with probability 0.5, turn left or left with equal probability 0.25
        """
        for i, vehicle in enumerate(self.vehicles):
            # update position every 100ms
            assert vehicle.velocity.size == 2
            delta_distance = vehicle.velocity * self.slow_period
            # for debugging
            # if np.linalg.norm(delta_distance) <=0.001:
            #     print(vehicle)
            #
            change_direction = True if np.random.uniform(0, 1) < 0.25 else False
            if vehicle.direction == UP:
                # delta_distance[1] > 0
                cross_left_lane_index = np.where((vehicle.position[1] <= self.left_lanes)\
                                                 & (vehicle.position[1] + delta_distance[1] >= self.left_lanes))
                cross_right_lane_index = np.where((vehicle.position[1] <= self.right_lanes)\
                                                 & (vehicle.position[1] + delta_distance[1] >= self.right_lanes))

                assert cross_left_lane_index[0].size + cross_right_lane_index[0].size <= 1
                # meets a cross
                if cross_left_lane_index[0].size != 0:
                    if change_direction:
                        index = cross_left_lane_index[0][0]
                        abs_delta_distance =delta_distance[1] - (self.left_lanes[index] - vehicle.position[1])
                        assert abs_delta_distance >= 0
                        vehicle.position = np.array([vehicle.position[0] - abs_delta_distance, self.left_lanes[index]])
                        vehicle.direction = LEFT
                        vehicle.velocity = np.array([-vehicle.velocity[1], 0.])
                        continue

                if cross_right_lane_index[0].size != 0:
                    if change_direction:
                        index = cross_right_lane_index[0][0]
                        abs_delta_distance =delta_distance[1] - (self.right_lanes[index] - vehicle.position[1])
                        assert abs_delta_distance >= 0
                        vehicle.position = np.array([vehicle.position[0] + abs_delta_distance, self.right_lanes[index]])
                        vehicle.direction = RIGHT
                        vehicle.velocity = np.array([vehicle.velocity[1], 0.])
                        continue
                # go straight
                vehicle.position += delta_distance

            elif vehicle.direction == DOWN:
                # delta_distance[1] < 0
                cross_left_lane_index = np.where((vehicle.position[1] >= self.left_lanes)\
                                                 & (vehicle.position[1] + delta_distance[1] <= self.left_lanes))
                cross_right_lane_index = np.where((vehicle.position[1] >= self.right_lanes)\
                                                 & (vehicle.position[1] + delta_distance[1] <= self.right_lanes))

                assert cross_left_lane_index[0].size + cross_right_lane_index[0].size <= 1
                # meets a cross
                if cross_left_lane_index[0].size != 0:
                    if change_direction:
                        index = cross_left_lane_index[0][0]
                        abs_delta_distance =delta_distance[1] - (self.left_lanes[index] - vehicle.position[1])
                        assert abs_delta_distance <= 0
                        vehicle.position = np.array([vehicle.position[0] + abs_delta_distance, self.left_lanes[index]])
                        vehicle.direction = LEFT
                        vehicle.velocity = np.array([vehicle.velocity[1], 0.])
                        continue

                if cross_right_lane_index[0].size != 0:
                    if change_direction:
                        index = cross_right_lane_index[0][0]
                        abs_delta_distance =delta_distance[1] - (self.right_lanes[index] - vehicle.position[1])
                        assert abs_delta_distance <= 0
                        vehicle.position = np.array([vehicle.position[0] - abs_delta_distance, self.right_lanes[index]])
                        vehicle.direction = RIGHT
                        vehicle.velocity = np.array([-vehicle.velocity[1], 0.])
                        continue
                # go straight
                vehicle.position += delta_distance

            elif vehicle.direction == LEFT:
                # delta_distance[0] < 0
                cross_up_lane_index = np.where((vehicle.position[0] >= self.up_lanes)\
                                                 & (vehicle.position[0] + delta_distance[0] <= self.up_lanes))
                cross_down_lane_index = np.where((vehicle.position[0] >= self.down_lanes)\
                                                 & (vehicle.position[0] + delta_distance[0] <= self.down_lanes))

                assert cross_up_lane_index[0].size + cross_down_lane_index[0].size <= 1
                if cross_up_lane_index[0].size != 0:
                    if change_direction:
                        index = cross_up_lane_index[0][0]
                        abs_delta_distance = delta_distance[0] - (self.up_lanes[index] - vehicle.position[0])
                        assert abs_delta_distance <= 0
                        vehicle.position = np.array([self.up_lanes[index], vehicle.position[1] - abs_delta_distance])
                        vehicle.direction = UP
                        vehicle.velocity = np.array([0., -vehicle.velocity[0]])
                        continue

                if cross_down_lane_index[0].size != 0:
                    if change_direction:
                        index = cross_down_lane_index[0][0]
                        abs_delta_distance = delta_distance[0] - (self.down_lanes[index] - vehicle.position[0])
                        assert abs_delta_distance <= 0
                        vehicle.position = np.array([self.down_lanes[index], vehicle.position[1] + abs_delta_distance])
                        vehicle.direction = DOWN
                        vehicle.velocity = np.array([0., vehicle.velocity[0]])
                        continue

                vehicle.position += delta_distance

            elif vehicle.direction == RIGHT:
                # delta_distance[0] > 0
                cross_up_lane_index = np.where((vehicle.position[0] <= self.up_lanes)\
                                                 & (vehicle.position[0] + delta_distance[0] >= self.up_lanes))
                cross_down_lane_index = np.where((vehicle.position[0] <= self.down_lanes)\
                                                 & (vehicle.position[0] + delta_distance[0] >= self.down_lanes))

                assert cross_up_lane_index[0].size + cross_down_lane_index[0].size <= 1
                if cross_up_lane_index[0].size != 0:
                    if change_direction:
                        index = cross_up_lane_index[0][0]
                        abs_delta_distance = delta_distance[0] - (self.up_lanes[index] - vehicle.position[0])
                        assert abs_delta_distance >= 0
                        vehicle.position = np.array([self.up_lanes[index], vehicle.position[1] + abs_delta_distance])
                        vehicle.direction = UP
                        vehicle.velocity = np.array([0., vehicle.velocity[0]])
                        continue

                if cross_down_lane_index[0].size != 0:
                    if change_direction:
                        index = cross_down_lane_index[0][0]
                        abs_delta_distance = delta_distance[0] - (self.down_lanes[index] - vehicle.position[0])
                        assert abs_delta_distance >= 0
                        vehicle.position = np.array([self.down_lanes[index], vehicle.position[1] - abs_delta_distance])
                        vehicle.direction = DOWN
                        vehicle.velocity = np.array([0., -vehicle.velocity[0]])
                        continue

                vehicle.position += delta_distance

            else:
                pass

            # come to the boundary
            # UP->RIGHT->DOWN->LEFT->UP
            x, y = vehicle.position[0], vehicle.position[1]
            velocity = np.linalg.norm(vehicle.velocity)

            if(x < 0 or y < 0 or x > self.width or y > self.height):
                if vehicle.direction == UP:
                    vehicle.direction = RIGHT
                    vehicle.position = np.array([x, self.right_lanes[-1]])
                    vehicle.velocity = np.array([velocity, 0.])
                elif vehicle.direction == DOWN:
                    vehicle.direction = LEFT
                    vehicle.position = np.array([x, self.left_lanes[0]])
                    vehicle.velocity = np.array([-velocity, 0.])
                elif vehicle.direction == LEFT:
                    vehicle.direction = UP
                    vehicle.position = np.array([self.up_lanes[0], y])
                    vehicle.velocity = np.array([0., velocity])
                elif vehicle.direction == RIGHT:
                    vehicle.direction = DOWN
                    vehicle.position = np.array([self.down_lanes[-1], y])
                    vehicle.velocity = np.array([0., -velocity])

        return


    
    def update_neighbor(self):
        """
        Update the neighbor for (V2V) vehicles, every 100ms
        """
        positions = np.array([[np.complex(vehicle.position[0], vehicle.position[1]) \
                                for vehicle in self.vehicles]])
        dist_matrix = np.abs(positions.T - positions)

        for i, vehicle in enumerate(self.vehicles):
            near_vehicles = np.argsort(dist_matrix[:, i])
            # distance to itself is 0
            vehicle.neighbor = near_vehicles[1]


    
    def update_interference(self):
        """
        Update the V2I and V2V interference from Tx to corresponding Rx on every Rb
        The V2V links which have finished the transmission (inactive) won't introduce interference
        """
        #? axis 0 => Tx id
        v2i_interference = np.zeros((self.num_vehicle, self.num_rb)) + NOISE_POWER
        #? axis 0 => Tx id, axis 1 => Rx id
        v2v_interference = np.zeros((self.num_vehicle, self.num_rb)) + NOISE_POWER


        #! we consider unicast
        #! interference from V2I links, number of V2I links = number of resource blocks = number of V2V links
        for i, vehicle_v2v_tx in enumerate(self.vehicles):
            vehicle_v2v_rx_id = vehicle_v2v_tx.neighbor
            #? traverse all vehicles whose V2I links produce interference
            for j in range(self.num_rb):
                #! assume V2I link (i) occupies resource block (i)
                interference_db =  POWER_LEVEL[0] - \
                                    self.v2v_channel.gain_with_fast_fading[j][vehicle_v2v_rx_id][j] + \
                                    2 * ANTENNA_GAIN_VEHICLE - NOISE_FIGURE_VEHICLE
                v2v_interference[i][j] += np.power(10,interference_db / 10)

        #! interference from V2V links
        for i, vehicle_v2v_tx in enumerate(self.vehicles):
            vehicle_v2v_rx_id = vehicle_v2v_tx.neighbor
            #? traverse all vehcicles whose V2V links produce interference
            for j, _ in enumerate(self.vehicles):
                if i == j or self.active_link[j] == False:
                    continue
                rb = self.selected_rb[j]
                interference_db = POWER_LEVEL[self.transmision_power[j]] - \
                                self.v2v_channel.gain_with_fast_fading[j, vehicle_v2v_rx_id, rb] + \
                                2 * ANTENNA_GAIN_VEHICLE - NOISE_FIGURE_VEHICLE
                assert not np.isinf(self.v2v_channel.gain_with_fast_fading[j, vehicle_v2v_rx_id, rb])
                assert not np.isinf(interference_db)
                v2v_interference[i, rb] += np.power(10, interference_db / 10)
        # assert not np.any(np.isnan(v2v_interference)) 
        # assert not np.any(np.isinf(v2v_interference))
        self.v2v_interference = 10 * np.log10(v2v_interference)

        #? necessary or not?
        #! interference from V2V links
        for i in range(self.num_vehicle):
            for j in range(self.num_vehicle):
                rb = self.selected_rb[j]
                if self.active_link[j] == False:
                    continue

                interference_db = POWER_LEVEL[self.transmision_power[j]] - \
                                    self.v2i_channel.gain_with_fast_fading[j, rb] + \
                                    ANTENNA_GAIN_BS + ANTENNA_GAIN_VEHICLE - NOISE_FIGURE_BS
                v2i_interference[i, rb] += np.power(10, interference_db / 10)
        self.v2i_interference = 10 * np.log10(v2i_interference)

    
    def update_interference_tx(self):
        """
        Update sensed interference at each transmitter, used in observation space
        """
        #? axis 0 => Tx id, axis 1 => Rx id
        v2v_interference = np.zeros((self.num_vehicle, self.num_rb)) + NOISE_POWER

        #! interference from V2I links
        for i in range(self.num_vehicle):
            #? traverse all vehicles whose V2I links produce interference
            for j in range(self.num_rb):
                #! assume V2I link (i) occupies resource block (i)
                interference_db =  POWER_LEVEL[0] - \
                                    self.v2v_channel.gain_with_fast_fading[j][i][j] + \
                                    2 * ANTENNA_GAIN_VEHICLE - NOISE_FIGURE_VEHICLE
                v2v_interference[i][j] += np.power(10,interference_db / 10)

        #! interference from V2V links
        for i in range(self.num_vehicle):
            #? traverse all vehcicles whose V2V links produce interference
            for j, _ in enumerate(self.vehicles):
                if i == j or self.active_link[j] == False:
                    continue
                rb = self.selected_rb[j]
                interference_db = POWER_LEVEL[self.transmision_power[j]] - \
                                self.v2v_channel.gain_with_fast_fading[j, i, rb] + \
                                2 * ANTENNA_GAIN_VEHICLE - NOISE_FIGURE_VEHICLE
                v2v_interference[i, rb] += np.power(10, interference_db / 10)
        # assert not np.any(np.isnan(v2v_interference)) 
        # assert not np.any(np.isinf(v2v_interference))
        self.v2v_interference_tx = 10 * np.log10(v2v_interference)


    #! The achievable rate of those V2V links which have finished the transmission (inactive) are also calculated
    #! Carefully deal with this in the reward calculation
    def calculate_rate(self):
        """
        Calculate the acheviable rate of each V2V and V2I link.
        """
        #! for V2I links, vehicle (i) occupies RB (i)
        assert self.num_vehicle == self.num_rb
        v2i_signal = POWER_LEVEL[0] - self.v2i_channel.gain_with_fast_fading.diagonal() + \
                        ANTENNA_GAIN_VEHICLE + ANTENNA_GAIN_BS - NOISE_FIGURE_BS
        v2i_signal = np.power(10, v2i_signal / 10)
        v2i_interference = np.power(10, self.v2i_interference.diagonal() / 10)
        self.v2i_rate = np.log2(1 + v2i_signal / v2i_interference)

        #! for each transmision link, consider the occupied RB only
        v2v_interference = np.zeros(self.num_vehicle)
        v2v_signal = np.zeros(self.num_vehicle)
        for i, vehicle in enumerate(self.vehicles):
            rb = self.selected_rb[i]
            rx_id = vehicle.neighbor
            v2v_interference[i] = self.v2v_interference[i, rb]
            v2v_signal[i] = POWER_LEVEL[self.transmision_power[i]] - \
                                self.v2v_channel.gain_with_fast_fading[i, rx_id, rb] + \
                                2 * ANTENNA_GAIN_VEHICLE - NOISE_FIGURE_VEHICLE

        v2v_interference = np.power(10, v2v_interference / 10)
        v2v_signal = np.power(10, v2v_signal / 10)
        self.v2v_rate = np.log2(1 + v2v_signal / v2v_interference)  #! [unit: bps]
        pass


    #todo integrate this code into the function "update _interference"
    def update_interference_benchmark(self):
        """
        Update the sensed interference for the benchmark method
        """
        #? axis 0 => Tx id
        v2i_interference = np.zeros((self.num_vehicle, self.num_rb)) + NOISE_POWER
        #? axis 0 => Tx id, axis 1 => Rx id
        v2v_interference = np.zeros((self.num_vehicle, self.num_rb)) + NOISE_POWER

        #! we consider unicast
        #! interference from V2I links, number of V2I links = number of resource blocks = number of V2V links
        for i, vehicle_v2v_tx in enumerate(self.vehicles):
            vehicle_v2v_rx_id = vehicle_v2v_tx.neighbor
            #? traverse all vehicles whose V2I links produce interference
            for j in range(self.num_rb):
                #! assume V2I link (i) occupies resource block (i)
                interference_db =  POWER_LEVEL[0] - \
                                    self.v2v_channel.gain_with_fast_fading[j][vehicle_v2v_rx_id][j] + \
                                    2 * ANTENNA_GAIN_VEHICLE - NOISE_FIGURE_VEHICLE
                v2v_interference[i][j] += np.power(10,interference_db / 10)

        #! interference from V2V links
        for i, vehicle_v2v_tx in enumerate(self.vehicles):
            vehicle_v2v_rx_id = vehicle_v2v_tx.neighbor
            #? traverse all vehcicles whose V2V links produce interference
            for j, _ in enumerate(self.vehicles):
                if i == j or self.active_link_benchmark[j] == False:
                    continue
                rb = self.selected_rb_benchmark[j]
                interference_db = POWER_LEVEL[self.transmision_power_benchmark[j]] - \
                                self.v2v_channel.gain_with_fast_fading[j, vehicle_v2v_rx_id, rb] + \
                                2 * ANTENNA_GAIN_VEHICLE - NOISE_FIGURE_VEHICLE

        self.v2v_interference_benchmark = 10 * np.log10(v2v_interference)

        #? necessary or not?
        #! interference from V2V links
        for i in range(self.num_vehicle):
            for j in range(self.num_vehicle):
                rb = self.selected_rb_benchmark[j]
                if self.active_link_benchmark[j] == False:
                    continue
                interference_db = POWER_LEVEL[self.transmision_power_benchmark[j]] - \
                                    self.v2i_channel.gain_with_fast_fading[j, rb] + \
                                    ANTENNA_GAIN_BS + ANTENNA_GAIN_VEHICLE - NOISE_FIGURE_BS
                v2i_interference[i, rb] += np.power(10, interference_db / 10)
        self.v2i_interference_benchmark = 10 * np.log10(v2i_interference)



    #todo integrate this part into the function "calculate_rate" 
    def calculate_rate_benchmark(self):
        """
        Calculate the acheviable rate for the benchmark method
        """
        #! for V2I links, vehicle (i) occupies RB (i)
        v2i_signal = POWER_LEVEL[0] - self.v2i_channel.gain_with_fast_fading.diagonal() + \
                        ANTENNA_GAIN_VEHICLE + ANTENNA_GAIN_BS - NOISE_FIGURE_BS
        v2i_signal = np.power(10, v2i_signal / 10)
        v2i_interference = np.power(10, self.v2i_interference_benchmark.diagonal() / 10)
        self.v2i_rate_benchmark = np.log2(1 + v2i_signal / v2i_interference)

        #! for each transmision link, consider the occupied RB only
        v2v_interference = np.zeros(self.num_vehicle)
        v2v_signal = np.zeros(self.num_vehicle)
        for i, vehicle in enumerate(self.vehicles):
            rb = self.selected_rb_benchmark[i]
            rx_id = vehicle.neighbor
            v2v_interference[i] = self.v2v_interference_benchmark[i, rb]
            v2v_signal[i] = POWER_LEVEL[self.transmision_power_benchmark[i]] - \
                                self.v2v_channel.gain_with_fast_fading[i, rx_id, rb] + \
                                2 * ANTENNA_GAIN_VEHICLE - NOISE_FIGURE_VEHICLE

        v2v_interference = np.power(10, v2v_interference / 10)
        v2v_signal = np.power(10, v2v_signal / 10)
        self.v2v_rate_benchmark = np.log2(1 + v2v_signal / v2v_interference)  #! [unit: bps]
        pass

    #todo specify weights of different part of reward, and minimum rate
    def calculate_reward(self):
        """
        Reward is composed of four parts:
        1. sum of V2I rate
        2. penalty of V2I rate that smaller than the minimum rate
        3. sum of V2V rate
        4. penalty of V2V rate that smaller than the minimum rate

        return the difference of reward between intended method and benchmark method
        """
        # Reward = namedtuple('Reward', ['total', 'r1', 'r2', 'r3', 'r4'])
        #todo reward weights should be adjustable hyper-parameters
        c1, c2, c3, c4 = self.reward_weights
        min_v2i_rate = 10        #! [unit: Mbps]

        reward_1 = np.sum(self.v2i_rate)

        penalty_v2i_rate = self.v2i_rate - min_v2i_rate
        penalty_v2i_rate[penalty_v2i_rate >= 0] = 1
        penalty_v2i_rate[penalty_v2i_rate < 0] = 0
        reward_2 = np.sum(penalty_v2i_rate)

        v2v_rate = copy.deepcopy(self.v2v_rate)
        v2v_rate[np.logical_not(self.active_link)] = 1
        # v2v_rate[self.active_link] = 0
        reward_3 = np.sum(v2v_rate)

        penalty_v2v_rate = self.v2v_rate - self.remaining_load / (self.remaining_time / 1000) / BANDWIDTH
        penalty_v2v_rate[penalty_v2v_rate >= 0] = 1
        penalty_v2v_rate[penalty_v2v_rate < 0] = 0
        penalty_v2v_rate[np.logical_not(self.active_link)] = 0
        reward_4 = np.sum(penalty_v2v_rate)
        # print("Baseline reward: ", reward_1, reward_2, reward_3, reward_4)
        total = c1 * reward_1 + c2 * reward_2 + c3 * reward_3 + c4 * reward_4

        #! reward of benchmark
        reward_1 = np.sum(self.v2i_rate_benchmark)

        penalty_v2i_rate = self.v2i_rate_benchmark - min_v2i_rate
        penalty_v2i_rate[penalty_v2i_rate >= 0] = 1
        penalty_v2i_rate[penalty_v2i_rate < 0] = 0
        reward_2 = np.sum(penalty_v2i_rate)

        v2v_rate = copy.deepcopy(self.v2v_rate_benchmark)
        v2v_rate[np.logical_not(self.active_link_benchmark)] = 1
        # v2v_rate[self.active_link] = 0
        reward_3 = np.sum(v2v_rate)

        penalty_v2v_rate = self.v2v_rate_benchmark - \
                                    self.remaining_load_benchmark / (self.remaining_time / 1000) / BANDWIDTH
        penalty_v2v_rate[penalty_v2v_rate >= 0] = 1
        penalty_v2v_rate[penalty_v2v_rate < 0] = 0
        penalty_v2v_rate[np.logical_not(self.active_link_benchmark)] = 0
        reward_4 = np.sum(penalty_v2v_rate)
        # print("Proposed reward: ", reward_1, reward_2, reward_3, reward_4)
        total_benchmark = c1 * reward_1 + c2 * reward_2 + c3 * reward_3 + c4 * reward_4

        #! normalize the reward
        return (total - total_benchmark)


    def normalize_observation(self):
        """
        Normalize the joint observations
        """
        raise NotImplementedError




#! Simplified simulation scenario, consider only one block which all vehicles move around
class SimplifiedEnvironment(Environment):
    def __init__(self, num_vehicle = 4, num_packet = 4, reward_weights = WEIGHTS, training = True) -> None:
        super().__init__(num_vehicle, num_packet, reward_weights, training)

        #! first axis corresponds to outside road, second axis corresponds to inside road
        self.up_lanes = 3.5 / 2
        self.down_lanes = 250 - 3.5 / 2
        self.left_lanes = 3.5 / 2
        self.right_lanes = 433 - 3.5 / 2
        self.width, self.height = np.array([250, 433])


    def init_vehicles(self):
        if len(self.vehicles) > 0:
            self.vehicles = []
        assert self.num_vehicle  == 4 or self.num_vehicle == 8
        if self.num_vehicle == 4:
            for i in range(self.num_vehicle):
                #! initialize the directional velocity
                direction = i % 4
                velocity = np.random.randint(VEHICLE_VELOCITY[0], VEHICLE_VELOCITY[1])
                position = None
                if direction == UP:
                    velocity = np.array([0., velocity])
                    position = np.array([self.up_lanes, np.random.randint(0.2 * self.height, 0.8 * self.height)])
                elif direction == DOWN:
                    velocity = np.array([0., -velocity])
                    position = np.array([self.down_lanes, np.random.randint(0.2 * self.height, 0.8 * self.height)])
                elif direction == LEFT:
                    velocity = np.array([-velocity, 0.])
                    position = np.array([np.random.randint(0.2 * self.width, 0.8 * self.width), self.left_lanes])
                elif direction == RIGHT:
                    velocity = np.array([velocity, 0.])
                    position = np.array([np.random.randint(0.2 * self.width, 0.8 * self.width), self.right_lanes])
                self.vehicles.append(Vehicle(i, position, direction, velocity))
        if self.num_vehicle == 8:
            for i in range(self.num_vehicle):
                #! initialize the directional velocity
                direction = i % 4
                bias = i // 4;
                velocity = np.random.randint(VEHICLE_VELOCITY[0], VEHICLE_VELOCITY[1])
                position = None
                if direction == UP:
                    velocity = np.array([0., velocity])
                    position = np.array([self.up_lanes, np.random.randint(0.2 * self.height, 0.3 * self.height) + self.height / 2 * bias])
                elif direction == DOWN:
                    velocity = np.array([0., -velocity])
                    position = np.array([self.down_lanes, np.random.randint(0.2 * self.height, 0.3 * self.height) + self.height / 2 * bias])
                elif direction == LEFT:
                    velocity = np.array([-velocity, 0.])
                    position = np.array([np.random.randint(0.2 * self.width, 0.3 * self.width) + self.width / 2 * bias, self.left_lanes])
                elif direction == RIGHT:
                    velocity = np.array([velocity, 0.])
                    position = np.array([np.random.randint(0.2 * self.width, 0.3 * self.width) + self.width / 2 * bias, self.right_lanes])
                self.vehicles.append(Vehicle(i, position, direction, velocity))



    def update_position(self):
        """
        Vehicle moves clockwise
        """
        for i, vehicle in enumerate(self.vehicles):
            delta_distance = vehicle.velocity * self.slow_period
            #! clockwise
            if vehicle.direction == UP:
                if (vehicle.position[1] + delta_distance[1]) >= (self.height - 3.5 / 2):
                    vehicle.direction = RIGHT
                    vehicle.velocity = np.array([vehicle.velocity[1], 0.])
                    vehicle.position = np.array([3.5 / 2, self.right_lanes])
                else:
                    vehicle.position += delta_distance
            elif vehicle.direction == DOWN:
                if (vehicle.position[1] + delta_distance[1]) <= (3.5 / 2):
                    vehicle.direction = LEFT
                    vehicle.velocity = np.array([vehicle.velocity[1], 0.])
                    vehicle.position = np.array([self.width - 3.5 / 2, self.left_lanes])
                else:
                    vehicle.position += delta_distance
            elif vehicle.direction == LEFT:
                if (vehicle.position[0] + delta_distance[0]) <= (3.5 / 2):
                    vehicle.direction = UP
                    vehicle.velocity = np.array([0., -vehicle.velocity[0]])
                    vehicle.position = np.array([self.up_lanes, 3.5 / 2])
                else:
                    vehicle.position += delta_distance
            elif vehicle.direction == RIGHT:
                if (vehicle.position[0] + delta_distance[0]) >= (self.width - 3.5 / 2):
                    vehicle.direction = DOWN
                    vehicle.velocity = np.array([0., -vehicle.velocity[0]])
                    vehicle.position = np.array([self.down_lanes, self.height - 3.5 / 2])
                else:
                    vehicle.position += delta_distance

        pass
