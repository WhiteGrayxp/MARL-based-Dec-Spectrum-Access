# This simulator follows the evaluation methodology in 3GPP TR 36.885, refer to Annex A and Urban case for details.

import numpy as np
import math


#! for simplified case, Area size is [250, 433]
simplified = False

if simplified:
    AREA_SIZE = (250.0, 433.0)
else:
    AREA_SIZE = (500.0, 866.0)

SLOW_PERIOD = 0.1              # ![uinit: s] = 100ms


class Vehicle(object):
    def __init__(self, id_, position, direction, velocity) -> None:
        self.id = id_
        self.type = None        # V2V or V2I

        self.position = position
        self.velocity = velocity
        self.direction = direction

        #! neighbor = receiver, consider unicast only
        self.neighbor = None


    def __str__(self) -> str:
        return 'ID : {}, Position : {}, Direction : {}, Velocity : {}'\
                    .format(self.id, self.position, self.direction, self.velocity)


# WINNER+ B1 model
class V2VChannel(object):
    """
    Simulation model of the V2V channel, including path loss and shadowing
    """
    def __init__(self, num_vehicle, num_rb) -> None:
        self.h_bs = 1.5     # base station antenna height, (m)
        self.h_ms = 1.5     # mobile station
        self.fc = 2         # carrier frequency, (GHz)
        self.decor_dist = 10    # decorrelation distance, (m)
        self.shadow_std = 3     # standard deviation of the shadowing, (dB)
        self.d_bp = 4 * (self.h_bs - 1) * (self.h_ms - 1) * self.fc * 10 / 3    # break down distance

        self.num_vehicle = num_vehicle
        self.num_rb = num_rb
        " record channel conditions for all vehicles"
        self.delta_dist = np.zeros([self.num_vehicle, self.num_vehicle])
        self.shadowing = np.random.normal(0, self.shadow_std, [self.num_vehicle, self.num_vehicle])
        self.gain = np.eye(self.num_vehicle, self.num_vehicle)
        self.fastfading = 20 * np.log10(np.random.rayleigh(1, [self.num_vehicle, self.num_vehicle, self.num_rb]))

    @property
    def gain_with_fast_fading(self):
        return np.repeat(self.gain[:, :, np.newaxis], self.num_rb, axis = 2) + self.fastfading

    
    def update_gain(self, vehicles):
        """
        Update the abs channel gain for all V2V links
        """
        assert len(vehicles) == self.num_vehicle

        for i in range(self.num_vehicle):
            for j in range(i+1, self.num_vehicle):
                self.gain[i][j] = self.gain[j][i] = self.update_pathloss(vehicles[i].position, vehicles[j].position)
                self.delta_dist[i][j] = self.delta_dist[j][i] = (np.linalg.norm(vehicles[i].velocity) \
                                                                 + np.linalg.norm(vehicles[j].velocity)) * SLOW_PERIOD
        # self.gain += 40 * np.eye(self.num_vehicle, self.num_vehicle)
        np.fill_diagonal(self.gain, 50)
        self.update_shadowing()
        self.gain += self.shadowing
        pass


    def update_pathloss(self, pos_1, pos_2) -> float:
        """
        Update the path loss at (100 * n) ms
        :param pos_1: position of transmitter
        :param pos_2: position of receiver
        """
        dist_1 = abs(pos_1[0] - pos_2[0])
        dist_2 = abs(pos_1[1] - pos_2[1])
        dist = math.hypot(dist_1, dist_2) + 0.001

        if min(dist_1, dist_2) < 7.0:
            return self.pathloss_los(dist)
        else:
            return min(self.pathloss_nlos(dist_1, dist_2), self.pathloss_nlos(dist_2, dist_1))


    def pathloss_los(self, dist):
        """
        Calculate the path loss when in line-of-sight
        """
        # Pathloss at 3 m is used if the distance is less than 3 m.
        if dist <= 3.0:
            return 22.7 * np.log10(3) + 41.0 + 20 * np.log10(self.fc / 5)
        else:
            if dist < self.d_bp:
                return 22.7 * np.log10(dist) + 41.0 + 20 * np.log10(self.fc / 5)
            else:
                #? self.h_bs - 1
                return 40.0 * np.log10(dist) + 9.45 - 17.3 * np.log10(self.h_bs) - 17.3 * np.log10(self.h_ms) + 2.7 * np.log10(self.fc / 5)


    def pathloss_nlos(self, dist_1, dist_2):
        """
        Calculate the path loss when in none-line-of-sight
        """
        nj = max(2.8 - 0.0024 * dist_2, 1.84)       # some confusing but useful number
        return self.pathloss_los(dist_1) + 20.0 - 12.5 * nj + 10 * nj * np.log10(dist_2) + 3.0 * np.log10(self.fc / 5)



    def update_shadowing(self):
        """
        Update the shadowing at (100 * n) ms
        :param delta_dist: change in distance of the link from time n-1 to time n
        :param prev_shadowing: shadowing at time n-1
        """
        assert self.delta_dist.shape == self.shadowing.shape
        shadowing =  np.exp(-1.0 * (self.delta_dist / self.decor_dist)) * self.shadowing + \
                np.sqrt(1 - np.exp(-2.0 * (self.delta_dist / self.decor_dist))) * np.random.normal(0, self.shadow_std, [self.num_vehicle, self.num_vehicle])
        # np.fill_diagonal(self.shadowing, 0)
        shadowing = np.triu(shadowing)
        shadowing += shadowing.T - np.diag(shadowing.diagonal())
        self.shadowing = shadowing

    
    def update_fast_fading(self):
        # ! [unit: dBm]
        self.fastfading = 20 * np.log10(np.random.rayleigh(1, [self.num_vehicle, self.num_vehicle, self.num_rb]))



class V2IChannel(object):
    """
    Simulation model of the V2V channel, including path loss and shadowing
    """
    def __init__(self, num_vehicle, num_rb) -> None:
        self.h_bs = 25      # base station antenna height, (m)
        self.h_ms = 1.5     # monile station
        self.fc = 2         # carrier frequency, (GHz)
        self.decor_dist = 50    # decorrelation frequency, (m)
        self.shadow_std = 8     # standard deviation of the shadowing, (dB)

        self.bs_pos = np.array(AREA_SIZE) / 2     # assume the base station is at the center of the grids

        self.num_vehicle = num_vehicle
        self.num_rb = num_rb
        " record channel conditions for all vehicles"
        self.delta_dist = np.zeros(self.num_vehicle)
        self.shadowing = np.random.normal(0, self.shadow_std, self.num_vehicle)
        self.gain = np.zeros(self.num_vehicle)
        self.fastfading = 20 * np.log10(np.random.rayleigh(1, [self.num_vehicle, self.num_rb]))

    @property
    def gain_with_fast_fading(self):
        return np.repeat(self.gain[:, np.newaxis], self.num_rb, axis = 1) + self.fastfading

    def update_gain(self, vehicles):
        """
        Update channel gain for all V2I links
        """
        assert len(vehicles) == self.num_vehicle

        for i in range(self.num_vehicle):
                self.gain[i] = self.update_pathloss(vehicles[i].position)
                self.delta_dist[i] = np.linalg.norm(vehicles[i].velocity)  * SLOW_PERIOD

        self.update_shadowing()
        self.gain += self.shadowing

    def update_pathloss(self, pos):
        """
        Update the path loss. Consider line-of-sight only.
        """
        # pos = np.array(pos)
        dist = np.linalg.norm(pos - self.bs_pos)
        # in kilometers
        R = np.sqrt(dist ** 2 + (self.h_bs - self.h_ms) ** 2) / 1000
        return 128.1 + 37.6 * np.log10(R)


    def update_shadowing(self):
        """
        Update the shadowing at time (100 * n) ms. Since we consider only one base station here, so modify slightly from standard 3GPP profile
        """
        self.shadowing = np.exp(-1.0 * (self.delta_dist / self.decor_dist)) * self.shadowing + \
                np.sqrt(1 - np.exp(-2.0 * (self.delta_dist / self.decor_dist))) * np.random.normal(0, self.shadow_std, self.delta_dist.shape)


    def update_fast_fading(self):
        # ! [unit: dBm]
        self.fastfading = 20 * np.log10(np.random.rayleigh(1, [self.num_vehicle, self.num_rb]))