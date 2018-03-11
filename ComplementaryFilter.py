import numpy as np
import quaternion

class ImuMsg(object):
    mag = np.zeros((3,1))
    acc = np.zeros((3,1))
    gyro = np.zeros((3,1))

class ComplementaryFilter(object):

    def __init__(self, initialReading):
        # 
        # Initial guess for the orientation will be a direct calculation.
        acc_norm = initialReading.acc / np.norm(initialReading.acc)
        a_x, a_y, a_z = acc_norm
        # 
        # Eq 25.
        q_acc = None
        if initialReading.acc[2] >= 0:
            tmp = 2*(a_z+1)
            q_acc_0 = np.sqrt((a_z + 1) / 2)
            q_acc_1 = - a_y / np.sqrt(tmp)
            q_acc_2 = a_x / np.sqrt(tmp)
            q_acc_3 = 0
            q_acc = np.quaternion(q_acc_0, q_acc_1,q_acc_2, q_acc_3)
        else:
            tmp = 2*(1-a_z)
            q_acc_0 = - a_y / np.sqrt(tmp)
            q_acc_1 = np.sqrt((1 - a_z) / 2)
            q_acc_2 = 0
            q_acc_3 = a_x / np.sqrt(tmp)
            q_acc = np.quaternion(q_acc_0, q_acc_1,q_acc_2, q_acc_3)
        # 
        # Eq 26.
        R_acc = quaternion.as_rotation_matrix(q_acc)
        l = np.dot(R_acc.T, mag)
        l_x, l_y, l_z = l
        # 
        # Eq 31.
        gamma = np.sqrt(l_x ** 2 + l_y ** 2)
        q_mag_0 = np.sqrt(gamma + l_x * np.sqrt(gamma)) / np.sqrt(2 * gamma)
        q_mag_1 = 0
        q_mag_2 = 0
        q_mag_3 = l_y / (np.sqrt(2) * np.sqrt(gamma + l_x * np.sqrt(gamma)))
        q_mag = np.quaternion(q_mag_0, q_mag_1, q_mag_2, q_mag_3)
        # 
        # Eq 36.
        q_body =  q_acc * q_mag

    def observation(self, imumsg):
        pass