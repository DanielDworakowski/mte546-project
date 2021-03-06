import numpy as np
import quaternion

class ImuMsg(object):
    mag = np.zeros((3,))
    acc = np.zeros((3,))
    gyro = np.zeros((3,))
    ts = -1

    def __repr__(self):
        ret = ''
        ret +='--------\n'
        ret += 'ts: ' + str(self.ts) + '\n'
        ret += 'MAG: \n'
        ret += str(self.mag)
        ret += '\nACC: \n'
        ret += str(self.acc)
        ret += '\nGYRO: \n'
        ret += str(self.gyro)
        ret += '\n--------'
        return ret

# 
# http://www.mdpi.com/1424-8220/15/8/19302/pdf
# http://docs.ros.org/lunar/api/imu_complementary_filter/html/complementary__filter_8cpp_source.html
class ComplementaryFilter(object):

    def __init__(self, initialReading, q_init = None):
        # 
        # Filter parameters.
        self.alpha = 0.009
        self.beta = 0.01
        self.gyroBias = np.zeros((3,)) # No prior. 
        self.gyroBias = np.array([-0.002, 0.020933373, 0.081622879]) # Dataset IMU.
        # self.gyroBias = np.array([0.011764301, -0.013778673, 0.015057202]) # Sparkfun IMU.
        self.prev_imumsg = initialReading
        self.g = 9.81
        # 
        # Initial guess for the orientation will be a direct calculation
        # as described in section 5.4.
        acc_norm = initialReading.acc / np.linalg.norm(initialReading.acc)
        a_x, a_y, a_z = acc_norm
        # 
        # Eq 25.
        q_acc = None
        if a_z >= 0:
            tmp = np.sqrt(2*(a_z+1))
            q_acc_0 = np.sqrt((a_z + 1) / 2)
            q_acc_1 = -a_y / tmp
            q_acc_2 =  a_x / tmp
            q_acc_3 = 0
            q_acc = np.quaternion(q_acc_0, q_acc_1,q_acc_2, q_acc_3)
        else:
            tmp = np.sqrt(2*(1-a_z))
            q_acc_0 = - a_y / tmp
            q_acc_1 = np.sqrt((1 - a_z) / 2)
            q_acc_2 = 0
            q_acc_3 = a_x / tmp
            q_acc = np.quaternion(q_acc_0, q_acc_1,q_acc_2, q_acc_3)
        # 
        # Orientation estimate based entirely off one magnetometer reading. 
        self.q_body = q_acc
        # 
        # If there is a magnetometer message proceed. 
        if initialReading.mag is not None:
            # 
            # Eq 26.
            R_acc = quaternion.as_rotation_matrix(q_acc)
            l = np.dot(R_acc.T, initialReading.mag)
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
            self.q_body = q_acc * q_mag
        # 
        # Check if there is a quaternion to initialize with instead of the direct guess.
        if q_init is not None:
            self.q_body = q_init.inverse() # State is inverse.

    def _predict(self, base, gyro, dt):
        # 
        # Eq 40.
        w_x, w_y, w_z = gyro
        sub_array = np.array([[0, -w_z, w_y], [w_z, 0, -w_x], [-w_y, w_x, 0]])
        omega = np.zeros((4,4))
        omega[0, 1:4] = gyro.T
        omega[1:4, 0] = -gyro
        omega[1:4, 1:4] = -sub_array
        # 
        # Eq 42. 
        npret = quaternion.as_float_array(base) + np.dot(omega, quaternion.as_float_array(base)) * dt * 0.5
        # 
        # Weird thing to allow for constructor to work with a numpy array.
        return np.quaternion(*npret).normalized()

    def _interpolate(self, q2, gain):
        qI = np.quaternion(1, 0, 0, 0)
        # 
        # Eq 50.
        qbar = None
        q2np = quaternion.as_float_array(q2)
        if q2np[0] > 0.9:
            qbar = (1 - gain) * qI + gain * q2
        else:
            # 
            # Eq 48.
            omega = np.arccos(q2np[0])
            qbar = np.sin((1 - gain) * omega) / np.sin(omega) * qI + np.sin(gain * omega) / np.sin(omega) * q2
        # 
        # Eq 51.
        qHat = qbar.normalized()
        return qHat

    def adaptiveGain(self, acc):
        # 
        # Eq 60.
        e_m  = np.abs(np.linalg.norm(acc) - self.g) / self.g
        # 
        # Fig 5. 
        ret = None
        if e_m < 0.1:
            ret = 1
        elif e_m > 0.2:
            ret = 0
        else:
            ret = (-10 * e_m + 2)
        # 
        # Scale wrt the normal gain.
        return ret * self.alpha

    def _correct(self, imumsg, q_pred):
        # 
        # Eq 44.
        acc = imumsg.acc / np.linalg.norm(imumsg.acc)
        g_pred = np.dot(quaternion.as_rotation_matrix(q_pred.inverse()), acc) ##### DOUBLE CHECK is it q_pred.inverse?????
        # 
        # Eq 47.
        g_x, g_y, g_z = g_pred
        dq_0 = np.sqrt((g_z + 1) / 2)
        dq_1 = -g_y / (2 * dq_0)
        dq_2 =  g_x / (2 * dq_0)
        dq_3 = 0

        dq = np.quaternion(dq_0, dq_1, dq_2, dq_3)
        # 
        # Correct dq.
        alpha = self.adaptiveGain(imumsg.acc)
        dqHat = self._interpolate(dq, alpha)
        # 
        # Update q_pred.
        q_pred_update = q_pred * dqHat
        if imumsg.mag is not None:
            # 
            # Rotate the magnetice field vector.
            # Eq 54.
            l = np.dot(quaternion.as_rotation_matrix(q_pred_update).T, imumsg.mag)
            l_x, l_y, l_z = l
            # 
            # Calculate the delta quaternion Eq 58.
            gamma = l_x ** 2 + l_y ** 2
            dq_0 = np.sqrt(gamma + l_x * np.sqrt(gamma)) / np.sqrt(2 * gamma)
            dq_1 = 0
            dq_2 = 0
            dq_3 = l_y / np.sqrt(2 * (gamma + l_x * np.sqrt(gamma)))
            dq = np.quaternion(dq_0, dq_1, dq_2, dq_3)
            dqHat = self._interpolate(dq, self.beta)
            # 
            # Eq 59.
            q_pred_update = q_pred_update * dqHat
        # 
        # Normalize.
        return q_pred_update.normalized()
    # 
    # http://docs.ros.org/lunar/api/imu_complementary_filter/html/complementary__filter_8cpp_source.html#l00540
    def _isSteadyState(self, imumsg):
        acc_norm = np.linalg.norm(imumsg.acc)
        accThresh = 0.1
        angVelThresh = 0.2
        deltaAngThresh = 0.01
        if np.abs(acc_norm - self.g) > accThresh:
            return False
        if np.any(np.abs(imumsg.gyro - self.prev_imumsg.gyro) > deltaAngThresh):
            return False
        if np.any(np.abs(imumsg.gyro - self.gyroBias) > angVelThresh):
            return False
        return True

    def _estimateBias(self, imumsg):
        # 
        # Estimate the bias here.
        if self._isSteadyState(imumsg):
            self.gyroBias += 0.01 * (imumsg.gyro - self.gyroBias)
        # 
        # Return the message with the bias removed.
        unbiased = imumsg
        unbiased.gyro -= self.gyroBias
        # print(self.gyroBias)
        return unbiased

    def observation(self, imumsg):
        # 
        # Calculate the dt. 
        dt = imumsg.ts - self.prev_imumsg.ts
        # 
        # Update the last measurement. 
        q_t_ = self.q_body
        # 
        # Bias estimation.
        filtered = self._estimateBias(imumsg)
        # 
        # Prediction. 
        pred = self._predict(q_t_, filtered.gyro, dt)
        # 
        # Correction. 
        self.q_body = self._correct(filtered, pred)
        # 
        # Store message as previous.
        self.prev_imumsg = imumsg

    def getEuler(self):
        b = quaternion.as_euler_angles(self.q_body.inverse()) # State is inverse.
        b *= 180. / np.pi
        return b

    def getState(self):
        return quaternion.as_float_array(self.q_body.inverse())

    def __repr__(self):
        return 'Body Pos: ' + str(self.getEuler())