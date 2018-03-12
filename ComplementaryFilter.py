import numpy as np
import quaternion

class ImuMsg(object):
    mag = np.zeros((3,1))
    acc = np.zeros((3,1))
    gyro = np.zeros((3,1))

class ComplementaryFilter(object):

    def __init__(self, initialReading):
        # 
        # Filter parameters.
        self.alpha = 0.01
        self.beta = 0.01
        self.gyroBias = np.zeros((3,1))
        self.prev_imumsg = ImuMsg()
        self.g = 9.81
        # 
        # Initial guess for the orientation will be a direct calculation
        # as described in section 5.4.
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
        self.q_body =  q_acc * q_mag
        self.q_t_ = self.q_body


    def predict(self, gyro, dt):
        # 
        # Eq 40.
        w_x, w_y, w_z = gyro
        sub_array = np.array([[0, -w_z, w_y], [w_z, 0, -w_x], [-w_y, w_x, 0]])
        omega = np.zeros((4,4))
        omega[0, 1:3] = gyro.T
        omega[1:3, 0] = -gyro
        omega[1:3, 1:3] = -sub_array
        # 
        # Eq 42. 
        return q_t_ + np.dot(omega, q_t_) * dt

    def _interpolate(self, q2, gain):
        qI = np.quaternion(1, 0, 0, 0)
        # 
        # Eq 50.
        qbar = None
        if q2[0] > 0.9:
            qbar = (1 - gain) * qI + gain * q2
        else:
            # 
            # Eq 48.
            omega = np.acos(q2[0])
            qbar = np.sin((1 - gain) * omega) / np.sin(omega) * qI + np.sin(gain * omega) / np.sin(omega) * q2
        # 
        # Eq 51.
        qHat = qbar / np.norm(qbar)
        return qHat

    def adaptiveGain(self, acc):
        # 
        # Eq 60.
        e_m  = np.abs(np.norm(acc) - self.g) / self.g
        # 
        # Fig 5. 
        if e_m < 0.1:
            return 1
        elif e_m > 0.2:
            return 0
        else:
            return -10 * e_m + 2

    def correct(self, imumsg, q_pred):
        # 
        # Eq 44.
        acc = imumsg.acc / np.norm(imumsg.acc)
        norm = np.norm(acc)
        g_pred = np.dot(quaternion.as_rotation_matrix(quaternion.inverse()), acc)
        # 
        # Eq 47.
        g_x, g_y, g_z = g_pred
        dq_0 = np.sqrt((g_z + 1) / 2)
        dq_1 = - g_y / (np.sqrt(2 * (g_z + 1)))
        dq_2 = g_x / (np.sqrt(2 * (g_z + 1)))
        dq_3 = 0
        dq = np.quaternion(dq_0, dq_1, dq_2, dq_3)
        # 
        # Correct dq.
        alpha = self.adaptiveGain(imumsg.acc)
        dqHat = self._interpolate(dq, alpha)
        # 
        # Update q_pred.
        q_pred_update = q_pred * dqHat
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
        self.q_body = q_pred_update / np.norm(q_pred_update)

    def _estimateBias(self, imumsg):
        # 
        # Estimate the bias here.
        #######
        #######
        #######
        # 
        # Return the message with the bias removed.
        unbiased = imumsg
        unbiased.gyro -= self.gyroBias
        return imumsg

    def observation(self, imumsg, dt):
        # 
        # Update the last measurement. 
        self.q_t_ = self.q_body
        # 
        # Bias estimation.
        filtered = self._estimateBias(imumsg)
        # 
        # Prediction. 
        q_pred = self.predict(filtered.gyro, dt)
        # 
        # Correction. 
        self.q_body = self.correct(filtered, q_pred)
        # 
        # Store message as previous.
        self.prev_imumsg = imumsg