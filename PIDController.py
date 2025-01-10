class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.prev_error = 0  # Previous error for derivative term
        self.integral = 0  # Cumulative error for integral term

    def compute(self, error, dt=1.0):
        """
        Compute the output of the PID controller.
        :param error: The current error (setpoint - current_value).
        :param dt: Time step since last computation (default = 1.0).
        :return: PID output.
        """
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output
