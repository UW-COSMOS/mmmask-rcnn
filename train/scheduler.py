"""
Helper class for 4 step training outlined in
https://arxiv.org/pdf/1506.01497.pdf
"""


class Scheduler:
    RPN = 0
    CLASS_HEAD = 1
    JOINT = 3

    def __init__(self, switch_period):
        """
        Initialize a scheduler object
        :param switch_period:
        """
        self.switch_period = switch_period
        self.period = self.RPN

    def step(self):
        if not self.period == self.JOINT:
            self.period += 1
