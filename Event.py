import numpy as np
import pandas as pd
import random
import math


class Event:
    def __init__(self, day, num_of_days, event_type):

        self.event_direction = random.choice([-1, 1])
        self.event_type = event_type
        self.event_magnitude = self.set_magnitude()
        self.event_duration = random.randint(0, num_of_days-day)
        self.max_point = random.randint(0, self.event_duration) + day
        self.sector = "none"

    def get_sector(self):
        return self.sector

    def get_event_duration(self):
        return self.event_duration

    def get_multiplier(self):
        return self.event_magnitude

    def get_max_point(self):
        return self.max_point

    def get_direction(self):
        direction = "positive"
        if self.event_direction < 0:
            direction = "negative"
        return direction

    def get_type(self):
        return self.event_type

    def set_magnitude(self):
        event_magnitude = 0
        if self.event_type == "small":
            event_magnitude = random.uniform(.01, .08)

        elif self.event_type == "large":
            event_magnitude = random.uniform(.15, .30)

        return event_magnitude

    def update_event_duration(self):
        self.event_duration = max(0, self.event_duration - 1)

    def decide_sector(self, active_sectors):
        # choose a random sector to effect
        choice = random.randint(0, len(active_sectors)-1)
        self.sector = active_sectors[choice]


