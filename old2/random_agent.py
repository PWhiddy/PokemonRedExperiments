from pyboy import WindowEvent
import random 

class RandomAgent:

    def __init__(self):
        pass

    def get_name(self):
        return 'random action agent v1'

    def reset_agent(self):
        pass

    def get_action(self, latest_state, rollout):
        # can use state and rollout history to make 
        # decision but agent doesn't care
        return random.choice([
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.PASS
        ])