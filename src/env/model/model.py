from ..events.event_manager import *
from .simulation.simulation import *


class Model:
    """
    This is the model of the whole program, which stores all data and controls the environment.
    """

    def __init__(self, event_manager: EventManager):
        """

        :param event_manager: Manager that allows posting events to the queue.
        """

        self.event_manager = event_manager
        self.state_machine = StateMachine()
        self.running = True
        self.pause = False
        self.debug_control = False
        event_manager.register_listener(self)
        self.simulation = Simulation()

    def toggle_debug_control(self):
        """
        Toggles debug control.
        """
        self.debug_control = not self.debug_control

    def toggle_pause(self):
        """
        Toggles pause.
        """
        self.pause = not self.pause

    def notify(self, event):
        """
        Called by event in event queue.

        :param event: The new event, which should be processed.
        """

        if isinstance(event, StateChangeEvent):

            if event.state is None:  # pop request
                if self.state_machine.pop() is None:  # no more states left
                    self.event_manager.post(QuitEvent())
            else:  # push new state on stack
                self.state_machine.push(event.state)
        elif isinstance(event, QuitEvent):
            self.running = False

    def initialize(self):
        """
        Initializes the model.
        """
        self.event_manager.post(InitializeEvent())
        self.state_machine.push(self.state_machine.IN_SIMULATION)

    def run(self):
        self.initialize()
        while self.running:
            self.step()

    def step(self):
        """
        Executes one step of the model and changes the model accordingly.
        """
        new_tick = TickEvent()
        if not self.pause:
            self.simulation.step(None)
        self.event_manager.post(new_tick)


class StateMachine(object):
    """
    Manages a stack based state machine.
    """

    """
    Constants for StateMachine
    """
    IN_SIMULATION = 0
    IN_MENU = 1

    def __init__(self):
        self.state_stack = []

    def peek(self):
        """
        Returns current state without altering the state or None if the stack is empty.

        :return: current state
        """
        if len(self.state_stack) > 0:
            return self.state_stack[-1]
        else:
            return None

    def pop(self):
        """
        Returns current state and removes it from the stack. Or returns None if the stack is empty.

        :return: current state
        """
        if len(self.state_stack) > 0:
            state = self.state_stack.pop(-1)
            return state
        else:
            return None

    def push(self, state):
        """
        Pushes new state on the stack and returns state.

        :return: pushed state
        """
        self.state_stack.append(state)
        return state

    def __str__(self):
        return str(self.state_stack)
