import pygame
from ..events.event_manager import *
from ..model.model import Model
from src.env.view.constants import *


class Controller(object):
    """
    Handles keyboard input.
    """

    def __init__(self, event_manager: EventManager, model: Model):
        """

        :param event_manager: The event manager that the view listens to.
        :param model: The model of the simulation.
        """
        self.event_manager = event_manager
        event_manager.register_listener(self)
        self.model = model

    def notify(self, event: Event):
        """
        Receive events posted to the message queue.

        :param event: The new event.
        """

        if isinstance(event, TickEvent):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.event_manager.post(QuitEvent())
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.event_manager.post(StateChangeEvent(None))
                    else:
                        current_state = self.model.state_machine.peek()
                        if current_state == self.model.state_machine.IN_MENU:
                            self.handle_menu_key_input(event)
                        elif current_state == self.model.state_machine.IN_SIMULATION:
                            self.handle_simulation_key_input(event)
                        # self.event_manager.post(InputEvent(unicode_char=event.unicode))
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    self.event_manager.post(MouseEvent(pos))

    def handle_menu_key_input(self, event):
        """
        Handles the key input for the menu.

        :param event: The pygame keydown event.
        """
        if event.key in [pygame.K_ESCAPE, pygame.K_RETURN]:
            self.event_manager.post(StateChangeEvent(None))

    def handle_simulation_key_input(self, event):
        """
        Handles the key input for the simulation.

        :param event: The pygame keydown event.
        """
        self.event_manager.post(KeyEvent(unicode_char=event.unicode))

        if event.key is pygame.K_p:
            self.model.toggle_pause()
        if event.key is pygame.K_q:
            self.model.toggle_debug_control()
        if event.key is pygame.K_t:
            if DEBUG and D_FORCE_SEED:
                self.model.simulation.reset(D_FORCE_SEED)
            else:
                self.model.simulation.reset()
            self.event_manager.post(ResetEvent())
