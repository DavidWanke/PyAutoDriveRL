class Event(object):
    """
    Superclass for events that might be created.
    """

    def __init__(self):
        self.name = "Default event"

    def __str__(self):
        return self.name


class QuitEvent(Event):
    """
    Quit event.
    """

    def __init__(self):
        self.name = "Quit event"


class ResetEvent(Event):
    """
    Reset event.
    """

    def __init__(self):
        self.name = "Reset event"


class InitializeEvent(Event):
    """
    Initialize event.
    """

    def __init__(self):
        self.name = "Initialize event"


class StateChangeEvent(Event):
    """
    Change model state machine.
    A None will pop() instead of push.
    """

    def __init__(self, state):
        self.name = "State change event"
        self.state = state

    def __str__(self):
        if self.state is None:
            return f"{self.name}: Popped"
        else:
            return f"{self.name}: Pushed State {self.state}"


class MouseEvent(Event):
    """
    Mouse input event.
    """

    def __init__(self, click_pos):
        self.name = "Mouse event"
        self.click_pos = click_pos

    def __str__(self):
        return f'{self.name}: click_pos={self.click_pos}'


class KeyEvent(Event):
    """
    Keyboard input event.
    """

    def __init__(self, unicode_char):
        self.name = "Key event"
        self.char = unicode_char

    def __str__(self):
        return f'{self.name}: char={self.char}'


class TickEvent(Event):
    """
    Tick event.
    """

    def __init__(self):
        self.name = "Tick event"


class EventManager(object):
    """
    Coordinates communication between model, view and controller.
    """

    def __init__(self):
        from weakref import WeakKeyDictionary
        self.listeners = WeakKeyDictionary()

    def register_listener(self, listener):
        """
        Adds a listener to the event queue.

        :param listener: The object which listens to the event queue.
        """
        self.listeners[listener] = 1

    def unregister_listener(self, listener):
        """
        Removes a listener from the event queue.

        :param listener: The object which should no longer listen to the event queue.
        """
        if listener in self.listeners.keys():
            del self.listeners[listener]

    def post(self, event):
        """
        Posts a new event to the event queue.

        :param event: The event which should be broadcasted.
        """
        if not isinstance(event, TickEvent):
            print(str(event))
        for listener in self.listeners.keys():
            listener.notify(event)
