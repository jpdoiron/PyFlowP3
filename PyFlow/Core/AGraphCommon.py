"""@file AGraphCommon.py

**AGraphCommon.py** is a common definitions file

this file is imported in almost all others files of the program
"""
import ctypes
import inspect
import math
import threading
from enum import IntEnum
from functools import wraps
from queue import Queue
from threading import Thread

from . import Enums

## determines step for all floating point input widgets
FLOAT_SINGLE_STEP = 0.01
## determines floating precision
FLOAT_DECIMALS = 10
## determines floating minimum value
FLOAT_RANGE_MIN = -1000000
## determines floating maximum value
FLOAT_RANGE_MAX = 1000000
## determines int minimum value
INT_RANGE_MIN = -1000000
## determines int maximum value
INT_RANGE_MAX = 1000000



## Performs a linear interpolation
# @param[in] start the value to interpolate from
# @param[in] end the value to interpolate to
# @param[in] alpha how far to interpolate
# @returns The result of the linear interpolation (float)
def lerp(start, end, alpha):
    return (start + alpha * (end - start))


## Computes the value of the first specified argument clamped to a range defined by the second and third specified arguments
# @param[in] n
# @param[in] vmin
# @param[in] vmax
# @returns The clamped value of n
def clamp(n, vmin, vmax):
    return max(min(n, vmax), vmin)


## Rounding up to sertain value.Used in grid snapping
# @param[in] x value to round
# @param[in] to value x will be rounded to
# @returns rounded value of x
def roundup(x, to):
    return int(math.ceil(x / to)) * to


## This function for establish dependencies bitween pins
# @param[in] affects_pin this pin affects other pins
# @param[in] affected_pin this pin affected by other pin
def pinAffects(affects_pin, affected_pin):
    affects_pin.affects.append(affected_pin)
    affected_pin.affected_by.append(affects_pin)


## Check for cycle connected nodes
# @param[in] src pin
# @param[in] dst pin
# @returns bool
def cycle_check(src, dst):
    # allow cycles on execs
    if src.dataType == DataTypes.Exec or dst.dataType == DataTypes.Exec:
        return False

    if src.direction == PinDirection.Input:
        src, dst = dst, src
    start = src
    if src in dst.affects:
        return True
    for i in dst.affects:
        if cycle_check(start, i):
            return True
    return False


## marks dirty all ports from start to the right
# this part of graph will be recomputed every tick
# @param[in] start_from pin from which recursion begins
def push(start_from):
    if not start_from.affects == []:
        start_from.setDirty()
        for i in start_from.affects:
            i.setDirty()
            push(i)


## This function clears property view's layout.
# @param[in] layout QLayout class
def clearLayout(layout):
    while layout.count():
        child = layout.takeAt(0)
        if child.widget() is not None:
            child.widget().deleteLater()
        elif child.layout() is not None:
            clearLayout(child.layout())


##  Decorator from <a href="https://wiki.python.org/moin/PythonDecoratorLibrary#Memoize">Python decorator library</a>
# @param[in] foo function to memorize
def memoize(foo):
    memo = {}

    @wraps(foo)
    def wrapper(*args):
        if args in memo:
            return memo[args]
        else:
            rv = foo(*args)
            memo[args] = rv
            return rv
    return wrapper


class REGISTER_ENUM(object):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, cls):
        Enums.appendEnumInstance(cls)
        return cls


## Data types identifires.
class DataTypes(IntEnum):
    Any = 0
    Float = 1
    Int = 2
    String = 3
    Bool = 4
    Array = 5
    ## This type represents Execution pins.
    # It doesn't carry any data, but it implements [call](@ref PyFlow.Pins.ExecPin.ExecPin#call) method.
    # Using pins of this type we can control execution flow of graph.
    Exec = 6
    ## Special type of data which represents value passed by reference using [IMPLEMENT_NODE](@ref PyFlow.Core.FunctionLibrary.IMPLEMENT_NODE) decorator.
    # For example see [factorial](@ref FunctionLibraries.MathLib.MathLib.factorial) function.
    # Here along with computation results we return additional info, whether function call succeeded or not.
    Reference = 7
    FloatVector3 = 8
    FloatVector4 = 9
    Matrix33 = 10
    Matrix44 = 11
    Quaternion = 12
    Enum = 13
    Layer = 14
    Files = 15


## Returns string representation of the data type identifier
# See [DataTypes](@ref PyFlow.Core.AGraphCommon.DataTypes)
# @param[in] data type identifier (int)
def getDataTypeName(inValue):
    for name, value in inspect.getmembers(DataTypes):
        if isinstance(value, int):
            if inValue == value:
                return name
    return None


def _async_raise(tid, excobj):
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(excobj))
    if res == 0:
        raise ValueError("nonexistent thread id")
    elif res > 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, 0)
        raise SystemError("PyThreadState_SetAsyncExc failed")


class Thread(threading.Thread):

    def __init__(self,**kwargs):
        super(Thread, self).__init__(**kwargs)
        print("keras memory")
        from tensorflow.python.keras.backend import set_session,get_session,clear_session
        import tensorflow as tf
        tf.logging.set_verbosity('DEBUG')
        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        self.sess = tf.Session(config=tfconfig)
        set_session(self.sess)  # set this TensorFlow session as the default session for Keras
        clear_session()
        print("init keras memory")



    def raise_exc(self, excobj):
        assert self.isAlive(), "thread must be started"
        for tid, tobj in threading._active.items():
            if tobj is self:
                _async_raise(tid, excobj)
                return

        # the thread was alive when we entered the loop, but was not found
        # in the dict, hence it must have been already terminated. should we raise
        # an exception here? silently ignore?

    def run(self):
        try:
            if self._target:
                self._target(*self._args, **self._kwargs)
        except Exception as e:
            import traceback
            import sys
            traceback.print_exception(type(e), e, sys.exc_info()[2], limit=1, file=sys.stdout)

        finally:
            # Avoid a refcycle if the thread is running a function with
            # an argument that has a member that points to the thread.
            del self._target, self._args, self._kwargs

            #if(self.sess is not None):
                # from tensorflow.python.keras.backend import clear_session
                # clear_session()  # get a new session
             #   print("memory cleared")

        print("function done")

    def terminate(self):
        # must raise the SystemExit type, instead of a SystemExit() instance
        # due to a bug in PyThreadState_SetAsyncExc
        print("RAISIN")
        self.raise_exc(SystemExit)


class Singleton:
    """
    A non-thread-safe helper class to ease implementing singletons.
    This should be used as a decorator -- not a metaclass -- to the
    class that should be a singleton.

    The decorated class can define one `__init__` function that
    takes only the `self` argument. Also, the decorated class cannot be
    inherited from. Other than that, there are no restrictions that apply
    to the decorated class.

    To get the singleton instance, use the `instance` method. Trying
    to use `__call__` will result in a `TypeError` being raised.

    """

    def __init__(self, decorated):
        self._decorated = decorated

    def instance(self):
        """
        Returns the singleton instance. Upon its first call, it creates a
        new instance of the decorated class and calls its `__init__` method.
        On all subsequent calls, the already created instance is returned.

        """
        try:
            return self._instance
        except AttributeError:
            self._instance = self._decorated()
            return self._instance

    def __call__(self):
        raise TypeError('Singletons must be accessed through `instance()`.')

    def __instancecheck__(self, inst):
        return isinstance(inst, self._decorated)


@Singleton
class StaticVar:
    currentProcessThread: Thread = None
    def __init__(self):
        print ('StaticVar created')



class asynchronous(object):
    def __init__(self, func):
        self.func = func

        def threaded(*args, **kwargs):
            self.queue.put(self.func(*args, **kwargs))

        self.threaded = threaded

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def start(self, *args, **kwargs):
        self.queue = Queue()
        thread = Thread(target=self.threaded, args=args, kwargs=kwargs)
        thread.start()
        return asynchronous.Result(self.queue, thread)

    class NotYetDoneException(Exception):
        def __init__(self, message):
            self.message = message

    class Result(object):
        def __init__(self, queue, thread):
            self.queue = queue
            self.thread = thread

        def is_done(self):
            return not self.thread.is_alive()

        def get_result(self):
            if not self.is_done():
                raise asynchronous.NotYetDoneException('the call has not yet completed its task')

            if not hasattr(self, 'result'):
                self.result = self.queue.get()

            return self.result


## [Circular buffer](https://en.wikipedia.org/wiki/Circular_buffer) container class.
# Useful for processing streaming data.
class CircularBuffer(object):
    def __init__(self, capacity):
        super(CircularBuffer, self).__init__()
        self._capacity = capacity
        self._ls = []
        self._current = 0

    def _is_full(self):
        return len(self._ls) == self.capacity()

    def append(self, item):
        if self._is_full():
            self._ls[self._current] = item
            self._current = (self._current + 1) % self.capacity()
        else:
            self._ls.append(item)

    def get(self):
        if self._is_full():
            return self._ls[self._current:] + self._ls[:self._current]
        else:
            return self._ls

    def capacity(self):
        return self._capacity


## Used in PyFlow.AbstractGraph.NodeBase.getPinByName for optimization purposes
class PinSelectionGroup(IntEnum):
    Inputs = 0
    Outputs = 1
    BothSides = 2


## Can be used for code generation
class AccessLevel(IntEnum):
    public = 0
    private = 1
    protected = 2


## Determines wheter it is input pin or output.
class PinDirection(IntEnum):
    Input = 0
    Output = 1


## Determines wheter it is callable node or pure.
# Callable node is a node with Exec pins
class NodeTypes(IntEnum):
    Callable = 0
    Pure = 1


@REGISTER_ENUM()
## Direction identifires. Used in [alignSelectedNodes](@ref PyFlow.Core.Widget.GraphWidget.alignSelectedNodes)
class Direction(IntEnum):
    Left = 0
    Right = 1
    Up = 2
    Down = 3
