from ..Core.FunctionLibrary import *
from ..Core.AGraphCommon import *
import math
import random
import pyrr


class MathLib(FunctionLibraryBase):
    """
    Python builtin math module wrapper
    """
    def __init__(self):
        super(MathLib, self).__init__()

    # ###################
    # builtin python math
    # ###################
    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Float, 0.0), meta={'Category': 'Math|Python math|Number-theoretic and representation functions', 'Keywords': []})
    ## Return the ceiling of x as a float, the smallest integer value greater than or equal to x
    def ceil(a=(DataTypes.Float, 0.0)):
        '''
        Return the ceiling of x as a float, the smallest integer value greater than or equal to x
        '''
        return math.ceil(a)

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Float, 0.0), meta={'Category': 'Math|Python math|Number-theoretic and representation functions', 'Keywords': []})
    ## Return x with the sign of y. On a platform that supports signed zeros, copysign(1.0, -0.0) returns -1.0
    def copysignf(a=(DataTypes.Float, 0.0), b=(DataTypes.Float, 0.0)):
        '''
        Return x with the sign of y. On a platform that supports signed zeros, copysign(1.0, -0.0) returns -1.0
        '''
        return math.copysign(a, b)

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Int, 0), meta={'Category': 'Math|Python math|Number-theoretic and representation functions', 'Keywords': []})
    ## Return x with the sign of y. On a platform that supports signed zeros, copysign(1.0, -0.0) returns -1.0
    def copysign(a=(DataTypes.Int, 0), b=(DataTypes.Int, 0)):
        '''
        Return x with the sign of y. On a platform that supports signed zeros, copysign(1.0, -0.0) returns -1.0
        '''
        return math.copysign(a, b)

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Int, 0), meta={'Category': 'Math|Python math|Number-theoretic and representation functions', 'Keywords': []})
    ## Return x factorial. Raises ValueError if x is not integral or is negative
    def factorial(a=(DataTypes.Int, 0), result=(DataTypes.Reference, (DataTypes.Bool, False))):
        '''
        Return x factorial. Raises ValueError if x is not integral or is negative
        '''
        try:
            f = math.factorial(a)
            result(True)
            return f
        except:
            result(False)
            return -1

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Float, 0.0), meta={'Category': 'Math|Python math|Number-theoretic and representation functions', 'Keywords': []})
    ## Return the floor of x as a float, the largest integer value less than or equal to x
    def floor(a=(DataTypes.Float, 0.0)):
        '''
        Return the floor of x as a float, the largest integer value less than or equal to x
        '''
        return math.floor(a)

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Float, 0.0), meta={'Category': 'Math|Python math|Number-theoretic and representation functions', 'Keywords': []})
    ## Return fmod(x, y), as defined by the platform C library
    def fmodf(a=(DataTypes.Float, 0.0), b=(DataTypes.Float, 0.0)):
        '''
        Return fmod(x, y), as defined by the platform C library
        '''
        return math.fmod(a, b)

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Int, 0), meta={'Category': 'Math|Python math|Number-theoretic and representation functions', 'Keywords': []})
    ## Python x % y operator
    def fmod(a=(DataTypes.Int, 0), b=(DataTypes.Int, 0)):
        '''
        Python x % y
        '''
        return a % b

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={'Category': 'Math|Python math|Number-theoretic and representation functions', 'Keywords': []})
    ## Return the mantissa and exponent of x as the pair (m, e). m is a float and e is an integer such that x == m * 2**e exactly
    def frexp(a=(DataTypes.Float, 0.0), m=(DataTypes.Reference, (DataTypes.Float, 0.0)), e=(DataTypes.Reference, (DataTypes.Int, 0))):
        '''
        Return the mantissa and exponent of x as the pair (m, e). m is a float and e is an integer such that x == m * 2**e exactly
        '''
        t = math.frexp(a)
        m(t[0])
        e(t[1])

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Float, 0.0), meta={'Category': 'Math|Python math|Number-theoretic and representation functions', 'Keywords': []})
    ## Return an accurate floating point sum of values in the iterable. Avoids loss of precision by tracking multiple intermediate partial sums
    def fsum(arr=(DataTypes.Array, []), result=(DataTypes.Reference, (DataTypes.Bool, False))):
        '''
        Return an accurate floating point sum of values in the iterable. Avoids loss of precision by tracking multiple intermediate partial sums
        '''
        try:
            s = math.fsum([float(i) for i in arr])
            result(True)
            return s
        except:
            result(False)
            return 0.0

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Bool, False), meta={'Category': 'Math|Python math|Number-theoretic and representation functions', 'Keywords': []})
    ## Check if the float x is positive or negative infinity
    def isinf(a=(DataTypes.Float, 0.0)):
        '''
        Check if the float x is positive or negative infinity
        '''
        return math.isinf(a)

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Bool, False), meta={'Category': 'Math|Python math|Number-theoretic and representation functions', 'Keywords': []})
    ## Check if the float x is a NaN (not a number)
    def isnan(a=(DataTypes.Float, 0.0)):
        '''
        Check if the float x is a NaN (not a number)
        .'''
        return math.isnan(a)

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Float, 0.0), meta={'Category': 'Math|Python math|Number-theoretic and representation functions', 'Keywords': []})
    ## Return x * (2**i). This is essentially the inverse of function frexp()
    def ldexp(a=(DataTypes.Float, 0.0), i=(DataTypes.Int, 0)):
        '''
        Return x * (2**i). This is essentially the inverse of function frexp()
        '''
        return math.ldexp(a, i)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={'Category': 'Math|Python math|Number-theoretic and representation functions', 'Keywords': []})
    ## Return the fractional and integer parts of x. Both results carry the sign of x and are floats
    def modf(a=(DataTypes.Int, 0), f=(DataTypes.Reference, (DataTypes.Float, 0.0)), i=(DataTypes.Reference, (DataTypes.Int, 0))):
        '''
        Return the fractional and integer parts of x. Both results carry the sign of x and are floats
        '''
        t = math.modf(a)
        f(t[0])
        i(t[1])

    @staticmethod
    @IMPLEMENT_NODE(returns=None, meta={'Category': 'Math|Python math|Number-theoretic and representation functions', 'Keywords': []})
    ## Return the fractional and integer parts of x. Both results carry the sign of x and are floats
    def fmodf(a=(DataTypes.Float, 0), f=(DataTypes.Reference, (DataTypes.Float, 0.0)), i=(DataTypes.Reference, (DataTypes.Int, 0))):
        '''
        Return the fractional and integer parts of x. Both results carry the sign of x and are floats
        '''
        t = math.modf(a)
        f(t[0])
        i(t[1])

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Int, 0), meta={'Category': 'Math|Python math|Number-theoretic and representation functions', 'Keywords': []})
    ## Return the Real value x truncated to an Integral (usually a long integer)
    def trunc(a=(DataTypes.Float, 0.0)):
        '''
        Return the Real value x truncated to an Integral (usually a long integer)
        '''
        return math.trunc(a)

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Float, 0.0), meta={'Category': 'Math|Python math|Power and logarithmic functions', 'Keywords': []})
    ## Return e**x
    def exp(a=(DataTypes.Float, 0.0)):
        '''
        Return e**x
        '''
        return math.exp(a)

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Float, 0.0), meta={'Category': 'Math|Python math|Power and logarithmic functions', 'Keywords': []})
    ## Return e**x - 1. For small floats x, the subtraction in exp(x) - 1 can result in a significant loss of precision
    def expm1(a=(DataTypes.Float, 0.1)):
        '''
        Return e**x - 1. For small floats x, the subtraction in exp(x) - 1 can result in a significant loss of precision
        '''
        return math.expm1(a)

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Float, 0.0), meta={'Category': 'Math|Python math|Power and logarithmic functions', 'Keywords': []})
    ## Return the logarithm of x to the given base, calculated as log(x)/log(base)
    def log(a=(DataTypes.Float, 1.0), base=(DataTypes.Float, math.e), result=(DataTypes.Reference, (DataTypes.Bool, False))):
        '''
        Return the logarithm of x to the given base, calculated as log(x)/log(base)
        '''
        try:
            result(True)
            return math.log(a, base)
        except:
            result(False)
            return -1

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Float, 0.0), meta={'Category': 'Math|Python math|Power and logarithmic functions', 'Keywords': []})
    ## Return the natural logarithm of 1+x (base e). The result is calculated in a way which is accurate for x near zero
    def log1p(a=(DataTypes.Float, 1.0), result=(DataTypes.Reference, (DataTypes.Bool, False))):
        '''
        Return the natural logarithm of 1+x (base e). The result is calculated in a way which is accurate for x near zero
        '''
        try:
            result(True)
            return math.log1p(a)
        except:
            result(False)
            return -1

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Float, 0.0), meta={'Category': 'Math|Python math|Power and logarithmic functions', 'Keywords': []})
    ## Return the base-10 logarithm of x. This is usually more accurate than log(x, 10)
    def log10(a=(DataTypes.Float, 1.0), result=(DataTypes.Reference, (DataTypes.Bool, False))):
        '''
        Return the base-10 logarithm of x. This is usually more accurate than log(x, 10)
        '''
        try:
            result(True)
            return math.log10(a)
        except:
            result(False)
            return -1

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Float, 0.0), meta={'Category': 'Math|Python math|Power and logarithmic functions', 'Keywords': []})
    ## Return x raised to the power y
    def power(a=(DataTypes.Float, 0.0), b=(DataTypes.Float, 0.0), result=(DataTypes.Reference, (DataTypes.Bool, False))):
        '''
        Return x raised to the power y
        '''
        try:
            result(True)
            return math.pow(a, b)
        except:
            result(False)
            return -1

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Float, 0.0), meta={'Category': 'Math|Python math|Power and logarithmic functions', 'Keywords': []})
    ## Return the square root of x
    def sqrt(a=(DataTypes.Float, 0.0), result=(DataTypes.Reference, (DataTypes.Bool, False))):
        '''
        Return the square root of x
        '''
        try:
            result(True)
            return math.sqrt(a)
        except:
            result(False)
            return -1

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Int, 0), meta={'Category': 'Math|Python math|Bult-in functions', 'Keywords': ['+']})
    ## Sums start and the items of an iterable from left to right and returns the total
    def Sum(arr=(DataTypes.Array, []), result=(DataTypes.Reference, (DataTypes.Bool, False))):
        '''
        Sums start and the items of an iterable from left to right and returns the total
        '''
        try:
            s = math.fsum([int(i) for i in arr])
            result(True)
            return s
        except:
            result(False)
            return 0

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Int, 0), meta={'Category': 'Math|Python math|Bult-in functions', 'Keywords': []})
    ## Return the absolute value of a number
    def absint(inp=(DataTypes.Int, 0)):
        '''
        Return the absolute value of a number
        '''
        return abs(inp)

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Float, 0.0), meta={'Category': 'Math|Python math|Trigonometry', 'Keywords': []})
    ## Return the cosine of x radians
    def cos(rad=(DataTypes.Float, 0.0)):
        '''
        Return the cosine of x radians
        '''
        return math.cos(rad)

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Float, 0.0), meta={'Category': 'Math|Python math|Trigonometry', 'Keywords': []})
    ## Return the arc cosine of x, in radians
    def acos(rad=(DataTypes.Float, 0.0)):
        '''
        Return the arc cosine of x, in radians
        '''
        return math.acos(rad)

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Float, 0.0), meta={'Category': 'Math|Python math|Trigonometry', 'Keywords': []})
    ## Return the sine of x radians
    def sin(rad=(DataTypes.Float, 0.0)):
        '''
        Return the sine of x radians
        '''
        return math.sin(rad)

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Float, 0.0), meta={'Category': 'Math|Python math|Trigonometry', 'Keywords': []})
    ## Return the arc sine of x, in radians
    def asin(rad=(DataTypes.Float, 0.0)):
        '''
        Return the arc sine of x, in radians
        '''
        return math.asin(rad)

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Float, 0.0), meta={'Category': 'Math|Python math|Trigonometry', 'Keywords': []})
    ## Return the tangent of x radians
    def tan(rad=(DataTypes.Float, 0.0)):
        '''
        Return the tangent of x radians
        '''
        return math.tan(rad)

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Float, 0.0), meta={'Category': 'Math|Python math|Trigonometry', 'Keywords': []})
    ## Return the arc tangent of x, in radians
    def atan(rad=(DataTypes.Float, 0.0)):
        '''
        Return the arc tangent of x, in radians
        '''
        return math.atan(rad)

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Float, 0.0), meta={'Category': 'Math|Python math|Trigonometry', 'Keywords': []})
    ## Return atan(a / b), in radians. The result is between -pi and pi.
    #  The vector in the plane from the origin to point (a, b) makes this angle
    #  with the positive X axis. The point of atan2() is that the signs of both
    #  inputs are known to it, so it can compute the correct quadrant for the angle.
    #  For example, atan(1) and atan2(1, 1) are both pi/4, but atan2(-1, -1) is -3*pi/4
    def atan2(a=(DataTypes.Float, 0.0), b=(DataTypes.Float, 0.0)):
        '''
        Return atan(a / b), in radians. The result is between -pi and pi.\nThe vector in the plane from the origin to point (a, b) makes this angle with the positive X axis. The point of atan2() is that the signs of both inputs are known to it, so it can compute the correct quadrant for the angle.\nFor example, atan(1) and atan2(1, 1) are both pi/4, but atan2(-1, -1) is -3*pi/4
        '''
        return math.atan2(a, b)

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Float, 0.0), meta={'Category': 'Math|Python math|Trigonometry', 'Keywords': []})
    ## Return the Euclidean norm, sqrt(x*x + y*y).
    #  This is the length of the vector from the origin to point (x, y)
    def hypot(a=(DataTypes.Float, 0.0), b=(DataTypes.Float, 0.0)):
        '''
        Return the Euclidean norm, sqrt(x*x + y*y). This is the length of the vector from the origin to point (x, y)
        .'''
        return math.hypot(a, b)

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Float, 0.0), meta={'Category': 'Math|Python math|Angular conversion', 'Keywords': []})
    ## Convert angle x from degrees to radians
    def degtorad(deg=(DataTypes.Float, 0.0)):
        '''
        Convert angle x from degrees to radians
        '''
        return math.radians(deg)

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Float, 0.0), meta={'Category': 'Math|Python math|Angular conversion', 'Keywords': []})
    ## Convert angle x from radians to degrees
    def radtodeg(rad=(DataTypes.Float, 0.0)):
        '''
        Convert angle x from radians to degrees
        '''
        return math.degrees(rad)

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Float, 0.0), meta={'Category': 'Math|Python math|Hyperbolic functions', 'Keywords': []})
    ## Return the inverse hyperbolic cosine of x
    def acosh(a=(DataTypes.Float, 0.0), Result=(DataTypes.Reference, (DataTypes.Bool, False))):
        '''
        Return the inverse hyperbolic cosine of x
        '''
        try:
            Result(True)
            return math.acosh(a)
        except:
            Result(False)
            return -1

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Float, 0.0), meta={'Category': 'Math|Python math|Hyperbolic functions', 'Keywords': []})
    ## Return the inverse hyperbolic sine of x
    def asinh(a=(DataTypes.Float, 0.0)):
        '''
        Return the inverse hyperbolic sine of x
        '''
        return math.asinh(a)

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Float, 0.0), meta={'Category': 'Math|Python math|Hyperbolic functions', 'Keywords': []})
    ## Return the inverse hyperbolic tangent of x
    def atanh(a=(DataTypes.Float, 0.0), Result=(DataTypes.Reference, (DataTypes.Bool, False))):
        '''
        Return the inverse hyperbolic tangent of x
        '''
        try:
            Result(True)
            return math.atanh(a)
        except:
            Result(False)
            return -1

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Float, 0.0), meta={'Category': 'Math|Python math|Hyperbolic functions', 'Keywords': []})
    ## Return the hyperbolic cosine of x
    def cosh(a=(DataTypes.Float, 0.0), Result=(DataTypes.Reference, (DataTypes.Bool, False))):
        '''
        Return the hyperbolic cosine of x
        '''
        try:
            Result(True)
            return math.cosh(a)
        except:
            Result(False)
            return -1

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Float, 0.0), meta={'Category': 'Math|Python math|Hyperbolic functions', 'Keywords': []})
    ## Return the hyperbolic sine of x
    def sinh(a=(DataTypes.Float, 0.0), Result=(DataTypes.Reference, (DataTypes.Bool, False))):
        '''
        Return the hyperbolic sine of x
        '''
        try:
            Result(True)
            return math.sinh(a)
        except:
            Result(False)
            return -1

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Float, 0.0), meta={'Category': 'Math|Python math|Hyperbolic functions', 'Keywords': []})
    ## Return the hyperbolic tangent of x
    def tanh(a=(DataTypes.Float, 0.0)):
        '''
        Return the hyperbolic tangent of x
        '''
        return math.tanh(a)

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Float, 0.0), meta={'Category': 'Math|Python math|Special functions', 'Keywords': []})
    ## Return the error function at x
    def erf(a=(DataTypes.Float, 0.0)):
        '''
        Return the error function at x
        '''
        return math.erf(a)

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Float, 0.0), meta={'Category': 'Math|Python math|Special functions', 'Keywords': []})
    ## Return the complementary error function at x
    def erfc(a=(DataTypes.Float, 0.0)):
        '''
        Return the complementary error function at x
        '''
        return math.erfc(a)

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Float, 0.0), meta={'Category': 'Math|Python math|Special functions', 'Keywords': []})
    ## Return the Gamma function at x
    def gamma(a=(DataTypes.Float, 0.0), Result=(DataTypes.Reference, (DataTypes.Bool, False))):
        '''
        Return the Gamma function at x
        '''
        try:
            Result(True)
            return math.gamma(a)
        except:
            Result(False)
            return -1

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Float, 0.0), meta={'Category': 'Math|Python math|Special functions', 'Keywords': []})
    ## Return the natural logarithm of the absolute value of the Gamma function at x
    def lgamma(a=(DataTypes.Float, 0.0), Result=(DataTypes.Reference, (DataTypes.Bool, False))):
        '''
        Return the natural logarithm of the absolute value of the Gamma function at x
        '''
        try:
            Result(True)
            return math.lgamma(a)
        except:
            Result(False)
            return -1

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Float, 0.0), meta={'Category': 'Math|Python math|Constants', 'Keywords': []})
    ## The mathematical constant e = 2.718281, to available precision
    def e():
        '''
        The mathematical constant e = 2.718281, to available precision
        '''
        return math.e

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Float, 0.0), meta={'Category': 'Math|Python math|Constants', 'Keywords': []})
    ## The mathematical constant = 3.141592, to available precision
    def pi():
        '''
        The mathematical constant = 3.141592, to available precision
        '''
        return math.pi
