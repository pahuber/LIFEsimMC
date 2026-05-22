from sympy import Matrix, sin, pi, cos, symbols, sqrt, atan, exp, I

t, tm, b = symbols('t tm b')


class BaseArrayConfiguration:
    acm = None


class XArrayConfiguration(BaseArrayConfiguration):
    q = 6
    acm = (b / 2
           * Matrix([[cos(2 * pi / tm * t), -sin(2 * pi / tm * t)],
                     [sin(2 * pi / tm * t), cos(2 * pi / tm * t)]])
           * Matrix([[q, q, -q, -q],
                     [1, -1, -1, 1]]))


class KiteArrayConfiguration(BaseArrayConfiguration):
    c = 1.69
    beta = 2 * atan(1 / c)
    th = [pi / 2 - beta, pi / 2, pi / 2 + beta, 3 * pi / 2]

    acm = (b / 2 * sqrt(1 + c ** 2)
           * Matrix([[cos(2 * pi / tm * t), -sin(2 * pi / tm * t)],
                     [sin(2 * pi / tm * t), cos(2 * pi / tm * t)]])
           * Matrix([[cos(th[0]), cos(th[1]), cos(th[2]), cos(th[3])],
                     [sin(th[0]), sin(th[1]), sin(th[2]), sin(th[3])]]))


class PentagonArrayConfiguration(BaseArrayConfiguration):
    th = [0, 2 * pi / 5, 4 * pi / 5, 6 * pi / 5, 8 * pi / 5]
    acm = (0.851 * b
           * Matrix([[cos(2 * pi / tm * t), -sin(2 * pi / tm * t)],
                     [sin(2 * pi / tm * t), cos(2 * pi / tm * t)]])
           * Matrix([[cos(th[0]), cos(th[1]), cos(th[2]), cos(th[3]), cos(th[4])],
                     [sin(th[0]), sin(th[1]), sin(th[2]), sin(th[3]), sin(th[4])]]))


class BaseBeamCombiner:
    catm = None
    diff_out = None
    sep_at_max_mod_eff = None


class DoubleBracewellBeamCombiner(BaseBeamCombiner):
    catm = 1 / 2 * Matrix([[0, 0, sqrt(2), sqrt(2)],
                           [sqrt(2), sqrt(2), 0, 0],
                           [1, -1, -exp(I * pi / 2), exp(I * pi / 2)],
                           [1, -1, exp(I * pi / 2), -exp(I * pi / 2)]])
    diff_out = [(2, 3)]
    sep_at_max_mod_eff = [0.6]


class Kernel4BeamCombiner(BaseBeamCombiner):
    ep = exp(I * pi / 2)
    em = exp(-I * pi / 2)

    catm = 1 / 4 * Matrix([[2, 2, 2, 2],
                           [1 + ep, 1 - ep, -1 + ep, -1 - ep],
                           [1 - em, -1 - em, 1 + em, -1 + em],
                           [1 + ep, 1 - ep, -1 - ep, -1 + ep],
                           [1 - em, -1 - em, -1 + em, 1 + em],
                           [1 + ep, -1 - ep, 1 - ep, -1 + ep],
                           [1 - em, -1 + em, -1 - em, 1 + em]])
    diff_out = [(1, 2), (3, 4), (5, 6)]
    sep_at_max_mod_eff = [0.4, 0.4, 0.4]


class Kernel5BeamCombiner(BaseBeamCombiner):
    e2 = exp(2 * pi * I / 5)
    e4 = exp(4 * pi * I / 5)
    e6 = exp(6 * pi * I / 5)
    e8 = exp(8 * pi * I / 5)

    catm = 1 / sqrt(5) * Matrix([[1, 1, 1, 1, 1],
                                 [1, e2, e4, e6, e8],
                                 [1, e4, e8, e2, e6],
                                 [1, e6, e2, e8, e4],
                                 [1, e8, e6, e4, e2]])

    diff_out = [(1, 4), (2, 3)]
    sep_at_max_mod_eff = [2.68, 1.03]


class LIFEBaselineArchitecture(XArrayConfiguration, DoubleBracewellBeamCombiner):
    pass


class Kernel4Kite(KiteArrayConfiguration, Kernel4BeamCombiner):
    pass


class Kernel5Pentagon(PentagonArrayConfiguration, Kernel5BeamCombiner):
    pass
