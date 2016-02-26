import math

import scipy
import math
import numpy
import scipy.misc
import scipy.optimize
import scipy.special
import sympy
from numpy.core.numeric import zeros
from sympy import sympify
import re
import imp
import numpy.random as rd


def func(x):
    return 1 + 2 + x


class QuasiNewton:
    def __init__(self, up_bound, low_bound, w, especial_param_p, especial_param_g, function, particles_count, stop_case):
        self.function = self.build_function(function)
        print(self.function([4, 3]))
        self.fsym = sympify(function)
        self.particles_count = particles_count
        self.low_bound = low_bound
        self.up_bound = up_bound
        self.w = w
        self.especial_param_p = especial_param_p
        self.especial_param_g = especial_param_g
        self.stop_case = stop_case
        self.dimension = len(self.variables)
        self.current_iteration = 0
        self.initial_step = 1
        self.factor = 2
        self.epsilon = 0.2

    def build_function(self, raw_function):
        raw_function = raw_function.replace("^", "**").replace(" ", "")
        self.variables = re.findall(r'[a-z]\d+', raw_function)
        fix_variables = []
        for i in self.variables:
            if i not in fix_variables:
                fix_variables.append(i)
        tuple_line = ",".join(fix_variables) + "=" + "x"
        function_define = \
            """def function(x,*args):
            from math import sin, cos, tan, atan, asin, acos
            {0}
            return {1}
            """.format(tuple_line, raw_function)
        # print function_define
        module = imp.new_module('myfunctions')
        exec function_define in module.__dict__
        return module.function

    def Gradiente(self, xk):
        gxk = [0 for _ in self.variables]
        for (k, i) in enumerate(self.variables):
            der = sympy.diff(self.fsym, i)
            derstr = str(der)
            for x in self.variables:
                derstr = derstr + "+%s*%s" %(x, "xx")
            der = sympify(derstr)
            dicc = {}
            for (j, x) in enumerate(self.variables):
                dicc[x] = xk[j]
            dicc["xx"] = 0
            gxk[k] = round(der.evalf(subs=dicc))

        return gxk

    def solve(self, xk, iteraciones):
        bk_matrix = self.Identity(numpy.size(xk))
        #xk=numpy.transpose(numpy.asmatrix(xk))
        xk1 = xk[:]
        def f(xk):
            a=xk[0]*xk[0]
            return a

        self.function= f

        for x in range(iteraciones):
            #gxk = self.Gradiente(xk)
            gxk = scipy.optimize.approx_fprime(xk,self.function,0.01)
            gxkt=numpy.transpose(numpy.asmatrix(gxk))
            dk_vector = - bk_matrix * gxkt
            def deriv(punto):
                return scipy.optimize.approx_fprime(xk,self.function,0.01)
            tupla = scipy.optimize.line_search(self.function,deriv,numpy.transpose(numpy.asmatrix(xk)),dk_vector,numpy.asmatrix(gxk))
            ak=tupla[0]
            #ak, _, _ = scipy.optimize.line_search(self.function, xk, dk_vector, gxk, self.function(xk))
            #ak, fevals, gfevals = scipy.optimize.line_search(self.function, None, self.xk_vector, self.dk_vector, gxk)
            dkvec=numpy.array(numpy.transpose(dk_vector))
            vec0=dkvec[0]
            xk1 = xk + ak * vec0
            xk = numpy.array(xk1)
        return xk1
            #self.bfgs()

    def Identity(self, n):
        a = zeros((n, n))

        for i in range(n):
            a[i, i] = 1
        return a

    def tunneling(self, vector_xk, max_iterations=20, lambda_init=1, lambda_step=0.1,
                  lambda_max=2):
        current_iteration = 0
        lambda_value = lambda_init

        def f(xk):
            return (self.function(xk) - self.function(vector_xk)) * math.exp(lambda_value / (scipy.linalg.norm(xk - vector_xk)))

        def polo(xk):
            return f(xk) * math.exp(lambda_value / (scipy.linalg.norm(xk - vector_xk)))

        t = f
        while 1:
            xk_temp = self.PSO_local()
            if t(xk_temp) < 0:
                lambda_value = lambda_init
                t = f
                current_iteration += 1

            elif current_iteration >= max_iterations:
                break

            else:
                lambda_value += lambda_step
                if lambda_value > lambda_max:
                    t = polo
                    lambda_value = lambda_init
        return xk_temp

    def bfgs(self, xk, xk1, bk_matrix):
        gk = self.Gradiente(xk)
        gk1 = self.Gradiente(xk1)
        #gk = scipy.misc.derivative(self.function, self.xk_vector, dx=.01)
        #gk1 = scipy.misc.derivative(self.function, self.xk1_vector, dx=.01)

        qk = gk1 - gk

        pk = xk1 - xk
        self.bk_matrix = bk_matrix + ((1 + numpy.transpose(qk) * bk_matrix * qk) / numpy.transpose(qk) * qk) * pk * \
                                          numpy.transpose(pk) / numpy.transpose(pk) * qk - (pk * numpy.transpose(qk) *
                                                                                            bk_matrix + bk_matrix * qk * numpy.transpose(qk)) / numpy.transpose(qk) * pk

    def PSO_global(self):
        _list = [rd.uniform(self.low_bound, self.up_bound, self.dimension) for x in range(0, self.particles_count, 1)]
        g_best = _list[0]
        position = _list
        vel = []
        for x in position:
            if self.function(x) < self.function(g_best):
                g_best = x
            vel.append(rd.uniform(-(abs(self.up_bound - self.low_bound)), abs(self.up_bound -self. low_bound), self.dimension))
        count = 0
        while count < self.stop_case:
            for x in range(len(_list)):
                for y in xrange(self.dimension):
                    r_g, r_p = rd.random(1), rd.random(1)
                    vel[x][y] = self.w * vel[x][y] + self.especial_param_p * r_p * (position[x][y] - _list[x][y]) + self.especial_param_g \
                                                                                                          * r_g * (
                                                                                                              g_best[y] -
                                                                                                              _list[x][y])
                _list[x] += vel[x]
                if self.function(_list[x]) < self.function(position[x]):
                    position[x] = _list[x]
                    if self.function(position[x]) < self.function(g_best):
                        g_best = position[x]
            count += 1
        return g_best


    def PSO_local(self):
        _list = [rd.uniform(self.low_bound, self.up_bound, n) for x in range(0, self.particles_count, 1)]
        l_best = _list
        position = _list
        vel = []
        pos = 0
        for x in range(0, len(l_best)):
            if not x:
                temp0 = self.function(_list[len(l_best) - 1])
                pos = len(l_best) - 1
            else:
                temp0 = self.function(_list[x - 1])
                pos = x - 1
            temp1 = self.function(_list[x])
            if temp0 > temp1:
                temp0 = temp1
                pos = x
            if x == len(l_best) - 1:
                temp2 = self.function(_list[0])
                if temp2 < temp0:
                    pos = 0
            else:
                temp2 = self.function(_list[x + 1])
                if temp2 < temp0:
                    pos = x + 1
            l_best[x] = pos
            vel.append(rd.uniform(-(abs(self.up_bound - self.low_bound)), abs(self.up_bound - self.low_bound), self.dimension))
        while self.stop_case:
            for x in range(len(_list)):
                for y in xrange(self.dimension):
                    r_g, r_p = rd.random(1), rd.random(1)
                    vel[x][y] = self.w * vel[x][y] + self.especial_param_p * r_p * (position[x][y] - _list[x][y]) + self.especial_param_g \
                                                                                                          * r_g * (
                                                                                                              l_best[x][y] -
                                                                                                              _list[x][y])
                _list[x] += vel[x]
                if self.function(_list[x]) < self.function(position[x]):
                    position[x] = _list[x]
                    if self.function(position[x]) < self.function(l_best[x]):
                        l_best[x] = position[x]
        return l_best

    def App(self, is_global):
        #xk = self.PSO_global() if is_global else self.PSO_local()
        #print(xk)
        xk=numpy.array([-100,-100])
        xk1 = self.solve(xk, self.stop_case)
        print(xk1)
        xk_final = self.tunneling(xk1)

        return xk_final