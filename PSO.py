import numpy.random as rd


def PSO_global(up_bound, low_bound, w, especial_param_p, especial_param_g, function, particles_count, stop_case, n):
    _list = [rd.uniform(low_bound, up_bound, n) for x in range(0, particles_count, 1)]
    g_best = _list[0]
    position = _list
    vel = []
    for x in position:
        if function(x) < function(g_best):
            g_best = x
        vel.append(rd.uniform(-(abs(up_bound - low_bound)), abs(up_bound - low_bound), n))
    while stop_case:
        for x in _list:
            for y in xrange(n):
                r_g, r_p = rd.random(1), rd.random(1)
                vel[x][y] = w * vel[x][y] + especial_param_p * r_p * (position[x][y] - _list[x][y]) + especial_param_g \
                                                                                                      * r_g * (
                                                                                                          g_best[y] -
                                                                                                          _list[x][y])
            _list[x] += vel[x]
            if function(_list[x]) < function(position[x]):
                position[x] = _list[x]
                if function(position[x]) < function(g_best):
                    g_best = position[x]
    return g_best


def PSO_local(up_bound, low_bound, w, especial_param_p, especial_param_g, function, particles_count, stop_case, n):
    _list = [rd.uniform(low_bound, up_bound, n) for x in range(0, particles_count, 1)]
    l_best = _list
    position = _list
    vel = []
    pos = 0
    for x in range(0, len(l_best)):
        if not x:
            temp0 = function(_list[len(l_best) - 1])
            pos = len(l_best) - 1
        else:
            temp0 = function(_list[x - 1])
            pos = x - 1
        temp1 = function(_list[x])
        if temp0 > temp1:
            temp0 = temp1
            pos = x
        if x == len(l_best) - 1:
            temp2 = function(_list[0])
            if temp2 < temp0:
                pos = 0
        else:
            temp2 = function(_list[x + 1])
            if temp2 < temp0:
                pos = x + 1
        l_best[x] = pos
        vel.append(rd.uniform(-(abs(up_bound - low_bound)), abs(up_bound - low_bound), n))
    while stop_case:
        for x in _list:
            for y in xrange(n):
                r_g, r_p = rd.random(1), rd.random(1)
                vel[x][y] = w * vel[x][y] + especial_param_p * r_p * (position[x][y] - _list[x][y]) + especial_param_g \
                                                                                                      * r_g * (
                                                                                                          l_best[x][y] -
                                                                                                          _list[x][y])
            _list[x] += vel[x]
            if function(_list[x]) < function(position[x]):
                position[x] = _list[x]
                if function(position[x]) < function(l_best[x]):
                    l_best[x] = position[x]
    return l_best