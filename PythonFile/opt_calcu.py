#!/usr/bin/python3
# -*- coding:utf-8 -*-

from PythonFile.my_func import my_func_5x
import numpy as np
from sko.DE import DE
from sko.GA import GA
# from sko.PSO import PSO


# 该类定义了优化模型所需的一切参数
class MathOpt:
    SMALL = 1e-7

    def __init__(self, func, lb, ub, constraint_eq, constraint_ueq, peak, opt_method, repeaT):
        self.lb, self.ub = None, None
        self.best_x, self.best_y = 0.0, 0.0
        self.out = 0
        self.func = func
        self.deal_lb_ub(lb, ub)
        self.eq_ls = []
        self.ueq_ls = []
        self.deal_constraint_eq(constraint_eq)
        self.deal_constraint_ueq(constraint_ueq)
        # self.deal_peak(peak)
        self.peak = peak
        self.repeaT = int(repeaT)
        self.deal_opt_method(opt_method)

    def deal_lb_ub(self, lb, ub):
        self.lb = eval(lb)
        self.ub = eval(ub)
        for i in range(len(self.lb)):
            if self.lb[i] == self.ub[i]:
                self.ub[i] += self.SMALL

    def deal_constraint_eq(self, constraint_eq):
        str_ = constraint_eq
        # try:
        ls = str_.split('\n')
        if ls != ['']:
            for i in ls:
                self.eq_ls.append(lambda x: eval(i))
        # except

    def deal_constraint_ueq(self, constraint_ueq):
        str_ = constraint_ueq
        ls = str_.split('\n')
        if ls != ['']:
            for i in ls:
                self.ueq_ls.append(lambda x: eval(i))

    def deal_opt_method(self, opt_method):
        if self.peak == '最小值':
            if opt_method == '差分进化算法':
                self.de_opt()
            elif opt_method == '遗传算法':
                self.ga_opt()
        elif self.peak == '最大值':
            if opt_method == '差分进化算法':
                self.de_opt(max_=True)
            elif opt_method == '遗传算法':
                self.ga_opt(max_=True)

    def de_opt(self, max_=False):
        if not max_:
            de = DE(func=self.func, n_dim=len(self.lb), size_pop=50, max_iter=int(self.repeaT), lb=self.lb, ub=self.ub,
                    constraint_eq=self.eq_ls, constraint_ueq=self.ueq_ls)
            self.best_x, self.best_y = de.run()
        elif max_:
            de = DE(func=-self.func, n_dim=len(self.lb), size_pop=50, max_iter=int(self.repeaT), lb=self.lb, ub=self.ub,
                    constraint_eq=self.eq_ls, constraint_ueq=self.ueq_ls)
            self.best_x, self.best_y = de.run()
            self.best_y = -self.best_y

    def ga_opt(self, max_=False):
        if not max_:
            ga = GA(func=self.func, n_dim=len(self.lb), size_pop=50, max_iter=int(self.repeaT), lb=self.lb, ub=self.ub,
                    constraint_eq=self.eq_ls, constraint_ueq=self.ueq_ls, precision=1e-5)
            self.best_x, self.best_y = ga.run()
        elif max_:
            ga = GA(func=-self.func, n_dim=len(self.lb), size_pop=50, max_iter=int(self.repeaT), lb=self.lb, ub=self.ub,
                    constraint_eq=self.eq_ls, constraint_ueq=self.ueq_ls, precision=1e-5)
            self.best_x, self.best_y = ga.run()
            self.best_y = -self.best_y


if __name__ == "__main__":
    # my_opt = MathOpt(func=my_func_5x, lb='[0,0,0,0,0]', ub='[1,1,1,1,1]', constraint_eq='',
    #                  constraint_ueq='x[0]+x[1]-1\nx[3]+x[4]-1', peak='最小值', opt_method='差分进化算法', repeaT='100')
    # x = my_opt.best_x
    # y = my_opt.best_y
    # print(x)
    # print(y)

    my_opt = MathOpt(func=my_func_5x, lb='[0,0,0,0,0]', ub='[1,1,1,1,1]', constraint_eq='',
                     constraint_ueq='x[0]+x[1]-1\nx[3]+x[4]-1', peak='最小值', opt_method='遗传算法', repeaT='100')
    x = my_opt.best_x
    y = my_opt.best_y
    print(x)
    print(y)

