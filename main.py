import pylab
import copy
from sympy import diff, symbols, cos, sin
import numpy as np
from scipy import optimize
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
from scipy.optimize import minimize
from sympy.solvers.solveset import linsolve
from scipy.optimize import rosen, rosen_der, rosen_hess, rosen_hess_prod
from matplotlib import cm


print_Hesse_matrix = True
print_Jakobi_matrix = True
step = 0
x_l = 0
y_l = 0

def graphics(x, y, z, point=None, with_point=None):
    fig = pylab.figure()
    axes = Axes3D(fig)
    axes.plot_surface(x, y, z, color="red")
    global step
    global x_l
    global y_l
    print()
    print("INFO: Finding invalid points...")
    linear_r_s = linear_inequality_right_side()
    non_linear_r_s = non_linear_inequality_right_side()
    new_x = copy.deepcopy(x)
    new_y = copy.deepcopy(y)
    new_z = copy.deepcopy(z)
    for i_index, x_item in enumerate(new_x[0]):
        for j_index, y_item in enumerate(new_y):
            if (g1([x_item, y_item[0]]) > linear_r_s[0]) or (g2([x_item, y_item[0]]) > linear_r_s[1])\
                    or (g3([x_item, y_item[0]]) > non_linear_r_s[0]):
                new_z[int((y_item[0]-y_l)/step)][int((x_item-x_l)/step)] = np.nan
    axes.plot_surface(new_x, new_y, new_z, color="green")
    if with_point:
        #axes.plot(*point, func([point[0],point[1]]), 'r*', markersize=20)
        axes.scatter(point[0],point[1],func([point[0],point[1]]), color='yellow', s=40, marker='o')

    pylab.show()


def get_intervals():
    global step
    global x_l
    global y_l
    try:
        x_l_ = float(input("Enter the left border on the X-axis : "))
        x_r = float(input("Enter the right border on the X-axis : "))
        y_l_ = float(input("Enter the left border on the Y-axis : "))
        y_r = float(input("Enter the right border on the Y-axis : "))
        step_ = float(input("Enter the step of the grid : "))
        step = step_
        x_l = x_l_
        y_l = y_l_
    except:
        print("ERROR: Check the correctness of the entered data. The value must be integers or real numbers.")
        return -1
    if x_l > x_r or y_l > y_r:
        print("ERROR: The left border exceeds the value of the right.")
        return -1
    return [x_l, x_r, y_l, y_r, step_]


def func(point):
    x, y = point
    #return 50 * (np.sin(3 * x) - y) ** 2 + 9 * x ** 2
    return 3*(y+x**2)**2 + (x**2 - 1)**2

def symb_func(point):
    x, y = point
    return 3*(y+x**2)**2 + (x**2 - 1)**2


def g1(point):
    x, y = point
    return y + 0.5 * x
    #return y + x


def g2(point):
    x, y = point
    return y
    #return y - x


def g3(point):
    x, y = point
    #return y ** 3 - x
    return -10 * (x + 1)**2 - (y - 2)**2


def linear_inequality_right_side():
    return [-0.5, -0.8]


def non_linear_inequality_right_side():
    return [-12]

def makeData(x, y, step):
    x_ = np.arange(x[0], x[1], step[0])
    y_ = np.arange(y[0], y[1], step[0])

    x_grid, y_grid = np.meshgrid(x_, y_)

    z_grid_f = func([x_grid, y_grid])
    return x_grid, y_grid, z_grid_f


def get_derivative():
    x, y = symbols('x y')
    variables = (x, y)
    symbol_func = sin(x) * cos(y)
    gr = (symbol_func.diff(x), symbol_func.diff(y))
    print("The function's gradient : ", gr)
    print("Enter the point to calculate the gradient")
    x_g = input("x : ")
    y_g = input("y : ")
    gr_x = gr[0].subs({x: x_g, y: y_g})
    gr_y = gr[1].subs({x: x_g, y: y_g})
    print("The gradient in the point is : (", gr_x, ",", gr_y, ")")
    derivatives = []
    for variable in variables:
        derivatives.append(diff(symbol_func, variable))
    print("The derivative of the function cos(x) * sin(x) is equal to : ", derivatives)


def cons_f(x):
    #return [x[1] ** 3 - x[0] + 1, x[1] ** 3 - x[0] + 1]
    return [-10 * (x[0] + 1)**2 - (x[1] - 2)**2]
    #return [-10 * (x[0] + 1)**2 - (x[1] - 2)**2]


def cons_J(x):
    global print_Jakobi_matrix
    x_, y_ = symbols('x y')
    non_lin_cons = cons_f([x_, y_])
    Jakobi = []

    for inequality in non_lin_cons:
        df_x_y = []
        df_x = inequality.diff(x_)
        df_y = inequality.diff(y_)
        df_x_y.append(df_x)
        df_x_y.append(df_y)
        Jakobi.append(df_x_y)
    if print_Jakobi_matrix:
        print("#" * 50)
        print("#\t\t\tJakobi matrix")
        print("#" * 50)
        print(Jakobi)
        print("\n\n")
        print_Jakobi_matrix = False
    results = []
    for Jakobi_item in Jakobi:
        tmp = []
        for item in Jakobi_item:
            tmp.append(float(item.subs({x_: x[0], y_: x[1]})))
        results.append(tmp)
    return results


def cons_H(x, v):
    global print_Hesse_matrix
    x_, y_ = symbols('x y')
    non_lin_cons = cons_f([x_, y_])
    Hesses = []

    for inequality in non_lin_cons:
        df_xx = inequality.diff(x_).diff(x_)
        df_yy = inequality.diff(y_).diff(y_)
        df_xy = inequality.diff(x_).diff(y_)
        df_yx = inequality.diff(y_).diff(x_)
        Hesses.append([[df_xx, df_xy], [df_yx, df_yy]])

    if print_Hesse_matrix:
        print("#" * 50)
        print("#\t\t\tHesse matrix")
        print("#" * 50)
        print(Hesses)
        print_Hesse_matrix = False
    results = []
    for Hesse in Hesses:
        calc_line = []
        for line in Hesse:
            tmp = []
            for item in line:
                tmp.append(float(item.subs({x_: x[0], y_: x[1]})))
            calc_line.append(tmp)
        results.append(calc_line)
    return v[0] * np.array(results[0])


def intersection():
    x, y = symbols('x y')
    variables = (x, y)
    sym_g1 = g1([x, y])
    sym_g2 = g2([x, y])
    linear_r_s = linear_inequality_right_side()
    d_x_1 = float(sym_g1.diff(x))
    d_y_1 = float(sym_g1.diff(y))
    d_x_2 = float(sym_g2.diff(x))
    d_y_2 = float(sym_g2.diff(y))
    return [[d_x_1, d_y_1], [d_x_2, d_y_2]], linear_r_s


def get_lambda(x_input):
    eps = 0.05
    active_intersection = []
    x, y = symbols('x y')
    g = [g1([x, y]), g2([x, y]), g3([x, y])]
    linear_r_s = linear_inequality_right_side()
    non_linear_r_s = non_linear_inequality_right_side()
    print("#" * 50)
    print("#\t\tAll constraints", " " * 25, "#")
    print("#" * 50)
    print(g[0])
    print(g[1])
    print(g[2])
    print("\n\n")
    if -eps <= g[0].subs({x: x_input[0], y: x_input[1]}) - linear_r_s[0] <= eps:
        active_intersection.append(g[0])
    if -eps <= g[1].subs({x: x_input[0], y: x_input[1]}) - linear_r_s[1] <= eps:
        active_intersection.append(g[1])
    if -eps <= g[2].subs({x: x_input[0], y: x_input[1]}) - non_linear_r_s[0] <= eps:
        active_intersection.append(g[2])
    print("The number of active intersection is : ", len(active_intersection))
    lambda_ = [symbols('l0')]
    for i in range(len(active_intersection)):
        lambda_.append(symbols('l' + str(i + 1)))
    for intersection in active_intersection:
        print(intersection)
    print("\n\n")
    L = lambda_[0] * symb_func([x, y])
    for index, inters in enumerate(active_intersection):
        L = L + lambda_[index + 1] * g[index]
    print("#" * 50)
    print("#\t\tLagrange function", " " * 23, "#")
    print("#" * 50)
    print(L)
    print("\n\n")
    d_f_x = L.diff(x)
    d_f_y = L.diff(y)
    d_f_x_in_point = d_f_x.subs({x: x_input[0], y: x_input[1]})
    d_f_y_in_point = d_f_y.subs({x: x_input[0], y: x_input[1]})
    print("#" * 50)
    print("#\t\tSystem of equations in point")
    print("#" * 50)
    print("df_x : ", d_f_x_in_point)
    print("df_y : ", d_f_y_in_point)

    print("\nINFO: Solving a linear system with respect to lambda...")
    solve = linsolve([d_f_x_in_point, d_f_y_in_point], lambda_)
    print(solve)
    print("INFO: Solving a linear system with respect to lambda...DONE")


def main():
    input_data = get_intervals()
    if input_data == -1:
        print("The program ended with an error.\nClose")
        return -1
    x, y, z = makeData(input_data[0:2], input_data[2:4], input_data[4:])

    #graphics(x, y, z)

    non_linear_r_s = non_linear_inequality_right_side()

    print("Enter the first approximation point:")
    x_i = float(input("x : "))
    y_i = float(input("y : "))
    matr, r_s = intersection()
    print("#" * 50)
    print("#\t\tMatrix of linear constraints", " " * 11, "#")
    print("#" * 50)
    print(matr)
    print("\n\n")
    print("#" * 50)
    print("#\t\tRight side of linear inequalities", " " * 6, "#")
    print("#" * 50)
    print(r_s)
    print("\n\n")
    linear_constraint = LinearConstraint(matr, [-np.inf, -np.inf], r_s)
    nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, non_linear_r_s[0],
                                               jac=cons_J, hess=cons_H)
    x0 = np.array([x_i, y_i])
    print("#"*50)
    print("#\tPreparation of conditions with restrictions...")
    print("#" * 50)
    print("\n")
    res = minimize(func, x0, method='trust-constr', jac=rosen_der, hess=rosen_hess,
                   constraints=[linear_constraint, nonlinear_constraint],
                   options={'verbose': 1})
    print("INFO: The value of the found minimum")
    print(res.x)
    opt = res.x.reshape(-1, 1)
    get_lambda(res.x)
    graphics(x, y, z, point=opt, with_point=True)
    print("g1 : ", g1([-0.31812601, -0.80041335]))
    print("g2 : ", g2([-0.31812601, -0.80041335]))
    print("g3 : ", g3([-0.31812601, -0.80041335]))


if __name__ == "__main__":
    main()
