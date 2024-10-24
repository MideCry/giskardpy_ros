from time import time

import gurobipy as gp
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from gurobipy import GRB

horizon = 20
dt = 0.0225
vel_limit = 1
acc_limit = GRB.INFINITY
jerk_limit = 30
vel_start = 0.
acc_start = 0
target = -1
alpha = 0.1
vel_weight = 0.01
acc_weight = 0.0
jerk_weight = 0.01
slack_weight = 1

vel_weights = []
acc_weights = []
jerk_weights = []
for i in range(horizon):
    w = (alpha * vel_weight + ((vel_weight - vel_weight * alpha) * i) / horizon) * (1 / vel_limit ** 2)
    vel_weights.append(w)
    w = (alpha * acc_weight + ((acc_weight - acc_weight * alpha) * i) / horizon) * (1 / acc_limit ** 2)
    acc_weights.append(w)
    w = (alpha * jerk_weight + ((jerk_weight - jerk_weight * alpha) * i) / horizon) * (1 / jerk_limit ** 2)
    jerk_weights.append(w)

def print_stats_f(model, Q):
    print(f'obj terms {model.getObjective().size()}')
    print(f'Q size {Q.shape}')
    print(f'constraints {len(model.getConstrs())}')
    print(f'Condition number {np.linalg.cond(Q)}')

def is_diagonal(matrix):
    # Create a diagonal matrix using the diagonal of the original matrix
    diagonal_matrix = np.diag(np.diag(matrix))

    # Check if the original matrix is equal to its diagonal matrix
    return np.allclose(matrix, diagonal_matrix)

def implicit_qp(vel_curr: float, acc_curr: float):
    # Define problem variables
    vel = cp.Variable(horizon)
    acc = cp.Variable(horizon)
    jerk = cp.Variable(horizon)
    slack = cp.Variable(1)

    # Define the quadratic objective
    vel_weights = []
    acc_weights = []
    jerk_weights = []
    for i in range(horizon):
        w = (alpha * vel_weight + ((vel_weight - vel_weight * alpha) * dt) / horizon) * (1 / vel_limit ** 2)
        vel_weights.append(w)
        w = (alpha * acc_weight + ((acc_weight - acc_weight * alpha) * dt) / horizon) * (1 / acc_limit ** 2)
        acc_weights.append(w)
        w = (alpha * jerk_weight + ((jerk_weight - jerk_weight * alpha) * dt) / horizon) * (1 / jerk_limit ** 2)
        jerk_weights.append(w)
    V = cp.quad_form(vel, np.diag(vel_weights))
    A = cp.quad_form(acc, np.diag(acc_weights))
    J = cp.quad_form(jerk, np.diag(jerk_weights))
    S = cp.quad_form(slack, np.eye(1) * slack_weight)
    Q = V + A + J + S

    # Define constraints
    direct_limits = [vel <= vel_limit,
                     vel >= -vel_limit,
                     acc <= acc_limit,
                     acc >= -acc_limit,
                     jerk <= jerk_limit,
                     jerk >= -jerk_limit]

    model = [vel_curr == vel[0] - acc[0] * dt,
             acc_curr == acc[0] - jerk[0] * dt]
    for i in range(1, horizon):
        model.append(vel[i] == vel[i - 1] + acc[i] * dt)
        model.append(acc[i] == acc[i - 1] + jerk[i] * dt)
    model.append(vel[-1] == 0)
    model.append(acc[-1] == 0)

    goal = [target == cp.sum(vel) * dt + slack[0]]

    constraints = direct_limits + model + goal

    # Define the optimization problem
    problem = cp.Problem(cp.Minimize(Q), constraints)
    print("Number of variables:", problem.size_metrics.num_scalar_variables)
    print("Number of constraints:", problem.size_metrics.num_scalar_eq_constr)
    print("Number of constraints:", problem.size_metrics.num_scalar_leq_constr)
    data = problem.get_problem_data(cp.OSQP)[0]
    Q_matrix = data['P'].toarray()
    print(is_diagonal(Q_matrix))

    # Solve the problem
    print(problem.solve(solver=cp.GUROBI))

    # print(vel.value)
    # print(sum(vel.value))
    # print(slack.value + sum(vel.value))
    # print(acc.value)
    # print(jerk.value)
    # Get the result
    return vel.value, acc.value, jerk.value

def implicit_qp_gurobi(pos_curr: float, vel_curr: float, acc_curr: float, target: float, print_states: bool):
    model = gp.Model("implicit_qp_vel_controlled")
    model.setParam('OutputFlag', 0)
    vars = model.addMVar(horizon*3+1, lb=[-vel_limit]*horizon+[-acc_limit]*horizon+[-jerk_limit]*horizon+[-GRB.INFINITY],
                         ub=[vel_limit]*horizon+[acc_limit]*horizon+[jerk_limit]*horizon+[GRB.INFINITY], name="x")
    vel = vars[:horizon]
    acc = vars[horizon: horizon*2]
    jerk = vars[horizon*2:-1]
    slack = vars[-1]

    # Define the quadratic objective
    Q = np.diag(vel_weights+acc_weights+jerk_weights+[slack_weight])
    model.setMObjective(Q, None, 0.0, None, None, None, sense=GRB.MINIMIZE)

    model.addConstr(vel_curr == vel[0] - acc[0] * dt, name="initial vel")
    model.addConstr(acc_curr == acc[0] - jerk[0] * dt, name="initial acc")
    for i in range(1, horizon):
        model.addConstr(vel[i] == vel[i - 1] + acc[i] * dt, name='vel ' + str(i))
        model.addConstr(acc[i] == acc[i - 1] + jerk[i] * dt, name='acc ' + str(i))
    model.addConstr(vel[horizon-1] == 0, name="final vel")
    model.addConstr(acc[horizon-1] == 0, name="final acc")

    expr = vel.sum() * dt
    expr += slack  # Slack variable
    model.addConstr(expr == target - pos_curr, name="target_constraint")


    # Define the optimization problem
    start_time = time()
    model.optimize()
    solve_time = time() - start_time
    if print_states:
        print_stats_f(model, Q)

    try:
        vel = model.X[:horizon]
        acc = model.X[horizon:horizon*2]
        jerk = model.X[horizon*2:-1]
    except Exception as e:
        pass
    # print(vel.value)
    # print(sum(vel.value))
    # print(slack.value + sum(vel.value))
    # print(acc.value)
    # print(jerk.value)
    # Get the result
    return vel, acc, jerk, solve_time


def implicit_qp_vel_controlled_gurobi(pos_curr: float, vel_curr: float, acc_curr: float, target: float, print_states: bool, jerk_limits: bool):
    n = horizon + 1  # Adding 1 for the slack variable
    # Create a new Gurobi model
    model = gp.Model("implicit_qp_vel_controlled")

    # Set Gurobi to be silent
    model.setParam('OutputFlag', 0)

    # Define decision variables
    vel = model.addMVar(n, lb=-vel_limit, ub=vel_limit, name="x")
    # Set variable bound for slack variable (last variable)
    vel[horizon].lb = -GRB.INFINITY  # Slack is typically non-negative
    vel[horizon].ub = GRB.INFINITY  # Slack is typically non-negative

    # Initialize Q matrix and c vector
    Q = np.zeros((n, n))
    c = np.zeros(n)

    # Velocity term V contributes to Q
    for i in range(horizon):
        Q[i, i] += 2 * vel_weights[i]

    # Acceleration term A
    # Acceleration: acc_i = (vel_i - vel_{i-1}) / dt
    # For i = 0:
    # acc_0 = (vel_0 - vel_curr) / dt
    # For i >= 1:
    # acc_i = (vel_i - vel_{i-1}) / dt
    if jerk_limits:
        for i in range(horizon):
            w = 2 * acc_weights[i] / (dt ** 2)
            if i == 0:
                # Terms involving vel_0 and constant vel_curr
                Q[i, i] += w
                c[i] -= w * vel_curr
                const_term = w * vel_curr ** 2  # This is constant and can be ignored in optimization
            else:
                # Terms involving vel_i and vel_{i-1}
                Q[i, i] += w
                Q[i - 1, i - 1] += w
                Q[i, i - 1] -= w
                Q[i - 1, i] -= w  # Symmetric

        # Jerk term J
        # Jerk: jerk_i = (acc_i - acc_{i-1}) / dt
        # Since acc_i depends on velocities, expand jerk_i in terms of velocities

        for i in range(horizon):
            w = 2 * jerk_weights[i] / (dt ** 4)
            if i == 0:
                # Jerk involves vel_0, vel_curr, and acc_curr
                Q[i, i] += w
                c[i] -= w * (2 * vel_curr - dt * acc_curr)
                const_term = w * (vel_curr - dt * acc_curr) ** 2  # Can be ignored
            elif i == 1:
                # Jerk involves vel_0, vel_1, and vel_curr
                Q[i - 1, i - 1] += w
                Q[i, i] += w
                Q[i - 1, i] -= w
                Q[i, i - 1] -= w  # Symmetric
                c[i - 1] += w * vel_curr
                c[i] -= w * vel_curr
                const_term = w * vel_curr ** 2  # Can be ignored
            else:
                # Jerk involves vel_{i-2}, vel_{i-1}, vel_i
                Q[i - 2, i - 2] += w
                Q[i - 1, i - 1] += 4 * w
                Q[i, i] += w
                Q[i - 2, i - 1] -= 2 * w
                Q[i - 1, i - 2] -= 2 * w  # Symmetric
                Q[i - 1, i] -= 2 * w
                Q[i, i - 1] -= 2 * w  # Symmetric
                Q[i - 2, i] += w
                Q[i, i - 2] += w  # Symmetric

    # Slack term S
    Q[n - 1, n - 1] += 2 * slack_weight

    # Set the objective
    model.setMObjective(Q, c, 0.0, vel, vel, sense=GRB.MINIMIZE)
    model.addConstr(vel[horizon - 1] == 0, name="final_vel_zero")

    # Target Constraint: sum(vel) + slack == target
    expr = gp.LinExpr()
    for i in range(horizon):
        expr += vel[i] * dt
    expr += vel[horizon]  # Slack variable
    model.addConstr(expr == target - pos_curr, name="target_constraint")

    # Velocity limits are already set via variable bounds

    # Acceleration and Jerk Limits
    # Acceleration constraints:
    acc = []
    for i in range(horizon):
        if i == 0:
            acc.append((vel[i] - vel_curr) / dt)
        else:
            acc.append((vel[i] - vel[i - 1]) / dt)
        model.addConstr(acc[i] <= acc_limit, name=f"acc_limit_upper_{i}")
        model.addConstr(acc[i] >= -acc_limit, name=f"acc_limit_lower_{i}")
    model.addConstr(acc[i] == 0, name='final acc 0')
    # Jerk constraints:
    jerk = []
    for i in range(horizon):
        if i == 0:
            jerk.append((acc[i] - acc_curr) / dt)
        else:
            jerk.append((acc[i] - acc[i - 1]) / dt)
        model.addConstr(jerk[i] <= jerk_limit, name=f"jerk_limit_upper_{i}")
        model.addConstr(jerk[i] >= -jerk_limit, name=f"jerk_limit_lower_{i}")

    # Optimize the model
    start_time = time()
    model.optimize()
    solve_time = time() - start_time
    if print_states:
        print_stats_f(model, Q)

    # Check if the optimization was successful
    if model.status != GRB.OPTIMAL:
        print(f"Optimization was not successful. Status: {model.status}")
        return None, None, None

    # Retrieve optimized velocities
    vel_values = model.X[:-1]

    # Calculate accelerations and jerks based on optimized velocities
    acc_values = []
    for i in range(horizon):
        if i == 0:
            acc_i = (vel_values[i] - vel_curr) / dt
        else:
            acc_i = (vel_values[i] - vel_values[i - 1]) / dt
        acc_values.append(acc_i)

    jerk_values = []
    for i in range(horizon):
        if i == 0:
            jerk_i = (acc_values[i] - acc_curr) / dt
        else:
            jerk_i = (acc_values[i] - acc_values[i - 1]) / dt
        jerk_values.append(jerk_i)

    # Optionally, verify if Q matrix is diagonal (for informational purposes)
    model.printStats()
    return vel_values, acc_values, jerk_values, solve_time


def implicit_qp_jerk_controlled_gurobi(pos_curr: float, vel_curr: float, acc_curr: float, target: float, print_states: bool):
    """
    Solves the jerk-controlled MPC problem using Gurobi with an explicit Q matrix.
    """

    # Create a new Gurobi model
    model = gp.Model("implicit_qp_jerk_controlled")

    # Set Gurobi to be silent
    model.setParam('OutputFlag', 0)

    # Total number of variables (jerk + slack)
    n = horizon + 1  # Adding 1 for the slack variable

    # Define decision variables
    vars = model.addVars(n, lb=-jerk_limit, ub=jerk_limit, name="x")
    # Set variable bounds for jerk
    vars[horizon].lb = -GRB.INFINITY
    vars[horizon].ub = GRB.INFINITY

    vel_weights_np = np.array(vel_weights)
    acc_weights_np = np.array(acc_weights)
    jerk_weights_np = np.array(jerk_weights)

    # Initialize Q matrix and c vector
    Q = np.zeros((n, n))
    c = np.zeros(n)

    # Construct matrices to represent acc and vel in terms of jerk
    # Acceleration: acc = A_acc * jerk + b_acc
    # Velocity: vel = A_vel * jerk + b_vel

    # Matrix A_acc: lower triangular matrix with dt on and below the diagonal
    A_acc = np.tril(np.ones((horizon, horizon))) * dt

    # b_acc: vector with acc_curr
    b_acc = np.full(horizon, acc_curr)

    # Matrix A_vel: constructed using cumulative sums
    A_vel = np.zeros((horizon, horizon))
    for i in range(horizon):
        for j in range(horizon):
            if j <= i:
                A_vel[i, j] = dt**2 * (i - j + 1)
    # b_vel: vector with vel_curr + dt * acc_curr * (i + 1)
    b_vel = vel_curr + acc_curr * dt * np.arange(1, horizon + 1)

    # Quadratic terms for jerk weights (J)
    for i in range(horizon):
        Q[i, i] += 2 * jerk_weights_np[i]

    # Quadratic terms for acceleration (A)
    Q_acc = A_acc.T @ np.diag(2 * acc_weights_np) @ A_acc
    c_acc = A_acc.T @ (2 * acc_weights_np * b_acc)
    Q[:horizon, :horizon] += Q_acc
    c[:horizon] += -c_acc

    # Quadratic terms for velocity (V)
    Q_vel = A_vel.T @ np.diag(2 * vel_weights_np) @ A_vel
    c_vel = A_vel.T @ (2 * vel_weights_np * b_vel)
    Q[:horizon, :horizon] += Q_vel
    c[:horizon] += -c_vel

    # Slack term (S)
    Q[n - 1, n - 1] += 2 * slack_weight

    # Set the objective
    vars_list = [vars[i] for i in range(n)]
    model.setMObjective(Q, c, 0.0, vars_list, None, sense=GRB.MINIMIZE)

    # Final Conditions: vel[-1] == 0, acc[-1] == 0
    # Compute final acceleration and velocity expressions
    acc_expr = b_acc[-1] + gp.quicksum(A_acc[-1, j] * vars[j] for j in range(horizon))
    vel_expr = b_vel[-1] + gp.quicksum(A_vel[-1, j] * vars[j] for j in range(horizon))
    model.addConstr(acc_expr == 0, name="final_acc_zero")
    model.addConstr(vel_expr == 0, name="final_vel_zero")

    # Target Constraint: sum(vel) + slack == target
    vel_sums = b_vel + A_vel @ np.array([vars[i] for i in range(horizon)])
    total_vel = gp.quicksum(vel_sums[i] * dt for i in range(horizon))
    model.addConstr(total_vel + vars[horizon] == target - pos_curr, name="target_constraint")

    # Acceleration and Velocity Limits
    for i in range(horizon):
        # Acceleration expressions
        acc_i = b_acc[i] + gp.quicksum(A_acc[i, j] * vars[j] for j in range(horizon))
        model.addConstr(acc_i <= acc_limit, name=f"acc_limit_upper_{i}")
        model.addConstr(acc_i >= -acc_limit, name=f"acc_limit_lower_{i}")
        # Velocity expressions
        vel_i = b_vel[i] + gp.quicksum(A_vel[i, j] * vars[j] for j in range(horizon))
        model.addConstr(vel_i <= vel_limit, name=f"vel_limit_upper_{i}")
        model.addConstr(vel_i >= -vel_limit, name=f"vel_limit_lower_{i}")

    # Jerk Limits (already set via variable bounds)

    # Optimize the model
    start_time = time()
    model.optimize()
    solve_time = time() - start_time
    if print_states:
        print_stats_f(model, Q)

    # Check if the optimization was successful
    if model.status != GRB.OPTIMAL:
        print(f"Optimization was not successful. Status: {model.status}")
        return None, None, None

    # Retrieve optimized jerk values
    jerk_values = [vars[i].X for i in range(horizon)]
    slack_value = vars[horizon].X

    # Calculate accelerations and velocities based on optimized jerk values
    jerk_array = np.array(jerk_values)
    acc_values = b_acc + A_acc @ jerk_array
    vel_values = b_vel + A_vel @ jerk_array

    return vel_values.tolist(), acc_values.tolist(), jerk_values, solve_time


def explicit_qp_jerk_controlled(vel_curr: float, acc_curr: float, target: float):
    # Define problem variables
    jerk = cp.Variable(horizon)
    slack = cp.Variable(1)

    # Define the quadratic objective
    weights = []
    for i in range(horizon):
        w = (alpha * jerk_weight + ((jerk_weight - jerk_weight * alpha) * dt) / horizon) * (1 / jerk_limit ** 2)
        weights.append(w)
    J = cp.quad_form(jerk, np.diag(weights))
    S = cp.quad_form(slack, np.eye(1) * 10000)
    Q = J + S

    # Define constraints
    direct_limits = [jerk <= jerk_limit,
                     jerk >= -jerk_limit]

    acc = [acc_curr]
    vel = [vel_curr + acc_curr * dt]
    for i in range(1, horizon):
        acc.append(acc[-1] + jerk[i - 1] * dt)
        direct_limits.append(acc[i] <= acc_limit)
        direct_limits.append(acc[i] >= -acc_limit)
        vel.append(vel[-1] + acc[i - 1] * dt)
        direct_limits.append(vel[i] <= vel_limit)
        direct_limits.append(vel[i] >= -vel_limit)

    model = []
    model.append(0 == acc[-1])
    model.append(0 == vel[-1])

    _sum = slack[0]
    for i in range(horizon):
        _sum += vel[i]
    goal = [target == _sum]

    constraints = direct_limits + model + goal

    # Define the optimization problem
    problem = cp.Problem(cp.Minimize(Q), constraints)

    # Solve the problem
    print(problem.solve())
    vel = [x.value if hasattr(x, 'value') else x for x in vel]
    acc = [x.value if hasattr(x, 'value') else x for x in acc]

    print(vel)
    print(sum(vel))
    print(slack.value + sum(vel))
    print(acc)
    print(jerk.value)
    # Get the result
    return vel, acc, jerk.value


def explicit_qp_vel_controlled(vel_curr: float, acc_curr: float, target: float):
    # Define problem variables
    vel = cp.Variable(horizon)
    slack = cp.Variable(1)

    # Define the quadratic objective
    weights = []
    for i in range(horizon):
        w = (alpha * vel_weight + ((vel_weight - vel_weight * alpha) * dt) / horizon) * (1 / vel_limit ** 2)
        weights.append(w)
    J = cp.quad_form(vel, np.diag(weights))
    S = cp.quad_form(slack, np.eye(1) * 10000)
    Q = J + S

    # Define constraints
    direct_limits = [vel <= vel_limit,
                     vel >= -vel_limit]

    acc = []
    jerk = []
    for i in range(horizon-1):
        acc.append((vel[i+1] - vel[i])/dt)
        direct_limits.append(acc[i] <= acc_limit)
        direct_limits.append(acc[i] >= -acc_limit)
    for i in range(horizon-2):
        jerk.append((acc[i+1] - acc[i])/dt)
        direct_limits.append(jerk[i] <= jerk_limit)
        direct_limits.append(jerk[i] >= -jerk_limit)

    model = []
    model.append(vel_curr + acc_curr * dt == vel[0])
    model.append(acc_curr == acc[0])
    model.append(vel[-1] == 0)
    model.append(acc[-1] == 0)

    _sum = slack[0]
    for i in range(horizon):
        _sum += vel[i]
    goal = [target == _sum]

    constraints = direct_limits + model + goal

    # Define the optimization problem
    problem = cp.Problem(cp.Minimize(Q), constraints)

    # Solve the problem
    print(problem.solve())
    vel = [x.value if hasattr(x, 'value') else x for x in vel]
    acc = [x.value if hasattr(x, 'value') else x for x in acc]
    jerk = [x.value if hasattr(x, 'value') else x for x in jerk]

    acc = np.pad(acc, (0, horizon - len(acc)), 'constant')
    jerk = np.pad(jerk, (0, horizon - len(jerk)), 'constant')

    print(vel)
    print(sum(vel))
    print(slack.value + sum(vel))
    print(acc)
    print(jerk)
    # Get the result
    return vel, acc, jerk


def interpolate_velocity(vel_mpc, dt_mpc, dt_control):
    # Linear interpolation between the first two points of the MPC velocity trajectory
    v0 = vel_mpc[0]
    v1 = vel_mpc[1]

    # Time in the control loop
    t_control = dt_control

    # Linear interpolation formula
    interpolated_vel = v0 + (t_control / dt_mpc) * (v1 - v0)

    return interpolated_vel



def mpc(method):
    pos = 0
    vel = 0
    acc = 0
    target = 1
    times = []
    print_stats = True
    vel_value, acc_value, jerk_value, solve_time = method(pos, vel, acc, target, print_stats)
    times.append(solve_time)
    print(f'avg time {np.average(times)}')
    pos_traj = []
    for v in vel_value:
        pos += v * dt
        pos_traj.append(pos)
    return pos_traj, [0]*len(vel_value), vel_value, acc_value, jerk_value


time_limit = 5
ctrl_dt = 0.05

def simulate(method):
    pos = 0
    vel = 0
    acc = 0
    target = 1
    target_traj = []
    pos_traj = []
    vel_traj = []
    acc_traj = []
    jerk_traj = []
    times = []
    print_stats = True
    swap_interval = 1.5
    for i in range(int(time_limit/ctrl_dt)):
        target_traj.append(target)
        vel_value, acc_value, jerk_value, solve_time = method(pos, vel, acc, target, print_stats)
        jerk = jerk_value[0]
        acc += jerk * ctrl_dt
        vel += acc * ctrl_dt
        pos += vel * ctrl_dt
        pos_traj.append(pos)
        vel_traj.append(vel)
        acc_traj.append(acc)
        jerk_traj.append(jerk)
        times.append(solve_time)
        if i*ctrl_dt >= swap_interval:
            target *= -1
            swap_interval += swap_interval
        print_stats = False
    print(f'avg time {np.average(times)}')
    print(f'overshoot {np.max(np.abs(pos_traj))}')
    print('================================================')
    return pos_traj, target_traj, vel_traj, acc_traj, jerk_traj

# f = mpc
f = simulate

print('old')
data = [
    ('old', f(implicit_qp_gurobi)),
    ('vel only', f(lambda po, v, a, t, p,: implicit_qp_vel_controlled_gurobi(po, v, a, t, p, False))),
    # ('vel w/ jerk', f(lambda po, v, a, t, p,: implicit_qp_vel_controlled_gurobi(po, v, a, t, p, True))),
    # ('jerk w/ vel',f(implicit_qp_jerk_controlled_gurobi))
]
# horizon = 9
# dt = 0.05
# data.append(('old', f(implicit_qp_gurobi)))
# data.append(('vel only', f(lambda po, v, a, t, p,: implicit_qp_vel_controlled_gurobi(po, v, a, t, p, False))))


# Plotting
time_axis = np.arange(len(data[0][1][0])) * ctrl_dt

plt.figure(figsize=(10, 10))

# Plot position
plt.subplot(4, 1, 1)
plt.plot(time_axis, data[0][1][1], label="Target", color='k')
plt.plot(time_axis, data[0][1][0], label="Position", linewidth=5, color='lightgray')
for name, d in data[1:]:
    plt.plot(time_axis, d[0], label=f"Position {name}")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(time_axis, data[0][1][2], label="Vel",  linewidth=5, color='lightgray')
for name, d in data[1:]:
    plt.plot(time_axis, d[2], label=f"Vel {name}")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(time_axis, data[0][1][3], label="Acc",  linewidth=5, color='lightgray')
for name, d in data[1:]:
    plt.plot(time_axis, d[3], label=f"Acc {name}")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(time_axis, data[0][1][4], label="Jerk",  linewidth=5, color='lightgray')
for name, d in data[1:]:
    plt.plot(time_axis, d[4], label=f"Jerk {name}")
plt.grid(True)
plt.legend()

# Show the plots
plt.tight_layout()
plt.savefig(f'h:{horizon} dt:{dt} ctlr dt: {ctrl_dt} time:{time_limit}.pdf')
plt.savefig('mpc.pdf')
plt.show()
