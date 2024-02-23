using SphericalFunctions
using Plots
using DifferentialEquations
using LinearAlgebra

# Parameters
l_max = 4  # Maximum degree of spherical harmonics
sigma = 0.2  # Width of the initial condition
t_max = 10.0  # Maximum time
n_points = 100  # Number of points for theta and phi

# Define theta and phi arrays
theta = range(0, stop=π, length=n_points)
phi = range(0, stop=2π, length=n_points)

# Mesh grid creation using broadcasting
Theta = repeat(reshape(theta, 1, :), length(phi), 1)
Phi = repeat(phi, 1, length(theta))

# Initial condition function
initial_condition(θ, σ) = exp(-θ^2 / (2σ^2))

# Project initial condition onto spherical harmonics basis
function project_initial_condition(l, m, sigma, theta, phi)
    integral = 0.0
    dθ = theta[2] - theta[1]
    dφ = phi[2] - phi[1]
    for θ in theta
        for φ in phi
            Y_lm = conj(𝕐(l, m, θ, φ))
            integral += initial_condition(θ, sigma) * Y_lm * sin(θ)
        end
    end
    return integral * dθ * dφ
end

# ODE system for c_lm(t) and psi_lm(t)
function ode_system!(dydt, y, p, t)
    l_max = p
    for l in 0:l_max
        for m in l𝕞range(l)
            idx = l^2 + l + m
            dydt[idx] = y[idx+1]  # d/dt c_lm(t) = psi_lm(t)
            dydt[idx+1] = -l * (l + 1) * y[idx]  # d/dt psi_lm(t) = -l(l+1)c_lm(t)
        end
    end
end

# Solve the ODE system
function solve_ode(l_max, sigma, t_max, theta, phi)
    # Number of coefficients
    n_coeffs = (l_max + 1)^2
    # Initial conditions for c_lm(0) and psi_lm(0)
    y0 = zeros(2 * n_coeffs)
    for l in 0:l_max
        for m in l𝕞range(l)
            idx = l^2 + l + m
            y0[idx] = project_initial_condition(l, m, sigma, theta, phi)
        end
    end
    # Time span
    t_span = (0.0, t_max)
    # Solve ODE
    prob = ODEProblem(ode_system!, y0, t_span, l_max)
    sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)
    return sol
end

# Visualization of the solution at a given time t
function visualize_solution(sol, t, l_max, theta, phi, n_points)
    Z = zeros(n_points, n_points)
    for l in 0:l_max
        for m in l𝕞range(l)
            idx = l^2 + l + m
            c_lm = sol(t)[idx]
            for (i, θ) in enumerate(theta)
                for (j, φ) in enumerate(phi)
                    Z[i, j] += c_lm * real(𝕐(l, m, θ, φ))
                end
            end
        end
    end
    # Plot
    p = contour(phi, theta, Z', levels=100, cmap="viridis")
    display(p)
end

# Main execution
l_max_values = [4, 8, 12]  # Different l_max values for comparison
for l_max in l_max_values
    sol = solve_ode(l_max, sigma, t_max, theta, phi)
    # Visualize at different time steps
    for t in range(0, stop=t_max, length=5)
        visualize_solution(sol, t, l_max, theta, phi, n_points)
    end
end
