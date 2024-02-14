using Plots
using DifferentialEquations
using SpecialFunctions

# Parameters
l_max = 4
sigma = 0.2
t_max = 10.0
n_points = 100

# Define theta and phi arrays
theta = range(0, stop=π, length=n_points)
phi = range(0, stop=2π, length=n_points)
Theta, Phi = ndgrid(theta, phi)

# Initial condition function
initial_condition(θ, σ) = exp(-θ^2 / (2σ^2))

# Associated Legendre function (simplified version)
function assoc_legendre(l, m, x)
    return ((-1)^m) * sqrt((2l + 1) / (4π) * factorial(l - m) / factorial(l + m)) * legendre(l, m, x)
end

# Spherical harmonics function
function spherical_harmonics(m, l, ϕ, θ)
    legendre_val = assoc_legendre(l, abs(m), cos(θ))
    harmonic = exp(im * m * ϕ) * legendre_val
    return m < 0 ? (-1)^m * conj(harmonic) : harmonic
end

# Project initial condition onto spherical harmonics basis
function project_initial_condition(l, m, σ, θ_vals, ϕ_vals)
    integral = 0.0
    for θ_val in θ_vals, ϕ_val in ϕ_vals
        Y_lm = conj(spherical_harmonics(m, l, ϕ_val, θ_val))
        integral += initial_condition(θ_val, σ) * Y_lm * sin(θ_val)
    end
    Δθ = θ_vals[2] - θ_vals[1]
    Δϕ = ϕ_vals[2] - ϕ_vals[1]
    return integral * Δθ * Δϕ
end

# ODE system for c_lm(t) and psi_lm(t)
function ode_system(du, u, p, t)
    l_max = p
    for l in 0:l_max
        for m in -l:l
            idx = l^2 + l + m + 1
            du[idx] = u[idx+1]
            du[idx+1] = -l * (l + 1) * u[idx]
        end
    end
end

# Solve the ODE system
function solve_ode(l_max, σ, t_max, θ, ϕ)
    n_coeffs = (l_max + 1)^2
    y0 = zeros(2 * n_coeffs)
    for l in 0:l_max, m in -l:l
        idx = l^2 + l + m + 1
        y0[idx] = project_initial_condition(l, m, σ, θ, ϕ)
    end
    tspan = (0.0, t_max)
    prob = ODEProblem(ode_system, y0, tspan, l_max)
    sol = solve(prob, Tsit5(), reltol=1e-8)
    return sol
end

# Visualization
function visualize_solution(sol, t, l_max, θ, ϕ)
    Z = [real(sol(t)[l^2+l+m+1] * spherical_harmonics(m, l, ϕ, θ)) for θ in theta, ϕ in phi, l in 0:l_max, m in -l:l]
    p = contourf(ϕ, θ, Z, title="Solution at t=$t", xlabel="Phi", ylabel="Theta", color=:viridis)
    display(p)
end

# Main execution
l_max_values = [4, 8, 12]
for l_max_val in l_max_values
    sol = solve_ode(l_max_val, sigma, t_max, theta, phi)
    for t in range(0, stop=t_max, length=5)
        visualize_solution(sol, t, l_max_val, theta, phi)
    end
end
