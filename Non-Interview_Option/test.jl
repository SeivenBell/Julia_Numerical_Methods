using LinearAlgebra

println("Starting the script...")

# Function to generate a symmetric positive definite matrix
function generate_spd_matrix(n)
    R = rand(n, n)
    return R' * R + 0.1 * I
end

# Quadratic form f
function f(A, b, x)
    return 0.5 * x' * A * x - b' * x
end

# Gradient of f
function df(A, b, x)
    return A * x - b
end

# Gradient descent method
function gradient_descent(A, b, x0; gamma=0.0001, tol=1e-5, max_iters=1e5)
    n = length(b)
    next_x = fill(x0, n)
    i = 1
    cvg = false
    println("Starting descent with gamma = $gamma")
    while i <= max_iters
        curr_x = next_x
        next_x = curr_x - gamma * df(A, b, curr_x)
        step = next_x - curr_x
        if norm(step, 2) / (norm(next_x, 2) + eps(Float64)) <= tol
            cvg = true
            break
        end
        if i == 1 || i % 100 == 0
            println("Iteration $i ; f(x)= ", f(A, b, next_x))
        end
        i += 1
    end

    if cvg
        println("Minimum found in $i iterations.")
    else
        println("No convergence within $max_iters iterations.")
    end
    return next_x
end

# Main script
n = 1000  # Size of the matrix
A = generate_spd_matrix(n)

# Ensure A is positive definite
eigenvalues = eigvals(A)
println("Minimum eigenvalue: ", minimum(eigenvalues))
if minimum(eigenvalues) > 0
    println("Matrix A is confirmed to be positive definite.")
else
    println("Matrix A might not be positive definite, adjust the matrix generation.")
    return
end

b = randn(n)  # Random vector b
x0 = 0.0  # Initial guess

println("Solving Ax = b using gradient descent...")
x_sol = gradient_descent(A, b, x0)

# Debug: Check for NaN in solution
if any(isnan.(x_sol))
    println("Solution contains NaN. Consider adjusting gamma or checking the functions' implementations.")
else
    println("Solution computed successfully.")
end

# Optional: Enable for further verification, might be computationally expensive
# println("Comparing with direct solver...")
# x_direct = A \ b
# println("Are the solutions close? ", isapprox(x_direct, x_sol))
