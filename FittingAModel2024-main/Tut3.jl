
println("Hello Julia")
println("Pkg loading")
import Pkg;
Pkg.add("CSV");
import Pkg;
Pkg.add("DataFrames");
import Pkg;
Pkg.add("WGLMakie");
println("Pkg loading done")

using CSV
using DataFrames
using WGLMakie


println("Import done")

alldata = CSV.read("data.csv", DataFrame);

println("Data read")
println(alldata)

f = Figure()
Axis(f[1, 1], title="Data", xlabel="X", ylabel="Y")
errorbars!(data.x, data.y, data.sigma_y)
# You can plot multiple things on the same plot using these commands ending with "!".
# The "!" is a common thing in Julia that means "this function modifies the state".
scatter!(data.x, data.y, markersize=10, color=:maroon)
f

#####################################################################

import Pkg;
Pkg.add("Optim");

using Optim

