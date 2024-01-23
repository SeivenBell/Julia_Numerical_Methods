using WGLMakie
# Set the default resolution to something that fits the Documenter theme
set_theme!(resolution = (80, 40))
scatter(1:4, color = 1:4)
