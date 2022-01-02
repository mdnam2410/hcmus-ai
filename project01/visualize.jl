using Luxor

function visualize_state(file_name, state, num_pred, num_prey, step)
    Drawing(610, 610, file_name)
    background("white")
    origin()
    
    pred_color = "red"
    prey_color = "blue"
    agent_point_size = 8
    
    # List of grid index
    a = [2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23]

    # List of hex cell index
    b = [13, 12, 11, 14, 4, 3, 10, 15, 5, 1, 2, 9, 16, 6, 7, 8, 17, 18, 19, 20]
    
    # Map from grid index to hex cell index
    d = Dict([(i, j) for (i, j) in zip(a, b)])
    
    # Map from hex cell index to grid index
    rd = Dict(value => key for (key, value) in d)
    
    # Map from grid index to a tuple t, where
    #  t[1]: a list of predators at this grid index
    #  t[2]: a list of prey at this grid index
    agents_to_place = Dict([(i, ([], [])) for i=1:23])
    
    preds = state[1:num_pred]
    preys = state[(num_pred+1):(num_pred+num_prey)]
    
    for (pidx, p) in enumerate(preds)
        push!(agents_to_place[rd[p]][1], pidx)
    end
    
    for (pidx, p) in enumerate(preys)
        push!(agents_to_place[rd[p]][2], pidx)
    end
    
    radius = 50
    grid = GridHex(Point(-250, -125), radius, 100)
    for i in 1:23
        # Determine if this center point belong to the map
        hex_cell_idx = get(d, i, nothing)
        if hex_cell_idx !== nothing
            sethue("black")
        else
            # Don't draw the hex tile (i.e. color it white) if this is not in the map
            sethue("white")
        end
    
        # Get the center of the grid cell
        grid_point = nextgridpoint(grid)
    
        # Draw a hexagon at this center, rotate it 45 degree
        setline(3)
        ngon(grid_point, radius - 10, 6, pi/2, :stroke)
    
        if hex_cell_idx !== nothing
            # Get a list of agents to place at this cell
            (p, q) = agents_to_place[i]
    
            # Vector of predator colors
            pc = Array{Any}(undef, length(p))
            fill!(pc, pred_color)
    
            # Vector of prey colors
            qc = Array{Any}(undef, length(q))
            fill!(qc, prey_color)
    
            # Drawing multiple agents in the same grid cell
            agents = vcat(p, q)
            agent_colors = vcat(pc, qc)
            j = 1
            if !isempty(agents)
                # Place the agents at the vertices of a transparent polygon
                sethue("white")
                poly = ngon(grid_point, 10, length(agents), 0)
                prettypoly(
                    poly,
                    :fill,
                    () ->
                    begin
                        # Draw the agents as circles with their corresponding colors
                        sethue(agent_colors[j])
                        circle(O, agent_point_size, :fill)
                        # Label the index of the agents
                        sethue("black")
                        text(string(agents[j]), O, halign=:center, valign=:bottom)
                        j += 1
                    end,
                    close = true)
            end
        end
    end
    
    sethue("black")
    fontsize(20)
    text("Step $step", Point(-260, -200))
    finish()
end
