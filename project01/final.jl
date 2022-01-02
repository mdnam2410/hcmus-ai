using IterTools
using SparseArrays
using StatsBase
using LinearAlgebra

include("visualize.jl")

mutable struct PredatorPreyHexWorld
    G
    m::Int64
    n::Int64
    alpha::Float64
    num_vertices::Int64
    L::Int64

    prey_penalty::Float64
    actions
    S
    A

    p
    r
    R
    T
    U
    discount_factor

    """Create a new hex world
    """
    function PredatorPreyHexWorld(num_predators::Int64, num_prey::Int64, alpha::Float64; prey_penalty::Float64=1.0)
        default_graph = 
            [Dict(["E", "NE", "NW", "W", "SW", "SE"] .=> [2, 3, 4, 5, 6, 7]),
            Dict(["E", "NE", "NW", "W", "SW", "SE"] .=> [9, 10, 3, 1, 7, 8]),
            Dict(["E", "NE", "NW", "W", "SW", "SE"] .=> [10, 11, 12, 4, 1, 2]),
            Dict(["E", "NE", "NW", "W", "SW", "SE"] .=> [3, 12, 13, 14, 5, 1]),
            Dict(["E", "NE", "NW", "W", "SW", "SE"] .=> [1, 4, 14, 15, 16, 6]),
            Dict(["E", "NE", "NW", "W", "SW", "SE"] .=> [7, 1, 5, 16, 17, 18]),
            Dict(["E", "NE", "NW", "W", "SW", "SE"] .=> [8, 2, 1, 6, 18, 19]),
            Dict(["NE", "NW", "W", "SW", "SE"] .=> [9, 2, 7, 19, 20]),
            Dict(["NW", "W", "SW"] .=> [10, 2, 8]),
            Dict(["NW", "W", "SW", "SE"] .=> [11, 3, 2, 9]),
            Dict(["W", "SW", "SE"] .=> [12, 3, 10]),
            Dict(["E", "W", "SW", "SE"] .=> [11, 13, 4, 3]),
            Dict(["E", "SW", "SE"] .=> [12, 14, 4]),
            Dict(["E", "NE", "SW", "SE"] .=> [4, 13, 15, 5]),
            Dict(["E", "NE", "SE"] .=> [5, 14, 16]),
            Dict(["E", "NE", "NW", "SE"] .=> [6, 5, 15, 17]),
            Dict(["E", "NE", "NW"] .=> [18, 8, 16]),
            Dict(["E", "NE", "NW", "W"] .=> [19, 7, 6, 17]),
            Dict(["E", "NE", "NW", "W"] .=> [20, 8, 7, 18]),
            Dict(["NW", "W"] .=> [8, 19])]
            
        return new(
            default_graph,
            num_predators,
            num_prey,
            alpha,
            length(default_graph),
            num_predators + num_prey,
            prey_penalty,
            ["E", "NE", "NW", "W", "SW", "SE"],
            vec(collect(product([1:length(default_graph) for _ in 1:(num_predators + num_prey)]...))),
            vec(collect(product([["E", "NE", "NW", "W", "SW", "SE"] for _ in 1:(num_predators + num_prey)]...))),
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            0.9
        )
    end
end

function to_number(action)
    d = Dict(["E", "NE", "NW", "W", "SW", "SE"] .=> 1:6)
    m = Array{Int64}(undef, length(action))
    for (i, direction) in enumerate(action)
        m[i] = d[direction]
    end
    return m
end

function get_index(a)
    s = 0
    l = 0
    if isa(a, String)
        return Dict(["E", "NE", "NW", "W", "SW", "SE"] .=> 1:6)[a]
    elseif isa(a[1], String)
        a = to_number(a)
        l = 6
    else
        l = 20
    end
    for i = 1:length(a)
        s += (a[i] - 1) * l ^ (i - 1)
    end
    s += 1
    return s
end

"""Return available actions at loc
"""
function avail_actions(world::PredatorPreyHexWorld, loc)
    return keys(world.G[loc])
end

"""Return adjacent vertices of loc
"""
function adj(world::PredatorPreyHexWorld, loc)
    return values(world.G[loc])
end

function actions(world::PredatorPreyHexWorld, state)
    return vec(collect(product([avail_actions(world, state[i]) for i=1:length(state)]...)))
end

function result(world::PredatorPreyHexWorld, state, action)
    return [world.G[state[i]][a] for (i, a) in enumerate(action)]
end

""" Return a vector of predators currently in `location`
"""
function predators_at(world::PredatorPreyHexWorld, state, location)
    r = Vector{Int64}()
    for i=1:world.m
        if state[i] == location && !(i in r)
            push!(r, i)
        end
    end
    return r
end

""" Return a vector of prey currently in `location`
"""
function prey_at(world::PredatorPreyHexWorld, state, location)
    r = Vector{Int64}()
    for i=(world.m + 1):world.L
        if state[i] == location && !(i in r)
            push!(r, i)
        end
    end
    return r
end

"""Reward for `i` if it executes `action` under `state`
"""
function reward(world::PredatorPreyHexWorld, i, state, action)
    if !(action in actions(world, state))
        return 0
    end
    
    nstate = result(world, state, action)
    loc = nstate[i]
    predators = predators_at(world, nstate, loc)
    preys = prey_at(world, nstate, loc)

    # i is predator
    if i <= world.m
        return -1 + length(preys) / length(predators)
    # i is prey
    else
        return length(predators) == 0 ? 0 : -world.prey_penalty
    end
end

""" Obtain a random location in the hex world where no agent is currently
occupying
"""
function free_location(world::PredatorPreyHexWorld, state)
    occupied = sort(unique(state[1:world.L]))
    list = []
    for i=1:world.num_vertices
        if !(i in occupied)
            push!(list, i)
        end
    end
    return list[rand(1:length(list))]
end

"""Return the probability distribution over the actions
"""
function predator_policy(world::PredatorPreyHexWorld, i, state)
    available_actions = avail_actions(world, state[i])
    v = world.G[state[i]]
    k = 0
    n = 0

    d = Dict(world.actions .=> [0.0 for _=1:length(world.actions)])
    
    for (a, ns) in v
        count = length(prey_at(world, state, ns))
        d[a] = count
        k += count
        n += count
        if count == 0
            n += 1
        end
    end

    for a in world.actions
        if !(a in available_actions)
            continue
        end

        # No prey around
        if k == 0
            # Each available cell gets equal probability
            d[a] = 1 / n
        else
            if d[a] != 0
                d[a] = d[a] * world.alpha / k
            else
                d[a] = (1 - world.alpha) / (n - k)
            end
        end
    end
    return d
end

function prey_policy(world::PredatorPreyHexWorld, i, state)
    d = Dict(world.actions .=> [0.0 for _=1:length(world.actions)])
    available_actions = avail_actions(world, state[i])
    v = world.G[state[i]]

    k = 0
    n = 0
    for (a, ns) in v
        count = length(predators_at(world, state, ns))
        d[a] +=  count
        if count == 0
            k += 1
        else
            n += 1 / count
        end
    end

    for a in world.actions
        if !(a in available_actions)
            continue
        end

        # All cells are safe
        if n == 0
            d[a] = 1 / length(available_actions)
        else
            if d[a] == 0
                d[a] = world.alpha / k
            else
                d[a] = (1 - world.alpha) / (d[a] * n)
            end
        end
    end
    return d
end

"""Probability distribution of the actions that agent `i` will choose at
`state`

Return a `Dict{String, Float64}` `d` where:
* `d["E"]`: the probability that agent `i` will select action `E` at this state
* `d["NW"]`: the probability that agent `i` will select action `NW` at this
state
* and so on
"""
function common_policy(world::PredatorPreyHexWorld, i, state)
    if i <= world.m
        return predator_policy(world, i, state)
    else
        return prey_policy(world, i, state)
    end
end


function is_applicable(world::PredatorPreyHexWorld, next_state, state, action)
    for i=1:length(state)
        e = avail_actions(world, state[i])
        v = adj(world, state[i])

        if !(next_state[i] in v) || !(e in action)
            return false
        end
    end
    return true
end

function compute_p(world::PredatorPreyHexWorld)
    m = Array{Float64}(undef, length(world.A), length(world.S), length(world.actions), world.L)
    p(a, s, act, i) = 
        act in a ? prod([(j == i ? 1 : common_policy(world, j, s)[a[j]]) for j=1:world.L]) : 0
    for i=1:world.L
        for act in world.actions
            for a in world.A
                for s in world.S
                    m[get_index(a), get_index(s), get_index(act), i] = p(a, s, act, i)
                end
            end
        end
    end
    return m
end

function compute_r(world::PredatorPreyHexWorld)
    m = Array{Float64}(undef, length(world.A), length(world.S), world.L)
    for i in 1:world.L
        m[:, :, i] = [reward(world, i, s, a) for a in world.A, s in world.S]
    end
    return m
end

function compute_R(world::PredatorPreyHexWorld)
    r = world.r
    p = world.p

    res = Array{Float64}(undef, length(world.S), length(world.actions), world.L)
    for i=1:world.L
        for act in world.actions
            for s in world.S
                sum = 0
                for a in world.A
                    sum += p[get_index(a), get_index(s), get_index(act), i] * r[get_index(a), get_index(s), i]
                end
                res[get_index(s), get_index(act), i] = sum
            end
        end
    end
    return res
end

function compute_T(world::PredatorPreyHexWorld)
    p = world.p
    t = Array{Float64}(undef, length(world.S), length(world.S), length(world.actions), world.L)
    for i=1:world.L
        for act in world.actions
            for ns in world.S
                for s in world.S
                    sum = 0
                    guarantee = [a for a in actions(world, s) if a[i] == act]
                    for a in guarantee
                        sum += p[get_index(a), get_index(s), get_index(act), i]
                    end
                    t[get_index(ns), get_index(s), get_index(act), i] = sum
                end
            end
        end
    end
    return t
end

function compute_U(world::PredatorPreyHexWorld)
    U = Array{Float64}(undef, length(world.S), length(world.actions), world.L)

    for i=1:world.L
        for act in world.actions
            U[:, get_index(act), i] = (I - world.discount_factor * world.T[:, :, get_index(act), i]) \ world.R[:, get_index(act), i]
        end
    end
    return U
end

function compute!(world::PredatorPreyHexWorld)
    world.p = compute_p(world)
    println("Done p, $(size(world.p))")
    world.r = compute_r(world)
    println("Done r, $(size(world.r))")
    world.R = compute_R(world)
    println("Done R, $(size(world.R))")
    world.T = compute_T(world)
    println("Done T, $(size(world.T))")
    world.U = compute_U(world)
    println("Done U, $(size(world.U))")
end

function best_response(world::PredatorPreyHexWorld, state, i)
    max = -Inf
    select = []
    for a in avail_actions(world, state[i])
        u =  world.U[get_index(state), get_index(a), i]
        if u > max
            max = u
            select = [a]
        elseif u == max
            push!(select, a)
        end
    end
    return sample(select)
end

function next_action(world::PredatorPreyHexWorld, state)
    return [best_response(world, state, i) for i in 1:world.L]
end

function apply!(world::PredatorPreyHexWorld, state, action)
    new_state = result(world, state, action)

    prey_locations = state[(world.m + 1):world.L]
    for loc in prey_locations
        prey = prey_at(world, new_state, loc)

        if length(prey) != 0
            for p in prey
                new_state[p + world.m] = free_location(world, new_state)
            end
        end
    end

    return new_state
end

function run(world::PredatorPreyHexWorld, init_state, folder, times::Int64)
    compute!(world)
    state = init_state
    for i=1:times
        visualize_state("assets/$folder/step$(lpad(i, 3, "0")).png", state, world.m, world.n, i)
        action = next_action(world, state)
        next_state = apply!(world, state, action)
        state = next_state
    end
end


function test()
    num_tests = 2
    times = 30
    folders = ["test1", "test2"]
    alpha = [0.75, 1.0]
    prey_penalty = 100.0
    
    for i=1:num_tests
        println("Running test $i...")
        world = PredatorPreyHexWorld(1, 1, alpha[i]; prey_penalty=prey_penalty)
        init_state = sample(1:world.num_vertices, world.L, replace=false)
        run(world, init_state, folders[i], times)
    end
end

test()
