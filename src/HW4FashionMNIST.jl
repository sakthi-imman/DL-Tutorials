using Lux, Optimisers, Zygote, MLDatasets, OneHotArrays
using Flux: onecold, logitcrossentropy
using Statistics, Random, Plots, CUDA
using Base.Iterators: partition

DEVICE = CUDA.functional() ? gpu : cpu


# 1. Load FashionMNIST Dataset
function load_data()
    x_train, y_train = FashionMNIST.traindata(Float32)
    x_test, y_test = FashionMNIST.testdata(Float32)

    x_train = reshape(x_train, :, size(x_train, 3)) |> DEVICE
    x_test = reshape(x_test, :, size(x_test, 3)) |> DEVICE

    y_train = onehotbatch(y_train, 0:9) |> DEVICE
    y_test = onehotbatch(y_test, 0:9) |> DEVICE

    return (x_train, y_train), (x_test, y_test)
end

# 2. Model Builder
function build_model(hidden_size)
    Lux.Chain(
        Lux.FlattenLayer(),
        Lux.Dense(28^2 => hidden_size, relu),
        Lux.Dense(hidden_size => 10)
    )
end

# 3. Loss Function
function loss_fn(model, ps, st, x, y)
    ŷ, st_ = Lux.apply(model, x, ps, st)
    return mean(logitcrossentropy(ŷ, y)), st_
end

# 4. Accuracy Metric
function accuracy(model, ps, st, x, y)
    ŷ, _ = Lux.apply(model, x, ps, st)
    y_pred = onecold(ŷ, 0:9)
    y_true = onecold(y, 0:9)
    return mean(y_pred .== y_true)
end

# 5. Training Loop
function train(; h=128, bs=128, η=1e-3, epochs=10, decay=0.0, seed=42)
    rng = MersenneTwister(seed)
    model = build_model(h) |> DEVICE
    ps, st = Lux.setup(rng, model)
    opt = Optimisers.Adam(η)
    opt_state = Optimisers.setup(opt, ps)

    (x_train, y_train), (x_test, y_test) = load_data()

    for epoch in 1:epochs
        for batch in partition(1:size(x_train, 2), bs)
            xb = x_train[:, batch]
            yb = y_train[:, batch]

            loss, back = Zygote.pullback(p -> loss_fn(model, Optimisers.getdata(p), st, xb, yb)[1], ps)
            grads = back(1f0)[1]
            ps, opt_state = Optimisers.update(opt_state, ps, grads)
        end
    end

    return accuracy(model, ps, st, x_test, y_test)
end


# --------------------
# Q1. Hidden Layer Size vs Accuracy
# --------------------
function q1_hidden_layer_size()
    sizes = [10, 20, 40, 50, 100, 300]
    results = Dict()

    for h in sizes
        println("Training with hidden size $h")
        acc = train(h=h, epochs=10)
        results[h] = acc
    end

    bar(string.(keys(results)), values(results) .* 100,
        xlabel="Hidden Size", ylabel="Accuracy (%)",
        title="Q1: Hidden Layer Size vs Accuracy", legend=false)
end

# --------------------
# Q2. Batch Size vs Accuracy
# --------------------
function q2_initialisation_effect()
    hidden_size = 30
    accuracies = Float64[]

    for seed in 1:10
        println("Run $seed")
        acc = train(h=hidden_size, seed=seed)
        push!(accuracies, acc)
    end

    μ = mean(accuracies)
    σ = std(accuracies)
    println("Mean Accuracy: $(round(μ * 100, digits=2))%")
    println("Std Dev: $(round(σ * 100, digits=2))%")

    scatter(1:10, accuracies .* 100,
        xlabel="Run", ylabel="Accuracy (%)",
        title="Q2: Accuracy for 10 Random Initialisations",
        label="Accuracy", legend=true)
    hline!([μ * 100], label="Mean")
end

# --------------------
# Q3. Learning Rate vs Accuracy
# --------------------
function train_with_decay(; h=128, bs=32, η=1e-2, epochs=25, decay_factor=0.9)
    rng = MersenneTwister(42)
    model = build_model(h) |> DEVICE
    ps, st = Lux.setup(rng, model)
    opt = Optimisers.Adam(η)
    opt_state = Optimisers.setup(opt, ps)

    (x_train, y_train), (x_test, y_test) = load_data()

    for epoch in 1:epochs
        lr = η * decay_factor^epoch
        opt = Optimisers.Adam(lr)
        for batch in partition(1:size(x_train, 2), bs)
            xb = x_train[:, batch]
            yb = y_train[:, batch]

            loss, back = Zygote.pullback(p -> loss_fn(model, p, st, xb, yb)[1], ps)
            grads = back(1f0)[1]
            ps, opt_state = Optimisers.update(opt_state, ps, grads)
        end
        acc = accuracy(model, ps, st, x_test, y_test)
        println("Epoch $epoch | Accuracy: $(round(acc * 100, digits=2))% | LR: $(round(lr, sigdigits=2))")
    end
end

# --------------------
# Q4. Weight Decay vs Accuracy
# --------------------
function q4_grid_search()
    bss = [32, 64]
    lrs = [1e-3, 5e-3, 1e-2]
    results = Dict()

    for bs in bss, lr in lrs
        println("Trying BS=$bs, LR=$lr")
        acc = train(h=128, bs=bs, η=lr, epochs=10)
        results[(bs, lr)] = acc
    end

    for ((bs, lr), acc) in results
        println("BS=$bs, LR=$lr => Accuracy=$(round(acc * 100, digits=2))%")
    end
end


function q5_train_best()
    best_bs = 64
    best_lr = 1e-2

    println("Training best config: BS=$best_bs, LR=$best_lr")
    final_acc = train(h=128, bs=best_bs, η=best_lr, epochs=25)
    println("Final Accuracy with Best Params: $(round(final_acc * 100, digits=2))%")
end

# --------------------
# Run All HW4 Questions
# --------------------
function run_all_hw4()
    q1_hidden_layer_size(); gui()
    q2_initialisation_effect(); gui()
    train_with_decay();  # Q3
    q4_grid_search()     # Q4
    q5_train_best()      # Q5
end


# Run the full experiment
run_hw4_all()
