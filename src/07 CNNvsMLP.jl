using Flux, JLD2, CSV, DataFrames, Statistics, Logging
using Lux, MLUtils, Optimisers, OneHotArrays, Random, Statistics, Printf, Zygote, JLD2, Plots

# folder = "mnist"  # sub-directory in which to save
# isdir(folder) || mkdir(folder)

# Load CSV MNIST data (downloaded separately or pre-saved)
train_data = CSV.read(joinpath(folder, "mnist_train.csv"), DataFrame)
test_data = CSV.read(joinpath(folder, "mnist_test.csv"), DataFrame)

# ========== DATA LOADERS ==========

# For CNN: input shape (28,28,1,batch)
function loader_cnn(data::DataFrame; batchsize=512)
    x4dim = reshape(permutedims(Matrix{Float32}(select(data, Not(:label)))), 28, 28, 1, :)
    x4dim = mapslices(x -> reverse(permutedims(x ./ 255), dims=1), x4dim, dims=(1,2))
    yhot = Flux.onehotbatch(Vector(data.label), 0:9)
    Flux.DataLoader((x4dim, yhot); batchsize=batchsize, shuffle=true)
end

# For MLP: flattened input shape (784, batch)
function loader_mlp(data::DataFrame; batchsize=512)
    x = Matrix{Float32}(select(data, Not(:label)))' ./ 255.0   # (features, samples)
    y = Flux.onehotbatch(Vector(data.label), 0:9)
    Flux.DataLoader((x, y); batchsize=batchsize, shuffle=true)
end

# ========== MODELS ==========

# CNN: LeNet-like model
lenet = Chain(
    Conv((5, 5), 1 => 6, relu),
    MeanPool((2, 2)),
    Conv((5, 5), 6 => 16, relu),
    MeanPool((2, 2)),
    Flux.flatten,
    Dense(256 => 120, relu),
    Dense(120 => 84, relu),
    Dense(84 => 10),
)

# MLP: simple 3-layer fully connected network
# mlp = Chain(
#     Dense(28*28, 128, relu),
#     Dense(128, 64, relu),
#     Dense(64, 10),
# )
mlp = Chain(
    Flux.flatten,                   
    Dense(784 => 52, relu),        
    Dense(52 => 14, relu),        
    Dense(14 => 10)                
)


# ========== LOSS AND ACCURACY ==========

function loss_and_accuracy_cnn(model, data)
    (x, y) = only(loader_cnn(data; batchsize=size(data, 1)))
    ŷ = model(x)
    loss = Flux.logitcrossentropy(ŷ, y)
    acc = round(100 * mean(Flux.onecold(ŷ) .== Flux.onecold(y)); digits=2)
    return loss, acc
end

function loss_and_accuracy_mlp(model, data)
    (x, y) = only(loader_mlp(data; batchsize=size(data, 1)))
    ŷ = model(x)
    loss = Flux.logitcrossentropy(ŷ, y)
    acc = round(100 * mean(Flux.onecold(ŷ) .== Flux.onecold(y)); digits=2)
    return loss, acc
end

# ========== TRAINING FUNCTION ==========
function train_flux!(model, data_loader, loss_and_acc_fn; epochs=10)
    opt = ADAM(0.001)
    ps = Flux.params(model)
    start_time = time()

    for epoch in 1:epochs
        for (x, y) in data_loader
            gs = gradient(() -> Flux.logitcrossentropy(model(x), y), ps)
            Flux.Optimise.update!(opt, ps, gs)
        end
        if epoch % 2 == 1
            loss, acc = loss_and_acc_fn(model, train_data)
            test_loss, test_acc = loss_and_acc_fn(model, test_data)
            println("Epoch $epoch: Train loss = $loss, Train acc = $acc%, Test acc = $test_acc%")
        end
    end

    return round(time() - start_time, digits=2)
end

function train_lux!(model, data_loader, loss_and_acc_fn; epochs=10)
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)
    opt = Optimisers.Adam(0.001)
    state = Optimisers.setup(opt, ps)

    start_time = time()

    for epoch in 1:epochs
        for (x, y) in data_loader
            loss, back = Zygote.pullback(ps) do p
                ŷ, _ = model(x, p, st)
                Lux.logitcrossentropy(ŷ, y)
            end
            grads = back(1f0)[1]
            state, ps = Optimisers.update(state, ps, grads)
        end

        if epoch % 2 == 1
            loss, acc = loss_and_accuracy_cnn(model, ps, train_data)
            test_loss, test_acc = loss_and_accuracy_mlp(model, ps, test_data)
            println("Epoch $epoch: Train loss = $loss, Train acc = $acc%, Test acc = $test_acc%")
        end
    end

    return round(time() - start_time, digits=2)
end

# ========== RUN TRAINING ==========

println("Training CNN (LeNet) on MNIST")
train_data_loader_cnn = loader_cnn(train_data; batchsize=64)
cnn_time = train_and_time!(lenet, train_data_loader_cnn, loss_and_accuracy_cnn; epochs=10)
println("CNN training time: $(round(cnn_time, digits=2)) seconds")

println("\nTraining MLP on MNIST")
train_data_loader_mlp = loader_mlp(train_data; batchsize=64)
mlp_time = train_and_time!(mlp, train_data_loader_mlp, loss_and_accuracy_mlp; epochs=10)
println("MLP training time: $(round(mlp_time, digits=2)) seconds")

println("\nSummary:")
if cnn_time < mlp_time
    println("CNN is faster by $(round(mlp_time - cnn_time, digits=2)) seconds")
else
    println("MLP is faster by $(round(cnn_time - mlp_time, digits=2)) seconds")
end
