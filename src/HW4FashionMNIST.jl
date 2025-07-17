import Pkg; Pkg.add("PrettyTables")

using Lux, Optimisers, Random, MLDatasets, Statistics, Plots, Flux, Zygote


# -------------------------Question 1 ------------------------------
# Implement a one hidden layer MLP, and vary the size of the hidden layer (10, 20, 40, 50, 100, 300) 
# and train for 10 Epochs on FashionMNIST and store the final test accuracy. Then plot
# the accuracy as a function of the hidden layer size.
# -------------------------------------------------------------------

# Set seed for reproducibility
Random.seed!(1234)

# Load FashionMNIST dataset
train_x, train_y = FashionMNIST.traindata()
test_x, test_y = FashionMNIST.testdata()

# Normalize and reshape data
function preprocess(x)
    Float32.(reshape(x, :, size(x, 3))) ./ 255.0
end

x_train = preprocess(train_x)
y_train = Flux.onehotbatch(train_y .+ 1, 1:10)

x_test = preprocess(test_x)
y_test = Flux.onehotbatch(test_y .+ 1, 1:10)

# Training parameters
learning_rate = 0.01
batch_size = 128
epochs = 10

# Accuracy function
function accuracy(model, x, y)
    ŷ = model(x)
    mean(Flux.onecold(ŷ) .== Flux.onecold(y))
end

# Training function
function train_model(hidden_size; seed=1234)
    Random.seed!(seed)
    model = Lux.Chain(
        Lux.Dense(28^2, hidden_size, relu),
        Lux.Dense(hidden_size, 10)
    )
    ps, st = Lux.setup(Random.default_rng(), model)
    opt = Optimisers.setup(Optimisers.Adam(learning_rate), ps)

    for epoch in 1:epochs
        for i in 1:batch_size:size(x_train, 2)
            last = min(i + batch_size - 1, size(x_train, 2))
            x_batch = x_train[:, i:last]
            y_batch = y_train[:, i:last]

            # Define loss function
            function loss_fun(p)
                ŷ, _ = model(x_batch, p, st)
                Flux.logitcrossentropy(ŷ, y_batch)
            end

            # Compute gradients using Zygote
            grads = Zygote.gradient(loss_fun, ps)[1]

            # Update parameters
            opt, ps = Optimisers.update(opt, ps, grads)
        end
    end

    ŷ_test, _ = model(x_test, ps, st)
    return accuracy((x) -> first(model(x, ps, st)), x_test, y_test)
end

# Run experiment
hidden_sizes = [10, 20, 40, 50, 100, 300]
accuracies = [train_model(h) for h in hidden_sizes]

# Plot results
plot(
        hidden_sizes, accuracies, xlabel="Hidden Layer Size", ylabel="Test Accuracy", 
        title="Test Accuracy vs Hidden Layer Size", lw=2, marker=:circle
    )


#---------------------------------Question 2------------------------------------------
# Use the same network with fixed hidden layer size of 30 to estimate the impact of random initialisation.
# Run the network 10 times with different weight initialisation. Compute standard
# deviation and mean. Visualize the datapoints in a plot to make the fluctuations of the final
# test accuracy visible.
#-------------------------------------------------------------------------------------

# Fixed hidden size
hidden_size_fixed = 30
n_runs = 10
seeds = rand(1:10^6, n_runs)

# Train the model with different random initializations
accuracies_random_init = [train_model(hidden_size_fixed, seed=s) for s in seeds]

# Compute statistics
mean_accuracy = mean(accuracies_random_init)
std_accuracy = std(accuracies_random_init)

println("Mean Accuracy: ", round(mean_accuracy * 100, digits=2), "%")
println("Standard Deviation: ", round(std_accuracy * 100, digits=2), "%")

# Plot results for random initializations
scatter(
    1:n_runs, accuracies_random_init,
    xlabel = "Run Index",
    ylabel = "Test Accuracy",
    title = "Test Accuracy for 10 Random Initializations (Hidden Size = 30)",
    legend = false,
    marker = :circle
)
hline!([mean_accuracy], label="Mean", linestyle=:dash)


#---------------------------------Question 3-------------------------------------------
# Train the model with a batch size of 32 for 25 epochs. Use a decaying learning rate schedule
# of your choice.
#--------------------------------------------------------------------------------------

batch_size_q3 = 32
epochs_q3 = 25

# Exponential decay function for learning rate
function learning_rate_schedule(initial_lr, epoch, decay_rate=0.9)
    return initial_lr * (decay_rate ^ (epoch - 1))
end

# Training function with learning rate schedule
function train_with_decay(hidden_size; seed=1234, initial_lr=0.01, decay_rate=0.9)
    Random.seed!(seed)
    model = Lux.Chain(
        Lux.Dense(28^2, hidden_size, relu),
        Lux.Dense(hidden_size, 10)
    )
    ps, st = Lux.setup(Random.default_rng(), model)
    opt_state = nothing

    for epoch in 1:epochs_q3
        current_lr = learning_rate_schedule(initial_lr, epoch, decay_rate)
        opt = Optimisers.setup(Optimisers.Adam(current_lr), ps)

        for i in 1:batch_size_q3:size(x_train, 2)
            last = min(i + batch_size_q3 - 1, size(x_train, 2))
            x_batch = x_train[:, i:last]
            y_batch = y_train[:, i:last]

            # Define loss function
            function loss_fun(p)
                ŷ, _ = model(x_batch, p, st)
                Flux.logitcrossentropy(ŷ, y_batch)
            end

            # Compute gradients using Zygote
            grads = Zygote.gradient(loss_fun, ps)[1]

            # Update parameters
            opt, ps = Optimisers.update(opt, ps, grads)
        end

        println("Epoch $epoch complete. Learning rate: $(round(current_lr, digits=5))")
    end

    ŷ_test, _ = model(x_test, ps, st)
    return accuracy((x) -> first(model(x, ps, st)), x_test, y_test)
end

# Run training for hidden layer size of 50
final_accuracy = train_with_decay(50)
println("Final Test Accuracy (Hidden Size = 50): ", round(final_accuracy * 100, digits=2), "%")


#----------------------------------Question 4------------------------------------------
# Optimise the batch size and the learning rate schedule via a small grid search.
#--------------------------------------------------------------------------------------

hidden_size_q4 = 50
epochs_q4 = 10

# Grid parameters
batch_sizes = [16, 32, 64, 128]
initial_lrs = [0.001, 0.005, 0.01]
decay_rates = [0.9, 0.95, 1.0]  # 1.0 means no decay

# Store results
results = []

function train_with_grid(hidden_size, batch_size, initial_lr, decay_rate; seed=1234)
    Random.seed!(seed)
    model = Lux.Chain(
        Lux.Dense(28^2, hidden_size, relu),
        Lux.Dense(hidden_size, 10)
    )
    ps, st = Lux.setup(Random.default_rng(), model)

    for epoch in 1:epochs_q4
        current_lr = initial_lr * (decay_rate ^ (epoch - 1))
        opt = Optimisers.setup(Optimisers.Adam(current_lr), ps)

        for i in 1:batch_size:size(x_train, 2)
            last = min(i + batch_size - 1, size(x_train, 2))
            x_batch = x_train[:, i:last]
            y_batch = y_train[:, i:last]

            # Loss
            function loss_fun(p)
                ŷ, _ = model(x_batch, p, st)
                Flux.logitcrossentropy(ŷ, y_batch)
            end

            # Gradient and update
            grads = Zygote.gradient(loss_fun, ps)[1]
            opt, ps = Optimisers.update(opt, ps, grads)
        end
    end

    ŷ_test, _ = model(x_test, ps, st)
    return accuracy((x) -> first(model(x, ps, st)), x_test, y_test)
end

# Run grid search
for bs in batch_sizes
    for lr in initial_lrs
        for dr in decay_rates
            acc = train_with_grid(hidden_size_q4, bs, lr, dr)
            push!(results, (batch_size=bs, initial_lr=lr, decay_rate=dr, accuracy=acc))
            println("Batch Size=$bs, LR=$lr, Decay=$dr --> Accuracy = $(round(acc*100, digits=2))%")
        end
    end
end

# Convert to DataFrame for visualization
using DataFrames
df_results = DataFrame(results)

# Display results
using Statistics
using Plots
using Printf
using PrettyTables

pretty_table(df_results)

# Optional: visualize best combinations
best = sort(df_results, :accuracy, rev=true)[1:5, :]
println("\nTop 5 Configurations:")
pretty_table(best)


#----------------------------------Question 5------------------------------------------
# Use the parameters which yield the best and train the network. Did you improve your result in 3?
#--------------------------------------------------------------------------------------

sorted_results = sort(df_results, :accuracy, rev=true)
best_config = sorted_results[1, :]  # Highest accuracy row

# Display best configuration
println("\nBest configuration from grid search:")
@printf("Batch Size = %d, Initial LR = %.4f, Decay Rate = %.2f\n", 
    best_config.batch_size, best_config.initial_lr, best_config.decay_rate)

# Use the best configuration to retrain
function train_with_best_config(hidden_size, batch_size, initial_lr, decay_rate; seed=1234)
    Random.seed!(seed)
    model = Lux.Chain(
        Lux.Dense(28^2, hidden_size, relu),
        Lux.Dense(hidden_size, 10)
    )
    ps, st = Lux.setup(Random.default_rng(), model)

    for epoch in 1:epochs_q3  # Same number of epochs as Question 3
        current_lr = initial_lr * (decay_rate ^ (epoch - 1))
        opt = Optimisers.setup(Optimisers.Adam(current_lr), ps)

        for i in 1:batch_size:size(x_train, 2)
            last = min(i + batch_size - 1, size(x_train, 2))
            x_batch = x_train[:, i:last]
            y_batch = y_train[:, i:last]

            # Loss
            function loss_fun(p)
                ŷ, _ = model(x_batch, p, st)
                Flux.logitcrossentropy(ŷ, y_batch)
            end

            grads = Zygote.gradient(loss_fun, ps)[1]
            opt, ps = Optimisers.update(opt, ps, grads)
        end

        println("Epoch $epoch complete. LR: $(round(current_lr, digits=5))")
    end

    ŷ_test, _ = model(x_test, ps, st)
    return accuracy((x) -> first(model(x, ps, st)), x_test, y_test)
end

# Retrain using best config
final_accuracy_best = train_with_best_config(
    hidden_size_q4,
    best_config.batch_size,
    best_config.initial_lr,
    best_config.decay_rate
)

println("\nFinal Test Accuracy using Best Config: $(round(final_accuracy_best * 100, digits=2))%")

# Compare with Question 3 result
println("Final Test Accuracy from Question 3 (decaying LR, bs=32): $(round(final_accuracy * 100, digits=2))%")

if final_accuracy_best > final_accuracy
    println("\nThe optimized parameters improved the test accuracy.")
else
    println("\nNo improvement over previous setup.")
end


# ------------------------------------------------------------------------------------------------