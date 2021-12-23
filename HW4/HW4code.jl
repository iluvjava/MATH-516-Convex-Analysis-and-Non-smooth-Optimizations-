using LinearAlgebra
import UnicodePlots
import Plots
import Logging

# ------------------------------------------------------------------------------
mutable struct LogisticRegressionModel
    X::Matrix{Float64}       # The data
    y::Vector{Float64}       # the labels
    theta::Vector{Float64}   # The parameters
    lambda::Float64          # a regularization parameters
    
    function LogisticRegressionModel(X, y, theta; lambda=1)
        this = new(X, y, theta, lambda)
        return this
    end

end

function Base.copy(this::LogisticRegressionModel)
    return LogisticRegressionModel(
        copy(this.X),
        copy(this.y), 
        copy(this.theta), 
        lambda = copy(this.lambda)
        )
end

function ErrorOf(this::LogisticRegressionModel)
    theta = this.theta
    X = this.X
    y = this.y
    lambda = this.lambda

    Error = - y.*(X*theta)  # vector 
    Error = 1 .+ exp.(Error)  # vector
    Error .= log.(Error)  # vector
    Error = sum(Error)
    Error += (lambda/2)*dot(theta, theta)
    return Error
end

function GradientOf(this::LogisticRegressionModel)
    theta = this.theta
    X = this.X
    y = this.y
    lambda = this.lambda

    Expr1 = (x) -> exp(x)/(1 + exp(x))
    Summation = zeros(size(theta))
    for Index = 1:size(X, 1)
        x = X[Index, :]
        Expr2 = - y[Index]*x  # vector
        Expr3 = dot(theta, Expr2)  # scalar
        Summation += Expr1(Expr3) * Expr2  # Vector
    end
    # ----------------------------------------------
    # Summation = exp.(-y.*X*theta)
    # Summation = Summation./(Summation .+ 1)
    # Summation = X'*(-y.*Summation)
    return Summation + lambda*theta
end

function ParametersOf(this::LogisticRegressionModel)
    return this.theta
end



# ------------------------------------------------------------------------------
mutable struct GradientDescend
    beta::Float64
    objective::LogisticRegressionModel
    objective_vals::Vector{Float64}
    
    function GradientDescend(objective::LogisticRegressionModel, beta::Float64)
        this = new(beta, objective, Vector{Float64}())
        push!(this.objective_vals, ErrorOf(this.objective))  # initial error. 
        return this
    end
end


function GradientUpdate(this::GradientDescend)
    DeltaTheta = GradientOf(this.objective)
    Params = ParametersOf(this.objective)
    Params .-= DeltaTheta * (1/this.beta)
    push!(this.objective_vals, ErrorOf(this.objective))
    
    return norm(DeltaTheta)
end

# ------------------------------------------------------------------------------
mutable struct AcceleratedGradientDescend
    beta::Float64
    objective::LogisticRegressionModel
    objective_vals::Vector{Float64}
    t::Int64                    # Iteration number. 
    a::Dict{Int64, Float64}
    x::Dict{Int64, Vector{Float64}}

    function AcceleratedGradientDescend(objective::LogisticRegressionModel, beta::Float64)
        this = new( 
            beta, 
            objective, 
            Vector{Float64}(),
            0,
            Dict{Int64, Float64}(), 
            Dict{Int64, Vector{Float64}}()
        )
        push!(this.objective_vals, ErrorOf(this.objective))  # initial error. 
        a = this.a
        x = this.x
        a[-1] = a[0] = 1
        x[-1] = x[0] = this.objective.theta
        return this
    end
end

function GradientUpdate(this::AcceleratedGradientDescend)
    t = this.t
    x = this.x
    a = this.a
    beta = this.beta
    u = x[t] + a[t]*(a[t - 1]^(-1) - 1)*(x[t] - x[t - 1])
    Gradient = GradientOf(this.objective)
    x[t + 1] = u - (1/beta)*Gradient
    this.objective.theta .= x[t + 1]
    a[t + 1] = (sqrt(a[t]^4 + 4a[t]^2) - a[t]^2)/2
    this.t += 1
    push!(this.objective_vals, ErrorOf(this.objective))
    return norm(Gradient)
end



function Run()
    function Bernoulli(p)
        if rand() < p
            return 1
        end
        return -1
    end
    m, n = 100, 50
    beta = 100.0
    theta = ones(n)
    X = randn(m, n)
    z = X*theta
    p = 1 ./ (1 .+ exp.(-z))
    theta0 = randn(n)
    y = Bernoulli.(p)
    Objective = LogisticRegressionModel(X, y, theta0)
    Optim = GradientDescend(Objective, beta)
    OptimAcc = AcceleratedGradientDescend(copy(Objective), beta)
    println("running...")
    for _ in 1:100
        println(GradientUpdate(Optim))
        GradientUpdate(OptimAcc)
    end 
    Logging.@info "Objective Values (Smooth Gradient Descend):"
    FxnVals = Optim.objective_vals
    FxnValsAcc = OptimAcc.objective_vals
    display(FxnVals)
    Logging.@info "Objective Values (Acc Gradient Descend)"
    display(FxnValsAcc)

    # Plot this out ------------------------------------------------------------
    UPlot = UnicodePlots.lineplot(1:length(FxnVals), FxnVals)
    println(UnicodePlots.lineplot!(UPlot, 1:length(FxnValsAcc), FxnValsAcc))
    Plots.plot(collect(1:length(FxnVals)), log.(FxnVals), label="Smooth Gradient")
    Plots.plot!(collect(1:length(FxnValsAcc)), log.(FxnValsAcc), label="Accelerated Gradient")
    
    Plots.savefig("objectiveVals.png")

    return Optim
end

Optim = Run()

