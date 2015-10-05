---
output: pdf_document
---
This project investigates the use of machine learning techniques to estimate heterogeneous treatment effects.
For now we study the Mexican PROGRESA program but we hope to include evaluations of other randomized controlled trials as well.
Information about PROGRESA is included in this document for now but it will be split out when more RCTs are added to the repository.

# Setup

Randomized controlled trials (RCTs) are now popular in development economics.
RCTs are typically used to evaluate the impact of an intervention.
For example, in the PROGRESA program, randomized rollout was used to determine whether or not conditional cash transfers increased school enrollment.
These are population-level questions -- on average, what is the effect of an intervention?
In this project, we are interested in using RCT data to predict what the effect of an intervention will be on a particular individual.
By targeting an intervention to the individuals most likely to benefit from it, we could improve the efficacy of many programs.


## Heterogenous Treatment Effects

This framework follows [1].
Suppose we have $N$ individuals.
We are interested in the effect of an intervention $W$ on an outcome $Y$ given an individual's attributes $X$.
For a particular individual $i$, we define the following quantities:

* Covariates $X_i$ (observed)
* $W_i \in \{0, 1\}$ an indicator for whether she received the treatment (observed)
* $Y_i(W)$ the "potential outcome" for each of the two possible values of $W$. We only observe $Y_i(W_i)$. We definte $Y_i=Y_i(0) + W_i * (Y_i(1) - Y_i(0))$ to be the observed value $Y_i(W_i)$.

We are particularly interested in the quantity $t_i = Y_i(1) - Y_i(0)$.
For a targeting problem, we want to distribute the intervention to individuals for whom $t_i$ is large.

We will study the quantity $t(x) = E[t_i \mid X_i=x]$, the conditional average treatment effect (CATE).
Note that $t = E[t(x)]$ is the average treatment effect.
This is the typical quantity of interest in an RCT, the average effect of an intervention.

Our goal is to approximate the function $t(x)$.
The phrase "heterogeneous treatment effects" refers to situations in which $t(x)$ varies with $x$.

## Machine Learning

Machine learning gives us a number of methods for estimating conditional expectation functions from data.
We can think of $E[t_i \mid X_i=x]$ as a supervised machine learning problem, with the difficulty that we never actually observe $t_i$.
The typical prblem with data of this kind is the selection problem.

$$
\begin{aligned}
E[Y_i \mid X_i=x, W_i=1] - E[Y_i \mid X_i=x, W_i=0]  & = (E[Y_i(1) - Y_i(0) \mid X_i=x, W_i=1]) \\
& + (E[Y_i(1) \mid X_i=x, W_i=1] - E[Y_i(0) \mid X_i=x, w_i=0]) \\
& \neq E[Y_i(1) - Y_i(0) \mid X_i=x]
\end{aligned}
$$

If we have RCT data, $W_i$ is independent of $Y_i(1), Y_i(0)$ and the latter term is zero.
This suggests that we can estimate $t(x)$ using observed outcomes.
Athey and Imbens suggest a number of possible ways we could do this with trees.
I describe their proposals below.

### Single model

The most straightforward approach is to directly approximate the conditional expectation of the outcome given the covariates and the treatment variable.

$$
\begin{aligned}
f(x, w) & = \hat E [Y_i \mid X_i=x, W_i=w] \\
t(x) & \approx f(x, 1) - f(x, 0)
\end{aligned}
$$

### Two models

We can also create separate models for the outcome with and without treatment.

$$
\begin{aligned}
f_1(x) & = \hat E [Y_i \mid X_i=x, W_i=1] \\
f_0(x) & = \hat E [Y_i \mid X_i=x, W_i=0] \\
t(x) & \approx f_1(x) - f_0(x)
\end{aligned}
$$

The difference between a single model and two models depends on the class of functions used in the approximation.
For OLS without interaction terms, the single model approximation is constant with respect to $x$.
This is not a good model if we are assuming heterogeneous treatment effects.
For trees, the two model approximation is a restriction of the single model since the single model can always choose to first split on the treatment variable.
Since trees are not fit optimally it is possible that this could still be a useful restriction.

### Transformed outcome

Athey and Imbens make the following observation.
Let $p$ be the probability of treatment.
The arguments below hold if $p$ is a function of $X$ but for notational simplicity we assume they are independent.
If $W_i$ is independent of $Y_i(1)$ and $Y_i(0)$ (possibly only conditionally on $X$),

$$
E[Y_i * (W_i - p) / (p * (1 - p)) \mid X_i=X] = E[t_i \mid X_i=x]
$$

Proof (dropping the conditioning on X_i for notational simplicity):
$$
\begin{aligned}
E[Y_i * (W_i - p) / (p * (1 - p)] & = E[E[Y_i * (W_i - p) / (p * (1 - p)] \mid W_i] \\
& = p * E[Y_i * (W_i - p) / (p * (1 - p)) \mid W_i=1] + (1 - p)  * E[Y_i * (W_i - p) / (p * (1 - p) \mid W_i=0] \\
& = E[Y_i(1) \mid W_i=1] - E[Y_i(0) \mid W_i=0] \\
& = E[Y_i(1)] - E[Y_i(0)]
\end{aligned}
$$
Note that this relies on the independence assumption.
Define
$$
Y_i^* = Y_i * \frac{W_i - p} {p (1 - p)}
$$

The notation above hides how simple the transformation is in an RCT.
We are just multiplying targeted outcomes by $(1 - p) / p$ and untargeted outcomes by $-p / (1 - p)$.

We might expect machine learning methods are generally robust to these kinds of transformations.
On the other hand, we no longer give the algorithms access to the value of $W_i$.
Athey and Imbens nicely explain that this may be suboptimal, for example if the variance of $Y_i(1)$ and $Y_i(0)$ are zero.
We have already observed above that withholding $W_i$ from a decision tree restricts the function class under consideration.
Athey and Imbens note that this method introduces variance in a tree because the mean of the transformed outcome within a leaf is not equal to the difference in treated and untreated means within a leaf, because the fraction of treated observations within a leaf will diverge from $p$.

## Goodness of fit

The major problem with trying to predict $t(x)$ is that traditional goodness of fit measures do not apply.
Specifically we cannot evaluate the squared error
$$
\sum (Y_i(1) - Y_i(0) - t*(X_i))^2
$$

Because for every $i$, we only know one of $Y_i(1)$ or $Y_i(0)$.

Athey and Imbens propose using a goodness of fit measure based on the transformed outcome defined above 
$$
\sum (Y*_i - t*(X_i))^2
$$
The expected error is minimized when $t*(X_i) = t(X_i)$.
$$
E[(Y*_i - t*(X_i))^2] = E[(Y*_i - t(X_i))^2] + E[(t(X_i) - t*(X_i))^2]
$$

Note that the first component is the variance of $Y_i^*$
This measure is useful to us in deciding between algorithms, but it does not tell us how much we can expect to gain in real world terms.

## Welfare estimation

Bhattacharya and Dupas give a method for estimating welfare gains of a targeting strategy.
Let
$$
t_w(x) = E[Y_w \mid X=x]
$$

So
$$
t(x) = t_1(x) - t_0(x)
$$
Suppose $\hat t (x)$ is an unbiased estimator for $t(x)$.
Assume we have estimators for $t_0(x)$ and  $t_1(x)$
Consider the targeting strategy 
$$
W(x) = 1 \{ \hat t (x) > \gamma \}
$$
where $W(x)$ is the probability of assigning an individual with observed attributes $x$ to the treatment.
In practice $\gamma$ will also be estimated from the data.

This strategy has expected welfare
$$
\begin{aligned}
\rho_N = \int_{\mathcal{X}} & t_1(x) \, \mathrm{Pr} \{ \hat t (x) > \gamma \}  \\
 + & t_0(x) \, \mathrm{Pr} \{ \hat t (x) < \gamma \} \, dF(x)
\end{aligned}
$$
Where $F(x)$ is the distribution function for $x$.

Bhattacharya and Dupas show that the following is a valid estimator for $\rho_N$ 
$$
\hat \rho = \frac {1} {N} \sum_i \left [ 
    \hat t_1(X_i) - \hat t(X_i) \, \bar L \left (
        \frac {\hat \gamma  - \hat t(X_i)} {h_N}
    \right )
\right ]
$$
Where $\bar L$ is the CDF of a chosen kernel and $h_N \to 0$ as $N \to \infty$ is a bandwidth parameter.


# Data
Progresa


