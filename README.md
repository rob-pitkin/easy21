Implementation of the Easy21 assignment from the UCL course "Reinforcement Learning" by David Silver.

To run the code, simply execute the following command:
```
# mc control
python control/monte_carlo_control.py

# sarsa lambda control
python control/sarsa_lambda_control.py

# q learning control
python control/q_learning_control.py

# linear function approximation control
python control/sarsa_lambda_control_fa.py
```

Solution to #2: Monte Carlo Control in Easy21
![Easy21MonteCarloControl](./images/easy21.png?raw=true "Easy21MonteCarloControl")

Solution to #3: Sarsa(λ) in Easy21
![Easy21SarsaLambda0](./images/lambda0.png?raw=true "Easy21SarsaLambda0")
![Easy21SarsaLambda1](./images/lambda1.png?raw=true "Easy21SarsaLambda1")
![Easy21SarsaMseLambda](./images/mseVsLambda.png?raw=true "Easy21MseVsLambda")

Extension of #3: Q-learning in Easy21
![Easy21QLearning](./images/Qlearning.png?raw=true "Easy21QLearning")

Solution to #4: Linear Function Approximation in Easy21
![Easy21LinearFunctionApproximationLambda0](./images/sarsaFA0.png?raw=true "Easy21LinearFunctionApproximationLambda0")
![Easy21LinearFunctionApproximationLambda1](./images/SarsaFA1.png?raw=true "Easy21LinearFunctionApproximationLambda1")