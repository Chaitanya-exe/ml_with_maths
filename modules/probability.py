import numpy as np
import matplotlib.pyplot as plt

class Probability:
    def __init__(self):
        pass

    @classmethod
    def likelihood(self, theta, data):
        h = np.sum(data)
        t = len(data) - h
        return (theta**h) * ((1-theta)**t)

    def bayes_theorem_visual(self):
        np.random.seed(42)
        true_theta = 0.7
        n_flips = 20
        flips = np.random.binomial(1, true_theta, size=n_flips)
        print(f"Observed flips: {flips}")

        theta_values = np.linspace(0, 1, 1001)
        prior = np.ones_like(theta_values)

        lik = Probability.likelihood(theta_values, flips)

        unnormalized_posterior = prior * lik
        posterior = unnormalized_posterior / unnormalized_posterior.sum()

        posterior_mean = np.sum(theta_values * posterior)
        posterior_map = theta_values[np.argmax(posterior)]

        print(f"Posterior mean estimate: {posterior_mean}")
        print(f"Posterior map estimate: {posterior_map}")

        plt.figure(figsize=(10, 6))
        plt.plot(theta_values, prior/prior.sum(), label='Prior (uniform)/')
        plt.plot(theta_values, lik/lik.sum(), label='Likelihood (normalized)')
        plt.plot(theta_values, posterior, label='Posterior')
        plt.axvline(true_theta, color='red', linestyle='--', label='true (theta)')
        plt.xlabel("Coin bias")
        plt.ylabel("Probablity")
        plt.title("Bayes theorem visulisation")
        plt.legend()
        plt.show()

