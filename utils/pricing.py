from functools import cached_property
from typing import Literal

import numpy as np
from scipy.stats import norm


class BinomialTree:
    def __init__(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        dividend: dict[float, float] | float,
        n_steps: int,
        option_type: Literal['call', 'put'] = 'call',
        exercise_type: Literal['european', 'american'] = 'european',
        discount_factor_type: Literal['discrete', 'continuous'] = 'continuous',
    ):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.mu = r if isinstance(r, dict) else r - dividend
        self.sigma = sigma
        self.n_steps = n_steps
        self.dt = T / n_steps
        self.time_to_maturity = T
        self.dividend = dividend if isinstance(dividend, dict) else {}
        self.nodes = None
        self.option_type = option_type
        self.exercise_type = exercise_type
        self.discount_factor_type = discount_factor_type

        self.exercise_boundary = None

    def discount(self, r: float, t: float) -> float:
        if self.discount_factor_type == 'discrete':
            return 1 / (1 + r) ** t
        else:
            return np.exp(-r * t)

    @cached_property
    def u(self) -> float:
        return np.exp(self.sigma * np.sqrt(self.dt))

    @property
    def d(self) -> float:
        return 1 / self.u

    @property
    def p(self) -> float:
        return (self.discount(self.mu, -self.dt) - self.d) / (self.u - self.d)

    @property
    def q(self) -> float:
        return 1 - self.p

    def reset(self):
        self.time_to_maturity = self.T
        self.nodes = np.zeros((self.n_steps + 1, self.n_steps + 1))
        if self.exercise_type == 'american':
            self.exercise_boundary = np.full(self.n_steps, np.nan)

    def generate_nodes(self):
        S0 = self.S0
        for t, rate in self.dividend.items():
            S0 -= rate * self.discount(self.r, t)
        self.nodes[0, 0] = S0
        for i in range(1, self.n_steps + 1):
            self.nodes[0, i] = self.nodes[0, i - 1] * self.u
            for j in range(1, i + 1):
                self.nodes[j, i] = self.nodes[j - 1, i - 1] * self.d

        for t, rate in self.dividend.items():
            for i in range(self.n_steps + 1):
                if i * self.dt < t:
                    self.nodes[i, j] += rate * self.discount(self.r, t - i * self.dt)

    def payoff(self, stock_price: np.ndarray | float) -> float:
        if self.option_type == 'call':
            return np.maximum(stock_price - self.K, 0)
        else:
            return np.maximum(self.K - stock_price, 0)

    def price(self) -> float:
        self.reset()
        self.generate_nodes()
        prices = self.nodes.copy()
        prices[:, -1] = self.payoff(prices[:, -1])
        for i in range(self.n_steps - 1, -1, -1):
            for j in range(i + 1):
                price = (
                    self.discount(self.r, self.dt) * (self.p * prices[j, i + 1] + self.q * prices[j + 1, i + 1]).item()
                )
                if self.exercise_type == 'american':
                    exercise_price = self.payoff(prices[j, i])
                    if exercise_price > price:
                        price = exercise_price
                        if np.isnan(self.exercise_boundary[i]):
                            self.exercise_boundary[i] = self.nodes[j, i]
                prices[j, i] = price
        return prices[0, 0].item()


class BlackScholes:
    def __init__(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        dividend: float,
        option_type: Literal['call', 'put'] = 'call',
    ):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.dividend = dividend
        self.option_type = option_type

    @cached_property
    def d1(self) -> float:
        return (np.log(self.S0 / self.K) + (self.r - self.dividend + 0.5 * self.sigma**2) * self.T) / (
            self.sigma * np.sqrt(self.T)
        )

    @cached_property
    def d2(self) -> float:
        return self.d1 - self.sigma * np.sqrt(self.T)

    def price(self) -> float:
        if self.option_type == 'call':
            return self.S0 * np.exp(-self.dividend * self.T) * norm.cdf(self.d1) - self.K * np.exp(
                -self.r * self.T
            ) * norm.cdf(self.d2)
        else:
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2) - self.S0 * np.exp(
                -self.dividend * self.T
            ) * norm.cdf(-self.d1)
