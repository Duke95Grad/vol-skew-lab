"""Black-Scholes pricing and greeks with a continuous dividend yield.

All functions are vectorised with numpy. `cp` is +1 for calls and -1 for puts.
Times are in years, vols and rates as decimals.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

_SQRT2PI = np.sqrt(2.0 * np.pi)
_TMIN = 1e-8


def npdf(x):
    return np.exp(-0.5 * np.asarray(x) ** 2) / _SQRT2PI


def _d1d2(S, K, T, r, q, sigma):
    S = np.asarray(S, float)
    K = np.asarray(K, float)
    T = np.maximum(np.asarray(T, float), _TMIN)
    sigma = np.maximum(np.asarray(sigma, float), 1e-6)
    sd = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / sd
    return d1, d1 - sd, T, sigma


def bs_price(S, K, T, r, q, sigma, cp):
    """Option price; returns intrinsic value once T <= 0."""
    cp = np.asarray(cp, float)
    T_raw = np.asarray(T, float)
    d1, d2, Tc, _ = _d1d2(S, K, T, r, q, sigma)
    px = cp * (np.asarray(S) * np.exp(-q * Tc) * ndtr(cp * d1)
               - np.asarray(K) * np.exp(-r * Tc) * ndtr(cp * d2))
    intrinsic = np.maximum(cp * (np.asarray(S, float) - np.asarray(K, float)), 0.0)
    return np.where(T_raw <= _TMIN, intrinsic, px)


def bs_greeks(S, K, T, r, q, sigma, cp):
    """Per-option greeks.

    delta: per $1 move in the underlying
    gamma: change in delta per $1 move
    vega:  price change per 1 vol point (0.01)
    theta: price change per calendar day
    """
    cp = np.asarray(cp, float)
    T_raw = np.asarray(T, float)
    S_ = np.asarray(S, float)
    K_ = np.asarray(K, float)
    d1, d2, Tc, sig = _d1d2(S, K, T, r, q, sigma)
    eq, er = np.exp(-q * Tc), np.exp(-r * Tc)
    pdf = npdf(d1)
    delta = cp * eq * ndtr(cp * d1)
    gamma = eq * pdf / (S_ * sig * np.sqrt(Tc))
    vega = S_ * eq * pdf * np.sqrt(Tc) / 100.0
    theta = (-S_ * eq * pdf * sig / (2 * np.sqrt(Tc))
             - cp * r * K_ * er * ndtr(cp * d2)
             + cp * q * S_ * eq * ndtr(cp * d1)) / 365.0
    expired = T_raw <= _TMIN
    itm = cp * (S_ - K_) > 0
    delta = np.where(expired, np.where(itm, cp, 0.0), delta)
    zero = np.zeros_like(delta)
    return {
        "delta": delta,
        "gamma": np.where(expired, zero, gamma),
        "vega": np.where(expired, zero, vega),
        "theta": np.where(expired, zero, theta),
    }
