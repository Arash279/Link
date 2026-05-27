from __future__ import annotations

import platform
import sys

import numpy as np


def main() -> None:
    print("Python:", sys.version.split()[0])
    print("Platform:", platform.platform())

    try:
        import pymc as pm
        import pytensor
        import arviz as az
    except Exception as exc:
        print("IMPORT FAILED")
        print(repr(exc))
        raise

    print("PyMC:", pm.__version__)
    print("PyTensor:", pytensor.__version__)
    print("ArviZ:", az.__version__)

    rng = np.random.default_rng(0)
    y = rng.normal(loc=1.0, scale=0.5, size=50)

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=10.0)
        sigma = pm.HalfNormal("sigma", sigma=1.0)
        pm.Normal("y", mu=mu, sigma=sigma, observed=y)

        idata = pm.sample(
            draws=200,
            tune=200,
            chains=2,
            cores=1,
            progressbar=False,
            random_seed=0,
        )

    mu_mean = float(idata.posterior["mu"].mean())
    sigma_mean = float(idata.posterior["sigma"].mean())
    print("Sampling OK")
    print(f"posterior mean mu: {mu_mean:.6f}")
    print(f"posterior mean sigma: {sigma_mean:.6f}")


if __name__ == "__main__":
    main()