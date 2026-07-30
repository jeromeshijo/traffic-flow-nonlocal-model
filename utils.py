import numpy as np


def riccati(rho_l, rho_r, lambda_, eta, x):
    # using lambda_ with _ because without its a python keyword

    if x == 0:
        return rho_l

    argument_of_exp = (lambda_ / eta) * rho_r * x
    return (
        (
            rho_r
            * rho_l
            * np.exp(
                argument_of_exp
            )
        )
        /
        (
            rho_l
            *
            (
                np.exp(
                    argument_of_exp
                )
                - 1
            )
            +
            rho_r
        )
    )


default_params = {
    'lambda': 1,
    'eta': 1.1,
    'L': 1.0,
    'q_max': 10.0  
}

params_for_short_street = {
    'lambda': 1,
    'eta': 1.1,
    'L': .01,
    'q_max': 10.0  
}

rho_l_in = .8
rho_r_out = 0.45
