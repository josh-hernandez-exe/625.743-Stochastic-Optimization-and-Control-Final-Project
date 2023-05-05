from typing import Callable


import numpy as np
import sympy
import sympy.stats
from matplotlib import pyplot as plt


def make_func_with_additive_noise(
    func: Callable[list[float], np.ndarray] = None,
    noise_dist: sympy.stats.JointRV = None,
) -> Callable[[list[float], np.random.RandomState], np.ndarray]:
    '''Gradient of a loss func with additive noise'''
    def func_with_additive_noise(
        theta_k: np.ndarray,
        random_state: np.random.RandomState = None,
    ) -> np.ndarray:
        e_k = np.matrix(
            sympy.stats.sample(
                noise_dist,
                seed=random_state,
            ),
        ).T

        return func(theta_k) + e_k

    return func_with_additive_noise

def basic_root_finding(
    noisy_grad_loss_func: Callable[[list[float], np.random.RandomState], np.ndarray] = None,
    a_k_func: Callable[[int], sympy.Basic] = None,
    random_state: np.random.RandomState = None,
    num_iteration: int = None,
    theta_0: np.matrix = None,
):
    theta_k_list = [theta_0,]
    theta_k_bar_list = [theta_0,]
    theta_k = theta_0

    if not callable(noisy_grad_loss_func):
        raise Exception('No Loss function given')
    
    for kk in range(num_iteration):

        y_k = noisy_grad_loss_func(theta_k, random_state)

        theta_k = (
            theta_k
            -
            a_k_func(kk) * y_k
        )
        
        theta_k_list.append(theta_k)
        theta_k_bar_list.append(np.mean(theta_k_list, axis=0))

    return theta_k_list, theta_k_bar_list


def calc_mean_errors(
    theta_data: list[tuple[list[np.ndarray],...]],
    theta_star: np.ndarray,
):
    num_iteration = min([
        len(tn_list)
        for theta_n_data_collection in theta_data
        for tn_list in theta_n_data_collection
    ])

    theta_data_error = [
        [
            [
                (
                    delta**2 
                    if isinstance(delta:=t_item-theta_star, (float,np.float64)) 
                    else np.linalg.norm(delta, ord=2)
                )
                for t_item in tn_list
            ]
            for tn_list in theta_n_data_collection
        ]
        for theta_n_data_collection in theta_data
    ]
    
    mean_theta_n_error = [
            np.mean([
                t_n_item[ii]
                for t_n_item,_ in theta_data_error
            ])
            for ii in range(num_iteration)
    ]
    theta_n_final_error = [
        t_n_item[-1]
        for t_n_item,_ in theta_data_error
    ]

    mean_theta_bar_n_error = [
            np.mean([
                t_bar_n_item[ii]
                for _,t_bar_n_item in theta_data_error
            ])
            for ii in range(num_iteration)
    ]
    theta_bar_n_final_error = [
        t_bar_n_item[-1]
        for _,t_bar_n_item in theta_data_error
    ]

    return theta_data_error, mean_theta_n_error, mean_theta_bar_n_error,theta_n_final_error,theta_bar_n_final_error

def plot_error_data(
    theta_data: list[tuple[list[np.ndarray],...]],
    theta_star: np.ndarray,
    ylim:tuple[float,float]=None,
    should_show=False,
) -> None:
    
    theta_data_error, mean_theta_n_error, mean_theta_bar_n_error,_,_ = calc_mean_errors(
        theta_data=theta_data,
        theta_star=theta_star,
    )

    fig, ax = plt.subplots()

    for theta_n_error_list, theta_bar_n_error_bar in theta_data_error:
        l1 = ax.plot(
            theta_n_error_list,
            color='C0',
            alpha=0.01,
            marker='o',
            linestyle='',
            label=r'$\hat{\theta_{n}}_{n}$',
        )
        l2 = ax.plot(
            theta_bar_n_error_bar,
            color='C1',
            alpha=0.01,
            marker='x',
            linestyle='',
            label=r'$\bar{\theta}_{n}$',
        )

    l3 = ax.plot(
        mean_theta_n_error,
        color='C2',
        label=r'mean $\hat{\theta_{n}}_{n}$',
    )
    l4 = ax.plot(
        mean_theta_bar_n_error,
        color='C3',
        label=r'mean $\bar{\theta}_{n}$',
    )

    ax.set_ylabel('Square Error')
    ax.set_xlabel('Iteration Index')
    if ylim is not None:
        ax.set_ylim(ylim)

    lns = l1+l2+l3+l4
    labs = [l.get_label() for l in lns]
    legend = ax.legend(lns, labs, loc='upper right')
    for lh in legend.legendHandles:
        lh.set_alpha(1.0)

    if should_show:
        plt.show()

    print('Mean Theta N Error     : {}'.format(mean_theta_n_error[-1]))
    print('Mean Theta Bar N Error : {}'.format(mean_theta_bar_n_error[-1]))
    print('Mean Theta N Bar Error < Mean Theta N Error: {}'.format(mean_theta_bar_n_error[-1] < mean_theta_n_error[-1]))
    
    return fig, ax


def plot_error_histogram(
    theta_data: list[tuple[list[np.ndarray],...]],
    theta_star: np.ndarray,
    ylim:tuple[float,float]=None,
    should_show=False,
) -> None:
    
    _, _, _,theta_n_final_error,theta_bar_n_final_error = calc_mean_errors(
        theta_data=theta_data,
        theta_star=theta_star,
    )
    
    fig,ax = plt.subplots()
    ax.hist(
        theta_n_final_error,
        label=r'$\hat{\theta}_{n}$ error',
        alpha=0.5,
    )
    ax.hist(
        theta_bar_n_final_error,
        label=r'$\bar{\theta}_{n}$ error',
        alpha=0.5,
    )
    ax.set_title(r'Histograms of MSE between $\hat{\theta}_{n}$ and $\bar{\theta}_{n}$')
    ax.legend()
    ax.set_xlabel('Mean Square Error')
    ax.set_ylabel('Count')

    if should_show:
        plt.show()

    return fig, ax