""" This file contains the implementation of the ARIA algorithm according to the paper:
    https://doi.org/10.1145/3583131.3590498
"""
import torch
import numpy as np
import scipy.stats as stats

# RIM objective function

def objective(solution, target_cell, fitness, descriptor):
    """solution: a PyTorch tensor representing a neural network solution
    target_cell: a tuple of two numbers representing the target cell in the descriptor space
    fitness: a function that takes a solution and returns its fitness value
    descriptor: a function that takes a solution and returns its descriptor value
    returns: the objective value for RIM as in equation 1 of the paper"""

    # Evaluate the fitness and descriptor of the solution
    f = fitness(solution) #TODO set up fitness function with overcooked simulator
    d = descriptor(solution) #TODO set up descriptor function with overcooked simulator

    # Check if the descriptor is in the target cell
    if target_cell[0] <= d[0] < target_cell[0] + 1 and target_cell[1] <= d[1] < target_cell[1] + 1:
        # Return the fitness value
        return f
    else:
        # Return the fitness value minus the squared distance to the cell center
        cell_center = (target_cell[0] + 0.5, target_cell[1] + 0.5)
        return f - torch.norm(d - torch.tensor(cell_center))**2

# NES optimizer

def nes(solution, target_cell, fitness, descriptor, num_steps, num_samples, sigma):
    """solution: a PyTorch tensor representing a neural network solution
    target_cell: a tuple of two numbers representing the target cell in the descriptor space
    fitness: a function that takes a solution and returns its fitness value
    descriptor: a function that takes a solution and returns its descriptor value
    num_steps: the number of gradient steps to perform
    num_samples: the number of samples to draw per gradient step
    sigma: the standard deviation of the Gaussian distribution used for sampling
    returns: the optimized solution after num_steps gradient steps"""

    # Make a copy of the solution and set it to require gradient
    solution = solution.clone().detach().requires_grad_(True)

    # Create a normal distribution with zero mean and identity covariance
    normal = torch.distributions.Normal(
        torch.zeros_like(solution), torch.ones_like(solution))

    # Loop for num_steps gradient steps
    for _ in range(num_steps):
        # Sample num_samples perturbations from the normal distribution
        perturbations = normal.sample((num_samples,))

        # Add the perturbations to the solution scaled by sigma
        samples = solution + sigma * perturbations

        # Evaluate the objective function for each sample
        objectives = torch.tensor(
            [objective(sample, target_cell, fitness, descriptor) for sample in samples])

        # Rank the objectives and compute the utilities
        ranks = stats.rankdata(objectives, method='ordinal')
        utilities = torch.tensor(stats.rankdata(-ranks, method='dense'))

        # Compute the gradient estimate as the weighted average of the perturbations
        gradient_estimate = torch.mean(
            utilities[:, None] * perturbations, dim=0) / sigma

        # Perform a gradient ascent step on the solution
        solution = solution + 0.01 * gradient_estimate

        # Detach the solution from the previous computation graph
        solution = solution.detach().requires_grad_(True)

    # Return the optimized solution
    return solution

def aria(solutions, fitness, descriptor, num_steps, num_samples, sigma):
    """solutions: a list of PyTorch tensors representing neural network solutions
    fitness: a function that takes a solution and returns its fitness value
    descriptor: a function that takes a solution and returns its descriptor value
    num_steps: the number of gradient steps to perform in RIM
    num_samples: the number of samples to draw per gradient step in RIM
    sigma: the standard deviation of the Gaussian distribution used for sampling in RIM
    returns: a dictionary mapping each cell in the descriptor space to a solution"""

    # Initialize an empty archive
    archive = {}

    # Reproducibility Improvement Phase
    for solution in solutions:
        # Evaluate the solution num_samples times and compute the mean descriptor
        descriptors = torch.stack([descriptor(solution)
                                  for _ in range(num_samples)])
        mean_descriptor = torch.mean(descriptors, dim=0)

        # Find the cell containing the mean descriptor
        target_cell = (int(mean_descriptor[0]), int(mean_descriptor[1]))

        # Optimize the solution for the target cell using RIM and NES
        optimized_solution = nes(
            solution, target_cell, fitness, descriptor, num_steps, num_samples, sigma)

        # Add the optimized solution to the archive
        archive[target_cell] = optimized_solution

    # Archive Completion Phase
    # Get the list of explored and unexplored cells
    explored_cells = list(archive.keys())
    unexplored_cells = [(i, j) for i in range(32)
                        for j in range(32) if (i, j) not in explored_cells]

    # While there are unexplored cells
    while unexplored_cells:
        # Select two adjacent cells such that one is explored and the other is unexplored
        source_cell = np.random.choice(explored_cells)
        target_cell = np.random.choice([(i, j) for i in range(max(0, source_cell[0] - 1), min(31, source_cell[0] + 2))
                                       for j in range(max(0, source_cell[1] - 1), min(31, source_cell[1] + 2)) if (i, j) not in explored_cells])

        # Get the solution from the source cell
        source_solution = archive[source_cell]

        # Optimize the solution for the target cell using RIM and NES
        target_solution = nes(source_solution, target_cell,
                              fitness, descriptor, num_steps, num_samples, sigma)

        # Add the target solution to the archive
        archive[target_cell] = target_solution

        # Update the list of explored and unexplored cells
        explored_cells.append(target_cell)
        unexplored_cells.remove(target_cell)

    # Return the archive
    return archive
