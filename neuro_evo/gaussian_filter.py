"""A standalone script that add gaussian noise to the model in the model registry without using the cogment api that requires a runsession
"""
import os
import argparse
import torch

def get_latest_name(model_registry: str, model_id: str) -> str:
    """get the latest version of the model in a model registry folder

    Args:
        model_registry (str): path to the the model registry
        model_id (str): the id(name) of the model

    Returns:
        str: the latest name of the model in the format: {model_id}-v{version_number}.data
    """
    model_dir = os.path.join(model_registry, model_id)
    yaml_files = [f for f in os.listdir(model_dir) if f.endswith('.data')]

    # Extract version numbers from file names
    versions = [int(f.split('-v')[-1].split('.data')[0]) for f in yaml_files]

    # Find the file with the highest version number
    latest_version = max(versions)
    latest_version_model = f"{model_id}-v{str(latest_version).zfill(6)}.data"
    return latest_version_model

def get_model(model_id: str, model_registry: str) -> torch.nn.Module:
    """ a standalone functio that retrieve the model from the model registry without using the cogment api that requires a runsession

    args:
        model_id: the id of the model to retrieve
        model_registry: the path of the model registry

    returns:
        the model as a torch.nn.Module
    """
    # check if the model registry exists
    if not os.path.exists(model_registry):
        raise RuntimeError(f"Model registry not found at {model_registry}")
    # check if the model exists in the model registry path folder
    if not os.path.exists(os.path.join(model_registry, model_id)):
        raise RuntimeError(f"Model {model_id} not found in {model_registry}")

    model_name = get_latest_name(model_registry, model_id)
    loaded_data = torch.load(os.path.join(model_registry, model_id, model_name))
    return loaded_data

def update_model(model_id: str, model_registry: str, model: torch.nn.Module):
    """ a standalone function that update the model in the model registry without using the cogment api that requires a runsession

    args:
        model_id: the id of the model to retrieve
        model_registry: the path of the model registry
        model: the model to save
    """
    # check if the model registry exists
    if not os.path.exists(model_registry):
        raise RuntimeError(f"Model registry not found at {model_registry}")
    # check if the model exists in the model registry path folder
    if not os.path.exists(os.path.join(model_registry, model_id)):
        raise RuntimeError(f"Model {model_id} not found in {model_registry}")
    # save the model
    model_name = get_latest_name(model_registry, model_id)
    torch.save(model, os.path.join(model_registry, model_id, model_name))

def append_model(model_id: str, model_registry: str, model: torch.nn.Module):
    """ a standalone function that append the model in the model registry without using the cogment api that requires a runsession

    args:
        model_id: the id of the model to retrieve
        model_registry: the path of the model registry
        model: the model to save
    """
    # check if the model registry exists
    if not os.path.exists(model_registry):
        raise RuntimeError(f"Model registry not found at {model_registry}")
    # save the model
    model_name = get_latest_name(model_registry, model_id)
    torch.save(model, os.path.join(model_registry, model_id, f"{model_name}_mutated.data"))

def gaussian_filter(genotype_id, model_registry, mean=0., std=0.1, overwrite=False):
    """ a standalone function that add gaussian noise to the model in the model registry without using the cogment api that requires a runsession

    args:
        model_id: the id of the model to retrieve
        model_registry: the path of the model registry
        mean: the mean of the gaussian noise
        std: the standard deviation of the gaussian noise
    """
    # get model from model registry
    model = list(get_model(genotype_id, model_registry))  # convert tuple to list
    # print(model[0].values())
    weight_tuples = list(model[0].values())
    for weight in weight_tuples:
        # add noise to the weight located at the second index of the tuple
        weight += torch.randn(weight.size()) * std + mean

    # overwrite weights in model
    model[0] = dict(zip(model[0].keys(), weight_tuples))

    # update model in model registry
    if overwrite:
        update_model(genotype_id, model_registry, model)
    else:
        append_model(genotype_id, model_registry, model)

    return model

def compare_models(old_model, new_model):
    """compare two models and print the difference between them (should be different)

    Args:
        old_model (torch.nn.Module): the old model
        new_model (torch.nn.Module): the new model
    """
    for (old_key, old_param), (new_key, new_param) in zip(old_model[0].items(), new_model[0].items()):
        if old_key == new_key:
            print(f"Difference in {old_key}: {torch.sum(old_param - new_param)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--model_registry", type=str, required=True)
    parser.add_argument("--mean", type=float, default=0.)
    parser.add_argument("--std", type=float, default=0.1)
    parser.add_argument("--overwrite", type=bool, default=False)
    args = parser.parse_args()
    old_model = get_model(args.model_id, args.model_registry)
    print(type(old_model), old_model)
    new_model = gaussian_filter(args.model_id, args.model_registry,
                    args.mean, args.std, args.overwrite)
    compare_models(old_model, new_model)
    