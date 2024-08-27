#!/usr/bin/env python3

import torch
import numpy as np
from train import DeepCraft
import json
import argparse

def load_model(model_path, num_block_types, latent_dim, embedding_dim, device):
    model = DeepCraft(num_block_types, latent_dim=latent_dim, embedding_dim=embedding_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def load_structure(builds_path, structure_pattern=None):
    builds = np.load(builds_path)['arr_0']

    if structure_pattern:
        with open('structures_to_idx.json', 'r') as f:
            structures_to_idx = json.load(f)

        matching_structures = [s for s in structures_to_idx.keys() if structure_pattern in s]

        if matching_structures:
            selected_structure = np.random.choice(matching_structures)
            return torch.from_numpy(builds[structures_to_idx[selected_structure]]).int(), selected_structure
        else:
            print(f"No structures matching '{structure_pattern}' found. Selecting a random structure.")

    return torch.from_numpy(builds[np.random.randint(0, len(builds))]).int(), ""

def get_latent_representation(model, structure, device):
    with torch.no_grad():
        structure = structure.unsqueeze(0).to(device)
        embedded = model.embedding(structure)
        embedded = embedded.permute(0, 4, 1, 2, 3)
        mu, logvar = model.encode(embedded)
        z = model.reparameterize(mu, logvar)
    return z

def generate_build(model, z, std_dev=0.5, num_samples=5):
    with torch.no_grad():
        if z.dim() == 1:
            z = z.unsqueeze(0)

        if z.size(0) < num_samples:
            z = z.repeat(num_samples, 1)

        z_perturbed = z + torch.randn_like(z) * std_dev
        generated = model.decode(z_perturbed)
        return torch.argmax(generated, dim=1).cpu().numpy()

def interpolate_latent(z1, z2, num_steps=1):
    return torch.stack([z1 + (z2 - z1) * 0.5 for t in torch.linspace(0, 1, num_steps)])

def save_build(build, output_path):
    np.save(output_path, build)
    print(f"Generated build saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate Minecraft builds using a trained DeepCraft model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--builds_path", type=str, required=True, help="Path to the builds.npz file")
    parser.add_argument("--output_path", type=str, default="generated_build.npy", help="Path to save the generated build")
    parser.add_argument("--latent_dim", type=int, default=128, help="Latent dimension of the model")
    parser.add_argument("--embedding_dim", type=int, default=32, help="Embedding dimension of the model")
    parser.add_argument("--structure1", type=str, default=None, help="Pattern to match for the first structure")
    parser.add_argument("--structure2", type=str, default=None, help="Pattern to match for the second structure")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        with open('mat_to_id_mapping.json', 'r') as f:
            num_block_types = len(json.load(f))
    except FileNotFoundError:
        num_block_types = 394

    model = load_model(args.model_path, num_block_types, args.latent_dim, args.embedding_dim, device)

    random_structure1, structure_name1 = load_structure(args.builds_path, args.structure1)
    random_structure2, structure_name2 = load_structure(args.builds_path, args.structure2)

    z1 = get_latent_representation(model, random_structure1, device)
    z2 = get_latent_representation(model, random_structure2, device)

    interpolated_z = interpolate_latent(z1, z2, 1)

    generated_builds = generate_build(model, interpolated_z, num_samples=1)

    if len(generated_builds) == 1:
        save_build(generated_builds[0], args.output_path)
    else:
        for i, build in enumerate(generated_builds):
            save_build(build, f"generated_build_{i}.npy")

    save_build(random_structure1.numpy(), "original_build_1.npy")
    save_build(random_structure2.numpy(), "original_build_2.npy")
    print("Original builds saved to original_build_1.npy and original_build_2.npy")

    if structure_name1:
        print(f"First structure: {structure_name1}")
    if structure_name2:
        print(f"Second structure: {structure_name2}")

if __name__ == "__main__":
    main()
