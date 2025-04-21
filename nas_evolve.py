import torch

# Evolution function for NeuralArchitectureSearch class
def evolve_population(self, num_generations=3):
    """Evolve the population over multiple generations"""
    # Check if we have partial results from previous run
    start_gen = len(self.generation_history)

    # Initialize best architecture if none exists
    if self.best_architecture is None and len(self.population) > 0:
        self.best_architecture = self.population[0].copy()

    for generation in range(start_gen, start_gen + num_generations):
        print(f"Generation {generation+1}/{start_gen + num_generations}")

        # Evaluate all architectures in the population
        scores = []
        for i, config in enumerate(self.population):
            print(f"Evaluating architecture {i+1}/{len(self.population)}")
            score = self.evaluate_architecture(config)
            scores.append((config, score))
            print(f"Architecture {i+1} score: {score}")

            # Update best architecture
            if score < self.best_score:
                self.best_score = score
                self.best_architecture = config.copy()
                print(f"New best architecture found with score {score}")

            # Save checkpoint after each evaluation for resilience
            self.generation_history.append({"generation": generation, "scores": scores.copy()})
            self.save_checkpoint()

        # Sort architectures by score (lower is better)
        scores.sort(key=lambda x: x[1])

        if generation < start_gen + num_generations - 1:  # Don't evolve after last generation
            # Keep top 40% (elitism)
            elite_size = max(1, len(self.population) // 5)
            new_population = [config for config, _ in scores[:elite_size]]

            # Create new architectures through mutation and crossover
            while len(new_population) < len(self.population):
                # Select parent(s)
                if torch.rand(1).item() < 0.7:  # Crossover
                    parent1 = scores[torch.randint(0, len(scores)//2 + 1, (1,)).item()][0]
                    parent2 = scores[torch.randint(0, len(scores)//2 + 1, (1,)).item()][0]

                    # Crossover
                    child = {}
                    for key in parent1:
                        if key == "target_modules":
                            child[key] = parent1[key]  # Keep this constant
                        else:
                            # Randomly select from either parent
                            child[key] = parent1[key] if torch.rand(1).item() < 0.5 else parent2[key]

                    # Small mutation chance
                    if torch.rand(1).item() < 0.3:
                        if torch.rand(1).item() < 0.33:
                            # Mutate rank
                            child["r"] = max(4, child["r"] + torch.randint(-8, 9, (1,)).item() * 4)
                            # Make rank a multiple of 4
                            child["r"] = (child["r"] // 4) * 4
                            if child["r"] < 4:
                                child["r"] = 4
                        elif torch.rand(1).item() < 0.5:
                            # Mutate alpha
                            child["lora_alpha"] = max(1, child["lora_alpha"] + torch.randint(-4, 5, (1,)).item() * 8)
                        else:
                            # Mutate dropout
                            child["lora_dropout"] = max(0, min(0.2, child["lora_dropout"] + (torch.rand(1).item() - 0.5) * 0.1))

                    new_population.append(child)
                else:  # Mutation only
                    parent = scores[torch.randint(0, len(scores)//2 + 1, (1,)).item()][0].copy()

                    # Apply mutations
                    mutation_strength = 0.5
                    if torch.rand(1).item() < mutation_strength:
                        parent["r"] = max(4, parent["r"] + torch.randint(-8, 9, (1,)).item() * 4)
                        # Make rank a multiple of 4
                        parent["r"] = (parent["r"] // 4) * 4
                        if parent["r"] < 4:
                            parent["r"] = 4

                    if torch.rand(1).item() < mutation_strength:
                        parent["lora_alpha"] = max(1, parent["lora_alpha"] + torch.randint(-4, 5, (1,)).item() * 8)

                    if torch.rand(1).item() < mutation_strength:
                        parent["lora_dropout"] = max(0, min(0.2, parent["lora_dropout"] + (torch.rand(1).item() - 0.5) * 0.1))

                    if torch.rand(1).item() < 0.2:  # 20% chance to flip RSLoRA
                        parent["use_rslora"] = not parent["use_rslora"]

                    new_population.append(parent)

            self.population = new_population
            self.save_checkpoint()

    # Ensure best architecture exists before returning
    if self.best_architecture is None and len(self.population) > 0:
        # If no proper evaluation was made, just take the first configuration
        self.best_architecture = self.population[0].copy()
        print("Warning: No valid best architecture found during evolution. Using first configuration as default.")

    # Return best architecture found
    return self.best_architecture 