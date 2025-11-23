#!/usr/bin/env python3
"""
Genetic Algorithm Optimizer - Self-Contained Module
Evolves engine weight combinations and strategies.
"""

import os
import sys
import numpy as np
import random
from typing import Dict, List, Optional, Tuple, Any
import pickle


class Chromosome:
    """Represents a strategy as a chromosome."""
    
    def __init__(self, weights: Tuple[float, float, float] = None, indicator_prefs: Dict = None):
        self.weights = weights or (0.33, 0.33, 0.34)  # (statistical, technical, ml)
        self.indicator_prefs = indicator_prefs or {}
        self.fitness = 0.0
        self.accuracy_history = []
    
    def mutate(self, mutation_rate: float = 0.1):
        """Mutate chromosome."""
        # Mutate weights
        if random.random() < mutation_rate:
            # Add small random change
            mutation = np.random.normal(0, 0.1, 3)
            new_weights = np.array(self.weights) + mutation
            # Normalize to sum to 1.0
            new_weights = np.maximum(new_weights, 0.01)  # Ensure positive
            new_weights = new_weights / np.sum(new_weights)
            self.weights = tuple(new_weights)
    
    def crossover(self, other: 'Chromosome') -> Tuple['Chromosome', 'Chromosome']:
        """Crossover with another chromosome."""
        # Uniform crossover for weights
        child1_weights = tuple(
            self.weights[i] if random.random() < 0.5 else other.weights[i]
            for i in range(3)
        )
        # Normalize
        child1_weights = tuple(np.array(child1_weights) / sum(child1_weights))
        
        child2_weights = tuple(
            other.weights[i] if random.random() < 0.5 else self.weights[i]
            for i in range(3)
        )
        child2_weights = tuple(np.array(child2_weights) / sum(child2_weights))
        
        child1 = Chromosome(child1_weights, self.indicator_prefs.copy())
        child2 = Chromosome(child2_weights, other.indicator_prefs.copy())
        
        return child1, child2
    
    def calculate_fitness(self, accuracy: float, sharpe: float = 0.0) -> float:
        """Calculate fitness based on accuracy and risk metrics."""
        self.accuracy_history.append(accuracy)
        if len(self.accuracy_history) > 100:
            self.accuracy_history = self.accuracy_history[-100:]
        
        avg_accuracy = np.mean(self.accuracy_history) if self.accuracy_history else accuracy
        fitness = avg_accuracy * 10  # Scale to 0-10
        
        # Bonus for good risk-adjusted returns
        if sharpe > 1.0:
            fitness += 2.0
        elif sharpe > 0.5:
            fitness += 1.0
        
        self.fitness = fitness
        return fitness


class GeneticOptimizer:
    """Genetic algorithm for evolving strategies."""
    
    def __init__(self, population_size: int = 50, mutation_rate: float = 0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population: List[Chromosome] = []
        self.generation = 0
        self.best_chromosome: Optional[Chromosome] = None
        self.initialize_population()
    
    def initialize_population(self):
        """Initialize random population."""
        self.population = []
        for _ in range(self.population_size):
            # Random weights that sum to 1.0
            weights = np.random.dirichlet([1, 1, 1])
            chromosome = Chromosome(tuple(weights))
            self.population.append(chromosome)
    
    def select_parents(self, tournament_size: int = 5) -> Tuple[Chromosome, Chromosome]:
        """Select parents using tournament selection."""
        def tournament_select():
            tournament = random.sample(self.population, min(tournament_size, len(self.population)))
            return max(tournament, key=lambda c: c.fitness)
        
        return tournament_select(), tournament_select()
    
    def evolve(self):
        """Evolve population for one generation."""
        # Evaluate fitness (should be done externally with actual predictions)
        # Sort by fitness
        self.population.sort(key=lambda c: c.fitness, reverse=True)
        
        # Keep best 20%
        elite_size = max(1, int(self.population_size * 0.2))
        elite = self.population[:elite_size]
        
        # Create new population
        new_population = elite.copy()
        
        # Generate offspring
        while len(new_population) < self.population_size:
            parent1, parent2 = self.select_parents()
            child1, child2 = parent1.crossover(parent2)
            child1.mutate(self.mutation_rate)
            child2.mutate(self.mutation_rate)
            new_population.extend([child1, child2])
        
        # Trim to population size
        self.population = new_population[:self.population_size]
        self.generation += 1
        
        # Update best
        if self.population:
            self.best_chromosome = self.population[0]
    
    def get_best_weights(self) -> Tuple[float, float, float]:
        """Get weights from best chromosome."""
        if self.best_chromosome:
            return self.best_chromosome.weights
        return (0.33, 0.33, 0.34)  # Default balanced
    
    def update_fitness(self, chromosome_index: int, accuracy: float, sharpe: float = 0.0):
        """Update fitness for a specific chromosome."""
        if 0 <= chromosome_index < len(self.population):
            self.population[chromosome_index].calculate_fitness(accuracy, sharpe)
    
    def save(self, filepath: str):
        """Save optimizer state."""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'population': self.population,
                    'generation': self.generation,
                    'best_chromosome': self.best_chromosome
                }, f)
        except Exception:
            pass
    
    def load(self, filepath: str):
        """Load optimizer state."""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    self.population = data.get('population', [])
                    self.generation = data.get('generation', 0)
                    self.best_chromosome = data.get('best_chromosome')
                    if not self.population:
                        self.initialize_population()
        except Exception:
            self.initialize_population()


# Global genetic optimizer instance
_genetic_optimizer = None


def get_genetic_optimizer(population_size: int = 50) -> GeneticOptimizer:
    """Get or create global genetic optimizer instance."""
    global _genetic_optimizer
    if _genetic_optimizer is None:
        _genetic_optimizer = GeneticOptimizer(population_size)
        # Try to load saved state
        optimizer_path = os.path.join(os.path.dirname(__file__), "..", "memory", "genetic_optimizer.pkl")
        _genetic_optimizer.load(optimizer_path)
    return _genetic_optimizer
