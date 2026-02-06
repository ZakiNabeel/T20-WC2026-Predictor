"""
T20 World Cup 2026 Monte Carlo Simulator
=========================================

This module implements a Monte Carlo simulation to predict the T20 World Cup 2026 winner
by simulating the entire tournament thousands of times.

Methodology:
- Uses ELO ratings and team statistics to predict individual match outcomes
- Simulates each match using Poisson distribution for runs scored
- Accounts for venue, toss, form, and player strength
- Runs 10,000+ simulations to calculate probabilities
- Tracks each team's path through group stage, Super 8, and knockouts
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson
from typing import Dict, List, Tuple
import random
from dataclasses import dataclass
from collections import defaultdict
import json

@dataclass
class Team:
    """Represents a T20 cricket team with its statistics"""
    name: str
    group: str
    icc_ranking: int
    elo_rating: float
    avg_runs_scored: float
    avg_runs_conceded: float
    win_rate_last_10: float
    key_players_strength: float
    spin_strength: float  # Important for subcontinent conditions
    pace_strength: float
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        return self.name == other.name


class MatchPredictor:
    """Predicts individual T20 match outcomes using statistical models"""
    
    def __init__(self, home_advantage: float = 1.05):
        self.home_advantage = home_advantage
        self.toss_advantage = 1.03
        
    def calculate_win_probability(self, team_a: Team, team_b: Team, 
                                 venue: str = "neutral", 
                                 toss_winner: str = None) -> float:
        """
        Calculate win probability for team_a vs team_b
        
        Returns: probability between 0 and 1
        """
        # ELO-based probability
        elo_diff = team_a.elo_rating - team_b.elo_rating
        elo_prob = 1 / (1 + 10 ** (-elo_diff / 400))
        
        # Form-based probability
        form_diff = team_a.win_rate_last_10 - team_b.win_rate_last_10
        form_prob = 0.5 + (form_diff / 2)
        
        # Ranking-based probability
        rank_diff = team_b.icc_ranking - team_a.icc_ranking
        rank_prob = 0.5 + (rank_diff * 0.02)
        
        # Player strength
        player_diff = team_a.key_players_strength - team_b.key_players_strength
        player_prob = 0.5 + (player_diff / 2)
        
        # Weighted combination
        base_prob = (
            0.35 * elo_prob +
            0.25 * form_prob +
            0.25 * rank_prob +
            0.15 * player_prob
        )
        
        # Venue adjustment (for India/Sri Lanka)
        if venue in ["India", "Sri Lanka"]:
            if team_a.name in ["India", "Sri Lanka", "Pakistan", "Bangladesh", "Afghanistan"]:
                base_prob *= 1.08
        
        # Toss advantage
        if toss_winner == team_a.name:
            base_prob *= self.toss_advantage
        elif toss_winner == team_b.name:
            base_prob /= self.toss_advantage
            
        # Ensure probability is between 0 and 1
        return max(0.01, min(0.99, base_prob))
    
    def simulate_match(self, team_a: Team, team_b: Team, venue: str = "neutral") -> Tuple[str, int, int]:
        """
        Simulate a single T20 match using Poisson distribution
        
        Returns: (winner_name, team_a_score, team_b_score)
        """
        # Simulate toss
        toss_winner = random.choice([team_a.name, team_b.name])
        
        # Calculate expected runs
        lambda_a = team_a.avg_runs_scored * (team_b.avg_runs_conceded / 160)
        lambda_b = team_b.avg_runs_scored * (team_a.avg_runs_conceded / 160)
        
        # Adjust for venue (subcontinent favors spin)
        if venue in ["India", "Sri Lanka"]:
            if team_a.spin_strength > team_b.spin_strength:
                lambda_a *= 1.03
                lambda_b *= 0.97
            else:
                lambda_a *= 0.97
                lambda_b *= 1.03
        
        # Toss advantage (team batting first typically scores 5-8 more runs)
        batting_first_bonus = random.randint(5, 8)
        if toss_winner == team_a.name:
            lambda_a += batting_first_bonus
        else:
            lambda_b += batting_first_bonus
        
        # Generate scores using Poisson distribution
        score_a = int(poisson.rvs(lambda_a))
        score_b = int(poisson.rvs(lambda_b))
        
        # Ensure realistic T20 scores (120-200 range typically)
        score_a = max(100, min(250, score_a))
        score_b = max(100, min(250, score_b))
        
        # Handle ties (rare in T20s due to Super Overs)
        if score_a == score_b:
            # Super Over - slight edge to higher rated team
            win_prob = self.calculate_win_probability(team_a, team_b, venue, toss_winner)
            if random.random() < win_prob:
                score_a += 1
            else:
                score_b += 1
        
        winner = team_a.name if score_a > score_b else team_b.name
        return winner, score_a, score_b


class T20WorldCup2026Simulator:
    """
    Complete Monte Carlo simulator for T20 World Cup 2026
    """
    
    def __init__(self):
        self.predictor = MatchPredictor()
        self.teams = self._initialize_teams()
        self.groups = self._get_tournament_groups()
        self.results = defaultdict(lambda: {
            'group_winner': 0,
            'group_runner_up': 0,
            'super_8': 0,
            'semi_final': 0,
            'final': 0,
            'champion': 0
        })
        
    def _initialize_teams(self) -> Dict[str, Team]:
        """Initialize all 20 teams with their statistics"""
        teams_data = {
            # Group A
            "India": Team("India", "A", 1, 1300, 175, 155, 0.80, 0.95, 0.92, 0.88),
            "Pakistan": Team("Pakistan", "A", 6, 1200, 172, 158, 0.60, 0.90, 0.88, 0.85),
            "United States": Team("United States", "A", 24, 950, 148, 165, 0.45, 0.65, 0.55, 0.70),
            "Netherlands": Team("Netherlands", "A", 16, 1020, 145, 162, 0.40, 0.70, 0.60, 0.75),
            "Namibia": Team("Namibia", "A", 22, 970, 142, 168, 0.35, 0.62, 0.58, 0.72),
            
            # Group B
            "Australia": Team("Australia", "B", 2, 1280, 170, 157, 0.70, 0.92, 0.75, 0.90),
            "Sri Lanka": Team("Sri Lanka", "B", 9, 1150, 165, 160, 0.55, 0.85, 0.88, 0.78),
            "Ireland": Team("Ireland", "B", 13, 1040, 152, 162, 0.48, 0.72, 0.68, 0.75),
            "Zimbabwe": Team("Zimbabwe", "B", 17, 1000, 148, 165, 0.42, 0.68, 0.70, 0.72),
            "Oman": Team("Oman", "B", 20, 980, 140, 170, 0.38, 0.60, 0.65, 0.68),
            
            # Group C
            "England": Team("England", "C", 3, 1260, 172, 160, 0.68, 0.90, 0.72, 0.88),
            "West Indies": Team("West Indies", "C", 5, 1220, 168, 162, 0.62, 0.87, 0.70, 0.85),
            "Scotland": Team("Scotland", "C", 18, 990, 145, 168, 0.40, 0.65, 0.62, 0.70),
            "Nepal": Team("Nepal", "C", 19, 985, 142, 170, 0.38, 0.63, 0.72, 0.65),
            "Italy": Team("Italy", "C", 25, 920, 135, 175, 0.30, 0.55, 0.58, 0.62),
            
            # Group D (Group of Death!)
            "New Zealand": Team("New Zealand", "D", 4, 1240, 168, 158, 0.58, 0.88, 0.78, 0.87),
            "South Africa": Team("South Africa", "D", 7, 1210, 170, 159, 0.64, 0.89, 0.75, 0.88),
            "Afghanistan": Team("Afghanistan", "D", 8, 1180, 162, 161, 0.56, 0.83, 0.90, 0.75),
            "Canada": Team("Canada", "D", 23, 960, 138, 172, 0.36, 0.60, 0.55, 0.68),
            "United Arab Emirates": Team("United Arab Emirates", "D", 21, 975, 140, 170, 0.37, 0.62, 0.68, 0.65),
        }
        
        return teams_data
    
    def _get_tournament_groups(self) -> Dict[str, List[str]]:
        """Return the actual World Cup 2026 groups"""
        return {
            "A": ["India", "Pakistan", "United States", "Netherlands", "Namibia"],
            "B": ["Australia", "Sri Lanka", "Ireland", "Zimbabwe", "Oman"],
            "C": ["England", "West Indies", "Scotland", "Nepal", "Italy"],
            "D": ["New Zealand", "South Africa", "Afghanistan", "Canada", "United Arab Emirates"]
        }
    
    def simulate_group_stage(self) -> Dict[str, List[Tuple[str, int]]]:
        """
        Simulate group stage matches
        
        Returns: Dict mapping group to [(team_name, points)]
        """
        group_standings = {}
        
        for group_name, team_names in self.groups.items():
            teams = [self.teams[name] for name in team_names]
            points = {team.name: 0 for team in teams}
            nrr = {team.name: 0.0 for team in teams}  # Net Run Rate
            
            # Each team plays every other team once
            for i in range(len(teams)):
                for j in range(i + 1, len(teams)):
                    team_a = teams[i]
                    team_b = teams[j]
                    
                    # Determine venue
                    venue = "India" if group_name in ["A", "B"] else "Sri Lanka"
                    
                    winner, score_a, score_b = self.predictor.simulate_match(team_a, team_b, venue)
                    
                    # Award points
                    if winner == team_a.name:
                        points[team_a.name] += 2
                    else:
                        points[team_b.name] += 2
                    
                    # Calculate NRR contribution (simplified)
                    nrr[team_a.name] += (score_a - score_b) / 20
                    nrr[team_b.name] += (score_b - score_a) / 20
            
            # Sort by points, then by NRR
            standings = sorted(
                [(team, points[team], nrr[team]) for team in team_names],
                key=lambda x: (x[1], x[2]),
                reverse=True
            )
            
            group_standings[group_name] = standings
        
        return group_standings
    
    def simulate_super_8(self, qualified_teams: List[str]) -> List[str]:
        """
        Simulate Super 8 stage
        
        Returns: List of 4 semi-finalists
        """
        # Divide into two groups of 4
        # X1, Y2, X2, Y1 in one group
        # X3, Y4, X4, Y3 in the other
        
        # For simplicity, we'll create two balanced groups
        random.shuffle(qualified_teams)
        group_1 = qualified_teams[:4]
        group_2 = qualified_teams[4:]
        
        semi_finalists = []
        
        for group in [group_1, group_2]:
            points = {team: 0 for team in group}
            
            # Round robin
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    team_a = self.teams[group[i]]
                    team_b = self.teams[group[j]]
                    
                    winner, _, _ = self.predictor.simulate_match(team_a, team_b, "India")
                    points[winner] += 2
            
            # Top 2 advance
            sorted_teams = sorted(group, key=lambda x: points[x], reverse=True)
            semi_finalists.extend(sorted_teams[:2])
        
        return semi_finalists
    
    def simulate_knockout(self, semi_finalists: List[str]) -> str:
        """
        Simulate semi-finals and final
        
        Returns: Champion team name
        """
        # Semi-final 1
        winner_sf1, _, _ = self.predictor.simulate_match(
            self.teams[semi_finalists[0]],
            self.teams[semi_finalists[1]],
            "India"
        )
        
        # Semi-final 2
        winner_sf2, _, _ = self.predictor.simulate_match(
            self.teams[semi_finalists[2]],
            self.teams[semi_finalists[3]],
            "India"
        )
        
        # Final (Ahmedabad or Colombo depending on Pakistan qualification)
        final_venue = "India"  # Ahmedabad
        champion, _, _ = self.predictor.simulate_match(
            self.teams[winner_sf1],
            self.teams[winner_sf2],
            final_venue
        )
        
        return champion
    
    def run_single_simulation(self) -> Dict:
        """
        Run one complete tournament simulation
        
        Returns: Dict with results of this simulation
        """
        # Group stage
        group_standings = self.simulate_group_stage()
        
        # Extract qualified teams (top 2 from each group)
        qualified_teams = []
        for group_name in ["A", "B", "C", "D"]:
            standings = group_standings[group_name]
            qualified_teams.extend([standings[0][0], standings[1][0]])
        
        # Super 8
        semi_finalists = self.simulate_super_8(qualified_teams)
        
        # Knockouts
        champion = self.simulate_knockout(semi_finalists)
        
        return {
            'group_standings': group_standings,
            'super_8_teams': qualified_teams,
            'semi_finalists': semi_finalists,
            'champion': champion
        }
    
    def run_monte_carlo(self, n_simulations: int = 10000) -> Dict:
        """
        Run Monte Carlo simulation n times
        
        Returns: Aggregated probabilities for all teams
        """
        print(f"Running {n_simulations:,} Monte Carlo simulations...")
        print("This may take a few minutes...")
        print()
        
        for sim in range(n_simulations):
            if (sim + 1) % 1000 == 0:
                print(f"  Completed {sim + 1:,} simulations...")
            
            result = self.run_single_simulation()
            
            # Track champion
            champion = result['champion']
            self.results[champion]['champion'] += 1
            
            # Track semi-finalists
            for team in result['semi_finalists']:
                self.results[team]['semi_final'] += 1
            
            # Track Super 8 qualifiers
            for team in result['super_8_teams']:
                self.results[team]['super_8'] += 1
            
            # Track group winners and runners-up
            for group_name, standings in result['group_standings'].items():
                winner = standings[0][0]
                runner_up = standings[1][0]
                self.results[winner]['group_winner'] += 1
                self.results[runner_up]['group_runner_up'] += 1
        
        print(f"\n✓ Completed {n_simulations:,} simulations!")
        print()
        
        # Convert counts to probabilities
        probabilities = {}
        for team, counts in self.results.items():
            probabilities[team] = {
                'group_winner': (counts['group_winner'] / n_simulations) * 100,
                'group_runner_up': (counts['group_runner_up'] / n_simulations) * 100,
                'super_8': (counts['super_8'] / n_simulations) * 100,
                'semi_final': (counts['semi_final'] / n_simulations) * 100,
                'final': (counts['champion'] / n_simulations) * 100 * 2,  # Approximate
                'champion': (counts['champion'] / n_simulations) * 100
            }
        
        return probabilities


def main():
    """Run the simulation and display results"""
    print("=" * 80)
    print(" " * 20 + "T20 WORLD CUP 2026 PREDICTOR")
    print(" " * 15 + "Monte Carlo Simulation (10,000 iterations)")
    print("=" * 80)
    print()
    
    # Initialize simulator
    simulator = T20WorldCup2026Simulator()
    
    # Run Monte Carlo simulation
    probabilities = simulator.run_monte_carlo(n_simulations=10000)
    
    # Sort by championship probability
    sorted_teams = sorted(
        probabilities.items(),
        key=lambda x: x[1]['champion'],
        reverse=True
    )
    
    # Display results
    print("=" * 80)
    print(" " * 25 + "CHAMPIONSHIP PROBABILITIES")
    print("=" * 80)
    print()
    print(f"{'Rank':<6} {'Team':<25} {'Win %':<12} {'Final %':<12} {'Semi %':<12}")
    print("-" * 80)
    
    for rank, (team, probs) in enumerate(sorted_teams[:10], 1):
        print(f"{rank:<6} {team:<25} {probs['champion']:>10.2f}%  {probs['super_8']:>10.2f}%  {probs['semi_final']:>10.2f}%")
    
    print()
    print("=" * 80)
    print(" " * 25 + "GROUP STAGE PREDICTIONS")
    print("=" * 80)
    print()
    
    groups = simulator._get_tournament_groups()
    for group_name in ["A", "B", "C", "D"]:
        print(f"\nGroup {group_name}:")
        print(f"{'Team':<25} {'Qualify %':<15} {'Top Group %':<15}")
        print("-" * 60)
        
        group_teams = sorted(
            [(team, probabilities.get(team, {
                'group_winner': 0,
                'group_runner_up': 0,
                'super_8': 0,
                'semi_final': 0,
                'final': 0,
                'champion': 0
            })) for team in groups[group_name]],
            key=lambda x: x[1]['super_8'],
            reverse=True
        )
        
        for team, probs in group_teams:
            qualify_prob = probs['super_8']
            winner_prob = probs['group_winner']
            print(f"{team:<25} {qualify_prob:>13.1f}%  {winner_prob:>13.1f}%")
    
    print()
    print("=" * 80)
    print(" " * 30 + "PREDICTION SUMMARY")
    print("=" * 80)
    print()
    
    top_3 = sorted_teams[:3]
    print(f"Most Likely Champion:     {top_3[0][0]} ({top_3[0][1]['champion']:.1f}%)")
    print(f"Second Most Likely:       {top_3[1][0]} ({top_3[1][1]['champion']:.1f}%)")
    print(f"Third Most Likely:        {top_3[2][0]} ({top_3[2][1]['champion']:.1f}%)")
    print()
    
    print("Dark Horses (Long shots with >1% chance):")
    for team, probs in sorted_teams:
        if 1.0 <= probs['champion'] < 5.0:
            print(f"  • {team}: {probs['champion']:.2f}%")
    
    print()
    print("=" * 80)
    
    # Save results
    with open('/home/claude/simulation_results.json', 'w') as f:
        json.dump(probabilities, f, indent=2)
    
    print("\n✓ Results saved to: simulation_results.json")
    print("=" * 80)


if __name__ == "__main__":
    main()
