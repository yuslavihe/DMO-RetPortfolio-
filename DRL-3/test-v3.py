import numpy as np
import os
import torch
import time
import gymnasium as gym
from gymnasium import spaces
from scipy.special import softmax  # For normalizing actions
import matplotlib.pyplot as plt  # For plotting
import pandas as pd              # For data manipulation for distribution plots
import seaborn as sns            # For enhanced distribution plots

# Copy this entire block from your training script:
class RetirementEnv(gym.Env):
    """
    Custom Environment for Retirement Portfolio Optimization with Annuities,
    Health Shocks, and a Bequest Motive. Reduced END_AGE.
    Action: Flattened Box space (Portfolio weights + Annuity purchase signals).
    Observation: Box space (Age, Wealth, Weights, Forecasts, Annuity Ownership).
    Reward: Combination of QALYs (health-dependent), terminal/bequest wealth,
            purchase bonus, and penalties (drawdown, risk, shortfall).
    """
    metadata = {'render_modes': []}

    def __init__(self, **kwargs):
        super().__init__()

        # --- Store parameters ---
        self.initial_wealth = float(kwargs.get('initial_wealth', INITIAL_WEALTH))
        self.start_age = int(kwargs.get('start_age', START_AGE))
        # **key point**: END_AGE and TIME_HORIZON are taken from config
        self.end_age = int(kwargs.get('end_age', END_AGE))
        self.time_horizon = self.end_age - self.start_age

        self.baseline_living_expense = float(kwargs.get('baseline_living_expense', BASELINE_LIVING_EXPENSE))

        # Asset counts
        self.num_stocks = int(kwargs.get('num_stocks', NUM_STOCKS))
        self.num_commodities = int(kwargs.get('num_commodities', NUM_COMMODITIES))
        self.num_etfs = int(kwargs.get('num_etfs', NUM_ETFS))
        self.num_gov_bonds = int(kwargs.get('num_gov_bonds', NUM_GOV_BONDS))
        self.num_corp_bonds = int(kwargs.get('num_corp_bonds', NUM_CORP_BONDS))
        self.num_stable_inc = int(kwargs.get('num_stable_inc', NUM_STABLE_INC))
        self.total_assets = self.num_stocks + self.num_commodities + self.num_etfs + \
                            self.num_gov_bonds + self.num_corp_bonds + self.num_stable_inc

        # Survival params
        self.survival_a = float(kwargs.get('survival_a', SURVIVAL_A))
        self.survival_k = float(kwargs.get('survival_k', SURVIVAL_K))
        self.survival_g = float(kwargs.get('survival_g', SURVIVAL_G))
        self.survival_c = float(kwargs.get('survival_c', SURVIVAL_C))

        # Reward/Penalty params
        self.gamma_w = float(kwargs.get('gamma_w', GAMMA_W))
        # **key point**: Store bequest gamma
        self.gamma_b = float(kwargs.get('gamma_b', GAMMA_B))
        self.theta = float(kwargs.get('theta', THETA))
        self.beta_d = float(kwargs.get('beta_d', BETA_D))
        self.annuity_purchase_bonus = float(kwargs.get('annuity_purchase_bonus', ANNUITY_PURCHASE_BONUS))
        self.qaly_normal_year = float(kwargs.get('qaly_normal_year', QALY_NORMAL_YEAR))
        self.qaly_shock_untreated = float(kwargs.get('qaly_shock_untreated', QALY_SHOCK_UNTREATED))
        self.shortfall_penalty_multiplier = float(kwargs.get('shortfall_penalty_multiplier', SHORTFALL_PENALTY_MULTIPLIER))

        # Health Shock config
        self.health_shocks_config = kwargs.get('health_shocks_config', HEALTH_SHOCKS_CONFIG)
        self.health_shock_age_factor = kwargs.get('health_shock_age_factor', HEALTH_SHOCK_AGE_FACTOR)

        # Asset Indices
        self.risky_asset_indices = list(kwargs.get('risky_asset_indices', RISKY_ASSET_INDICES))
        self.critical_asset_index = int(kwargs.get('critical_asset_index', CRITICAL_ASSET_INDEX))

        # Forecast Noise
        self.forecast_return_noise_1yr = float(kwargs.get('forecast_return_noise_factor_1yr', FORECAST_RETURN_NOISE_FACTOR_1YR))
        self.forecast_return_noise_2yr = float(kwargs.get('forecast_return_noise_factor_2yr', FORECAST_RETURN_NOISE_FACTOR_2YR))
        self.forecast_mortality_noise_1yr = float(kwargs.get('forecast_mortality_noise_1yr', FORECAST_MORTALITY_NOISE_1YR))
        self.forecast_mortality_noise_2yr = float(kwargs.get('forecast_mortality_noise_2yr', FORECAST_MORTALITY_NOISE_2YR))

        # Annuities
        self.compulsory_annuity_payment = float(kwargs.get('compulsory_annuity_payment', COMPULSORY_ANNUITY_PAYMENT))
        self.opt_annuity_1_premium = float(kwargs.get('opt_annuity_1_premium', OPTIONAL_ANNUITY_1_PREMIUM))
        self.opt_annuity_1_payment = float(kwargs.get('opt_annuity_1_payment', OPTIONAL_ANNUITY_1_PAYMENT))
        self.opt_annuity_2_premium = float(kwargs.get('opt_annuity_2_premium', OPTIONAL_ANNUITY_2_PREMIUM))
        self.opt_annuity_2_payment = float(kwargs.get('opt_annuity_2_payment', OPTIONAL_ANNUITY_2_PAYMENT))

        self.verbose = int(kwargs.get('verbose', 0))
        self.np_random = None # Initialize RNG attribute

        # --- Define Action Space ---
        action_dim = self.total_assets + 2 # Portfolio weights + 2 annuity decisions
        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(action_dim,), dtype=np.float32)

        # --- Define Observation Space ---
        obs_dim = 1 + 1 + self.total_assets + self.total_assets * 2 + 2 + 2 # Age, LogWealth, Weights, 2xReturns, 2xMortality, 2xAnnuityOwned
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # --- Internal State ---
        self.current_age = self.start_age
        self.current_wealth = self.initial_wealth
        self.current_weights = np.ones(self.total_assets, dtype=np.float32) / self.total_assets
        self.current_step = 0
        self.peak_wealth_critical_asset = 0.0 # For drawdown penalty
        self.is_alive = True
        self.has_opt_annuity_1 = False
        self.has_opt_annuity_2 = False
        self.reward_components = {}
        self.current_health_shock = None # Stores dict of the shock if one occurs in step


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Ensure RNG is seeded/initialized
        if seed is not None:
             self.np_random, _ = gym.utils.seeding.np_random(seed)
        elif self.np_random is None: # Initialize if not seeded and not already existing
             self.np_random, _ = gym.utils.seeding.np_random(None)


        self.current_age = self.start_age
        self.current_wealth = self.initial_wealth
        self.current_weights = np.ones(self.total_assets, dtype=np.float32) / self.total_assets
        self.current_step = 0
        self.is_alive = True
        self.has_opt_annuity_1 = False
        self.has_opt_annuity_2 = False
        self.current_health_shock = None # Reset health shock

        # Calculate initial peak wealth for drawdown
        initial_critical_asset_value = self.current_wealth * self.current_weights[self.critical_asset_index] if self.critical_asset_index < self.total_assets else EPSILON
        self.peak_wealth_critical_asset = max(EPSILON, initial_critical_asset_value)

        # Initialize all reward components including bequest
        self.reward_components = {
            'qaly_comp': 0.0, 'dd_comp': 0.0, 'risky_comp': 0.0,
            'terminal_wealth_reward_comp': 0.0, 'shortfall_comp': 0.0,
            'annuity_bonus_comp': 0.0,
            'bequest_reward_comp': 0.0 # Added bequest component
        }

        observation = self._get_obs()
        info = self._get_info()

        # Safety check for NaN/Inf in initial observation
        if np.any(np.isnan(observation)) or np.any(np.isinf(observation)):
            print(f"!!! WARNING: NaN/Inf detected in initial observation: {observation} !!!")
            observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        # Ensure observation is numpy array
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation, dtype=np.float32)

        return observation, info

    def step(self, action):
        # Initialize step reward and components for this specific step
        step_reward = 0.0
        qaly_comp = 0.0
        dd_comp = 0.0
        risky_comp = 0.0
        shortfall_comp = 0.0
        annuity_bonus_this_step = 0.0
        terminal_wealth_reward_comp = 0.0
        bequest_reward_comp = 0.0 # Initialize bequest for the step
        healthcare_cost_this_step = 0.0 # Hs,t
        self.current_health_shock = None # Reset shock status for the step

        # --- 0. Check for Health Shock ---
        if self.is_alive: # Only check for shocks if alive at start of step
             shock_occurred, shock_details = self._generate_health_shock()
             if shock_occurred:
                 self.current_health_shock = shock_details
                 healthcare_cost_this_step = shock_details['cost']
                 if self.verbose > 0: print(f"Step {self.current_step} Age {self.current_age}: Health Shock '{shock_details['name']}' occurred! Cost: {healthcare_cost_this_step}")

        # --- 1. Portfolio Rebalancing & Annuity Purchase ---
        flat_action = action.flatten() # Ensure action is 1D
        portfolio_action = flat_action[:self.total_assets]
        buy_opt_1_signal = flat_action[self.total_assets]
        buy_opt_2_signal = flat_action[self.total_assets + 1]

        target_weights = softmax(portfolio_action).astype(np.float32)
        target_weights /= (np.sum(target_weights) + EPSILON) # Normalize
        self.current_weights = target_weights

        purchase_cost = 0.0
        if buy_opt_1_signal > 0.0 and not self.has_opt_annuity_1 and self.current_wealth >= self.opt_annuity_1_premium:
            purchase_cost += self.opt_annuity_1_premium
            self.has_opt_annuity_1 = True
            annuity_bonus_this_step += self.annuity_purchase_bonus
            if self.verbose > 1: print(f"Step {self.current_step}: Purchased Optional Annuity 1")

        if buy_opt_2_signal > 0.0 and not self.has_opt_annuity_2 and self.current_wealth - purchase_cost >= self.opt_annuity_2_premium:
            purchase_cost += self.opt_annuity_2_premium
            self.has_opt_annuity_2 = True
            annuity_bonus_this_step += self.annuity_purchase_bonus
            if self.verbose > 1: print(f"Step {self.current_step}: Purchased Optional Annuity 2")

        # Apply bonus to step reward and accumulate total bonus for the episode
        step_reward += annuity_bonus_this_step
        self.reward_components['annuity_bonus_comp'] += annuity_bonus_this_step

        # Deduct purchase cost immediately
        self.current_wealth -= purchase_cost
        self.current_wealth = max(WEALTH_FLOOR, self.current_wealth)

        # --- 2. Simulate Market Returns ---
        simulated_returns = self._simulate_market_returns()

        # --- 3. Calculate Wealth Before Expenses ---
        wealth_after_returns = self.current_wealth * np.sum(self.current_weights * (1 + simulated_returns))
        total_annuity_income = self.compulsory_annuity_payment
        if self.has_opt_annuity_1: total_annuity_income += self.opt_annuity_1_payment
        if self.has_opt_annuity_2: total_annuity_income += self.opt_annuity_2_payment
        wealth_before_expenses = wealth_after_returns + total_annuity_income

        # --- 4. Determine Expenses & Check Shortfall ---
        total_required_outflow = self.baseline_living_expense + healthcare_cost_this_step
        shortfall_amount = max(0, total_required_outflow - wealth_before_expenses)
        experienced_shortfall = shortfall_amount > EPSILON

        wealth_after_expenses = WEALTH_FLOOR # Default if shortfall
        if experienced_shortfall:
            shortfall_pct = shortfall_amount / (total_required_outflow + EPSILON)
            raw_shortfall_penalty = -self.beta_d * shortfall_pct * self.shortfall_penalty_multiplier
            shortfall_comp = np.clip(raw_shortfall_penalty, -MAX_PENALTY_MAGNITUDE_PER_STEP, 0)
            step_reward += shortfall_comp # Apply penalty
            if self.verbose > 0: print(f"Step {self.current_step}: SHORTFALL of {shortfall_amount:.2f}. Penalty: {shortfall_comp:.2f}")
            # Wealth is effectively floored due to shortfall
        else:
            # Deduct expenses normally
            wealth_after_expenses = max(WEALTH_FLOOR, wealth_before_expenses - total_required_outflow)

        # Update wealth post-expenses (or floor if shortfall)
        self.current_wealth = wealth_after_expenses


        # --- 5. Check Survival ---
        # This determines the 'is_alive' status for the *next* step and terminal conditions
        survival_prob_step = self._survival_prob(self.current_age)
        if not hasattr(self, 'np_random') or self.np_random is None: # Ensure RNG initialized
             self.np_random, _ = gym.utils.seeding.np_random(None)
        # Store if alive *before* the check for calculating this step's QALY correctly
        alive_at_start_of_survival_check = self.is_alive
        if alive_at_start_of_survival_check and (self.np_random.random() > survival_prob_step):
            self.is_alive = False # Agent dies this step
            if self.verbose > 0: print(f"Step {self.current_step} Age {self.current_age}: Agent did not survive.")

        # --- 6. Calculate Step Reward Components (QALY, Risky, Drawdown) ---
        # These depend on the state *during* the year (i.e., before potential death this step)

        if alive_at_start_of_survival_check: # Only accrue QALY/penalties if alive entering the year
            # a) QALY Reward
            if self.current_health_shock:
                # If shock occurred, QALY depends on whether treatment was affordable (no shortfall)
                qaly_comp = self.current_health_shock['qaly_outcome'] if not experienced_shortfall else self.qaly_shock_untreated
            else: # Normal year
                qaly_comp = self.qaly_normal_year
            # Apply penalty if shortfall occurred, potentially overriding shock outcome
            if experienced_shortfall:
                 qaly_comp = self.qaly_shock_untreated # Assign worst QALY on any shortfall
            step_reward += qaly_comp

            # b) Risky Asset Holding Penalty
            fraction_in_risky = np.sum(self.current_weights[self.risky_asset_indices])
            raw_risky_penalty = -self.theta * fraction_in_risky
            risky_comp = np.clip(raw_risky_penalty, -MAX_PENALTY_MAGNITUDE_PER_STEP, 0)
            step_reward += risky_comp

            # c) Drawdown Penalty (based on wealth before expenses)
            if self.critical_asset_index < self.total_assets:
                 wealth_for_dd = wealth_before_expenses # Use wealth before expenses/potential death
                 critical_asset_value = wealth_for_dd * self.current_weights[self.critical_asset_index]
                 critical_asset_value = max(EPSILON, critical_asset_value)
                 self.peak_wealth_critical_asset = max(self.peak_wealth_critical_asset, critical_asset_value)
                 drawdown_pct = (critical_asset_value - self.peak_wealth_critical_asset) / (self.peak_wealth_critical_asset + EPSILON)
                 raw_dd_penalty = self.beta_d * drawdown_pct # Note: Should be negative if drawdown occurs
                 dd_comp = np.clip(raw_dd_penalty if drawdown_pct < 0 else 0, -MAX_PENALTY_MAGNITUDE_PER_STEP, 0) # Only penalize negative drawdown
                 step_reward += dd_comp

        # Accumulate components for the episode log
        # Note: shortfall_comp already accumulated if it occurred
        self.reward_components['qaly_comp'] += qaly_comp
        self.reward_components['dd_comp'] += dd_comp
        self.reward_components['risky_comp'] += risky_comp
        self.reward_components['shortfall_comp'] += shortfall_comp # This was added earlier if shortfall occurred

        # --- 7. Update State and Check Termination ---
        self.current_step += 1
        self.current_age += 1
        # Termination check uses the 'is_alive' flag updated in step 5
        terminated = not self.is_alive or (self.current_age >= self.end_age)
        # Truncation based on time horizon (using updated TIME_HORIZON)
        truncated = self.current_step >= self.time_horizon

        # --- 8. Calculate Terminal / Bequest Reward ---
        # This happens only on the step the episode ends
        if terminated or truncated:
            current_final_wealth = max(WEALTH_FLOOR, self.current_wealth) # Use wealth at end of step
            if self.is_alive: # Reached end_age/horizon alive
                normalized_terminal_wealth = (current_final_wealth / (self.initial_wealth + EPSILON))
                raw_terminal_reward = normalized_terminal_wealth * self.gamma_w
                terminal_wealth_reward_comp = np.clip(raw_terminal_reward, 0, MAX_TERMINAL_REWARD_MAGNITUDE)
                step_reward += terminal_wealth_reward_comp # Add to this final step's reward
                self.reward_components['terminal_wealth_reward_comp'] = terminal_wealth_reward_comp
                self.reward_components['bequest_reward_comp'] = 0.0 # Ensure bequest is zero
                if self.verbose > 0: print(f"Step {self.current_step}: TERMINATED (ALIVE). Terminal Reward: {terminal_wealth_reward_comp:.2f}")
            # Check if died THIS step or was already dead before the final step check
            elif not self.is_alive:
                 # Calculate bequest based on wealth at end of step where death occurred
                 normalized_bequest_wealth = (current_final_wealth / (self.initial_wealth + EPSILON))
                 raw_bequest_reward = normalized_bequest_wealth * self.gamma_b # Use bequest gamma
                 bequest_reward_comp = np.clip(raw_bequest_reward, 0, MAX_BEQUEST_REWARD_MAGNITUDE) # Use separate clip
                 step_reward += bequest_reward_comp # Add to this final step's reward
                 self.reward_components['bequest_reward_comp'] = bequest_reward_comp
                 self.reward_components['terminal_wealth_reward_comp'] = 0.0 # Ensure terminal is zero
                 if self.verbose > 0: print(f"Step {self.current_step}: TERMINATED (DEAD). Bequest Reward: {bequest_reward_comp:.2f}")
            # Note: If truncated but dead, the bequest reward logic above still applies


        # --- 9. Get Observation and Info ---
        observation = self._get_obs()
        info = self._get_info(terminated or truncated) # Pass terminal flag
        info["healthcare_cost_in_step"] = healthcare_cost_this_step # Add cost to info

        # Safety checks for NaN/Inf in reward and observation
        if np.any(np.isnan(observation)) or np.any(np.isinf(observation)):
            print(f"!!! WARNING: NaN/Inf detected in observation: {observation} at step {self.current_step} !!!")
            observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            step_reward = -MAX_PENALTY_MAGNITUDE_PER_STEP * 2 # Heavy penalty

        if np.isnan(step_reward) or np.isinf(step_reward):
            print(f"!!! WARNING: NaN/Inf detected in reward: {step_reward} at step {self.current_step} !!!")
            step_reward = -MAX_PENALTY_MAGNITUDE_PER_STEP * 2 # Heavy penalty

        step_reward = float(step_reward) # Ensure reward is standard float
        # Ensure observation is numpy array
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation, dtype=np.float32)


        # Return state, reward, terminated, truncated, info (Gymnasium >= 0.26 format)
        return observation, step_reward, terminated, truncated, info

    def _generate_health_shock(self):
        """Generates a health shock based on configured probabilities and age factor."""
        if not hasattr(self, 'np_random') or self.np_random is None:
            self.np_random, _ = gym.utils.seeding.np_random(None)
        # Calculate age factor using the instance's end_age
        age_multiplier = self.health_shock_age_factor(self.current_age)
        rand_val = self.np_random.random()
        cumulative_prob = 0.0
        for shock in self.health_shocks_config:
            adjusted_prob = min(1.0, shock['base_prob'] * age_multiplier) # Cap prob at 1.0
            cumulative_prob += adjusted_prob
            if rand_val < cumulative_prob:
                return True, shock # Return the shock details
        return False, None # No shock occurred

    def _get_obs(self):
        """Construct the observation array including annuity ownership."""
        # Normalize age using the instance's end_age
        norm_age = (self.current_age - self.start_age) / (self.end_age - self.start_age) if (self.end_age - self.start_age) > 0 else 0.0
        log_wealth = np.log(max(WEALTH_FLOOR, self.current_wealth))

        # Forecasts
        return_forecast_1yr = self._get_noisy_forecast(base_value=0.05, noise_factor=self.forecast_return_noise_1yr, size=self.total_assets)
        return_forecast_2yr = self._get_noisy_forecast(base_value=0.05, noise_factor=self.forecast_return_noise_2yr, size=self.total_assets)
        mortality_forecast_1yr = self._get_noisy_forecast(base_value=self._survival_prob(self.current_age + 1), noise_factor=self.forecast_mortality_noise_1yr, size=1)
        mortality_forecast_2yr = self._get_noisy_forecast(base_value=self._survival_prob(self.current_age + 2), noise_factor=self.forecast_mortality_noise_2yr, size=1)

        # Annuity ownership flags
        opt_annuity_1_owned_flag = 1.0 if self.has_opt_annuity_1 else 0.0
        opt_annuity_2_owned_flag = 1.0 if self.has_opt_annuity_2 else 0.0

        obs = np.concatenate([
            np.array([norm_age], dtype=np.float32),
            np.array([log_wealth], dtype=np.float32),
            self.current_weights.astype(np.float32),
            return_forecast_1yr.astype(np.float32),
            return_forecast_2yr.astype(np.float32),
            mortality_forecast_1yr.astype(np.float32),
            mortality_forecast_2yr.astype(np.float32),
            np.array([opt_annuity_1_owned_flag], dtype=np.float32),
            np.array([opt_annuity_2_owned_flag], dtype=np.float32)
        ]).flatten()
        return obs.astype(np.float32)

    def _get_info(self, is_terminal=False):
        """Return supplementary info dictionary including accumulated rewards and health shock."""
        info = {
            "age": self.current_age,
            "wealth": self.current_wealth,
            "weights": self.current_weights.astype(np.float32),
            "is_alive": self.is_alive,
            "has_opt_annuity_1": self.has_opt_annuity_1,
            "has_opt_annuity_2": self.has_opt_annuity_2,
            # Include info about shock in the *last* step before reset/termination
            "health_shock_in_step": self.current_health_shock['name'] if self.current_health_shock else "None",
            # Note: healthcare_cost_in_step is added dynamically in step() return
        }
        if is_terminal:
            # Populate final components using .get() for safety
            # Ensure these keys match exactly what's stored in self.reward_components
            info["final_qaly_comp"] = float(self.reward_components.get('qaly_comp', 0.0))
            info["final_dd_comp"] = float(self.reward_components.get('dd_comp', 0.0))
            info["final_risky_comp"] = float(self.reward_components.get('risky_comp', 0.0))
            info["final_terminal_wealth_reward_comp"] = float(self.reward_components.get('terminal_wealth_reward_comp', 0.0))
            info["final_shortfall_comp"] = float(self.reward_components.get('shortfall_comp', 0.0))
            info["final_annuity_bonus_comp"] = float(self.reward_components.get('annuity_bonus_comp', 0.0))
            # Add final bequest component
            info["final_bequest_reward_comp"] = float(self.reward_components.get('bequest_reward_comp', 0.0))
            info["final_wealth"] = float(self.current_wealth) # Capture final wealth

            # Calculate final total reward from components for verification
            total_comp_reward = sum(v for v in self.reward_components.values() if isinstance(v, (int, float)))
            info["final_total_reward_from_components"] = float(total_comp_reward)

        return info

    def _survival_prob(self, age):
        """Calculate survival probability using Gompertz law."""
        if age < 0: return 1.0
        # Gompertz hazard rate: lambda(t) = a * exp(g * t) + c
        hazard_rate = self.survival_a * np.exp(self.survival_g * age) + self.survival_c
        # Survival probability over one year: S(t+1)/S(t) = exp(-hazard_rate(t)) approx
        prob = np.exp(-hazard_rate)
        return np.clip(prob, 0.0, 1.0)

    def _get_noisy_forecast(self, base_value, noise_factor, size):
        """Generate a noisy forecast around a base value."""
        if not hasattr(self, 'np_random') or self.np_random is None:
             self.np_random, _ = gym.utils.seeding.np_random(None) # Ensure RNG is initialized
        noise = self.np_random.normal(loc=0.0, scale=np.abs(base_value * noise_factor + EPSILON), size=size)
        forecast = base_value + noise
        return forecast

    def _simulate_market_returns(self):
        """Placeholder simulation logic - *only* returns asset returns."""
        # Example placeholder - Replace with your actual financial market simulation
        mean_returns = np.array([0.08, 0.06, 0.07, 0.02, 0.035, 0.01] * int(np.ceil(self.total_assets/6)))[:self.total_assets]
        volatilities = np.array([0.15, 0.18, 0.12, 0.03, 0.05, 0.005] * int(np.ceil(self.total_assets/6)))[:self.total_assets]

        if not hasattr(self, 'np_random') or self.np_random is None:
             self.np_random, _ = gym.utils.seeding.np_random(None) # Ensure RNG is initialized

        asset_returns = self.np_random.normal(loc=mean_returns, scale=volatilities).astype(np.float32)
        return asset_returns

    def close(self):
        """Perform any necessary cleanup."""
        pass # No specific cleanup needed in this version
# --- End of RetirementEnv Class ---

# Copy all these constant definitions from your training script:

# --- Environment Asset Structure ---
NUM_STOCKS = 25
NUM_COMMODITIES = 7
NUM_ETFS = 3
NUM_GOV_BONDS = 5
NUM_CORP_BONDS = 4
NUM_STABLE_INC = 3
TOTAL_ASSETS = NUM_STOCKS + NUM_COMMODITIES + NUM_ETFS + NUM_GOV_BONDS + NUM_CORP_BONDS + NUM_STABLE_INC

# --- Initial State & Environment Parameters ---
INITIAL_WEALTH = 1_500_000
START_AGE = 65
# **key point**: Reduce END_AGE for more plausible survival to term
END_AGE = 100 # Reduced from 120
TIME_HORIZON = END_AGE - START_AGE # T = 35 steps (Updated)

BASELINE_LIVING_EXPENSE = 40000.0 # Example: $40k/year non-healthcare needs

# --- Survival Probability Parameters ---
SURVIVAL_A = 0.00005
SURVIVAL_K = 5.6
SURVIVAL_G = 0.055
SURVIVAL_C = 0.0001

# --- Reward/Penalty Coefficients ---
GAMMA_W = 6.0    # Terminal Wealth Reward Scaling (if alive at END_AGE)
# **key point**: Add Bequest Reward Scaling (if dead before END_AGE)
GAMMA_B = 3.0    # Bequest Wealth Reward Scaling (e.g., half of GAMMA_W)

THETA = 0.05     # Risky Asset Holding Penalty
BETA_D = 0.5     # Drawdown & Shortfall Base Penalty
QALY_NORMAL_YEAR = 0.7 # QALY value in a year with no adverse health event
QALY_SHOCK_UNTREATED = -0.5 # QALY value if shock occurs and shortfall prevents treatment
SHORTFALL_PENALTY_MULTIPLIER = 3.0 # Multiplier for shortfall penalty based on BETA_D

ANNUITY_PURCHASE_BONUS = 5.0 # Small reward for buying an optional annuity

# --- Health Shock Configuration ---
HEALTH_SHOCKS_CONFIG = [
    {"name": "Minor", "cost": 5000,  "qaly_outcome": 0.5, "base_prob": 0.10}, # e.g., treatable infection
    {"name": "Moderate", "cost": 25000, "qaly_outcome": 0.2, "base_prob": 0.05}, # e.g., non-major surgery, chronic management
    {"name": "Major", "cost": 80000, "qaly_outcome": 0.0, "base_prob": 0.02}, # e.g., major surgery, serious illness
]
# **key point**: Probability multiplier based on age (simple linear increase example)
# Note: END_AGE change affects this slope slightly
HEALTH_SHOCK_AGE_FACTOR = lambda age: 1.0 + max(0, (age - START_AGE) / (END_AGE - START_AGE)) * 1.5 if (END_AGE - START_AGE) > 0 else 1.0

# --- Numerical Stability/Clipping ---
MAX_PENALTY_MAGNITUDE_PER_STEP = 100.0 # Used in Env step logic
MAX_TERMINAL_REWARD_MAGNITUDE = 1000.0 # Used in Env step logic
MAX_BEQUEST_REWARD_MAGNITUDE = MAX_TERMINAL_REWARD_MAGNITUDE / 2 # Used in Env step logic
WEALTH_FLOOR = 1.0
EPSILON = 1e-8

# --- Asset Indices Definitions ---
RISKY_ASSET_INDICES = list(range(NUM_STOCKS + NUM_COMMODITIES + NUM_ETFS))
CRITICAL_ASSET_INDEX = NUM_STOCKS + NUM_COMMODITIES # Index for drawdown penalty calculation

# --- Forecast Noise Parameters ---
FORECAST_RETURN_NOISE_FACTOR_1YR = 0.3
FORECAST_RETURN_NOISE_FACTOR_2YR = 0.6
FORECAST_MORTALITY_NOISE_1YR = 0.01
FORECAST_MORTALITY_NOISE_2YR = 0.03

# --- ANNUITY PARAMETERS ---
COMPULSORY_ANNUITY_PAYMENT = 10000.0
OPTIONAL_ANNUITY_1_PREMIUM = 150000.0
OPTIONAL_ANNUITY_1_PAYMENT = 7500.0
OPTIONAL_ANNUITY_2_PREMIUM = 225000.0
OPTIONAL_ANNUITY_2_PAYMENT = 12375.0
# --- End of Constants ---


# === Stable Baselines and Monitor Imports ===
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# === Evaluation Configuration ===
N_EVAL_SIMULATIONS = 10000  # Number of episodes to simulate for metrics (Monte Carlo)
CVAR_PERCENTILE = 0.05  # For Conditional Value at Risk (worst 5%)
STRESS_TEST_FLOOR_PCT = 0.70  # Wealth floor relative to initial wealth
OOP_BURDEN_TARGET = 0.15  # Target Out-of-Pocket Burden Ratio (<=15%)
SURPLUS_COVERAGE_TARGET = 1.2  # Target Surplus Coverage Ratio (>=1.2x)
ANNUITY_ADEQUACY_TARGET_AGE = 85  # Age threshold for annuity adequacy
ANNUITY_ADEQUACY_TARGET_RATIO = 0.80  # Target ratio (>=80%)
PLOT_SAVE_DIR = "evaluation_plots"  # Directory to save plots

# Specify the path where the final model and stats are saved
# !!! IMPORTANT: Set this path correctly based on your training output !!!
# Example: MODEL_SAVE_DIR = r"./models/1234567890_MyModelName"
MODEL_SAVE_DIR = r"C:\Users\jeanna\PycharmProjects\DMO RetPort\DRL-3\models\1743853292_Full_Embedded_HealthShocks_Bequest_V4_Continued" #<--- UPDATE THIS PATH
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "final_model.zip")
STATS_PATH = os.path.join(MODEL_SAVE_DIR, "final_vec_normalize.pkl")


# --- Helper: Get Environment Kwargs ---
def get_env_kwargs():
    """Returns the dictionary of kwargs used to initialize RetirementEnv."""
    # Ensure ALL parameters needed by RetirementEnv.__init__ are here
    # and match the constants defined above (which should match training).
    env_kwargs = {
        'initial_wealth': INITIAL_WEALTH, 'start_age': START_AGE, 'end_age': END_AGE,
        'baseline_living_expense': BASELINE_LIVING_EXPENSE,
        # Asset Counts (pass if needed by __init__)
        'num_stocks': NUM_STOCKS, 'num_commodities': NUM_COMMODITIES, 'num_etfs': NUM_ETFS,
        'num_gov_bonds': NUM_GOV_BONDS, 'num_corp_bonds': NUM_CORP_BONDS, 'num_stable_inc': NUM_STABLE_INC,
        # Reward/Penalty Params
        'gamma_b': GAMMA_B, 'gamma_w': GAMMA_W, 'theta': THETA, 'beta_d': BETA_D,
        'qaly_normal_year': QALY_NORMAL_YEAR, 'qaly_shock_untreated': QALY_SHOCK_UNTREATED,
        'shortfall_penalty_multiplier': SHORTFALL_PENALTY_MULTIPLIER,
        'annuity_purchase_bonus': ANNUITY_PURCHASE_BONUS,
        # Health Params
        'health_shocks_config': HEALTH_SHOCKS_CONFIG,
        'health_shock_age_factor': HEALTH_SHOCK_AGE_FACTOR,
        # Asset Indices
        'risky_asset_indices': RISKY_ASSET_INDICES,
        'critical_asset_index': CRITICAL_ASSET_INDEX,
        # Annuity Params
        'compulsory_annuity_payment': COMPULSORY_ANNUITY_PAYMENT,
        'opt_annuity_1_premium': OPTIONAL_ANNUITY_1_PREMIUM,
        'opt_annuity_1_payment': OPTIONAL_ANNUITY_1_PAYMENT,
        'opt_annuity_2_premium': OPTIONAL_ANNUITY_2_PREMIUM,
        'opt_annuity_2_payment': OPTIONAL_ANNUITY_2_PAYMENT,
        # Survival Params
        'survival_a': SURVIVAL_A, 'survival_k': SURVIVAL_K,
        'survival_g': SURVIVAL_G, 'survival_c': SURVIVAL_C,
        # Forecast Noise Params
        'forecast_return_noise_factor_1yr': FORECAST_RETURN_NOISE_FACTOR_1YR,
        'forecast_return_noise_factor_2yr': FORECAST_RETURN_NOISE_FACTOR_2YR,
        'forecast_mortality_noise_1yr': FORECAST_MORTALITY_NOISE_1YR,
        'forecast_mortality_noise_2yr': FORECAST_MORTALITY_NOISE_2YR,
        # Control verbosity for evaluation
        'verbose': 0
    }
    # Verify this dict matches the kwargs expected by YOUR RetirementEnv.__init__
    return env_kwargs


# === Strategy Definitions ===
def ppo_strategy(model, obs, deterministic=True):
    """Get action from the loaded PPO model."""
    # Note: obs should be the normalized observation from VecNormalize
    action, _ = model.predict(obs, deterministic=deterministic)
    return action


def static_60_40_strategy(obs, env):
    """Fixed 60% risky, 40% less risky. No optional annuities."""
    # env here is the unwrapped RetirementEnv instance
    target_weights = np.zeros(env.total_assets, dtype=np.float32)
    less_risky_indices = list(set(range(env.total_assets)) - set(env.risky_asset_indices))
    risky_weight = 0.60
    less_risky_weight = 0.40
    # Distribute weights equally within categories
    if len(env.risky_asset_indices) > 0:
        target_weights[env.risky_asset_indices] = risky_weight / len(env.risky_asset_indices)
    if len(less_risky_indices) > 0:
        target_weights[less_risky_indices] = less_risky_weight / len(less_risky_indices)

    target_weights /= (np.sum(target_weights) + EPSILON)  # Normalize

    # Action format: portfolio weights (pre-softmax) + annuity decisions
    # Use target weights * 10 as a simple proxy for pre-softmax values
    portfolio_action = target_weights * 10.0
    # Annuity actions: Large negative values to signify "don't buy"
    annuity_action = np.array([-10.0, -10.0], dtype=np.float32)

    action = np.concatenate([portfolio_action, annuity_action])
    return action


def age_based_glidepath_strategy(obs, env):
    """Glidepath: Risky allocation = max(0.1, min(1.0, (105 - age) / 100)). No optional annuities."""
    # obs is the *normalized* observation from VecNormalize
    # env is the unwrapped RetirementEnv instance
    # We need the current age, get it from the env instance directly
    current_age = env.current_age

    # Calculate target risky weight based on current age
    target_risky_weight = max(0.1, min(1.0, (105.0 - current_age) / 100.0))  # Example: 105-age rule
    target_less_risky_weight = 1.0 - target_risky_weight

    target_weights = np.zeros(env.total_assets, dtype=np.float32)
    less_risky_indices = list(set(range(env.total_assets)) - set(env.risky_asset_indices))

    if len(env.risky_asset_indices) > 0:
        target_weights[env.risky_asset_indices] = target_risky_weight / len(env.risky_asset_indices)
    if len(less_risky_indices) > 0:
        target_weights[less_risky_indices] = target_less_risky_weight / len(less_risky_indices)

    target_weights /= (np.sum(target_weights) + EPSILON)  # Normalize

    # Convert target weights to raw action values (proxy)
    portfolio_action = target_weights * 10.0
    # Annuity actions: Don't buy
    annuity_action = np.array([-10.0, -10.0], dtype=np.float32)

    action = np.concatenate([portfolio_action, annuity_action])
    return action


# === Simulation Function ===
def run_simulations(strategy_func, env, n_simulations, model=None):
    """Runs n_simulations for a given strategy and returns collected data."""
    all_results = []
    # Get the VecEnv's properties
    is_vecenv = hasattr(env, 'num_envs')
    if is_vecenv:
        env_instance = env.envs[0].unwrapped  # Get the unwrapped env instance for properties
    else:  # Should not happen with make_vec_env, but handle just in case
        env_instance = env.unwrapped

    initial_wealth_local = env_instance.initial_wealth  # Store initial wealth for stress test calc

    # Check if reset supports options (for SB3 > 2.0)
    reset_supports_options = False
    try:
        import inspect
        sig = inspect.signature(env.reset)
        reset_supports_options = 'options' in sig.parameters
    except Exception:
        pass  # Keep it False if inspection fails

    for i in range(n_simulations):
        if i % 100 == 0: print(f"  Simulating episode {i + 1}/{n_simulations}...")
        episode_data = {
            "steps": [], "final_wealth": 0, "final_age": 0, "survived": False,
            "total_reward": 0, "hit_stress_floor": False, "final_info": {}
        }

        # Reset environment - handle potential differences in return format
        if reset_supports_options:
            obs, info = env.reset(options={})
        else:
            obs = env.reset()
            # Handle different possible return formats from env.reset()
            if isinstance(obs, tuple) and len(obs) == 2 and isinstance(obs[1], dict):
                info = obs[1]
                obs = obs[0]
            else:  # Assume obs only if not a tuple or info is not dict
                info = {}

        # Ensure obs is in the correct shape (1, obs_dim) for VecEnv/PPO predict
        if is_vecenv:
            obs = obs.reshape(1, -1)

        terminated = truncated = False
        episode_reward_sum = 0.0
        min_wealth_in_episode = initial_wealth_local

        while not (terminated or truncated):
            # Get action from strategy
            if strategy_func == ppo_strategy:
                # PPO strategy needs the potentially normalized obs
                action = strategy_func(model, obs)
            else:
                # Baseline strategies might need direct access to the unwrapped env state
                # Pass the normalized obs AND the unwrapped env instance
                current_unwrapped_env = env.envs[0].unwrapped if is_vecenv else env.unwrapped
                action = strategy_func(obs[0], current_unwrapped_env)  # Pass obs[0] if obs has batch dim

                # Ensure action has the batch dimension if needed by env.step
                if is_vecenv and action.ndim == 1:
                    action = action.reshape(1, -1)

            # Step environment
            step_output = env.step(action)

            # Unpack step results carefully, handling different Gym/SB3 versions
            if len(step_output) == 5:  # SB3 >= 2.0, Gym >= 0.26: obs, reward, terminated, truncated, info
                new_obs, reward, terminated, truncated, info = step_output
                # In VecEnv, these are arrays; extract the first element for single env case
                if is_vecenv:
                    reward_val = reward[0]
                    terminated_val = terminated[0]
                    truncated_val = truncated[0]
                    info_val = info[0]
                else:
                    reward_val = reward
                    terminated_val = terminated
                    truncated_val = truncated
                    info_val = info

            elif len(step_output) == 4:  # Older Gym: obs, reward, done, info
                new_obs, reward, done, info = step_output
                if is_vecenv:
                    reward_val = reward[0]
                    done_val = done[0]  # 'done' is equivalent to terminated or truncated
                    info_val = info[0]
                else:
                    reward_val = reward
                    done_val = done
                    info_val = info
                terminated_val = done_val  # Assume done means terminated
                truncated_val = info_val.get("TimeLimit.truncated", False)  # Check truncation info if present

            else:
                raise ValueError(f"Unexpected number of return values from env.step: {len(step_output)}")

            # Reshape new_obs if needed
            if is_vecenv:
                new_obs = new_obs.reshape(1, -1)

            # --- Data Collection per Step ---
            # Access underlying env state safely
            current_env_state = env.envs[0].unwrapped if is_vecenv else env.unwrapped
            step_age = current_env_state.current_age - 1  # Age *during* the year just completed
            step_wealth = current_env_state.current_wealth  # Wealth at *end* of step
            step_medical_cost = info_val.get("healthcare_cost_in_step", 0.0)  # Get from info dict
            step_income = current_env_state.compulsory_annuity_payment + \
                          (current_env_state.opt_annuity_1_payment if current_env_state.has_opt_annuity_1 else 0) + \
                          (current_env_state.opt_annuity_2_payment if current_env_state.has_opt_annuity_2 else 0)

            episode_data["steps"].append({
                "age": step_age, "wealth": step_wealth, "income": step_income,
                "medical_cost": step_medical_cost, "baseline_expense": current_env_state.baseline_living_expense,
                "has_opt1": current_env_state.has_opt_annuity_1, "has_opt2": current_env_state.has_opt_annuity_2
            })

            # Track minimum wealth for stress test
            min_wealth_in_episode = min(min_wealth_in_episode, step_wealth)

            obs = new_obs
            episode_reward_sum += reward_val  # Accumulate reward

            # Check termination/truncation
            if terminated_val or truncated_val:
                # Use 'final_info' key if available (SB3 >= 2.0 convention)
                final_info = info_val.get("final_info", info_val)  # Fallback to info_val

                # Ensure final state values are captured correctly
                final_wealth = final_info.get("wealth", current_env_state.current_wealth)
                final_age = final_info.get("age", current_env_state.current_age)
                is_alive = final_info.get("is_alive", current_env_state.is_alive)

                episode_data["final_wealth"] = final_wealth
                episode_data["final_age"] = final_age
                episode_data["survived"] = is_alive
                episode_data["total_reward"] = episode_reward_sum
                episode_data["final_info"] = final_info  # Store the whole final info dict

                # Check if stress floor was hit
                stress_floor_value = initial_wealth_local * STRESS_TEST_FLOOR_PCT
                if min_wealth_in_episode < stress_floor_value:
                    episode_data["hit_stress_floor"] = True

                break  # Exit while loop

        all_results.append(episode_data)

    print(f"  Finished {n_simulations} simulations.")
    return all_results


# === Metric Calculation Functions ===
def calculate_cvar(results, initial_wealth, percentile=0.05):
    """Calculates CVaR based on final wealth relative loss."""
    final_wealths = np.array([r['final_wealth'] for r in results if r is not None and 'final_wealth' in r])
    if len(final_wealths) == 0: return np.nan
    final_wealths = final_wealths[np.isfinite(final_wealths)] # Filter out NaNs/Infs
    if len(final_wealths) == 0: return np.nan
    # Calculate relative loss (positive value indicates loss)
    relative_losses = (initial_wealth - final_wealths) / (initial_wealth + EPSILON)
    if len(relative_losses) == 0: return np.nan
    threshold_loss = np.percentile(relative_losses, (1 - percentile) * 100)
    worst_losses = relative_losses[relative_losses >= threshold_loss]
    if len(worst_losses) == 0: return 0.0  # No losses above threshold
    cvar = np.mean(worst_losses)
    # Return negative percentage loss (as per original table convention)
    return -cvar * 100


def calculate_stress_test_pass_rate(results):
    """Calculates the percentage of simulations that did NOT hit the stress floor."""
    valid_results = [r for r in results if r is not None and 'hit_stress_floor' in r]
    if not valid_results: return np.nan
    times_floor_hit = sum(1 for r in valid_results if r['hit_stress_floor'])
    pass_rate = 1.0 - (times_floor_hit / len(valid_results))
    return pass_rate * 100  # Return as percentage


def calculate_oop_burden_metrics(results):
    """Calculates metrics related to Out-of-Pocket Medical Burden."""
    all_burden_ratios = []
    n_shock_years = 0
    n_burden_violations = 0
    valid_results = [r for r in results if r is not None and 'steps' in r]
    for r in valid_results:
        for step_data in r.get('steps', []):
            medical_cost = step_data.get('medical_cost', 0.0)
            if medical_cost > EPSILON:
                n_shock_years += 1
                income = step_data.get('income', 0.0)
                if income < EPSILON:
                    burden_ratio = np.inf
                else:
                    burden_ratio = medical_cost / income
                all_burden_ratios.append(burden_ratio)
                if burden_ratio > OOP_BURDEN_TARGET: n_burden_violations += 1

    if not all_burden_ratios:
        return {"mean": 0.0, "median": 0.0, "max": 0.0, "violation_rate": 0.0}

    finite_ratios = [b for b in all_burden_ratios if np.isfinite(b)]
    mean_burden = np.mean(finite_ratios) if finite_ratios else 0.0
    median_burden = np.median(finite_ratios) if finite_ratios else 0.0
    # Calculate max carefully, handling potential only-inf cases
    finite_max = np.max(finite_ratios) if finite_ratios else 0.0
    has_inf = any(np.isinf(b) for b in all_burden_ratios)
    max_burden = float('inf') if has_inf else finite_max

    violation_rate = (n_burden_violations / n_shock_years) if n_shock_years > 0 else 0.0

    return {"mean": mean_burden * 100, "median": median_burden * 100,
            "max": max_burden * 100 if np.isfinite(max_burden) else float('inf'),
            "violation_rate": violation_rate * 100}


def calculate_surplus_coverage_metrics(results):
    """Calculates metrics related to Surplus Coverage Ratio."""
    all_coverage_ratios = []
    n_violations = 0
    total_steps = 0
    valid_results = [r for r in results if r is not None and 'steps' in r]
    for r in valid_results:
        for step_data in r.get('steps', []):
            total_steps += 1
            liquid_assets = step_data.get('wealth', 0.0)
            income = step_data.get('income', 0.0)
            min_required = step_data.get('baseline_expense', 0.0)
            medical_outlays = step_data.get('medical_cost', 0.0)
            total_required = min_required + medical_outlays

            if total_required < EPSILON:
                coverage_ratio = np.inf  # Favorable
            else:
                # Use wealth at END of step + income DURING step to cover expenses DURING step
                coverage_ratio = (liquid_assets + income) / total_required

            all_coverage_ratios.append(coverage_ratio)
            if coverage_ratio < SURPLUS_COVERAGE_TARGET: n_violations += 1

    if not all_coverage_ratios:
        return {"mean": np.nan, "median": np.nan, "min": np.nan, "violation_rate": np.nan}

    # Calculate min carefully, handle potential -inf
    finite_ratios = [r for r in all_coverage_ratios if np.isfinite(r)]
    mean_coverage = np.mean(finite_ratios) if finite_ratios else 0.0
    median_coverage = np.median(finite_ratios) if finite_ratios else 0.0
    # Check for negative infinity explicitly if wealth can hit floor severely
    has_neginf = any(r == -np.inf for r in all_coverage_ratios)
    min_coverage = -np.inf if has_neginf else (np.min(finite_ratios) if finite_ratios else 0.0)

    violation_rate = (n_violations / total_steps) if total_steps > 0 else 0.0

    return {"mean": mean_coverage, "median": median_coverage, "min": min_coverage,
            "violation_rate": violation_rate * 100}


def calculate_annuity_adequacy_metrics(results):
    """Calculates metrics related to Annuity Adequacy in later life."""
    ratios_in_later_life = []
    n_target_years = 0
    n_adequate_years = 0
    valid_results = [r for r in results if r is not None]
    for r in valid_results:
        # Check survival conditions carefully based on how final age/survival is stored
        final_age = r.get("final_age", START_AGE)
        survived = r.get("survived", False)

        reached_or_surpassed_target_age = False
        if survived and final_age >= ANNUITY_ADEQUACY_TARGET_AGE:
            reached_or_surpassed_target_age = True
        elif not survived and final_age > ANNUITY_ADEQUACY_TARGET_AGE: # Died *after* starting the target age year
             reached_or_surpassed_target_age = True

        if not reached_or_surpassed_target_age: continue # Skip if died before or didn't reach target age

        for step_data in r.get('steps', []):
            age = step_data.get('age', START_AGE -1) # Age during the year
            if age >= ANNUITY_ADEQUACY_TARGET_AGE:
                n_target_years += 1
                guaranteed_income = COMPULSORY_ANNUITY_PAYMENT + \
                                    (OPTIONAL_ANNUITY_1_PAYMENT if step_data.get('has_opt1', False) else 0) + \
                                    (OPTIONAL_ANNUITY_2_PAYMENT if step_data.get('has_opt2', False) else 0)
                essential_expenses = step_data.get('baseline_expense', BASELINE_LIVING_EXPENSE)

                if essential_expenses < EPSILON:
                    ratio = np.inf # Favorable
                else:
                    ratio = guaranteed_income / essential_expenses

                ratios_in_later_life.append(ratio)
                if ratio >= ANNUITY_ADEQUACY_TARGET_RATIO: n_adequate_years += 1

    if not ratios_in_later_life:
        # It's possible no sims reach target age, return 0 not NaN for rates
        # print(f"Warning: No simulation steps found >= age {ANNUITY_ADEQUACY_TARGET_AGE}.")
        return {"mean": 0.0, "median": 0.0, "pct_adequate_years": 0.0}

    finite_ratios = [r for r in ratios_in_later_life if np.isfinite(r)]
    mean_ratio = np.mean(finite_ratios) if finite_ratios else 0.0
    median_ratio = np.median(finite_ratios) if finite_ratios else 0.0
    pct_adequate = (n_adequate_years / n_target_years) if n_target_years > 0 else 0.0

    return {"mean": mean_ratio * 100,  # Pct
            "median": median_ratio * 100,  # Pct
            "pct_adequate_years": pct_adequate * 100}  # Pct of years >= target ratio


def calculate_qaly_estate_metrics(results):
    """Calculates average QALY and Final Estate Value."""
    valid_results = [r for r in results if r is not None and 'final_info' in r and 'final_wealth' in r]
    if not valid_results:
        return {"avg_qaly": np.nan, "avg_final_estate": np.nan}

    # Extract QALY component sum from final_info (using 'final_qaly_comp' key)
    qaly_values = [r['final_info'].get('final_qaly_comp', np.nan) for r in valid_results]
    qaly_values = [q for q in qaly_values if not np.isnan(q)] # Filter NaNs

    # Final estate value is simply the final wealth
    final_estate_values = [r['final_wealth'] for r in valid_results]
    final_estate_values = [w for w in final_estate_values if np.isfinite(w)] # Filter NaNs/Infs

    avg_qaly = np.mean(qaly_values) if qaly_values else 0.0
    avg_final_estate = np.mean(final_estate_values) if final_estate_values else 0.0

    return {"avg_qaly": avg_qaly, "avg_final_estate": avg_final_estate}


# === Summary Plotting Functions ===
def plot_objective1(metrics_summary, strategy_names, save_dir):
    """Plots metrics for Objective 1: Downside Risk Protection."""
    # Use negative CVaR values as per table convention
    cvar_values = [metrics_summary[name].get('CVaR (5%, % Loss)', np.nan) for name in strategy_names]
    stress_pass_rates = [metrics_summary[name].get('Stress Pass Rate (%)', np.nan) for name in strategy_names]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.suptitle('Objective 1: Downside Risk Protection', fontsize=16)
    # CVaR Plot (Plotting the negative value, so lower is better)
    bars_cvar = axes[0].bar(strategy_names, cvar_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0].set_ylabel('CVaR (5%, % Loss)')
    axes[0].set_title(f'Conditional Value at Risk (Lower is Better)')
    axes[0].grid(axis='y', linestyle='--')
    # axes[0].axhline(-CVAR_PERCENTILE * 100, color='red', linestyle=':', linewidth=2, label=f'Target? (-{CVAR_PERCENTILE*100:.0f}%)') # Target line might be confusing here
    # if any(~np.isnan(cvar_values)): axes[0].legend()
    axes[0].tick_params(axis='x', rotation=15)
    for bar in bars_cvar:
        yval = bar.get_height();
        if not np.isnan(yval): axes[0].text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.1f}%', va='bottom' if yval >=0 else 'top', ha='center')
    # Stress Test Pass Rate Plot
    bars_stress = axes[1].bar(strategy_names, stress_pass_rates, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[1].set_ylabel('Pass Rate (%)')
    axes[1].set_title(f'Stress Test Pass Rate (Floor: {STRESS_TEST_FLOOR_PCT * 100:.0f}% Init Wealth)')
    valid_rates = [r for r in stress_pass_rates if not np.isnan(r)]
    if valid_rates:
        axes[1].set_ylim(bottom=max(0, min(valid_rates) - 10), top=105)
    else:
        axes[1].set_ylim(bottom=0, top=105)
    axes[1].grid(axis='y', linestyle='--');
    axes[1].tick_params(axis='x', rotation=15)
    for bar in bars_stress:
        yval = bar.get_height();
        if not np.isnan(yval): axes[1].text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.1f}%', va='bottom', ha='center')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]);
    plt.savefig(os.path.join(save_dir, "summary_objective1_downside_risk.png"))
    plt.close(fig)


def plot_objective2(metrics_summary, strategy_names, save_dir):
    """Plots metrics for Objective 2: Declining Income & Healthcare Shocks."""
    oop_violation_rates = [metrics_summary[name].get('OOP Burden Violation Rate (%)', np.nan) for name in strategy_names]
    surplus_violation_rates = [metrics_summary[name].get('Surplus Coverage Violation Rate (%)', np.nan) for name in strategy_names]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.suptitle('Objective 2: Income & Healthcare Shocks (Violation Rates)', fontsize=16)
    # OOP Burden Violation Rate Plot
    bars_oop = axes[0].bar(strategy_names, oop_violation_rates, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0].set_ylabel('Violation Rate (%)');
    axes[0].set_title(f'OOP Burden Violation Rate (Burden > {OOP_BURDEN_TARGET * 100:.0f}%)')
    axes[0].grid(axis='y', linestyle='--'); axes[0].set_ylim(bottom=0, top=max(10, np.nanmax(oop_violation_rates)*1.1 if any(~np.isnan(oop_violation_rates)) else 10))
    axes[0].tick_params(axis='x', rotation=15)
    for bar in bars_oop:
        yval = bar.get_height();
        if not np.isnan(yval): axes[0].text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.1f}%', va='bottom', ha='center')
    # Surplus Coverage Violation Rate Plot
    bars_surplus = axes[1].bar(strategy_names, surplus_violation_rates, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[1].set_ylabel('Violation Rate (%)');
    axes[1].set_title(f'Surplus Coverage Violation Rate (Ratio < {SURPLUS_COVERAGE_TARGET:.1f}x)')
    axes[1].grid(axis='y', linestyle='--'); axes[1].set_ylim(bottom=0, top=max(10, np.nanmax(surplus_violation_rates)*1.1 if any(~np.isnan(surplus_violation_rates)) else 10))
    axes[1].tick_params(axis='x', rotation=15)
    for bar in bars_surplus:
        yval = bar.get_height();
        if not np.isnan(yval): axes[1].text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.1f}%', va='bottom', ha='center')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]);
    plt.savefig(os.path.join(save_dir, "summary_objective2_income_shocks.png"))
    plt.close(fig)


def plot_objective3(metrics_summary, strategy_names, save_dir):
    """Plots metrics for Objective 3: Longevity Risk Hedging."""
    adequacy_pct_years = [metrics_summary[name].get(f'Annuity Adequacy Pct Years Adequate (Age {ANNUITY_ADEQUACY_TARGET_AGE}+, %)', np.nan) for name in strategy_names]
    fig, ax = plt.subplots(1, 1, figsize=(8, 6));
    fig.suptitle('Objective 3: Longevity Risk Hedging (Annuity Adequacy)', fontsize=16)
    # Annuity Adequacy Plot
    bars = ax.bar(strategy_names, adequacy_pct_years, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_ylabel('Adequate Years (%)');
    ax.set_title(f'% Years Annuity Covers Baseline Exp. (Age {ANNUITY_ADEQUACY_TARGET_AGE}+, Ratio >= {ANNUITY_ADEQUACY_TARGET_RATIO:.0%})')
    ax.set_ylim(bottom=0, top=105);
    ax.grid(axis='y', linestyle='--')
    # ax.axhline(ANNUITY_ADEQUACY_TARGET_RATIO * 100, color='red', linestyle=':', linewidth=2, label=f'Target ({ANNUITY_ADEQUACY_TARGET_RATIO:.0%})')
    # if any(~np.isnan(adequacy_pct_years)): ax.legend()
    ax.tick_params(axis='x', rotation=15)
    for bar in bars:
        yval = bar.get_height();
        if not np.isnan(yval): ax.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.1f}%', va='bottom', ha='center')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]);
    plt.savefig(os.path.join(save_dir, "summary_objective3_longevity_risk.png"))
    plt.close(fig)


def plot_objective4(metrics_summary, strategy_names, save_dir):
    """Plots metrics for Objective 4: QALY and Final Estate Value."""
    avg_qaly = [metrics_summary[name].get('Average QALY', np.nan) for name in strategy_names]
    avg_estate = [metrics_summary[name].get('Average Final Estate ($)', np.nan) for name in strategy_names]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.suptitle('Objective 4: Quality of Life & Estate Value (Averages)', fontsize=16)
    # Average QALY Plot
    bars_qaly = axes[0].bar(strategy_names, avg_qaly, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0].set_ylabel('Average Accumulated QALY')
    axes[0].set_title('Average Quality-Adjusted Life Years')
    axes[0].grid(axis='y', linestyle='--')
    axes[0].tick_params(axis='x', rotation=15)
    for bar in bars_qaly:
        yval = bar.get_height()
        if not np.isnan(yval): axes[0].text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.2f}', va='bottom', ha='center')
    # Average Final Estate Plot
    bars_estate = axes[1].bar(strategy_names, avg_estate, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[1].set_ylabel('Average Final Estate Value ($)')
    axes[1].set_title('Average Final Wealth (Terminal/Bequest)')
    axes[1].grid(axis='y', linestyle='--')
    axes[1].tick_params(axis='x', rotation=15)
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ','))) # Format y-axis ticks
    # Adjust text placement if values are low
    min_estate = np.nanmin(avg_estate) if any(~np.isnan(avg_estate)) else 0
    max_estate = np.nanmax(avg_estate) if any(~np.isnan(avg_estate)) else 0
    text_offset = (max_estate - min_estate) * 0.05 if max_estate > 0 else 10000
    for bar in bars_estate:
        yval = bar.get_height()
        if not np.isnan(yval): axes[1].text(bar.get_x() + bar.get_width() / 2.0, yval + text_offset*0.1, f'${yval:,.0f}', va='bottom', ha='center')
    if max_estate > 0 : axes[1].set_ylim(top=max_estate * 1.15) # Adjust y-limit for text
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]);
    plt.savefig(os.path.join(save_dir, "summary_objective4_qaly_estate.png"))
    plt.close(fig)


# === NEW: Distribution Plotting Functions ===

def plot_distribution_final_outcomes(results, strategy_names, save_dir):
    """Plots the distribution of Final Wealth and Total QALY."""
    print("  Generating distribution plots for Final Wealth and QALY...")
    plot_data_wealth = []
    plot_data_qaly = []

    for strategy_name in strategy_names:
        sim_results = results.get(strategy_name, [])
        for i, sim_data in enumerate(sim_results):
            # Final Wealth
            final_wealth = sim_data.get('final_wealth')
            if final_wealth is not None and np.isfinite(final_wealth):
                plot_data_wealth.append({'Strategy': strategy_name, 'Simulation': i, 'Final Wealth': final_wealth})

            # Total QALY
            final_info = sim_data.get('final_info', {})
            final_qaly = final_info.get('final_qaly_comp')
            if final_qaly is not None and np.isfinite(final_qaly):
                 plot_data_qaly.append({'Strategy': strategy_name, 'Simulation': i, 'Total QALY': final_qaly})

    # --- Plot Final Wealth Distribution ---
    if plot_data_wealth:
        df_wealth = pd.DataFrame(plot_data_wealth)

        # Box Plot
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Strategy', y='Final Wealth', data=df_wealth, palette=['#1f77b4', '#ff7f0e', '#2ca02c'])
        plt.title('Distribution of Final Wealth across Strategies')
        plt.ylabel('Final Wealth ($)')
        plt.xlabel('Strategy')
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        plt.xticks(rotation=10)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "distribution_final_wealth_box.png"))
        plt.close()

        # Density Plot
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=df_wealth, x='Final Wealth', hue='Strategy', fill=True, common_norm=False, alpha=0.5, palette=['#1f77b4', '#ff7f0e', '#2ca02c'])
        plt.title('Density Distribution of Final Wealth across Strategies')
        plt.xlabel('Final Wealth ($)')
        plt.ylabel('Density')
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "distribution_final_wealth_density.png"))
        plt.close()
    else:
        print("    WARNING: No valid final wealth data found for distribution plots.")

    # --- Plot Total QALY Distribution ---
    if plot_data_qaly:
        df_qaly = pd.DataFrame(plot_data_qaly)

        # Box Plot
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Strategy', y='Total QALY', data=df_qaly, palette=['#1f77b4', '#ff7f0e', '#2ca02c'])
        plt.title('Distribution of Total Accumulated QALY across Strategies')
        plt.ylabel('Total QALY')
        plt.xlabel('Strategy')
        plt.xticks(rotation=10)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "distribution_total_qaly_box.png"))
        plt.close()

        # Density Plot
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=df_qaly, x='Total QALY', hue='Strategy', fill=True, common_norm=False, alpha=0.5, palette=['#1f77b4', '#ff7f0e', '#2ca02c'])
        plt.title('Density Distribution of Total Accumulated QALY across Strategies')
        plt.xlabel('Total QALY')
        plt.ylabel('Density')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "distribution_total_qaly_density.png"))
        plt.close()
    else:
        print("    WARNING: No valid total QALY data found for distribution plots.")

def plot_distribution_wealth_trajectory(results, strategy_names, save_dir):
    """Plots the median wealth trajectory with percentile bands."""
    print("  Generating distribution plot for Wealth Trajectory...")
    trajectory_data = []
    max_age_observed = START_AGE # Find the max age for padding

    for strategy_name in strategy_names:
        sim_results = results.get(strategy_name, [])
        for i, sim_data in enumerate(sim_results):
            for step_info in sim_data.get('steps', []):
                age = step_info.get('age')
                wealth = step_info.get('wealth')
                if age is not None and wealth is not None and np.isfinite(wealth):
                    trajectory_data.append({'Strategy': strategy_name, 'Simulation': i, 'Age': age, 'Wealth': wealth})
                    max_age_observed = max(max_age_observed, age)

    if trajectory_data:
        df_trajectory = pd.DataFrame(trajectory_data)

        plt.figure(figsize=(12, 7))
        # Using Seaborn's lineplot to show median and 25th-75th percentile interval
        sns.lineplot(data=df_trajectory, x='Age', y='Wealth', hue='Strategy', estimator='median', errorbar=('pi', 50), palette=['#1f77b4', '#ff7f0e', '#2ca02c'], legend='full')
        plt.title('Median Wealth Trajectory (with 25th-75th Percentile Range)')
        plt.xlabel('Age')
        plt.ylabel('Wealth ($)')
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        plt.grid(True, linestyle='--')
        plt.legend(title='Strategy')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "distribution_wealth_trajectory_quantiles.png"))
        plt.close()
    else:
        print("    WARNING: No valid wealth trajectory data found for plotting.")

def plot_distribution_age_at_death(results, strategy_names, save_dir):
    """Plots the distribution of age at death for simulations ending before END_AGE."""
    print("  Generating distribution plot for Age at Death...")
    death_age_data = []
    for strategy_name in strategy_names:
        sim_results = results.get(strategy_name, [])
        for i, sim_data in enumerate(sim_results):
             survived = sim_data.get('survived')
             final_age = sim_data.get('final_age')
             # Include only if agent did not survive AND final_age is valid and before scheduled end
             if survived == False and final_age is not None and final_age < END_AGE:
                 death_age_data.append({'Strategy': strategy_name, 'Simulation': i, 'Age at Death': final_age})

    if death_age_data:
        df_death_age = pd.DataFrame(death_age_data)

        plt.figure(figsize=(10, 6))
        sns.histplot(data=df_death_age, x='Age at Death', hue='Strategy', kde=True, element="step", stat="density", common_norm=False, palette=['#1f77b4', '#ff7f0e', '#2ca02c'])
        plt.title(f'Distribution of Age at Death (for simulations ending before Age {END_AGE})')
        plt.xlabel('Age at Death')
        plt.ylabel('Density')
        plt.legend(title='Strategy')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "distribution_age_at_death.png"))
        plt.close()
    else:
        print(f"    INFO: No simulations ended due to death before age {END_AGE} for plotting age at death distribution.")


def plot_distribution_intermediate_ratios(results, strategy_names, save_dir):
    """Plots distributions of OOP Burden Ratio and Surplus Coverage Ratio."""
    print("  Generating distribution plots for OOP Burden and Surplus Coverage Ratios...")
    oop_data = []
    surplus_data = []

    for strategy_name in strategy_names:
        sim_results = results.get(strategy_name, [])
        for i, sim_data in enumerate(sim_results):
            for step_info in sim_data.get('steps', []):
                age = step_info.get('age', START_AGE - 1)
                # --- OOP Burden ---
                medical_cost = step_info.get('medical_cost', 0.0)
                if medical_cost > EPSILON: # Only calculate for years with medical costs
                    income = step_info.get('income', 0.0)
                    if income < EPSILON:
                        burden_ratio = np.inf # Handle as a large number / special category
                    else:
                        burden_ratio = medical_cost / income
                    # Only plot finite ratios, maybe cap extremely large ones for visualization
                    if np.isfinite(burden_ratio):
                         # Cap ratio for visualization if desired, e.g., at 10 (1000%)
                         # burden_ratio_capped = min(burden_ratio, 10.0)
                         oop_data.append({'Strategy': strategy_name, 'OOP Burden Ratio': burden_ratio})

                # --- Surplus Coverage ---
                liquid_assets = step_info.get('wealth', 0.0)
                income_surplus = step_info.get('income', 0.0)
                min_required = step_info.get('baseline_expense', 0.0)
                medical_outlays_surplus = step_info.get('medical_cost', 0.0)
                total_required = min_required + medical_outlays_surplus

                if total_required < EPSILON:
                    coverage_ratio = np.inf # Favorable
                else:
                    coverage_ratio = (liquid_assets + income_surplus) / total_required

                # Only plot finite ratios, maybe cap extremely large/small ones
                if np.isfinite(coverage_ratio):
                     # Cap ratio for visualization if desired, e.g., at 100x
                     # coverage_ratio_capped = min(coverage_ratio, 100.0)
                     # Handle potential negative values if wealth floor is low/negative? Assume wealth >= 0
                     surplus_data.append({'Strategy': strategy_name, 'Surplus Coverage Ratio': coverage_ratio})


    # --- Plot OOP Burden Ratio Distribution ---
    if oop_data:
        df_oop = pd.DataFrame(oop_data)

        plt.figure(figsize=(10, 7))
        sns.violinplot(x='Strategy', y='OOP Burden Ratio', data=df_oop, palette=['#1f77b4', '#ff7f0e', '#2ca02c'], cut=0) # cut=0 avoids smoothing beyond data range
        plt.title('Distribution of OOP Burden Ratio (Years with Medical Costs > 0)')
        plt.ylabel('OOP Burden Ratio (Medical Cost / Annuity Income)')
        plt.xlabel('Strategy')
        # Optional: Limit y-axis to see detail if outliers are huge, e.g., using quantile
        q99 = df_oop['OOP Burden Ratio'].quantile(0.99) if not df_oop.empty else 1.0
        plt.ylim(bottom = -0.1, top=max(q99 * 1.1, OOP_BURDEN_TARGET * 1.5)) # Show target clearly
        plt.axhline(OOP_BURDEN_TARGET, color='red', linestyle=':', label=f'Target ({OOP_BURDEN_TARGET:.0%})')
        plt.legend()
        plt.xticks(rotation=10)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "distribution_oop_burden_ratio.png"))
        plt.close()
    else:
        print("    WARNING: No OOP burden data found for distribution plot.")

    # --- Plot Surplus Coverage Ratio Distribution ---
    if surplus_data:
        df_surplus = pd.DataFrame(surplus_data)

        plt.figure(figsize=(10, 7))
        sns.violinplot(x='Strategy', y='Surplus Coverage Ratio', data=df_surplus, palette=['#1f77b4', '#ff7f0e', '#2ca02c'], cut=0)
        plt.title('Distribution of Surplus Coverage Ratio (All Years)')
        plt.ylabel('Surplus Coverage Ratio ((Wealth + Income) / Total Expenses)')
        plt.xlabel('Strategy')
        # Optional: Limit y-axis, e.g., focus around the target
        q01 = df_surplus['Surplus Coverage Ratio'].quantile(0.01) if not df_surplus.empty else 0.0
        q99_surplus = df_surplus['Surplus Coverage Ratio'].quantile(0.99) if not df_surplus.empty else 10.0
        plt.ylim(bottom=max(-0.1, q01*0.9), top=min(q99_surplus * 1.1, 50)) # Cap max ylim for clarity
        plt.axhline(SURPLUS_COVERAGE_TARGET, color='red', linestyle=':', label=f'Target ({SURPLUS_COVERAGE_TARGET:.1f}x)')
        plt.legend()
        plt.xticks(rotation=10)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "distribution_surplus_coverage_ratio.png"))
        plt.close()
    else:
        print("    WARNING: No Surplus Coverage data found for distribution plot.")


# === Main Evaluation Execution ===
if __name__ == "__main__":
    print("--- Starting Model Evaluation & Plotting ---")

    # ** Verify Model Path **
    if "UPDATE THIS PATH" in MODEL_SAVE_DIR or "<YOUR" in MODEL_SAVE_DIR: # More generic check
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! ERROR: Please update the 'MODEL_SAVE_DIR' variable in the script with the correct      !!!")
        print("!!!        path to your trained model directory (containing .zip and .pkl files).          !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        exit()
    if not os.path.exists(MODEL_PATH) or not os.path.exists(STATS_PATH):
        raise FileNotFoundError(f"Model ({MODEL_PATH}) or Stats ({STATS_PATH}) not found. "
                                f"Ensure MODEL_SAVE_DIR ('{MODEL_SAVE_DIR}') is set correctly and training artifacts exist.")

    # Create plot directory
    os.makedirs(PLOT_SAVE_DIR, exist_ok=True)
    print(f"Plots will be saved in: {PLOT_SAVE_DIR}")

    # --- Load Environment ---
    print(f"\nLoading environment definition...")
    env_kwargs = get_env_kwargs()
    # Create a dummy VecEnv for loading normalization stats
    # Use Monitor to ensure final_info is populated correctly
    print(f"Creating dummy VecEnv and loading normalization stats from: {STATS_PATH}")
    try:
        # Wrap the env creation in a lambda for make_vec_env
        env_lambda = lambda: Monitor(RetirementEnv(**env_kwargs))
        eval_env = make_vec_env(env_lambda, n_envs=1, vec_env_cls=DummyVecEnv)

        # Load the VecNormalize stats
        eval_env = VecNormalize.load(STATS_PATH, eval_env)
        eval_env.training = False  # Set to evaluation mode
        eval_env.norm_reward = False  # Do not normalize reward to get true values
        print("Environment and normalization stats loaded successfully.")
    except FileNotFoundError:
         print(f"--- ERROR loading normalization stats: File not found at {STATS_PATH} ---")
         print("Please ensure the path is correct and the file exists.")
         exit()
    except Exception as e:
        print(f"--- ERROR loading environment or normalization stats ---")
        print(f"Error Type: {type(e).__name__}, Error: {e}")
        print("Please ensure:")
        print("  1. You have pasted the EXACT RetirementEnv class and constants used for training.")
        print(f"  2. The VecNormalize stats file '{STATS_PATH}' exists and corresponds to the environment structure used during training.")
        print("  3. All necessary libraries (gymnasium, stable-baselines3, torch, numpy, matplotlib, pandas, seaborn) are installed.")
        exit()

    # --- Load PPO Model ---
    print(f"\nLoading trained PPO model from: {MODEL_PATH}")
    try:
        ppo_model = PPO.load(MODEL_PATH, env=eval_env) # Pass the VecNormalize env here
        print("PPO model loaded successfully.")
    except FileNotFoundError:
         print(f"--- ERROR loading PPO model: File not found at {MODEL_PATH} ---")
         print("Please ensure the path is correct and the file exists.")
         exit()
    except Exception as e:
        print(f"--- ERROR loading PPO model ---")
        print(f"Error Type: {type(e).__name__}, Error: {e}")
        print(f"Please ensure '{MODEL_PATH}' exists and is a valid PPO model compatible with the loaded environment and SB3 version.")
        exit()

    # --- Run Simulations for Each Strategy ---
    results = {} # This dictionary will hold the detailed simulation results
    strategies = {
        "DRL Agent": (ppo_strategy, ppo_model),
        "Static 60/40": (static_60_40_strategy, None),
        "Glidepath": (age_based_glidepath_strategy, None)
    }
    strategy_names = list(strategies.keys())

    # Store the initial wealth from the loaded env instance (unwrapped)
    try:
        env_instance = eval_env.envs[0].unwrapped
        INITIAL_WEALTH_FROM_ENV = env_instance.initial_wealth
        print(f"Initial wealth read from environment: ${INITIAL_WEALTH_FROM_ENV:,.0f}")
    except Exception as e:
        print(f"Warning: Could not read initial wealth from unwrapped env. Using constant. Error: {e}")
        INITIAL_WEALTH_FROM_ENV = INITIAL_WEALTH # Fallback

    for name, (strategy_func, model_or_none) in strategies.items():
        print(f"\nRunning {N_EVAL_SIMULATIONS} simulations for strategy: {name}...")
        start_sim_time = time.time()
        # Pass the VecNormalize env to the simulation function
        results[name] = run_simulations(strategy_func, eval_env, N_EVAL_SIMULATIONS, model=model_or_none)
        end_sim_time = time.time()
        print(f"  Simulation for {name} took {end_sim_time - start_sim_time:.2f} seconds.")

    # --- Calculate Summary Metrics ---
    print("\nCalculating summary metrics for all strategies...")
    metrics_summary = {name: {} for name in strategy_names}
    for name, sim_results in results.items():
        print(f"  Calculating for: {name}")
        if not sim_results:
            print(f"    WARNING: No simulation results found for {name}. Skipping metrics.")
            continue

        # Recalculate metrics using the collected simulation results
        metrics_summary[name]['CVaR (5%, % Loss)'] = calculate_cvar(sim_results, INITIAL_WEALTH_FROM_ENV, CVAR_PERCENTILE)
        metrics_summary[name]['Stress Pass Rate (%)'] = calculate_stress_test_pass_rate(sim_results)
        oop_metrics = calculate_oop_burden_metrics(sim_results)
        metrics_summary[name]['OOP Burden Mean (%)'] = oop_metrics['mean']
        metrics_summary[name]['OOP Burden Median (%)'] = oop_metrics['median']
        metrics_summary[name]['OOP Burden Max (%)'] = oop_metrics['max']
        metrics_summary[name]['OOP Burden Violation Rate (%)'] = oop_metrics['violation_rate']
        surplus_metrics = calculate_surplus_coverage_metrics(sim_results)
        metrics_summary[name]['Surplus Coverage Mean (x)'] = surplus_metrics['mean']
        metrics_summary[name]['Surplus Coverage Median (x)'] = surplus_metrics['median']
        metrics_summary[name]['Surplus Coverage Min (x)'] = surplus_metrics['min']
        metrics_summary[name]['Surplus Coverage Violation Rate (%)'] = surplus_metrics['violation_rate']
        annuity_metrics = calculate_annuity_adequacy_metrics(sim_results)
        metrics_summary[name][f'Annuity Adequacy Mean (Age {ANNUITY_ADEQUACY_TARGET_AGE}+, %)'] = annuity_metrics['mean']
        metrics_summary[name][f'Annuity Adequacy Median (Age {ANNUITY_ADEQUACY_TARGET_AGE}+, %)'] = annuity_metrics['median']
        metrics_summary[name][f'Annuity Adequacy Pct Years Adequate (Age {ANNUITY_ADEQUACY_TARGET_AGE}+, %)'] = annuity_metrics['pct_adequate_years']
        qaly_estate_metrics = calculate_qaly_estate_metrics(sim_results)
        metrics_summary[name]['Average QALY'] = qaly_estate_metrics['avg_qaly']
        metrics_summary[name]['Average Final Estate ($)'] = qaly_estate_metrics['avg_final_estate']

    # --- Generate Summary Bar Plots ---
    print("\nGenerating summary objective plots...")
    try:
        plot_objective1(metrics_summary, strategy_names, PLOT_SAVE_DIR)
        plot_objective2(metrics_summary, strategy_names, PLOT_SAVE_DIR)
        plot_objective3(metrics_summary, strategy_names, PLOT_SAVE_DIR)
        plot_objective4(metrics_summary, strategy_names, PLOT_SAVE_DIR)
        print(f"Summary objective plots saved successfully to '{PLOT_SAVE_DIR}'.")
    except Exception as e:
        print(f"--- ERROR generating summary plots ---")
        print(f"Error Type: {type(e).__name__}, Error: {e}")
        print("Plotting failed. Please check the calculated metrics and plotting functions.")

    # --- Generate Distribution Plots ---
    print("\nGenerating distribution plots...")
    try:
        # Pass the raw simulation results dictionary
        plot_distribution_final_outcomes(results, strategy_names, PLOT_SAVE_DIR)
        plot_distribution_wealth_trajectory(results, strategy_names, PLOT_SAVE_DIR)
        plot_distribution_age_at_death(results, strategy_names, PLOT_SAVE_DIR)
        plot_distribution_intermediate_ratios(results, strategy_names, PLOT_SAVE_DIR)
        print(f"Distribution plots saved successfully to '{PLOT_SAVE_DIR}'.")
    except ImportError:
         print("--- ERROR: Could not generate distribution plots. ---")
         print("--- Please ensure 'pandas' and 'seaborn' are installed (`pip install pandas seaborn`) ---")
    except Exception as e:
        print(f"--- ERROR generating distribution plots ---")
        print(f"Error Type: {type(e).__name__}, Error: {e}")
        print("Plotting failed. Please check the collected simulation data and distribution plotting functions.")


    # --- Print Numerical Summary Table ---
    print("\n--- Numerical Metrics Summary ---")
    if not metrics_summary or not any(metrics_summary.values()):
        print("  No metrics calculated to display.")
    else:
        # Dynamically get metric names from the first strategy with results
        first_valid_strategy = next((name for name in strategy_names if metrics_summary.get(name)), None)
        if first_valid_strategy:
            # Define preferred order or use keys directly
            metric_names_print = [
                 'CVaR (5%, % Loss)', 'Stress Pass Rate (%)',
                 'OOP Burden Mean (%)', 'OOP Burden Median (%)', 'OOP Burden Max (%)', 'OOP Burden Violation Rate (%)',
                 'Surplus Coverage Mean (x)', 'Surplus Coverage Median (x)', 'Surplus Coverage Min (x)', 'Surplus Coverage Violation Rate (%)',
                 f'Annuity Adequacy Mean (Age {ANNUITY_ADEQUACY_TARGET_AGE}+, %)', f'Annuity Adequacy Median (Age {ANNUITY_ADEQUACY_TARGET_AGE}+, %)', f'Annuity Adequacy Pct Years Adequate (Age {ANNUITY_ADEQUACY_TARGET_AGE}+, %)',
                 'Average QALY', 'Average Final Estate ($)'
            ]
            # Filter out metrics that might not have been calculated if Annuity Adequacy failed etc.
            metric_names_print = [m for m in metric_names_print if m in metrics_summary[first_valid_strategy]]

            # Determine column width dynamically
            max_name_len = max(len(name) for name in strategy_names)
            col_width = max(max_name_len + 2, 18) # Ensure minimum width

            header = f"{'Metric':<55}" + "".join([f"{name:>{col_width}}" for name in strategy_names])
            print("=" * len(header));
            print(header);
            print("-" * len(header))
            for metric in metric_names_print:
                row = f"{metric:<55}"
                for name in strategy_names:
                    value = metrics_summary.get(name, {}).get(metric, np.nan) # Safe access
                    fmt = ".1f" # Default format
                    if "Rate" in metric or "%" in metric:
                        fmt = ".1f" # Keep CVaR as %.1f
                    elif "Coverage" in metric or "(x)" in metric:
                         fmt = ".2f"
                         if "Min" in metric and value == -np.inf: value = "-inf" # Display infinity nicely
                    elif "Estate" in metric or "($)" in metric:
                         fmt = ",.0f" # Format estate
                    elif "QALY" in metric:
                         fmt = ".2f" # Format QALY
                    elif "Max" in metric and value == np.inf:
                         value = "+inf" # Display infinity nicely
                    else: # Handles Mean/Median OOP Burden, Annuity Adequacy Mean/Median
                         fmt = ".1f"

                    # Handle NaN and Inf for printing
                    if isinstance(value, str): # Already formatted as inf/-inf
                        display_val = value
                    elif np.isnan(value):
                        display_val = "NaN"
                    elif np.isinf(value): # Catch any other Infs
                        display_val = "+inf" if value > 0 else "-inf"
                    else:
                        display_val = f"{value:{fmt}}"

                    # Add % sign back for display where appropriate
                    if "%" in metric and display_val not in ["NaN", "+inf", "-inf"]:
                         display_val += "%"

                    row += f"{display_val:>{col_width}}"
                print(row)
            print("=" * len(header))
        else:
            print("  No valid metrics found for any strategy.")

    # --- Clean up ---
    print("\nClosing evaluation environment...")
    if 'eval_env' in locals() and eval_env is not None:
        try:
            eval_env.close()
        except Exception as e:
            print(f"Error closing environment: {e}")
    print("\nEvaluation finished.")

