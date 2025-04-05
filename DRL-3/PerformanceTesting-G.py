# coding=utf-8
import numpy as np
import os
import torch
import wandb
import time
import gymnasium as gym
from gymnasium import spaces
from scipy.special import softmax  # For normalizing actions
import argparse  # To control mode (train/evaluate)

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
# Import SubprocVecEnv for true parallelism
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.vec_env import VecEnv

from wandb.integration.sb3 import WandbCallback as OfficialWandbCallback  # Use the official one

# ===-----------------------------------------------------------------------===
#    CONTROL FLAGS (Set which part of the script to run)
# ===-----------------------------------------------------------------------===
RUN_TRAINING = True  # Set to True to run the training process
RUN_EVALUATION = True  # Set to True to run the evaluation after training (or independently if model exists)
# If RUN_EVALUATION is True and RUN_TRAINING is False, specify a pre-existing model directory:
PRE_TRAINED_MODEL_DIR = None  # e.g., "models/1712290349_Full_Embedded_HealthShocks_Bequest_V4" # Set if not training

# ===-----------------------------------------------------------------------===
#    CONFIGURATION PARAMETERS (V4 - Reduced End Age & Bequest)
# ===-----------------------------------------------------------------------===

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
END_AGE = 100  # Reduced from 120
TIME_HORIZON = END_AGE - START_AGE  # T = 35 steps (Updated)

BASELINE_LIVING_EXPENSE = 40000.0  # Example: $40k/year non-healthcare needs

# --- Survival Probability Parameters ---
SURVIVAL_A = 0.00005
SURVIVAL_K = 5.6
SURVIVAL_G = 0.055
SURVIVAL_C = 0.0001

# --- Reward/Penalty Coefficients ---
GAMMA_W = 6.0  # Terminal Wealth Reward Scaling (if alive at END_AGE)
# **key point**: Add Bequest Reward Scaling (if dead before END_AGE)
GAMMA_B = 3.0  # Bequest Wealth Reward Scaling (e.g., half of GAMMA_W)

THETA = 0.05  # Risky Asset Holding Penalty
BETA_D = 0.5  # Drawdown & Shortfall Base Penalty
QALY_NORMAL_YEAR = 0.7  # QALY value in a year with no adverse health event
QALY_SHOCK_UNTREATED = -0.5  # QALY value if shock occurs and shortfall prevents treatment
SHORTFALL_PENALTY_MULTIPLIER = 3.0  # Multiplier for shortfall penalty based on BETA_D

ANNUITY_PURCHASE_BONUS = 5.0  # Small reward for buying an optional annuity

# --- Health Shock Configuration ---
HEALTH_SHOCKS_CONFIG = [
    {"name": "Minor", "cost": 5000, "qaly_outcome": 0.5, "base_prob": 0.10},  # e.g., treatable infection
    {"name": "Moderate", "cost": 25000, "qaly_outcome": 0.2, "base_prob": 0.05},
    # e.g., non-major surgery, chronic management
    {"name": "Major", "cost": 80000, "qaly_outcome": 0.0, "base_prob": 0.02},  # e.g., major surgery, serious illness
]
# **key point**: Probability multiplier based on age (simple linear increase example)
# Note: END_AGE change affects this slope slightly
HEALTH_SHOCK_AGE_FACTOR = lambda age: 1.0 + max(0, (age - START_AGE) / (END_AGE - START_AGE)) * 1.5 if (
                                                                                                                   END_AGE - START_AGE) > 0 else 1.0

# --- Numerical Stability/Clipping ---
MAX_PENALTY_MAGNITUDE_PER_STEP = 100.0
MAX_TERMINAL_REWARD_MAGNITUDE = 1000.0  # For GAMMA_W
MAX_BEQUEST_REWARD_MAGNITUDE = MAX_TERMINAL_REWARD_MAGNITUDE / 2  # Separate clip for bequest
WEALTH_FLOOR = 1.0
EPSILON = 1e-8

# --- Asset Indices Definitions ---
RISKY_ASSET_INDICES = list(range(NUM_STOCKS + NUM_COMMODITIES + NUM_ETFS))
CRITICAL_ASSET_INDEX = NUM_STOCKS + NUM_COMMODITIES  # Index for drawdown penalty calculation

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

# --- Training Run Identification ---
RUN_NAME_SUFFIX = "Full_Embedded_HealthShocks_Bequest_V4"  # Updated suffix
# MODEL_SAVE_DIR will be set dynamically in train_model() if training is run

# --- WandB Configuration ---
WANDB_PROJECT_NAME = "DMO_RetPortfolio_V4"
WANDB_ENTITY = ''  # Your W&B username or team name, or None for default

# --- Training & PPO Configuration ---
TOTAL_TRAINING_TIMESTEPS = 5_000_000  # Adjust as needed
# **key point**: Utilize available CPU cores
NUM_PARALLEL_ENVS = os.cpu_count() if os.cpu_count() else 16
NUM_PARALLEL_ENVS = max(1, NUM_PARALLEL_ENVS)  # Ensure at least 1
SEED = 42

# Check for CUDA availability
if torch.cuda.is_available():
    print(f"CUDA (GPU) available! Using device: cuda ({torch.cuda.get_device_name(0)})")
    DEVICE = "cuda"
else:
    print("CUDA not available, using CPU.")
    DEVICE = "cpu"

# PPO Hyperparameters
PPO_CONFIG = {
    "learning_rate": 1e-4,
    "n_steps": 2048,  # Steps per env per update
    "batch_size": 512,  # Minibatch size for optimization
    "n_epochs": 10,  # Optimization epochs per update
    "gamma": 0.99,  # Discount factor for future rewards
    "gae_lambda": 0.95,  # Factor for Generalized Advantage Estimation
    "clip_range": 0.2,  # Clipping parameter for PPO policy gradient loss
    "ent_coef": 0.005,  # Entropy coefficient for exploration
    "vf_coef": 0.5,  # Value function coefficient in loss
    "max_grad_norm": 0.5,  # Max gradient norm clipping
    "device": DEVICE,  # Use GPU if available
}

# Training Evaluation and Logging Settings
EVAL_FREQ_TOTAL_STEPS = 50_000  # Evaluate every N *total* steps across all envs
N_EVAL_EPISODES = 30  # Number of episodes for each evaluation during training
LOG_INTERVAL = 20  # Log training stats every N updates

# === Evaluation Configuration (Added Section) ===
N_EVAL_SIMULATIONS = 1000  # Number of episodes to simulate for final metrics
CVAR_PERCENTILE = 0.05  # For Conditional Value at Risk (worst 5%)
STRESS_TEST_FLOOR_PCT = 0.70  # Wealth floor relative to initial wealth
OOP_BURDEN_TARGET = 0.15  # Target Out-of-Pocket Burden Ratio (<=15%)
SURPLUS_COVERAGE_TARGET = 1.2  # Target Surplus Coverage Ratio (>=1.2x)
ANNUITY_ADEQUACY_TARGET_AGE = 85  # Age threshold for annuity adequacy
ANNUITY_ADEQUACY_TARGET_RATIO = 0.80  # Target ratio (>=80%)


# ===-----------------------------------------------------------------------===
#    RETIREMENT ENVIRONMENT DEFINITION (V4 - With Bequest Reward)
# ===-----------------------------------------------------------------------===

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

        # Asset counts (allow passing or calculate from constants)
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
        self.shortfall_penalty_multiplier = float(
            kwargs.get('shortfall_penalty_multiplier', SHORTFALL_PENALTY_MULTIPLIER))

        # Health Shock config
        self.health_shocks_config = kwargs.get('health_shocks_config', HEALTH_SHOCKS_CONFIG)
        self.health_shock_age_factor = kwargs.get('health_shock_age_factor', HEALTH_SHOCK_AGE_FACTOR)

        # Asset Indices
        self.risky_asset_indices = list(kwargs.get('risky_asset_indices', RISKY_ASSET_INDICES))
        self.critical_asset_index = int(kwargs.get('critical_asset_index', CRITICAL_ASSET_INDEX))

        # Forecast Noise
        self.forecast_return_noise_1yr = float(
            kwargs.get('forecast_return_noise_factor_1yr', FORECAST_RETURN_NOISE_FACTOR_1YR))
        self.forecast_return_noise_2yr = float(
            kwargs.get('forecast_return_noise_factor_2yr', FORECAST_RETURN_NOISE_FACTOR_2YR))
        self.forecast_mortality_noise_1yr = float(
            kwargs.get('forecast_mortality_noise_1yr', FORECAST_MORTALITY_NOISE_1YR))
        self.forecast_mortality_noise_2yr = float(
            kwargs.get('forecast_mortality_noise_2yr', FORECAST_MORTALITY_NOISE_2YR))

        # Annuities
        self.compulsory_annuity_payment = float(kwargs.get('compulsory_annuity_payment', COMPULSORY_ANNUITY_PAYMENT))
        self.opt_annuity_1_premium = float(kwargs.get('opt_annuity_1_premium', OPTIONAL_ANNUITY_1_PREMIUM))
        self.opt_annuity_1_payment = float(kwargs.get('opt_annuity_1_payment', OPTIONAL_ANNUITY_1_PAYMENT))
        self.opt_annuity_2_premium = float(kwargs.get('opt_annuity_2_premium', OPTIONAL_ANNUITY_2_PREMIUM))
        self.opt_annuity_2_payment = float(kwargs.get('opt_annuity_2_payment', OPTIONAL_ANNUITY_2_PAYMENT))

        self.verbose = int(kwargs.get('verbose', 0))

        # --- Define Action Space ---
        action_dim = self.total_assets + 2  # Portfolio weights + 2 annuity decisions
        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(action_dim,), dtype=np.float32)

        # --- Define Observation Space ---
        obs_dim = 1 + 1 + self.total_assets + self.total_assets * 2 + 2 + 2  # Age, LogWealth, Weights, 2xReturns, 2xMortality, 2xAnnuityOwned
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # --- Internal State ---
        self.current_age = self.start_age
        self.current_wealth = self.initial_wealth
        self.current_weights = np.ones(self.total_assets, dtype=np.float32) / self.total_assets
        self.current_step = 0
        self.peak_wealth_critical_asset = 0.0  # For drawdown penalty
        self.is_alive = True
        self.has_opt_annuity_1 = False
        self.has_opt_annuity_2 = False
        self.reward_components = {}
        self.current_health_shock = None  # Stores dict of the shock if one occurs in step

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_age = self.start_age
        self.current_wealth = self.initial_wealth
        self.current_weights = np.ones(self.total_assets, dtype=np.float32) / self.total_assets
        self.current_step = 0
        self.is_alive = True
        self.has_opt_annuity_1 = False
        self.has_opt_annuity_2 = False
        self.current_health_shock = None  # Reset health shock

        # Calculate initial peak wealth for drawdown
        initial_critical_asset_value = self.current_wealth * self.current_weights[
            self.critical_asset_index] if self.critical_asset_index < self.total_assets else EPSILON
        self.peak_wealth_critical_asset = max(EPSILON, initial_critical_asset_value)

        # **key point**: Initialize all reward components including bequest
        self.reward_components = {
            'qaly_comp': 0.0, 'dd_comp': 0.0, 'risky_comp': 0.0,
            'terminal_wealth_reward_comp': 0.0, 'shortfall_comp': 0.0,
            'annuity_bonus_comp': 0.0,
            'bequest_reward_comp': 0.0  # Added bequest component
        }

        observation = self._get_obs()
        info = self._get_info()

        # Safety check for NaN/Inf in initial observation
        if np.any(np.isnan(observation)) or np.any(np.isinf(observation)):
            print(f"!!! WARNING: NaN/Inf detected in initial observation: {observation} !!!")
            observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # Ensure observation matches space if needed (especially for Monitor/VecEnv)
        if not self.observation_space.contains(observation):
            print(
                f"!!! WARNING: Initial observation {observation.shape} does not match space {self.observation_space.shape} !!!")
            # Attempt to fix - common issue is float64 vs float32
            observation = observation.astype(np.float32)
            if not self.observation_space.contains(observation):
                # If still mismatching, might need padding or clipping depending on the cause
                print(f"!!! ERROR: Observation mismatch persists after type cast. Check dimensions/bounds. !!!")
                # Add more specific fixing logic if the cause is known

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
        bequest_reward_comp = 0.0  # Initialize bequest for the step
        healthcare_cost_this_step = 0.0  # Hs,t
        self.current_health_shock = None  # Reset shock status for the step

        # --- 0. Check for Health Shock ---
        if self.is_alive:  # Only check for shocks if alive at start of step
            shock_occurred, shock_details = self._generate_health_shock()
            if shock_occurred:
                self.current_health_shock = shock_details
                healthcare_cost_this_step = shock_details['cost']
                if self.verbose > 0: print(
                    f"Step {self.current_step} Age {self.current_age}: Health Shock '{shock_details['name']}' occurred! Cost: {healthcare_cost_this_step}")

        # --- 1. Portfolio Rebalancing & Annuity Purchase ---
        flat_action = action.flatten()  # Ensure action is 1D

        # Handle potential dimension mismatch if action space is simplified in baseline
        if len(flat_action) < self.total_assets + 2:
            # Example: Baseline might only provide portfolio weights
            # Pad with 'do not buy annuity' signals
            portfolio_action = flat_action[:self.total_assets]
            buy_opt_1_signal = -10.0  # Default don't buy
            buy_opt_2_signal = -10.0  # Default don't buy
            if len(flat_action) > self.total_assets: buy_opt_1_signal = flat_action[self.total_assets]
            if len(flat_action) > self.total_assets + 1: buy_opt_2_signal = flat_action[self.total_assets + 1]
        else:
            portfolio_action = flat_action[:self.total_assets]
            buy_opt_1_signal = flat_action[self.total_assets]
            buy_opt_2_signal = flat_action[self.total_assets + 1]

        # Use softmax for portfolio weights regardless of input action scale
        target_weights = softmax(portfolio_action).astype(np.float32)
        target_weights /= (np.sum(target_weights) + EPSILON)  # Normalize
        self.current_weights = target_weights

        purchase_cost = 0.0
        # Check annuity signals - use a threshold (e.g., > 0)
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

        wealth_after_expenses = WEALTH_FLOOR  # Default if shortfall
        if experienced_shortfall:
            shortfall_pct = shortfall_amount / (total_required_outflow + EPSILON)
            raw_shortfall_penalty = -self.beta_d * shortfall_pct * self.shortfall_penalty_multiplier
            shortfall_comp = np.clip(raw_shortfall_penalty, -MAX_PENALTY_MAGNITUDE_PER_STEP, 0)
            step_reward += shortfall_comp  # Apply penalty
            if self.verbose > 0: print(
                f"Step {self.current_step}: SHORTFALL of {shortfall_amount:.2f}. Penalty: {shortfall_comp:.2f}")
            # Wealth is effectively floored due to shortfall
        else:
            # Deduct expenses normally
            wealth_after_expenses = max(WEALTH_FLOOR, wealth_before_expenses - total_required_outflow)

        # Update wealth post-expenses (or floor if shortfall)
        self.current_wealth = wealth_after_expenses

        # --- 5. Check Survival ---
        # This determines the 'is_alive' status for the *next* step and terminal conditions
        survival_prob_step = self._survival_prob(self.current_age)
        if not hasattr(self, 'np_random'): self.np_random, _ = gym.utils.seeding.np_random(None)
        # Store if alive *before* the check for calculating this step's QALY correctly
        alive_at_start_of_survival_check = self.is_alive
        if alive_at_start_of_survival_check and (self.np_random.random() > survival_prob_step):
            self.is_alive = False  # Agent dies this step
            if self.verbose > 0: print(f"Step {self.current_step} Age {self.current_age}: Agent did not survive.")

        # --- 6. Calculate Step Reward Components (QALY, Risky, Drawdown) ---
        # These depend on the state *during* the year (i.e., before potential death this step)

        if alive_at_start_of_survival_check:  # Only accrue QALY/penalties if alive entering the year
            # a) QALY Reward
            if self.current_health_shock:
                # If shock occurred, QALY depends on whether treatment was affordable (no shortfall)
                qaly_comp = self.current_health_shock[
                    'qaly_outcome'] if not experienced_shortfall else self.qaly_shock_untreated
            else:  # Normal year
                qaly_comp = self.qaly_normal_year
            # Apply penalty if shortfall occurred, potentially overriding shock outcome
            if experienced_shortfall:
                qaly_comp = self.qaly_shock_untreated  # Assign worst QALY on any shortfall
            step_reward += qaly_comp

            # b) Risky Asset Holding Penalty
            fraction_in_risky = np.sum(self.current_weights[self.risky_asset_indices])
            raw_risky_penalty = -self.theta * fraction_in_risky
            risky_comp = np.clip(raw_risky_penalty, -MAX_PENALTY_MAGNITUDE_PER_STEP, 0)
            step_reward += risky_comp

            # c) Drawdown Penalty (based on wealth before expenses)
            if self.critical_asset_index < self.total_assets:
                wealth_for_dd = wealth_before_expenses  # Use wealth before expenses/potential death
                critical_asset_value = wealth_for_dd * self.current_weights[self.critical_asset_index]
                critical_asset_value = max(EPSILON, critical_asset_value)
                self.peak_wealth_critical_asset = max(self.peak_wealth_critical_asset, critical_asset_value)
                drawdown_pct = (critical_asset_value - self.peak_wealth_critical_asset) / (
                            self.peak_wealth_critical_asset + EPSILON)
                # Only penalize negative drawdown
                raw_dd_penalty = self.beta_d * min(0, drawdown_pct)  # Penalty only if drawdown_pct < 0
                dd_comp = np.clip(raw_dd_penalty, -MAX_PENALTY_MAGNITUDE_PER_STEP, 0)
                step_reward += dd_comp

        # Accumulate components for the episode log
        # Note: shortfall_comp already accumulated if it occurred
        self.reward_components['qaly_comp'] += qaly_comp
        self.reward_components['dd_comp'] += dd_comp
        self.reward_components['risky_comp'] += risky_comp
        self.reward_components['shortfall_comp'] += shortfall_comp

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
            current_final_wealth = max(WEALTH_FLOOR, self.current_wealth)  # Use wealth at end of step
            if self.is_alive:  # Reached end_age/horizon alive
                normalized_terminal_wealth = (current_final_wealth / (self.initial_wealth + EPSILON))
                raw_terminal_reward = normalized_terminal_wealth * self.gamma_w
                terminal_wealth_reward_comp = np.clip(raw_terminal_reward, 0, MAX_TERMINAL_REWARD_MAGNITUDE)
                step_reward += terminal_wealth_reward_comp  # Add to this final step's reward
                self.reward_components['terminal_wealth_reward_comp'] = terminal_wealth_reward_comp
                self.reward_components['bequest_reward_comp'] = 0.0  # Ensure bequest is zero
                if self.verbose > 0: print(
                    f"Step {self.current_step}: TERMINATED (ALIVE). Terminal Reward: {terminal_wealth_reward_comp:.2f}")
            # Check if died this step OR was already dead but episode terminated/truncated
            elif not self.is_alive:  # Died this step or previously
                normalized_bequest_wealth = (current_final_wealth / (self.initial_wealth + EPSILON))
                raw_bequest_reward = normalized_bequest_wealth * self.gamma_b  # Use bequest gamma
                bequest_reward_comp = np.clip(raw_bequest_reward, 0, MAX_BEQUEST_REWARD_MAGNITUDE)  # Use separate clip
                step_reward += bequest_reward_comp  # Add to this final step's reward
                self.reward_components['bequest_reward_comp'] = bequest_reward_comp
                self.reward_components['terminal_wealth_reward_comp'] = 0.0  # Ensure terminal is zero
                if self.verbose > 0: print(
                    f"Step {self.current_step}: TERMINATED (DEAD). Bequest Reward: {bequest_reward_comp:.2f}")

        # --- 9. Get Observation and Info ---
        observation = self._get_obs()
        info = self._get_info(terminated or truncated)  # Pass terminal flag

        # Safety checks for NaN/Inf in reward and observation
        if np.any(np.isnan(observation)) or np.any(np.isinf(observation)):
            print(f"!!! WARNING: NaN/Inf detected in observation: {observation} at step {self.current_step} !!!")
            observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            step_reward = -MAX_PENALTY_MAGNITUDE_PER_STEP * 2  # Heavy penalty

        if np.isnan(step_reward) or np.isinf(step_reward):
            print(f"!!! WARNING: NaN/Inf detected in reward: {step_reward} at step {self.current_step} !!!")
            step_reward = -MAX_PENALTY_MAGNITUDE_PER_STEP * 2  # Heavy penalty

        step_reward = float(step_reward)  # Ensure reward is standard float

        # Ensure observation fits the space (type and shape) before returning
        if not self.observation_space.contains(observation):
            print(
                f"!!! WARNING: Final observation {observation.shape} does not match space {self.observation_space.shape} !!!")
            observation = observation.astype(np.float32)  # Try casting type
            # Add more fixing logic if needed (clipping, reshaping)

        # Ensure terminated/truncated are booleans
        terminated = bool(terminated)
        truncated = bool(truncated)

        # Return state, reward, terminated, truncated, info
        return observation, step_reward, terminated, truncated, info

    def _generate_health_shock(self):
        """Generates a health shock based on configured probabilities and age factor."""
        if not hasattr(self, 'np_random'): self.np_random, _ = gym.utils.seeding.np_random(None)
        # Calculate age factor using the instance's end_age
        age_multiplier = self.health_shock_age_factor(self.current_age)
        rand_val = self.np_random.random()
        cumulative_prob = 0.0
        for shock in self.health_shocks_config:
            adjusted_prob = min(1.0, shock['base_prob'] * age_multiplier)  # Cap prob at 1.0
            cumulative_prob += adjusted_prob
            if rand_val < cumulative_prob:
                return True, shock  # Return the shock details
        return False, None  # No shock occurred

    def _get_obs(self):
        """Construct the observation array including annuity ownership."""
        # Normalize age using the instance's end_age
        norm_age = (self.current_age - self.start_age) / (self.end_age - self.start_age) if (
                                                                                                        self.end_age - self.start_age) > 0 else 0.0
        log_wealth = np.log(max(WEALTH_FLOOR, self.current_wealth))

        # Forecasts
        return_forecast_1yr = self._get_noisy_forecast(base_value=0.05, noise_factor=self.forecast_return_noise_1yr,
                                                       size=self.total_assets)
        return_forecast_2yr = self._get_noisy_forecast(base_value=0.05, noise_factor=self.forecast_return_noise_2yr,
                                                       size=self.total_assets)
        # Ensure mortality forecasts are probabilities (0-1), even with noise
        mortality_forecast_1yr = np.clip(
            self._get_noisy_forecast(base_value=(1.0 - self._survival_prob(self.current_age + 1)),
                                     noise_factor=self.forecast_mortality_noise_1yr, size=1), 0.0, 1.0)
        mortality_forecast_2yr = np.clip(
            self._get_noisy_forecast(base_value=(1.0 - self._survival_prob(self.current_age + 2)),
                                     noise_factor=self.forecast_mortality_noise_2yr, size=1), 0.0, 1.0)

        # Annuity ownership flags
        opt_annuity_1_owned_flag = 1.0 if self.has_opt_annuity_1 else 0.0
        opt_annuity_2_owned_flag = 1.0 if self.has_opt_annuity_2 else 0.0

        obs = np.concatenate([
            np.array([norm_age], dtype=np.float32),
            np.array([log_wealth], dtype=np.float32),
            self.current_weights.astype(np.float32),
            return_forecast_1yr.astype(np.float32),
            return_forecast_2yr.astype(np.float32),
            mortality_forecast_1yr.astype(np.float32),  # Forecast of *mortality* (1-survival)
            mortality_forecast_2yr.astype(np.float32),  # Forecast of *mortality* (1-survival)
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
            "healthcare_cost_in_step": self.current_health_shock['cost'] if self.current_health_shock else 0.0,
        }
        if is_terminal:
            # Populate final components using .get() for safety
            info["final_qaly_comp"] = float(self.reward_components.get('qaly_comp', 0.0))
            info["final_dd_comp"] = float(self.reward_components.get('dd_comp', 0.0))
            info["final_risky_comp"] = float(self.reward_components.get('risky_comp', 0.0))
            info["final_terminal_wealth_reward_comp"] = float(
                self.reward_components.get('terminal_wealth_reward_comp', 0.0))
            info["final_shortfall_comp"] = float(self.reward_components.get('shortfall_comp', 0.0))
            info["final_annuity_bonus_comp"] = float(self.reward_components.get('annuity_bonus_comp', 0.0))
            # **key point**: Add final bequest component
            info["final_bequest_reward_comp"] = float(self.reward_components.get('bequest_reward_comp', 0.0))
            info["final_wealth"] = float(self.current_wealth)

            # Calculate final total reward from components for verification
            # Ensure all values are numeric before summing
            total_comp_reward = sum(v for v in self.reward_components.values() if isinstance(v, (int, float)))
            info["final_total_reward_from_components"] = float(total_comp_reward)

        return info

    def _survival_prob(self, age):
        """Calculate survival probability using Gompertz law."""
        if age < self.start_age: return 1.0  # Assume survival before start age
        if age >= self.end_age: return 0.0  # No survival at or after end age by definition
        # Gompertz hazard rate: lambda(t) = a * exp(g * t) + c
        # Use age relative to some baseline if needed, or absolute age. Let's use absolute age.
        hazard_rate = self.survival_a * np.exp(self.survival_g * age) + self.survival_c
        # Survival probability over one year: S(t+1)/S(t) = exp(-hazard_rate(t)) approx
        prob = np.exp(-hazard_rate)
        return np.clip(prob, 0.0, 1.0)

    def _get_noisy_forecast(self, base_value, noise_factor, size):
        """Generate a noisy forecast around a base value."""
        if not hasattr(self, 'np_random'):
            self.np_random, _ = gym.utils.seeding.np_random(None)  # Ensure RNG is initialized
        noise = self.np_random.normal(loc=0.0, scale=np.abs(base_value * noise_factor) + EPSILON,
                                      size=size)  # Use abs(base_value)
        forecast = base_value + noise
        return forecast

    def _simulate_market_returns(self):
        """Placeholder simulation logic - *only* returns asset returns."""
        # Example placeholder - Replace with your actual financial market simulation
        # Ensure means/vols match TOTAL_ASSETS length
        base_means = [0.08] * NUM_STOCKS + [0.06] * NUM_COMMODITIES + [0.07] * NUM_ETFS + \
                     [0.02] * NUM_GOV_BONDS + [0.035] * NUM_CORP_BONDS + [0.01] * NUM_STABLE_INC
        base_vols = [0.15] * NUM_STOCKS + [0.18] * NUM_COMMODITIES + [0.12] * NUM_ETFS + \
                    [0.03] * NUM_GOV_BONDS + [0.05] * NUM_CORP_BONDS + [0.005] * NUM_STABLE_INC

        mean_returns = np.array(base_means, dtype=np.float32)
        volatilities = np.array(base_vols, dtype=np.float32)

        if not hasattr(self, 'np_random'):
            self.np_random, _ = gym.utils.seeding.np_random(None)  # Ensure RNG is initialized

        asset_returns = self.np_random.normal(loc=mean_returns, scale=volatilities).astype(np.float32)
        return asset_returns

    def close(self):
        """Perform any necessary cleanup."""
        pass  # No specific cleanup needed in this version


# ===-----------------------------------------------------------------------===
#    CUSTOM CALLBACKS (For Training)
# ===-----------------------------------------------------------------------===

# ===-----------------------------------------------------------------------===
#    CUSTOM EVALUATION CALLBACK DEFINITION (CORRECTED INIT & FREQ) - V2 FIX
# ===-----------------------------------------------------------------------===

# Import SB3 VecEnv base class for type checking if needed
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

class EvalCallbackWithComponents(BaseCallback):
    """
    Callback for evaluating the agent and logging reward components from info dict.
    Assumes eval_env is a VecEnv. Handles VecNormalize saving. Logs bequest reward.
    Corrected __init__ to accept pre-vectorized env and uses total steps for eval_freq.
    V2 Fix: Handles potential (1,1) reward shape from VecEnv(n_envs=1).
    """
    def __init__(self, eval_env: VecEnv, n_eval_episodes: int = 5, eval_freq: int = 10000,
                 log_path: str = None, best_model_save_path: str = None,
                 deterministic: bool = True, verbose: int = 1):
        """
        :param eval_env: The vectorized evaluation environment.
        :param n_eval_episodes: The number of episodes to run per evaluation.
        :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback, based on total timesteps.
        :param log_path: Path to save evaluation log files (optional).
        :param best_model_save_path: Path to save the best model according to evaluation (optional).
        :param deterministic: Whether the evaluation should use deterministic actions.
        :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages.
        """
        super().__init__(verbose=verbose)

        # **key point**: Directly assign the provided VecEnv. No internal wrapping needed.
        self.eval_env = eval_env
        # Ensure the eval environment is not accidentally training its normalization stats
        if isinstance(self.eval_env, VecNormalize):
            self.eval_env.training = False
            # **key point**: Also ensure reward normalization is OFF for evaluation clarity
            self.eval_env.norm_reward = False

        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq # Store the frequency (based on total timesteps)
        self.log_path = log_path
        self.best_model_save_path = best_model_save_path
        self.deterministic = deterministic
        self.best_mean_reward = -np.inf
        self.last_eval_step_total = 0 # Track total timesteps for trigger

        # Create directories if they don't exist
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        if self.best_model_save_path is not None:
             # Use exist_ok=True for the directory containing the model file
            os.makedirs(self.best_model_save_path, exist_ok=True)

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        :return: True if training should continue, False to stop training.
        """
        # **key point**: Trigger evaluation based on total environment steps (self.num_timesteps)
        if self.num_timesteps - self.last_eval_step_total >= self.eval_freq:
            self.last_eval_step_total = self.num_timesteps # Update last eval step count

            if self.verbose > 0:
                print(f"\nEvaluating agent at {self.num_timesteps} training steps...")

            all_episode_rewards = []
            all_episode_lengths = []
            all_final_infos = []

            # --- Evaluation loop ---
            try: # Add try-except block for evaluation robustness
                # Handle potential differences in reset() return format based on Gym version
                reset_output = self.eval_env.reset()
                if isinstance(reset_output, tuple) and len(reset_output) == 2:
                    current_obs, reset_info = reset_output # Gym >= 0.26 likely returns obs, info
                else:
                    current_obs = reset_output # Older Gym might just return obs
                    # Note: VecEnv reset might still return obs as shape (1, obs_dim)

                # Ensure current_obs is correctly shaped (n_envs, obs_dim)
                current_obs = np.array(current_obs).reshape(self.eval_env.num_envs, -1)

                episodes_done = 0
                num_envs = self.eval_env.num_envs # Should typically be 1 for eval
                # **key point**: Initialize ep_rewards correctly with shape (n_envs,)
                ep_rewards = np.zeros(num_envs, dtype=np.float32)
                ep_lengths = np.zeros(num_envs, dtype=np.int32)

                while episodes_done < self.n_eval_episodes:
                     # _deterministic action in eval callback
                     with torch.no_grad(): # Reduce memory for evaluation
                        action, _states = self.model.predict(current_obs, deterministic=self.deterministic)

                     # Ensure action has the correct shape if needed, though predict usually handles VecEnv
                     # action = action.reshape(num_envs, -1) # Typically not needed for predict output

                     new_obs, rewards, dones, infos = self.eval_env.step(action)

                     # --- FIX: Ensure rewards is shape (n_envs,) ---
                     # The VecEnv wrappers might return rewards as (n_envs, 1) or even just scalar if n_envs=1 unnormed
                     rewards_flat = np.array(rewards).flatten() # Use flatten() to ensure shape (n_envs,)
                     # ----------------------------------------------

                     if self.verbose > 2: # Add debug prints if needed
                         print(f"  Eval Step Debug: ep_rewards shape={ep_rewards.shape}, rewards shape={np.array(rewards).shape}, rewards_flat shape={rewards_flat.shape}")

                     # **key point**: Add the flattened rewards
                     ep_rewards += rewards_flat
                     ep_lengths += 1

                     # Important: dones is also array of shape (n_envs,)
                     for i in range(num_envs):
                          # Check if the i-th environment is done
                          if dones[i]:
                               if episodes_done < self.n_eval_episodes:
                                    if self.verbose > 1: print(f"  Eval Episode {episodes_done+1} finished. Reward: {ep_rewards[i]:.2f}, Length: {ep_lengths[i]}")
                                    all_episode_rewards.append(ep_rewards[i])
                                    all_episode_lengths.append(ep_lengths[i])

                                    # Extract final info dictionary correctly from the list of infos
                                    # Look for "final_info" provided by Monitor wrapper
                                    final_info = infos[i].get("final_info", {})
                                    # If "final_info" is None (e.g., truncated), use current info
                                    if final_info is None: final_info = infos[i]

                                    all_final_infos.append(final_info)
                                    episodes_done += 1
                               # Reset only the completed environment's state trackers
                               ep_rewards[i] = 0
                               ep_lengths[i] = 0
                     # Update observation - ensure it has the correct shape for the next prediction step
                     current_obs = np.array(new_obs).reshape(self.eval_env.num_envs, -1)

            except Exception as e:
                print(f"!!! ERROR during evaluation loop: {e} !!!")
                import traceback
                traceback.print_exc()
                # Ensure evaluation stops cleanly if an error occurs
                # Log NaNs or zeros? For now, just report error.
                mean_reward = np.nan
                std_reward = np.nan
                mean_ep_length = np.nan
                # Skip logging detailed components if eval failed badly
                self.logger.record("eval/mean_reward", mean_reward)
                self.logger.record("eval/mean_ep_length", mean_ep_length)
                self.logger.dump(step=self.num_timesteps)
                return True # Continue training maybe? Or False to stop? Let's continue.


            # --- Logging --- (Moved outside the try-except for eval loop, but uses results from it)
            if not all_episode_rewards: # Handle case where no episodes finished (e.g., error early)
                 mean_reward = np.nan
                 std_reward = np.nan
                 mean_ep_length = np.nan
                 print("Warning: No complete evaluation episodes finished.")
            else:
                mean_reward = np.mean(all_episode_rewards)
                std_reward = np.std(all_episode_rewards)
                mean_ep_length = np.mean(all_episode_lengths)

            if self.verbose > 0:
                print(f"Eval results ({len(all_episode_rewards)}/{self.n_eval_episodes} episodes finished): Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}, Mean Ep Length: {mean_ep_length:.2f}")

            # Log standard eval metrics
            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/mean_ep_length", mean_ep_length)
            self.logger.record("eval/std_reward", std_reward)
            self.logger.record("eval/count", len(all_episode_rewards))

            # Filter for valid final info dictionaries
            valid_infos = [info for info in all_final_infos if info and isinstance(info, dict)]

            # Helper function to safely extract and calculate mean from info dicts
            def get_mean_component(key, default_val=np.nan):
                vals = [info.get(key, default_val) for info in valid_infos]
                # Ensure values are numeric before processing
                numeric_vals = [v for v in vals if isinstance(v, (int, float))]
                if not numeric_vals: return 0.0 # Return 0 if no valid numeric data found
                # Use nanmean to handle potential NaNs if default_val was NaN and got included
                mean_val = np.nanmean(numeric_vals)
                # Return 0 if mean is NaN (e.g., all inputs were NaN)
                return mean_val if not np.isnan(mean_val) else 0.0

            # Log all reward components
            self.logger.record("eval/mean_qaly_comp", get_mean_component("final_qaly_comp"))
            self.logger.record("eval/mean_dd_comp", get_mean_component("final_dd_comp"))
            self.logger.record("eval/mean_risky_comp", get_mean_component("final_risky_comp"))
            self.logger.record("eval/mean_terminal_wealth_reward_comp", get_mean_component("final_terminal_wealth_reward_comp"))
            self.logger.record("eval/mean_shortfall_comp", get_mean_component("final_shortfall_comp"))
            self.logger.record("eval/mean_annuity_bonus_comp", get_mean_component("final_annuity_bonus_comp"))
            self.logger.record("eval/mean_bequest_reward_comp", get_mean_component("final_bequest_reward_comp")) # Log bequest
            self.logger.record("eval/mean_final_wealth", get_mean_component("final_wealth"))
            self.logger.record("eval/mean_total_reward_from_components", get_mean_component("final_total_reward_from_components"))

            # Log health shock and annuity status from final step info
            shocks_in_final_step = [info.get("health_shock_in_step", "None") != "None" for info in valid_infos]
            healthcare_costs_final_step = [info.get("healthcare_cost_in_step", 0.0) for info in valid_infos]
            # Calculate mean only if list is not empty to avoid NaN warnings
            self.logger.record("eval/frac_health_shock_final_step", np.mean(shocks_in_final_step) if shocks_in_final_step else 0)
            self.logger.record("eval/mean_healthcare_cost_final_step", np.mean(healthcare_costs_final_step) if healthcare_costs_final_step else 0)

            opt1_bought_final = [info.get("has_opt_annuity_1", False) for info in valid_infos]
            opt2_bought_final = [info.get("has_opt_annuity_2", False) for info in valid_infos]
            self.logger.record("eval/frac_bought_opt_annuity_1", np.mean(opt1_bought_final) if opt1_bought_final else 0)
            self.logger.record("eval/frac_bought_opt_annuity_2", np.mean(opt2_bought_final) if opt2_bought_final else 0)

            # Dump all recorded values to the logger (e.g., TensorBoard, WandB)
            self.logger.dump(step=self.num_timesteps)

            # --- Save best model logic ---
            # Only save if the evaluation didn't error and produced a valid mean_reward
            if not np.isnan(mean_reward) and mean_reward > self.best_mean_reward:
                 self.best_mean_reward = mean_reward
                 if self.best_model_save_path is not None:
                      save_path_model = os.path.join(self.best_model_save_path, "best_model.zip")
                      save_path_stats = os.path.join(self.best_model_save_path, "best_model_vecnormalize.pkl")
                      if self.verbose > 0: print(f"New best model found! Saving model to {save_path_model}")
                      self.model.save(save_path_model)
                      # Save VecNormalize statistics if the training env uses it
                      # Need to get the VecNormalize wrapper from the *training* environment
                      training_env = self.model.get_env()
                      # Check if training_env itself is VecNormalize or if it wraps one
                      vec_norm_env = None
                      if isinstance(training_env, VecNormalize):
                          vec_norm_env = training_env
                      elif hasattr(training_env, 'venv') and isinstance(training_env.venv, VecNormalize):
                           vec_norm_env = training_env.venv # Handle case where Monitor wraps VecNormalize

                      if vec_norm_env is not None:
                          if self.verbose > 0: print(f"Saving VecNormalize stats from training env to {save_path_stats}")
                          vec_norm_env.save(save_path_stats)
                      else:
                           if self.verbose > 0: print("Warning: Training environment is not VecNormalize or structure not recognized, stats not saved with best model.")
        return True


class HParamCallback(BaseCallback):
    """
    Saves hyperparameters and metrics at the start of training for WandB's HParams tab.
    """

    def _on_training_start(self) -> None:
        clip_range_val = self.model.clip_range
        if callable(clip_range_val):
            clip_range_val = clip_range_val(0.0)  # Initial value

        hparam_dict = {
            # Algorithm & PPO params
            "algorithm": self.model.__class__.__name__,
            "learning_rate": self.model.learning_rate,
            "gamma": self.model.gamma,
            "gae_lambda": self.model.gae_lambda,
            "ent_coef": self.model.ent_coef,
            "vf_coef": self.model.vf_coef,
            "max_grad_norm": self.model.max_grad_norm,
            "n_steps": self.model.n_steps,
            "batch_size": self.model.batch_size,
            "n_epochs": self.model.n_epochs,
            "clip_range": float(clip_range_val),  # Ensure float
            "policy": "MlpPolicy",
            "action_space_type": "Box (Flattened)",
            # Env/training params
            "total_timesteps": TOTAL_TRAINING_TIMESTEPS,
            "num_parallel_envs": NUM_PARALLEL_ENVS,
            "vec_env_type": SubprocVecEnv.__name__,
            "device": PPO_CONFIG['device'],
            "seed": SEED,
            "end_age": END_AGE,
            # Reward coefficients
            "reward_gamma_w": GAMMA_W,
            "reward_gamma_b": GAMMA_B,  # **key point**: Log bequest gamma
            "reward_theta": THETA,
            "reward_beta_d": BETA_D,
            "reward_annuity_bonus": ANNUITY_PURCHASE_BONUS,
            # Health/QALY/expense params
            "baseline_living_expense": BASELINE_LIVING_EXPENSE,
            "qaly_normal_year": QALY_NORMAL_YEAR,
            "qaly_shock_untreated": QALY_SHOCK_UNTREATED,
            "shortfall_penalty_multiplier": SHORTFALL_PENALTY_MULTIPLIER,
            "num_health_shock_types": len(HEALTH_SHOCKS_CONFIG),
            # Annuity hparams
            "compulsory_annuity_payment": COMPULSORY_ANNUITY_PAYMENT,
            "opt_annuity_1_premium": OPTIONAL_ANNUITY_1_PREMIUM,
            "opt_annuity_1_payment": OPTIONAL_ANNUITY_1_PAYMENT,
            "opt_annuity_2_premium": OPTIONAL_ANNUITY_2_PREMIUM,
            "opt_annuity_2_payment": OPTIONAL_ANNUITY_2_PAYMENT,
        }

        metric_dict = {
            "rollout/ep_rew_mean": 0.0,
            "train/value_loss": 0.0,
            "eval/mean_reward": 0.0,
            "eval/mean_qaly_comp": 0.0,
            "eval/mean_terminal_wealth_reward_comp": 0.0,
            "eval/mean_bequest_reward_comp": 0.0,  # **key point**: Add bequest metric
            "eval/mean_final_wealth": 0.0,
            "eval/mean_annuity_bonus_comp": 0.0,
            "eval/frac_bought_opt_annuity_1": 0.0,
            "eval/frac_bought_opt_annuity_2": 0.0,
            "eval/mean_dd_comp": 0.0,
            "eval/mean_risky_comp": 0.0,
            "eval/mean_shortfall_comp": 0.0,
            "eval/mean_total_reward_from_components": 0.0,
            "eval/frac_health_shock_final_step": 0.0,
            "eval/mean_healthcare_cost_final_step": 0.0,
        }
        try:
            self.logger.record("hparams", HParam(hparam_dict, metric_dict), exclude=("stdout", "log", "json", "csv"))
            print("Hyperparameters logged for WandB HParams tab.")
        except Exception as e:
            print(f"Warning: Failed to record HParams via logger.record: {e}")

    def _on_step(self) -> bool:
        return True


# ===-----------------------------------------------------------------------===
#    HELPER FUNCTION TO GET ENV KWARGS (Used by Training & Evaluation)
# ===-----------------------------------------------------------------------===

def get_env_kwargs():
    """Returns the dictionary of kwargs used to initialize RetirementEnv."""
    env_kwargs = {
        'initial_wealth': INITIAL_WEALTH, 'start_age': START_AGE,
        'end_age': END_AGE,
        'baseline_living_expense': BASELINE_LIVING_EXPENSE,
        'num_stocks': NUM_STOCKS, 'num_commodities': NUM_COMMODITIES, 'num_etfs': NUM_ETFS,
        'num_gov_bonds': NUM_GOV_BONDS, 'num_corp_bonds': NUM_CORP_BONDS, 'num_stable_inc': NUM_STABLE_INC,
        'survival_a': SURVIVAL_A, 'survival_k': SURVIVAL_K, 'survival_g': SURVIVAL_G, 'survival_c': SURVIVAL_C,
        'gamma_w': GAMMA_W, 'gamma_b': GAMMA_B, 'theta': THETA, 'beta_d': BETA_D,
        'annuity_purchase_bonus': ANNUITY_PURCHASE_BONUS,
        'qaly_normal_year': QALY_NORMAL_YEAR, 'qaly_shock_untreated': QALY_SHOCK_UNTREATED,
        'shortfall_penalty_multiplier': SHORTFALL_PENALTY_MULTIPLIER,
        'health_shocks_config': HEALTH_SHOCKS_CONFIG,
        'health_shock_age_factor': HEALTH_SHOCK_AGE_FACTOR,
        'risky_asset_indices': RISKY_ASSET_INDICES,
        'critical_asset_index': CRITICAL_ASSET_INDEX,
        'forecast_return_noise_factor_1yr': FORECAST_RETURN_NOISE_FACTOR_1YR,
        'forecast_return_noise_factor_2yr': FORECAST_RETURN_NOISE_FACTOR_2YR,
        'forecast_mortality_noise_1yr': FORECAST_MORTALITY_NOISE_1YR,
        'forecast_mortality_noise_2yr': FORECAST_MORTALITY_NOISE_2YR,
        'compulsory_annuity_payment': COMPULSORY_ANNUITY_PAYMENT,
        'opt_annuity_1_premium': OPTIONAL_ANNUITY_1_PREMIUM,
        'opt_annuity_1_payment': OPTIONAL_ANNUITY_1_PAYMENT,
        'opt_annuity_2_premium': OPTIONAL_ANNUITY_2_PREMIUM,
        'opt_annuity_2_payment': OPTIONAL_ANNUITY_2_PAYMENT,
        'verbose': 0  # Keep low for training/eval runs unless debugging
    }
    return env_kwargs


# ===-----------------------------------------------------------------------===
#    TRAINING FUNCTION
# ===-----------------------------------------------------------------------===

def train_model():
    """Sets up and runs the PPO training."""
    # Generate model save directory based on current time and suffix
    current_model_save_dir = f"models/{int(time.time())}_{RUN_NAME_SUFFIX}"
    os.makedirs(current_model_save_dir, exist_ok=True)
    print(f"Created save directory for this run: {current_model_save_dir}")

    # --- Print Configuration Summary ---
    print("=" * 50)
    print("--- TRAINING CONFIGURATION (V4 - Health Shocks & Bequest Motive) ---")
    print("=" * 50)
    print(f"Run Suffix: {RUN_NAME_SUFFIX}")
    print(f"Save Directory: {current_model_save_dir}")
    print(f"Total Timesteps: {TOTAL_TRAINING_TIMESTEPS:,}")
    print(f"Parallel Envs: {NUM_PARALLEL_ENVS} (Using {SubprocVecEnv.__name__})")
    print(f"Initial Wealth: {INITIAL_WEALTH:,}")
    # **key point**: Print updated age/horizon
    print(f"**key point**: Time Horizon: {START_AGE}-{END_AGE} ({TIME_HORIZON} years)")
    print(
        f"Assets: S={NUM_STOCKS}, C={NUM_COMMODITIES}, E={NUM_ETFS}, GB={NUM_GOV_BONDS}, CB={NUM_CORP_BONDS}, SI={NUM_STABLE_INC} (Total={TOTAL_ASSETS})")
    print(f"Baseline Living Expense (L): {BASELINE_LIVING_EXPENSE:,}")
    # **key point**: Print GAMMA_W and GAMMA_B
    print(
        f"Reward Coeffs: GAMMA_W={GAMMA_W}, **key point**: GAMMA_B={GAMMA_B}, THETA={THETA}, BETA_D={BETA_D}, SHORTFALL_MULT={SHORTFALL_PENALTY_MULTIPLIER}, ANNUITY_BONUS={ANNUITY_PURCHASE_BONUS}")
    print(f"QALY Params: Normal={QALY_NORMAL_YEAR}, ShockUntreated={QALY_SHOCK_UNTREATED}")
    print(
        f"Health Shocks: {len(HEALTH_SHOCKS_CONFIG)} types configured, AgeFactor Range: ~{HEALTH_SHOCK_AGE_FACTOR(START_AGE):.2f}-{HEALTH_SHOCK_AGE_FACTOR(END_AGE - 1):.2f}")
    print(
        f"Annuities: Compulsory={COMPULSORY_ANNUITY_PAYMENT:,.0f}/yr; Opt1={OPTIONAL_ANNUITY_1_PAYMENT:,.0f}/yr (Premium={OPTIONAL_ANNUITY_1_PREMIUM:,.0f}); Opt2={OPTIONAL_ANNUITY_2_PAYMENT:,.0f}/yr (Premium={OPTIONAL_ANNUITY_2_PREMIUM:,.0f})")
    print(
        f"PPO Config: n_steps={PPO_CONFIG['n_steps']}, batch_size={PPO_CONFIG['batch_size']}, lr={PPO_CONFIG['learning_rate']}, ent_coef={PPO_CONFIG['ent_coef']}")
    print(f"PPO Device: {PPO_CONFIG['device']}")
    print(f"Eval Freq (Training): {EVAL_FREQ_TOTAL_STEPS:,} total steps, N Episodes: {N_EVAL_EPISODES}")
    print("=" * 50)

    # --- Environment Setup ---
    env_kwargs = get_env_kwargs()

    print(f"Creating {NUM_PARALLEL_ENVS} parallel training environments using {SubprocVecEnv.__name__}...")
    # **key point**: Wrap the environment creation with Monitor INSIDE the lambda
    train_env = make_vec_env(
        lambda: Monitor(RetirementEnv(**env_kwargs)),
        n_envs=NUM_PARALLEL_ENVS,
        seed=SEED,
        vec_env_cls=SubprocVecEnv
    )
    print("Normalizing training environment observations...")
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10., gamma=PPO_CONFIG['gamma'])

    print("Creating evaluation environment (n_envs=1) for training callback...")
    # **key point**: Wrap the environment creation with Monitor here too
    eval_env_train_callback = make_vec_env(
        lambda: Monitor(RetirementEnv(**env_kwargs)),
        n_envs=1,
        seed=SEED + NUM_PARALLEL_ENVS,  # Use different seed for eval
        vec_env_cls=DummyVecEnv
    )
    print("Normalizing evaluation environment observations using stats from training env...")
    eval_env_train_callback = VecNormalize(eval_env_train_callback, norm_obs=True, norm_reward=False, clip_obs=10.,
                                           training=False, gamma=PPO_CONFIG['gamma'])
    # Load stats from training env
    eval_env_train_callback.obs_rms = train_env.obs_rms

    # --- Initialize WandB ---
    print("Initializing WandB...")
    # **key point**: CORRECTED dictionary merging using **
    combined_config = {
        "policy": "MlpPolicy",
        "action_space_type": "Box (Flattened)",
        "total_timesteps": TOTAL_TRAINING_TIMESTEPS,
        "num_parallel_envs": NUM_PARALLEL_ENVS,
        "vec_env_type": SubprocVecEnv.__name__,
        "seed": SEED,
        **PPO_CONFIG,
        **env_kwargs  # Log env params too
    }
    serializable_config = {k: v for k, v in combined_config.items() if not callable(v)}  # Filter non-serializable

    run = wandb.init(
        project=WANDB_PROJECT_NAME,
        entity=WANDB_ENTITY,
        name=f"{os.path.basename(current_model_save_dir)}",  # Use dir name for WandB run name
        config=serializable_config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    # --- Callback Setup ---
    print("Setting up training callbacks...")
    checkpoint_save_freq_total_steps = EVAL_FREQ_TOTAL_STEPS * 2
    # Ensure save_freq is at least n_steps and calculated correctly based on total steps
    checkpoint_save_freq_per_env = max(1, checkpoint_save_freq_total_steps // NUM_PARALLEL_ENVS)

    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_save_freq_per_env,
        save_path=os.path.join(current_model_save_dir, 'checkpoints'),
        name_prefix='ppo_retire_v4',
        save_replay_buffer=False,
        save_vecnormalize=True
    )
    print(
        f"Checkpoint Callback: Save Freq approx every {checkpoint_save_freq_total_steps:,} total steps ({checkpoint_save_freq_per_env} steps per env)")

    evaluation_callback = EvalCallbackWithComponents(
        eval_env=eval_env_train_callback,
        n_eval_episodes=N_EVAL_EPISODES,
        eval_freq=EVAL_FREQ_TOTAL_STEPS,  # Trigger based on total steps
        log_path=os.path.join(current_model_save_dir, 'eval_logs_training'),
        best_model_save_path=os.path.join(current_model_save_dir, 'best_model'),
        deterministic=True,
        verbose=1,
    )
    print(f"Evaluation Callback (Training): Eval Freq every {EVAL_FREQ_TOTAL_STEPS:,} total steps")

    wandb_callback = OfficialWandbCallback(
        gradient_save_freq=0,
        model_save_path=os.path.join(current_model_save_dir, f"wandb_models/{run.id}"),
        model_save_freq=0,  # Use EvalCallbackWithComponents for best model saving
        verbose=2
    )

    hparam_callback = HParamCallback()
    callback_list = CallbackList([checkpoint_callback, evaluation_callback, hparam_callback, wandb_callback])

    # --- Model Definition ---
    print(f"Defining PPO model with MlpPolicy on device: {PPO_CONFIG['device']}...")
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        seed=SEED,
        tensorboard_log=f"runs/{run.id}",  # Local log dir for WandB sync
        **PPO_CONFIG
    )

    # --- Training ---
    print("=" * 50)
    print(f"--- STARTING TRAINING (V4 - {TOTAL_TRAINING_TIMESTEPS:,} steps) ---")
    print(f"--- Using {NUM_PARALLEL_ENVS} parallel environments ---")
    print(
        f"--- Monitor CPU ({os.cpu_count()} cores) and GPU ({torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'N/A'}) utilization ---")
    print("=" * 50)
    start_time = time.time()
    training_successful = False
    try:
        model.learn(
            total_timesteps=TOTAL_TRAINING_TIMESTEPS,
            callback=callback_list,
            log_interval=LOG_INTERVAL,
            reset_num_timesteps=True
        )
        training_successful = True
    except Exception as e:
        print(f"\n!!! TRAINING ERROR: {e} !!!\n")
        import traceback
        traceback.print_exc()
    finally:
        duration = time.time() - start_time
        print("=" * 50)
        print(f"--- TRAINING FINISHED ---")
        print(f"Duration: {duration:.2f}s | {duration / 60:.2f} min | {duration / 3600:.2f} hr")
        print(f"Model assets saved in: {current_model_save_dir}")
        print("=" * 50)

        if training_successful:
            # --- Save Final Model and Normalization Stats ---
            final_model_path = os.path.join(current_model_save_dir, "final_model.zip")
            final_stats_path = os.path.join(current_model_save_dir, "final_vec_normalize.pkl")
            print(f"Saving final model to: {final_model_path}")
            model.save(final_model_path)
            print(f"Saving final VecNormalize stats to: {final_stats_path}")
            _train_env = model.get_env()
            if _train_env and isinstance(_train_env, VecNormalize):
                _train_env.save(final_stats_path)
            elif _train_env and hasattr(_train_env, 'venv') and isinstance(_train_env.venv, VecNormalize):
                _train_env.venv.save(final_stats_path)
            else:
                print(f"Warning: Could not save final VecNormalize stats (env type: {type(_train_env)})")
        else:
            print("Skipping final model save due to training error.")

        # --- Clean up ---
        print("Closing training environments...")
        try:
            if 'train_env' in locals() and train_env is not None: train_env.close()
            if 'eval_env_train_callback' in locals() and eval_env_train_callback is not None: eval_env_train_callback.close()
        except Exception as e:
            print(f"Error closing training environments: {e}")

        if run:
            print("Finishing WandB run...")
            run.finish()

    return current_model_save_dir if training_successful else None


# ===-----------------------------------------------------------------------===
#    EVALUATION SECTION (Strategies, Simulation, Metrics)
# ===-----------------------------------------------------------------------===

# === Strategy Definitions ===

def ppo_strategy(model, obs, deterministic=True):
    """Get action from the loaded PPO model."""
    action, _ = model.predict(obs, deterministic=deterministic)
    return action


def static_60_40_strategy(obs, env_unwrapped):
    """
    Fixed 60% risky, 40% less risky allocation. No optional annuities.
    Outputs raw action values (pre-softmax for portfolio).
    Requires unwrapped env instance for direct access to parameters like total_assets.
    """
    target_weights = np.zeros(env_unwrapped.total_assets, dtype=np.float32)

    # Use risky asset indices defined in the env instance
    less_risky_indices = list(set(range(env_unwrapped.total_assets)) - set(env_unwrapped.risky_asset_indices))

    risky_weight = 0.60
    less_risky_weight = 1.0 - risky_weight

    # Distribute weights evenly within categories
    if len(env_unwrapped.risky_asset_indices) > 0:
        target_weights[env_unwrapped.risky_asset_indices] = risky_weight / len(env_unwrapped.risky_asset_indices)
    if len(less_risky_indices) > 0:
        target_weights[less_risky_indices] = less_risky_weight / len(less_risky_indices)

    target_weights /= (np.sum(target_weights) + EPSILON)  # Ensure normalization

    # Use target weights directly as action proxy (softmax happens in env.step)
    portfolio_action = target_weights * 10.0  # Scale up values

    # Annuity actions: Large negative values to signify "don't buy"
    annuity_action = np.array([-10.0, -10.0], dtype=np.float32)

    action = np.concatenate([portfolio_action, annuity_action])
    # Ensure action matches the env's action space shape if necessary
    if action.shape != env_unwrapped.action_space.shape:
        # Pad or truncate if needed, though should match total_assets + 2
        padded_action = np.full(env_unwrapped.action_space.shape, -10.0, dtype=np.float32)
        len_to_copy = min(len(action), len(padded_action))
        padded_action[:len_to_copy] = action[:len_to_copy]
        action = padded_action

    return action


def age_based_glidepath_strategy(obs, env_unwrapped):
    """
    Glidepath: Risky allocation = max(0.1, min(1.0, (105 - age) / 100)). No optional annuities.
    Outputs raw action values. Uses unwrapped env for parameters and needs obs for age.
    """
    # Extract normalized age from observation and denormalize using env parameters
    norm_age = obs[0]  # Assuming age is the first element
    current_age = int(norm_age * (env_unwrapped.end_age - env_unwrapped.start_age) + env_unwrapped.start_age)

    # Calculate target risky weight (e.g., 105-Age rule)
    target_risky_weight = max(0.1, min(1.0, (105.0 - current_age) / 100.0))
    target_less_risky_weight = 1.0 - target_risky_weight

    target_weights = np.zeros(env_unwrapped.total_assets, dtype=np.float32)
    less_risky_indices = list(set(range(env_unwrapped.total_assets)) - set(env_unwrapped.risky_asset_indices))

    if len(env_unwrapped.risky_asset_indices) > 0:
        target_weights[env_unwrapped.risky_asset_indices] = target_risky_weight / len(env_unwrapped.risky_asset_indices)
    if len(less_risky_indices) > 0:
        target_weights[less_risky_indices] = target_less_risky_weight / len(less_risky_indices)

    target_weights /= (np.sum(target_weights) + EPSILON)  # Normalize

    # Use target weights directly as action proxy
    portfolio_action = target_weights * 10.0

    # Annuity actions: Don't buy
    annuity_action = np.array([-10.0, -10.0], dtype=np.float32)

    action = np.concatenate([portfolio_action, annuity_action])
    # Ensure action matches the env's action space shape
    if action.shape != env_unwrapped.action_space.shape:
        padded_action = np.full(env_unwrapped.action_space.shape, -10.0, dtype=np.float32)
        len_to_copy = min(len(action), len(padded_action))
        padded_action[:len_to_copy] = action[:len_to_copy]
        action = padded_action
    return action


# === Simulation Function ===

def run_simulations(strategy_func, eval_env, n_simulations, model=None):
    """Runs n_simulations for a given strategy and returns collected data."""
    all_results = []

    # Get the initial wealth directly from the unwrapped env instance
    env_unwrapped = eval_env.envs[0].unwrapped
    initial_wealth_sim = env_unwrapped.initial_wealth  # Use for stress test calc

    for i in range(n_simulations):
        if i % (max(1, n_simulations // 10)) == 0:  # Print progress periodically
            print(f"  Simulating episode {i + 1}/{n_simulations}...")

        episode_data = {
            "steps": [],
            "final_wealth": np.nan,  # Initialize with NaN
            "final_age": np.nan,
            "survived": False,
            "total_reward": 0,
            "hit_stress_floor": False,
            "final_info": {}
        }

        # Reset environment
        try:
            obs = eval_env.reset()
        except Exception as e:
            print(f"Error during env reset: {e}")
            continue  # Skip simulation if reset fails

        if isinstance(obs, tuple):  # Handle (obs, info) tuple if returned by reset
            obs = obs[0]
        # Ensure obs is numpy array and has batch dimension for model.predict
        if not isinstance(obs, np.ndarray): obs = np.array(obs)
        if len(obs.shape) == 1: obs = obs.reshape(1, -1)

        terminated = truncated = False
        episode_reward_sum = 0.0
        min_wealth_in_episode = initial_wealth_sim  # Use actual initial wealth

        while not (terminated or truncated):
            # Get action from strategy
            if strategy_func == ppo_strategy:
                action = strategy_func(model, obs)
            else:
                # Baselines need unwrapped env and potentially normalized obs[0]
                action = strategy_func(obs[0], eval_env.envs[0].unwrapped)

            # Step environment
            try:
                new_obs, reward, terminated_flag, truncated_flag, info = eval_env.step(action)
                # Ensure flags are single booleans if envs=1
                terminated = bool(terminated_flag[0]) if isinstance(terminated_flag, (list, np.ndarray)) else bool(
                    terminated_flag)
                truncated = bool(truncated_flag[0]) if isinstance(truncated_flag, (list, np.ndarray)) else bool(
                    truncated_flag)
                current_info = info[0] if isinstance(info, (list, np.ndarray)) else info  # Get dict for single env

            except Exception as e:
                print(f"Error during env step: {e}")
                # Attempt to get final state if possible, otherwise mark as error
                current_env_state = eval_env.envs[0].unwrapped
                episode_data["final_wealth"] = current_env_state.current_wealth
                episode_data["final_age"] = current_env_state.current_age
                episode_data["survived"] = current_env_state.is_alive
                terminated = True  # Force termination on error
                break

            # Ensure new_obs is numpy array and has batch dimension
            if not isinstance(new_obs, np.ndarray): new_obs = np.array(new_obs)
            if len(new_obs.shape) == 1: new_obs = new_obs.reshape(1, -1)

            # --- Data Collection per Step ---
            current_env_state = eval_env.envs[0].unwrapped  # Access current state
            step_age = current_env_state.current_age - 1  # Age *during* the year just completed
            step_wealth = current_env_state.current_wealth  # Wealth at *end* of step
            # Get cost from info dict provided by Monitor wrapper if possible
            step_medical_cost = current_info.get("healthcare_cost_in_step", 0.0)
            step_income = current_env_state.compulsory_annuity_payment + \
                          (current_env_state.opt_annuity_1_payment if current_env_state.has_opt_annuity_1 else 0) + \
                          (current_env_state.opt_annuity_2_payment if current_env_state.has_opt_annuity_2 else 0)

            episode_data["steps"].append({
                "age": step_age,
                "wealth": step_wealth,
                "income": step_income,
                "medical_cost": step_medical_cost,
                "baseline_expense": current_env_state.baseline_living_expense,
                "has_opt1": current_env_state.has_opt_annuity_1,
                "has_opt2": current_env_state.has_opt_annuity_2
            })

            min_wealth_in_episode = min(min_wealth_in_episode, step_wealth)
            obs = new_obs
            # Accumulate reward (handle scalar if envs=1)
            current_reward = reward[0] if isinstance(reward, (list, np.ndarray)) else reward
            episode_reward_sum += current_reward

            if terminated or truncated:
                final_info = current_info.get("final_info", current_info)  # Get final info from Monitor if wrapped
                episode_data["final_wealth"] = final_info.get("wealth", current_env_state.current_wealth)
                episode_data["final_age"] = final_info.get("age", current_env_state.current_age)
                episode_data["survived"] = final_info.get("is_alive", current_env_state.is_alive)
                episode_data["total_reward"] = episode_reward_sum
                episode_data["final_info"] = final_info

                stress_floor_value = initial_wealth_sim * STRESS_TEST_FLOOR_PCT
                if min_wealth_in_episode < stress_floor_value:
                    episode_data["hit_stress_floor"] = True
                break  # Exit while loop

        all_results.append(episode_data)

    print(f"  Finished {n_simulations} simulations.")
    return all_results


# === Metric Calculation Functions ===

def calculate_cvar(results, initial_wealth, percentile=0.05):
    """Calculates CVaR based on final wealth relative loss."""
    final_wealths = np.array(
        [r['final_wealth'] for r in results if r and 'final_wealth' in r and np.isfinite(r['final_wealth'])])
    if len(final_wealths) == 0: return np.nan

    relative_losses = (initial_wealth - final_wealths) / (initial_wealth + EPSILON)
    threshold_index = int(np.ceil(len(relative_losses) * percentile))  # Index for the worst losses

    if threshold_index == 0: return 0.0  # No losses or only one data point

    # Sort losses descending (worst first)
    sorted_losses = np.sort(relative_losses)[::-1]
    worst_losses = sorted_losses[:threshold_index]

    if len(worst_losses) == 0: return 0.0

    cvar = np.mean(worst_losses)
    return cvar * 100  # Percentage loss


def calculate_stress_test_pass_rate(results):
    """Calculates the percentage of simulations that did NOT hit the stress floor."""
    if not results: return np.nan
    valid_results = [r for r in results if r]  # Filter out potential None results if sim failed
    if not valid_results: return np.nan
    times_floor_hit = sum(1 for r in valid_results if r.get('hit_stress_floor', False))
    pass_rate = 1.0 - (times_floor_hit / len(valid_results))
    return pass_rate * 100  # Percentage


def calculate_oop_burden_metrics(results):
    """Calculates metrics related to Out-of-Pocket Medical Burden."""
    all_burden_ratios = []
    n_shock_years_with_income = 0  # Count years with shock AND positive income
    n_burden_violations = 0  # Count violations *in those years*

    for r in results:
        if not r: continue  # Skip failed simulations
        for step_data in r.get('steps', []):
            medical_cost = step_data.get('medical_cost', 0.0)
            income = step_data.get('income', 0.0)

            # Only calculate burden if there was a medical cost
            if medical_cost > EPSILON:
                if income > EPSILON:
                    n_shock_years_with_income += 1
                    burden_ratio = medical_cost / income
                    all_burden_ratios.append(burden_ratio)
                    if burden_ratio > OOP_BURDEN_TARGET:
                        n_burden_violations += 1
                else:
                    # Shock occurred but income was zero - handle as extreme case
                    all_burden_ratios.append(np.inf)
                    # Count as violation if target is finite
                    if np.isfinite(OOP_BURDEN_TARGET):
                        n_burden_violations += 1  # Count zero-income case as violation
                        n_shock_years_with_income += 1  # Include in denominator for rate

    if not all_burden_ratios:
        return {"mean": 0.0, "median": 0.0, "max": 0.0, "violation_rate": 0.0}

    finite_ratios = [b for b in all_burden_ratios if np.isfinite(b)]
    mean_burden = np.mean(finite_ratios) if finite_ratios else 0.0
    median_burden = np.median(finite_ratios) if finite_ratios else 0.0
    # Max can be inf if zero income occurred with costs
    max_burden = np.max(all_burden_ratios) if all_burden_ratios else 0.0
    # Violation rate: violations / (years with shock and positive income + zero-income shock years)
    violation_rate = (n_burden_violations / n_shock_years_with_income) if n_shock_years_with_income > 0 else 0.0

    return {
        "mean": mean_burden * 100,  # Pct
        "median": median_burden * 100,  # Pct
        "max": max_burden * 100 if np.isfinite(max_burden) else float('inf'),  # Pct or Inf
        "violation_rate": violation_rate * 100  # Pct of relevant shock years exceeding target
    }


def calculate_surplus_coverage_metrics(results):
    """Calculates metrics related to Surplus Coverage Ratio."""
    all_coverage_ratios = []
    n_violations = 0
    total_steps = 0

    for r in results:
        if not r: continue
        for step_data in r.get('steps', []):
            total_steps += 1
            # Resources: End-of-step wealth + Income received during step
            liquid_assets = step_data.get('wealth', 0.0)
            income = step_data.get('income', 0.0)

            # Required Spending: Baseline + Medical Outlays for the step
            min_required = step_data.get('baseline_expense', BASELINE_LIVING_EXPENSE)  # Use constant if missing
            medical_outlays = step_data.get('medical_cost', 0.0)
            total_required = min_required + medical_outlays

            if total_required < EPSILON:
                coverage_ratio = np.inf  # Favorable undefined case
            else:
                coverage_ratio = (liquid_assets + income) / total_required

            all_coverage_ratios.append(coverage_ratio)
            if np.isfinite(coverage_ratio) and coverage_ratio < SURPLUS_COVERAGE_TARGET:
                n_violations += 1

    if not all_coverage_ratios:
        return {"mean": np.nan, "median": np.nan, "min": np.nan, "violation_rate": np.nan}

    finite_ratios = [r for r in all_coverage_ratios if np.isfinite(r)]
    mean_coverage = np.mean(finite_ratios) if finite_ratios else 0.0
    median_coverage = np.median(finite_ratios) if finite_ratios else 0.0
    min_coverage = np.min(finite_ratios) if finite_ratios else 0.0  # Min of finite values
    violation_rate = (n_violations / total_steps) if total_steps > 0 else 0.0

    return {
        "mean": mean_coverage,
        "median": median_coverage,
        "min": min_coverage,
        "violation_rate": violation_rate * 100  # Pct of steps below target
    }


def calculate_annuity_adequacy_metrics(results):
    """Calculates metrics related to Annuity Adequacy in later life."""
    ratios_in_later_life = []
    n_target_years_lived = 0  # Count person-years lived at or above target age
    n_adequate_years = 0

    for r in results:
        if not r: continue
        # Check if agent survived to the target age at least
        survived_to_target = False
        final_age = r.get('final_age', 0)
        survived_episode = r.get('survived', False)

        # Determine if the target age was reached while alive
        if survived_episode and final_age >= ANNUITY_ADEQUACY_TARGET_AGE:
            survived_to_target = True
        elif not survived_episode and final_age > ANNUITY_ADEQUACY_TARGET_AGE:  # Died after reaching target age
            survived_to_target = True

        if not survived_to_target:
            continue  # Skip simulations that ended before reaching target age alive

        # Iterate through steps for simulations that reached the target age
        for step_data in r.get('steps', []):
            age = step_data.get('age', 0)
            if age >= ANNUITY_ADEQUACY_TARGET_AGE:
                n_target_years_lived += 1  # Count this person-year

                # Calculate guaranteed income based on annuity ownership in that step
                guaranteed_income = COMPULSORY_ANNUITY_PAYMENT + \
                                    (OPTIONAL_ANNUITY_1_PAYMENT if step_data.get('has_opt1', False) else 0) + \
                                    (OPTIONAL_ANNUITY_2_PAYMENT if step_data.get('has_opt2', False) else 0)
                essential_expenses = step_data.get('baseline_expense', BASELINE_LIVING_EXPENSE)

                if essential_expenses < EPSILON:
                    ratio = np.inf  # Favorable
                else:
                    ratio = guaranteed_income / essential_expenses

                ratios_in_later_life.append(ratio)
                if np.isfinite(ratio) and ratio >= ANNUITY_ADEQUACY_TARGET_RATIO:
                    n_adequate_years += 1

    if not ratios_in_later_life:
        print(f"Warning: No simulation person-years found lived at or above age {ANNUITY_ADEQUACY_TARGET_AGE}.")
        return {"mean": np.nan, "median": np.nan, "pct_adequate_years": 0.0}

    finite_ratios = [r for r in ratios_in_later_life if np.isfinite(r)]
    mean_ratio = np.mean(finite_ratios) if finite_ratios else 0.0
    median_ratio = np.median(finite_ratios) if finite_ratios else 0.0
    # Pct adequate: years meeting target / total person-years lived >= target age
    pct_adequate = (n_adequate_years / n_target_years_lived) if n_target_years_lived > 0 else 0.0

    return {
        "mean": mean_ratio * 100,  # Pct
        "median": median_ratio * 100,  # Pct
        "pct_adequate_years": pct_adequate * 100  # Pct of person-years >= target ratio
    }


# ===-----------------------------------------------------------------------===
#    EVALUATION FUNCTION
# ===-----------------------------------------------------------------------===

def evaluate_model(model_save_dir):
    """Loads a trained model and runs the full evaluation against baselines."""
    print("\n" + "=" * 60)
    print("--- Starting Model Evaluation ---")
    print(f"--- Using Model From: {model_save_dir} ---")
    print("=" * 60)

    model_path = os.path.join(model_save_dir, "final_model.zip")
    stats_path = os.path.join(model_save_dir, "final_vec_normalize.pkl")
    best_model_path = os.path.join(model_save_dir, "best_model", "best_model.zip")
    best_stats_path = os.path.join(model_save_dir, "best_model", "best_model_vecnormalize.pkl")

    # Decide which model to load: best or final? Let's prefer 'best'.
    if os.path.exists(best_model_path) and os.path.exists(best_stats_path):
        print("Loading BEST model and stats...")
        load_model_path = best_model_path
        load_stats_path = best_stats_path
    elif os.path.exists(model_path) and os.path.exists(stats_path):
        print("Loading FINAL model and stats...")
        load_model_path = model_path
        load_stats_path = stats_path
    else:
        print(f"!!! ERROR: Model/Stats not found in {model_save_dir} or {os.path.join(model_save_dir, 'best_model')}")
        print("Cannot run evaluation.")
        return

    # --- Load Environment ---
    print(f"Loading environment and normalization stats from: {load_stats_path}")
    env_kwargs_eval = get_env_kwargs()
    # Create a *single* environment for evaluation simulation runs
    eval_env = make_vec_env(
        lambda: Monitor(RetirementEnv(**env_kwargs_eval)),  # Wrap with Monitor
        n_envs=1,
        vec_env_cls=DummyVecEnv
    )
    # Load the normalization stats
    try:
        eval_env = VecNormalize.load(load_stats_path, eval_env)
    except Exception as e:
        print(f"!!! ERROR loading VecNormalize stats: {e}")
        print("    Attempting evaluation without normalization stats (may perform poorly).")
        # Keep the unnormalized eval_env

    eval_env.training = False  # Set to evaluation mode
    eval_env.norm_reward = False  # Typically don't normalize reward for analysis

    # --- Load PPO Model ---
    print(f"Loading trained PPO model from: {load_model_path}")
    try:
        ppo_model = PPO.load(load_model_path, env=eval_env)
    except Exception as e:
        print(f"!!! ERROR loading PPO model: {e}")
        print("Cannot run evaluation.")
        eval_env.close()
        return

    # --- Run Simulations for Each Strategy ---
    results = {}
    strategies = {
        "PPO": (ppo_strategy, ppo_model),
        "Static 60/40": (static_60_40_strategy, None),
        "Glidepath (105-Age)": (age_based_glidepath_strategy, None)
    }

    # Get initial wealth from the properly initialized unwrapped env
    # This ensures it matches the parameters used for normalization/training
    env_instance = eval_env.envs[0].unwrapped
    INITIAL_WEALTH_FROM_ENV = env_instance.initial_wealth

    for name, (strategy_func, model_or_none) in strategies.items():
        print(f"\nRunning {N_EVAL_SIMULATIONS} simulations for strategy: {name}...")
        # Pass the VecNormalize env to run_simulations
        results[name] = run_simulations(strategy_func, eval_env, N_EVAL_SIMULATIONS, model=model_or_none)

    # --- Calculate and Compare Metrics ---
    print("\n" + "=" * 60)
    print("--- Evaluation Results ---")
    print("=" * 60)
    metrics_summary = {name: {} for name in strategies}
    calculation_successful = True

    for name, sim_results in results.items():
        print(f"\nCalculating metrics for: {name}")
        if not sim_results:
            print("  No simulation results found, skipping metrics.")
            metrics_summary[name] = {k: np.nan for k in
                                     metrics_summary.get(list(strategies.keys())[0], {})}  # Fill with NaN
            continue

        try:
            # Objective 1: Downside Risk
            metrics_summary[name]['CVaR (5%, % Loss)'] = calculate_cvar(sim_results, INITIAL_WEALTH_FROM_ENV,
                                                                        CVAR_PERCENTILE)
            metrics_summary[name]['Stress Pass Rate (%)'] = calculate_stress_test_pass_rate(sim_results)

            # Objective 2: Income & Healthcare Shocks
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

            # Objective 3: Longevity Risk
            annuity_metrics = calculate_annuity_adequacy_metrics(sim_results)
            metrics_summary[name][f'Annuity Adequacy Mean (Age {ANNUITY_ADEQUACY_TARGET_AGE}+, %)'] = annuity_metrics[
                'mean']
            metrics_summary[name][f'Annuity Adequacy Median (Age {ANNUITY_ADEQUACY_TARGET_AGE}+, %)'] = annuity_metrics[
                'median']
            metrics_summary[name][f'Annuity Adequacy Pct Years Adequate (Age {ANNUITY_ADEQUACY_TARGET_AGE}+, %)'] = \
            annuity_metrics['pct_adequate_years']

            # Add average final wealth for context
            valid_final_wealths = [r['final_wealth'] for r in sim_results if
                                   r and np.isfinite(r.get('final_wealth', np.nan))]
            avg_final_wealth = np.mean(valid_final_wealths) if valid_final_wealths else np.nan
            metrics_summary[name]['Average Final Wealth'] = avg_final_wealth

        except Exception as e:
            print(f"!!! ERROR calculating metrics for {name}: {e}")
            import traceback
            traceback.print_exc()
            calculation_successful = False
            # Fill metrics with NaN on error for this strategy
            for k in metrics_summary.get(list(strategies.keys())[0], {}):
                if k not in metrics_summary[name]: metrics_summary[name][k] = np.nan

    # --- Print Comparison Table ---
    if not calculation_successful:
        print("\n--- METRICS COMPARISON (INCOMPLETE DUE TO ERRORS) ---")
    else:
        print("\n--- METRICS COMPARISON ---")

    # Dynamically get metric names from the first strategy's results if available
    if metrics_summary and list(strategies.keys())[0] in metrics_summary:
        metric_names = list(metrics_summary[list(strategies.keys())[0]].keys())
    else:
        metric_names = ["Error: No metrics calculated"]  # Placeholder

    strategy_names = list(strategies.keys())

    # Header
    header = f"{'Metric':<45}" + "".join([f"{name:>25}" for name in strategy_names])
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    # Target values (for context)
    targets = {
        'CVaR (5%, % Loss)': f"(Target <= {CVAR_PERCENTILE * 100:.0f}%)",
        'Stress Pass Rate (%)': f"(Target >= {100 - (1 - STRESS_TEST_FLOOR_PCT) * 100:.0f}%)",  # Pct Pass
        'OOP Burden Violation Rate (%)': f"(Target Ratio <= {OOP_BURDEN_TARGET * 100:.0f}%)",
        'Surplus Coverage Violation Rate (%)': f"(Target Ratio >= {SURPLUS_COVERAGE_TARGET:.1f}x)",
        f'Annuity Adequacy Pct Years Adequate (Age {ANNUITY_ADEQUACY_TARGET_AGE}+, %)': f"(Target Ratio >= {ANNUITY_ADEQUACY_TARGET_RATIO * 100:.0f}%)"
    }

    # Rows
    for metric in metric_names:
        target_str = targets.get(metric, "")
        row = f"{metric:<45}"
        for name in strategy_names:
            value = metrics_summary.get(name, {}).get(metric, np.nan)  # Safe access
            # Format based on metric type/name patterns
            format_str = ">25.2f"  # Default float
            if isinstance(value, (int, float)) and np.isnan(value):
                row += f"{'NaN':>25}"
            elif isinstance(value, float) and np.isinf(value):
                row += f"{'Inf':>25}"
            elif "Rate" in metric or "%" in metric or "CVaR" in metric:
                format_str = ">25.2f"  # Percentage like
            elif "Coverage" in metric or "(x)" in metric:
                format_str = ">25.2f"  # Ratio like
            elif "Wealth" in metric:
                format_str = ">25,.0f"  # Integer wealth
            row += f"{value:{format_str}}"

        # Add target info if available
        if target_str:
            row += f"  {target_str}"
        print(row)

    print("-" * len(header))

    # --- Clean up ---
    print("\nClosing evaluation environment...")
    if 'eval_env' in locals() and eval_env is not None:
        eval_env.close()
    print("Evaluation finished.")


# ===-----------------------------------------------------------------------===
#    MAIN EXECUTION GUARD
# ===-----------------------------------------------------------------------===
if __name__ == "__main__":

    print(f"** Script Mode: Training={RUN_TRAINING}, Evaluation={RUN_EVALUATION} **")

    # Check CUDA availability early
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version detected by PyTorch: {torch.version.cuda}")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU.")

    # --- Run Training ---
    model_directory_for_eval = PRE_TRAINED_MODEL_DIR
    if RUN_TRAINING:
        # Crucial for SubprocVecEnv multiprocessing safety, especially on Windows
        # Needs to be inside the main guard for multiprocessing spawn safety
        try:
            # If using 'spawn' start method (sometimes needed on MacOS/Windows)
            # import multiprocessing as mp
            # mp.set_start_method('spawn', force=True)
            pass  # Keep 'fork' (default on Linux) unless issues arise
        except RuntimeError as e:
            print(f"Note: Could not set multiprocessing start method if not first time: {e}")

        trained_model_dir = train_model()
        if trained_model_dir:
            model_directory_for_eval = trained_model_dir  # Use the newly trained model for eval
        else:
            print("\n!!! Training failed or was skipped, cannot proceed to evaluation with newly trained model. !!!")
            # If a pre-trained model is specified, evaluation might still proceed below.
            # If not, evaluation will likely fail or be skipped.

    # --- Run Evaluation ---
    if RUN_EVALUATION:
        if model_directory_for_eval:
            evaluate_model(model_directory_for_eval)
        else:
            print("\n!!! Evaluation Skipped: No model directory specified or training failed. !!!")
            print("    (Set PRE_TRAINED_MODEL_DIR if you want to evaluate an existing model)")

    print("\nScript finished.")
