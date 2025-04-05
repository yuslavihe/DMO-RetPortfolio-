import numpy as np
import os
import torch
import wandb
import time
import gymnasium as gym
from gymnasium import spaces
from scipy.special import softmax # For normalizing actions

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
# Import SubprocVecEnv for true parallelism
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
from wandb.integration.sb3 import WandbCallback as OfficialWandbCallback # Use the official one

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
# key point: Reduce END_AGE for more plausible survival to term
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
# key point: Add Bequest Reward Scaling (if dead before END_AGE)
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
# key point: Probability multiplier based on age (simple linear increase example)
# Note: END_AGE change affects this slope slightly
HEALTH_SHOCK_AGE_FACTOR = lambda age: 1.0 + max(0, (age - START_AGE) / (END_AGE - START_AGE)) * 1.5 if (END_AGE - START_AGE) > 0 else 1.0

# --- Numerical Stability/Clipping ---
MAX_PENALTY_MAGNITUDE_PER_STEP = 100.0
MAX_TERMINAL_REWARD_MAGNITUDE = 1000.0 # For GAMMA_W
MAX_BEQUEST_REWARD_MAGNITUDE = MAX_TERMINAL_REWARD_MAGNITUDE / 2 # Separate clip for bequest
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

# --- Training Run Identification ---
RUN_NAME_SUFFIX = "Full_Embedded_HealthShocks_Bequest_V4" # Updated suffix
# **key point**: Directory for saving NEW training run artifacts
MODEL_SAVE_DIR_BASE = "models" # Base directory
# **key point**: Flag to indicate if we are continuing training
LOAD_PREVIOUS_MODEL = True # <<< SET TO True TO CONTINUE TRAINING
# **key point**: Directory containing the model/stats to LOAD FROM
PREVIOUS_MODEL_LOAD_DIR = r"C:\Users\jeanna\PycharmProjects\DMO RetPort\DRL-3\models\1743832862_Full_Embedded_HealthShocks_Bequest_V4\best_model" # <<< SET THIS PATH
# **key point**: Specify which model file and stats file to load (e.g., 'best_model' or 'final_model')
PREVIOUS_MODEL_FILENAME = "best_model.zip" # or "final_model.zip" or specific checkpoint name
PREVIOUS_STATS_FILENAME = "best_model_vecnormalize.pkl" # or "final_vec_normalize.pkl" or specific checkpoint stats name


import os # Make sure os is imported

# ** Path to the BASE directory **
load_base_dir = r"C:\Users\jeanna\PycharmProjects\DMO RetPort\DRL-3\models\1743832862_Full_Embedded_HealthShocks_Bequest_V4"

# ** Construct the specific path to the model zip file INSIDE the subfolder **
load_model_subdir = "best_model"
load_model_filename = "best_model.zip"
load_model_path = os.path.join(load_base_dir, load_model_subdir, load_model_filename) # <-- NOTE THE CHANGE HERE

# ** Specify the correct VecNormalize stats filename and path (directly in base dir) **
load_stats_filename = "final_vec_normalize.pkl" # <<<--- USE THIS EXACT NAME
load_stats_path = os.path.join(load_base_dir, load_stats_filename)

# **key point**: Modify save dir/run name if continuing
if LOAD_PREVIOUS_MODEL:
    RUN_NAME_SUFFIX += "_Continued"
    # Optionally, create a subdirectory for the continued run
    # MODEL_SAVE_DIR = os.path.join(PREVIOUS_MODEL_LOAD_DIR, f"continued_{int(time.time())}")
    # Or save in the standard structure but with the modified name
    MODEL_SAVE_DIR = os.path.join(MODEL_SAVE_DIR_BASE, f"{int(time.time())}_{RUN_NAME_SUFFIX}")
else:
    MODEL_SAVE_DIR = os.path.join(MODEL_SAVE_DIR_BASE, f"{int(time.time())}_{RUN_NAME_SUFFIX}")

use_vec_normalize = True

# --- WandB Configuration ---
WANDB_PROJECT_NAME = "DMO_RetPortfolio_V4"
WANDB_ENTITY = '' # Your W&B username or team name, or None for default

# --- Training & PPO Configuration ---
# **key point**: Set the TOTAL timesteps for the combined training (original + this run)
# If original was 5M and you want 5M more, set this to 10M
# If loading, this is the target total steps for the *entire* process
TOTAL_TRAINING_TIMESTEPS = 10_000_000 # Example: Train for 5M more steps

# key point: Utilize available CPU cores
NUM_PARALLEL_ENVS = os.cpu_count() if os.cpu_count() else 16
NUM_PARALLEL_ENVS = max(1, NUM_PARALLEL_ENVS) # Ensure at least 1
SEED = 42 # Using the same seed might be relevant if stochasticity needs to be consistent, but usually ok to change for continued run

# Check for CUDA availability
if torch.cuda.is_available():
    print(f"CUDA (GPU) available! Using device: cuda ({torch.cuda.get_device_name(0)})")
    DEVICE = "cuda"
else:
    print("CUDA not available, using CPU.")
    DEVICE = "cpu"

# PPO Hyperparameters - **key point**: These should generally match the original training unless fine-tuning
PPO_CONFIG = {
    "learning_rate": 1e-4, # You might want to DECREASE this for fine-tuning (e.g., 3e-5)
    "n_steps": 2048, # Steps per env per update
    "batch_size": 512, # Minibatch size for optimization
    "n_epochs": 10, # Optimization epochs per update
    "gamma": 0.99, # Discount factor for future rewards
    "gae_lambda": 0.95, # Factor for Generalized Advantage Estimation
    "clip_range": 0.2, # Clipping parameter for PPO policy gradient loss
    "ent_coef": 0.005, # Entropy coefficient for exploration
    "vf_coef": 0.5, # Value function coefficient in loss
    "max_grad_norm": 0.5, # Max gradient norm clipping
    "device": DEVICE, # Use GPU if available
}

# Evaluation and Logging Settings
EVAL_FREQ_TOTAL_STEPS = 50_000 # Evaluate every N *total* steps across all envs
N_EVAL_EPISODES = 30 # Number of episodes for each evaluation
LOG_INTERVAL = 20 # Log training stats every N updates

# ===-----------------------------------------------------------------------===
#    RETIREMENT ENVIRONMENT DEFINITION (V4 - With Bequest Reward)
# ===-----------------------------------------------------------------------===
# (RetirementEnv class definition remains unchanged - pasted below for completeness)
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
        # key point: END_AGE and TIME_HORIZON are taken from config
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
        # key point: Store bequest gamma
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

        # key point: Initialize all reward components including bequest
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
        flat_action = action
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
        if not hasattr(self, 'np_random'): self.np_random, _ = gym.utils.seeding.np_random(None)
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
                 raw_dd_penalty = self.beta_d * drawdown_pct
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
            current_final_wealth = max(WEALTH_FLOOR, self.current_wealth) # Use wealth at end of step
            if self.is_alive: # Reached end_age/horizon alive
                normalized_terminal_wealth = (current_final_wealth / (self.initial_wealth + EPSILON))
                raw_terminal_reward = normalized_terminal_wealth * self.gamma_w
                terminal_wealth_reward_comp = np.clip(raw_terminal_reward, 0, MAX_TERMINAL_REWARD_MAGNITUDE)
                step_reward += terminal_wealth_reward_comp # Add to this final step's reward
                self.reward_components['terminal_wealth_reward_comp'] = terminal_wealth_reward_comp
                self.reward_components['bequest_reward_comp'] = 0.0 # Ensure bequest is zero
                if self.verbose > 0: print(f"Step {self.current_step}: TERMINATED (ALIVE). Terminal Reward: {terminal_wealth_reward_comp:.2f}")
            elif alive_at_start_of_survival_check: # Died this step (or was already dead)
                normalized_bequest_wealth = (current_final_wealth / (self.initial_wealth + EPSILON))
                raw_bequest_reward = normalized_bequest_wealth * self.gamma_b # Use bequest gamma
                bequest_reward_comp = np.clip(raw_bequest_reward, 0, MAX_BEQUEST_REWARD_MAGNITUDE) # Use separate clip
                step_reward += bequest_reward_comp # Add to this final step's reward
                self.reward_components['bequest_reward_comp'] = bequest_reward_comp
                self.reward_components['terminal_wealth_reward_comp'] = 0.0 # Ensure terminal is zero
                if self.verbose > 0: print(f"Step {self.current_step}: TERMINATED (DEAD). Bequest Reward: {bequest_reward_comp:.2f}")
            # If truncated but dead, still assign bequest reward.

        # --- 9. Get Observation and Info ---
        observation = self._get_obs()
        info = self._get_info(terminated or truncated) # Pass terminal flag

        # Safety checks for NaN/Inf in reward and observation
        if np.any(np.isnan(observation)) or np.any(np.isinf(observation)):
            print(f"!!! WARNING: NaN/Inf detected in observation: {observation} at step {self.current_step} !!!")
            observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            step_reward = -MAX_PENALTY_MAGNITUDE_PER_STEP * 2 # Heavy penalty

        if np.isnan(step_reward) or np.isinf(step_reward):
            print(f"!!! WARNING: NaN/Inf detected in reward: {step_reward} at step {self.current_step} !!!")
            step_reward = -MAX_PENALTY_MAGNITUDE_PER_STEP * 2 # Heavy penalty

        step_reward = float(step_reward) # Ensure reward is standard float

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
            "healthcare_cost_in_step": self.current_health_shock['cost'] if self.current_health_shock else 0.0,
        }
        if is_terminal:
            # Populate final components using .get() for safety
            info["final_qaly_comp"] = float(self.reward_components.get('qaly_comp', 0.0))
            info["final_dd_comp"] = float(self.reward_components.get('dd_comp', 0.0))
            info["final_risky_comp"] = float(self.reward_components.get('risky_comp', 0.0))
            info["final_terminal_wealth_reward_comp"] = float(self.reward_components.get('terminal_wealth_reward_comp', 0.0))
            info["final_shortfall_comp"] = float(self.reward_components.get('shortfall_comp', 0.0))
            info["final_annuity_bonus_comp"] = float(self.reward_components.get('annuity_bonus_comp', 0.0))
            # key point: Add final bequest component
            info["final_bequest_reward_comp"] = float(self.reward_components.get('bequest_reward_comp', 0.0))
            info["final_wealth"] = float(self.current_wealth)

            # Calculate final total reward from components for verification
            # Ensure all values are numeric before summing
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
        if not hasattr(self, 'np_random'):
             self.np_random, _ = gym.utils.seeding.np_random(None) # Ensure RNG is initialized
        noise = self.np_random.normal(loc=0.0, scale=np.abs(base_value * noise_factor + EPSILON), size=size)
        forecast = base_value + noise
        return forecast

    def _simulate_market_returns(self):
        """Placeholder simulation logic - *only* returns asset returns."""
        # Example placeholder - Replace with your actual financial market simulation
        mean_returns = np.array([0.08, 0.06, 0.07, 0.02, 0.035, 0.01] * int(np.ceil(self.total_assets/6)))[:self.total_assets]
        volatilities = np.array([0.15, 0.18, 0.12, 0.03, 0.05, 0.005] * int(np.ceil(self.total_assets/6)))[:self.total_assets]

        if not hasattr(self, 'np_random'):
             self.np_random, _ = gym.utils.seeding.np_random(None) # Ensure RNG is initialized

        asset_returns = self.np_random.normal(loc=mean_returns, scale=volatilities).astype(np.float32)
        return asset_returns

    def close(self):
        """Perform any necessary cleanup."""
        pass # No specific cleanup needed in this version

# ===-----------------------------------------------------------------------===
#    CUSTOM EVALUATION CALLBACK DEFINITION (CORRECTED INIT & FREQ)
# ===-----------------------------------------------------------------------===
# (EvalCallbackWithComponents class definition remains unchanged - pasted below for completeness)
# Import SB3 VecEnv base class for type checking if needed, although not strictly required now
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

class EvalCallbackWithComponents(BaseCallback):
    """
    Callback for evaluating the agent and logging reward components from info dict.
    Assumes eval_env is a VecEnv. Handles VecNormalize saving. Logs bequest reward.
    Corrected __init__ to accept pre-vectorized env and uses total steps for eval_freq.
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
        # Check if enough steps have passed *since the last evaluation*
        # self.num_timesteps is the TOTAL steps since training started (including previous runs if reset_num_timesteps=False)
        if self.num_timesteps - self.last_eval_step_total >= self.eval_freq:
            # **key point**: Ensure last_eval_step is updated correctly relative to the *current* total steps
            # To evaluate every eval_freq steps:
            self.last_eval_step_total = self.num_timesteps

            if self.verbose > 0:
                print(f"\nEvaluating agent at {self.num_timesteps} total training steps...")

            all_episode_rewards = []
            all_episode_lengths = []
            all_final_infos = []

            # --- Evaluation loop ---
            current_obs = self.eval_env.reset()
            episodes_done = 0
            num_envs = self.eval_env.num_envs # Should typically be 1 for eval
            ep_rewards = np.zeros(num_envs)
            ep_lengths = np.zeros(num_envs)

            while episodes_done < self.n_eval_episodes:
                 # _deterministic action in eval callback
                 with torch.no_grad(): # Reduce memory for evaluation
                    # **key point**: Ensure using self.model for predictions
                    action, _states = self.model.predict(current_obs, deterministic=self.deterministic)

                 new_obs, rewards, dones, infos = self.eval_env.step(action)

                 ep_rewards += rewards
                 ep_lengths += 1

                 for i in range(num_envs):
                      if dones[i]:
                           if episodes_done < self.n_eval_episodes:
                                if self.verbose > 1: print(f"  Eval Episode {episodes_done+1} finished. Reward: {ep_rewards[i]:.2f}, Length: {ep_lengths[i]}")
                                all_episode_rewards.append(ep_rewards[i])
                                all_episode_lengths.append(ep_lengths[i])
                                # Extract final info dictionary correctly
                                final_info = infos[i].get("final_info", {})
                                all_final_infos.append(final_info)
                                episodes_done += 1
                           # Reset only the completed environment's state trackers
                           ep_rewards[i] = 0
                           ep_lengths[i] = 0
                           # Important: If the eval env resets automatically, we might need to get the new obs here
                           # but VecEnv usually handles this; the new_obs outside the loop should be correct.
                           # We might need to reset manually if using non-standard VecEnv wrappers
                           # new_obs[i] = self.eval_env.reset()[i] # Typically not needed for standard VecEnvs

                 # Update observation for the next step
                 current_obs = new_obs

            # --- Logging ---
            mean_reward = np.mean(all_episode_rewards) if all_episode_rewards else 0
            std_reward = np.std(all_episode_rewards) if all_episode_rewards else 0
            mean_ep_length = np.mean(all_episode_lengths) if all_episode_lengths else 0

            if self.verbose > 0:
                print(f"Eval results ({self.n_eval_episodes} episodes): Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}, Mean Ep Length: {mean_ep_length:.2f}")

            # Log standard eval metrics using the global step counter
            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/mean_ep_length", mean_ep_length)
            self.logger.record("eval/std_reward", std_reward)
            self.logger.record("eval/count", len(all_episode_rewards))

            # Filter for valid final info dictionaries
            valid_infos = [info for info in all_final_infos if info and isinstance(info, dict)]

            # Helper function to safely extract and calculate mean from info dicts
            def get_mean_component(key, default_val=np.nan):
                vals = [info.get(key, default_val) for info in valid_infos]
                numeric_vals = [v for v in vals if isinstance(v, (int, float))]
                if not numeric_vals: return 0.0
                mean_val = np.nanmean(numeric_vals)
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
            self.logger.record("eval/frac_health_shock_final_step", np.mean(shocks_in_final_step) if shocks_in_final_step else 0)
            self.logger.record("eval/mean_healthcare_cost_final_step", np.mean(healthcare_costs_final_step) if healthcare_costs_final_step else 0)
            opt1_bought_final = [info.get("has_opt_annuity_1", False) for info in valid_infos]
            opt2_bought_final = [info.get("has_opt_annuity_2", False) for info in valid_infos]
            self.logger.record("eval/frac_bought_opt_annuity_1", np.mean(opt1_bought_final) if opt1_bought_final else 0)
            self.logger.record("eval/frac_bought_opt_annuity_2", np.mean(opt2_bought_final) if opt2_bought_final else 0)

            # Dump all recorded values to the logger (e.g., TensorBoard, WandB)
            # **key point**: Ensure logging uses the correct global step
            self.logger.dump(step=self.num_timesteps)

            # --- Save best model logic ---
            if mean_reward > self.best_mean_reward:
                 self.best_mean_reward = mean_reward
                 if self.best_model_save_path is not None:
                      save_path_model = os.path.join(self.best_model_save_path, "best_model.zip")
                      save_path_stats = os.path.join(self.best_model_save_path, "best_model_vecnormalize.pkl")
                      if self.verbose > 0: print(f"New best model found! Saving model to {save_path_model} (Mean Reward: {mean_reward:.2f})")
                      # **key point**: Save the current model state
                      self.model.save(save_path_model)
                      # Save VecNormalize statistics if the training env uses it
                      # **key point**: Get the VecNormalize wrapper instance correctly
                      training_env = self.model.get_env() # This might be the VecNormalize wrapper directly or wrapped again
                      vec_norm_env = None
                      if isinstance(training_env, VecNormalize):
                          vec_norm_env = training_env
                      elif hasattr(training_env, 'venv') and isinstance(training_env.venv, VecNormalize):
                          vec_norm_env = training_env.venv # Access the wrapped VecNormalize

                      if vec_norm_env is not None:
                          if self.verbose > 0: print(f"Saving VecNormalize stats from training env to {save_path_stats}")
                          vec_norm_env.save(save_path_stats)
                      else:
                           if self.verbose > 0: print("Warning: Training environment is not VecNormalize or VecNormalize wrapper not found, stats not saved with best model.")
        return True

# ===-----------------------------------------------------------------------===
#    MAIN TRAINING EXECUTION SCRIPT (MODIFIED FOR CONTINUATION)
# ===-----------------------------------------------------------------------===

def main():
    """Main function to set up and run the PPO training, with option to continue."""
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    print(f"MODEL_SAVE_DIR for this run: {MODEL_SAVE_DIR}")
    if LOAD_PREVIOUS_MODEL:
        print(f"Attempting to continue training from: {PREVIOUS_MODEL_LOAD_DIR}")

    # --- Print Configuration Summary ---
    print("="*50)
    print("--- TRAINING CONFIGURATION (V4 - Health Shocks & Bequest Motive) ---")
    if LOAD_PREVIOUS_MODEL:
        print("*** CONTINUING PREVIOUS TRAINING ***")
    print("="*50)
    print(f"Run Suffix: {RUN_NAME_SUFFIX}")
    # **key point**: Clarify total timesteps meaning based on loading
    if LOAD_PREVIOUS_MODEL:
        print(f"Target Total Timesteps (Combined): {TOTAL_TRAINING_TIMESTEPS:,}")
    else:
        print(f"Total Timesteps (New Training): {TOTAL_TRAINING_TIMESTEPS:,}")
    print(f"Parallel Envs: {NUM_PARALLEL_ENVS} (Using {SubprocVecEnv.__name__})")
    print(f"Initial Wealth: {INITIAL_WEALTH:,}")
    print(f"**key point**: Time Horizon: {START_AGE}-{END_AGE} ({TIME_HORIZON} years)")
    print(f"Assets: S={NUM_STOCKS}, C={NUM_COMMODITIES}, E={NUM_ETFS}, GB={NUM_GOV_BONDS}, CB={NUM_CORP_BONDS}, SI={NUM_STABLE_INC} (Total={TOTAL_ASSETS})")
    print(f"Baseline Living Expense (L): {BASELINE_LIVING_EXPENSE:,}")
    print(f"Reward Coeffs: GAMMA_W={GAMMA_W}, **key point**: GAMMA_B={GAMMA_B}, THETA={THETA}, BETA_D={BETA_D}, SHORTFALL_MULT={SHORTFALL_PENALTY_MULTIPLIER}, ANNUITY_BONUS={ANNUITY_PURCHASE_BONUS}")
    print(f"QALY Params: Normal={QALY_NORMAL_YEAR}, ShockUntreated={QALY_SHOCK_UNTREATED}")
    print(f"Health Shocks: {len(HEALTH_SHOCKS_CONFIG)} types configured (see code), AgeFactor Range: ~{HEALTH_SHOCK_AGE_FACTOR(START_AGE):.2f}-{HEALTH_SHOCK_AGE_FACTOR(END_AGE-1):.2f}")
    print(f"Annuities: Compulsory={COMPULSORY_ANNUITY_PAYMENT:,.0f}/yr; Opt1={OPTIONAL_ANNUITY_1_PAYMENT:,.0f}/yr (Premium={OPTIONAL_ANNUITY_1_PREMIUM:,.0f}); Opt2={OPTIONAL_ANNUITY_2_PAYMENT:,.0f}/yr (Premium={OPTIONAL_ANNUITY_2_PREMIUM:,.0f})")
    print(f"PPO Config: n_steps={PPO_CONFIG['n_steps']}, batch_size={PPO_CONFIG['batch_size']}, lr={PPO_CONFIG['learning_rate']}, ent_coef={PPO_CONFIG['ent_coef']}")
    print(f"PPO Device: {PPO_CONFIG['device']}")
    print(f"Eval Freq: {EVAL_FREQ_TOTAL_STEPS:,} total steps, N Episodes: {N_EVAL_EPISODES}")
    print("="*50)

    # --- Environment Setup ---
    # **key point**: Environment parameters MUST match the original training if loading
    env_kwargs = {
        'initial_wealth': INITIAL_WEALTH, 'start_age': START_AGE,
        'end_age': END_AGE, # Pass updated end_age
        'gamma_b': GAMMA_B, # Pass bequest gamma
        'baseline_living_expense': BASELINE_LIVING_EXPENSE,
        'health_shocks_config': HEALTH_SHOCKS_CONFIG,
        'health_shock_age_factor': HEALTH_SHOCK_AGE_FACTOR,
        'qaly_normal_year': QALY_NORMAL_YEAR,
        'qaly_shock_untreated': QALY_SHOCK_UNTREATED,
        'shortfall_penalty_multiplier': SHORTFALL_PENALTY_MULTIPLIER,
        'num_stocks': NUM_STOCKS, 'num_commodities': NUM_COMMODITIES, 'num_etfs': NUM_ETFS,
        'num_gov_bonds': NUM_GOV_BONDS, 'num_corp_bonds': NUM_CORP_BONDS, 'num_stable_inc': NUM_STABLE_INC,
        'survival_a': SURVIVAL_A, 'survival_k': SURVIVAL_K, 'survival_g': SURVIVAL_G, 'survival_c': SURVIVAL_C,
        'gamma_w': GAMMA_W, 'theta': THETA, 'beta_d': BETA_D,
        'annuity_purchase_bonus': ANNUITY_PURCHASE_BONUS,
        'risky_asset_indices': RISKY_ASSET_INDICES, 'critical_asset_index': CRITICAL_ASSET_INDEX,
        'forecast_return_noise_factor_1yr': FORECAST_RETURN_NOISE_FACTOR_1YR,
        'forecast_return_noise_factor_2yr': FORECAST_RETURN_NOISE_FACTOR_2YR,
        'forecast_mortality_noise_1yr': FORECAST_MORTALITY_NOISE_1YR,
        'forecast_mortality_noise_2yr': FORECAST_MORTALITY_NOISE_2YR,
        'compulsory_annuity_payment': COMPULSORY_ANNUITY_PAYMENT,
        'opt_annuity_1_premium': OPTIONAL_ANNUITY_1_PREMIUM, 'opt_annuity_1_payment': OPTIONAL_ANNUITY_1_PAYMENT,
        'opt_annuity_2_premium': OPTIONAL_ANNUITY_2_PREMIUM, 'opt_annuity_2_payment': OPTIONAL_ANNUITY_2_PAYMENT,
        'verbose': 0 # Set to 1 or 2 for more detailed env output during training/eval
    }

    # Create parallel training environments
    print(f"Creating {NUM_PARALLEL_ENVS} parallel training environments using {SubprocVecEnv.__name__}...")
    train_env_no_norm = make_vec_env(
        lambda: Monitor(RetirementEnv(**env_kwargs)), # Wrap with Monitor
        n_envs=NUM_PARALLEL_ENVS,
        seed=SEED,
        vec_env_cls=SubprocVecEnv
    )
    # Wrap with VecNormalize **AFTER** make_vec_env
    print("Wrapping training environment with VecNormalize...")
    train_env = VecNormalize(train_env_no_norm, norm_obs=True, norm_reward=False, clip_obs=10., gamma=PPO_CONFIG['gamma'])


    # Create a *single* evaluation environment
    print("Creating evaluation environment (n_envs=1)...")
    eval_env_no_norm = make_vec_env(
        lambda: Monitor(RetirementEnv(**env_kwargs)), # Wrap with Monitor
        n_envs=1,
        seed=SEED + NUM_PARALLEL_ENVS, # Different seed for eval env
        vec_env_cls=DummyVecEnv # DummyVecEnv is fine for single eval env
    )
    # Wrap with VecNormalize **AFTER** make_vec_env
    print("Wrapping evaluation environment with VecNormalize...")
    eval_env = VecNormalize(eval_env_no_norm, norm_obs=True, norm_reward=False, clip_obs=10., training=False, gamma=PPO_CONFIG['gamma'])


    # --- Model Definition / Loading ---
    model = None
    if LOAD_PREVIOUS_MODEL:
        model_path = os.path.join(PREVIOUS_MODEL_LOAD_DIR, PREVIOUS_MODEL_FILENAME)
        stats_path = os.path.join(PREVIOUS_MODEL_LOAD_DIR, PREVIOUS_STATS_FILENAME)

        if os.path.exists(model_path) and os.path.exists(stats_path):
            print(f"Loading model from: {model_path}")
            # **key point**: Load the model, providing the *current* train_env
            # Pass custom_objects if you changed activation functions etc.
            # Pass PPO hyperparameters again to ensure consistency or apply changes (like reduced LR)
            model = PPO.load(
                model_path,
                env=train_env, # Link loaded model to the newly created VecNormalize env
                device=DEVICE,
                custom_objects={'learning_rate': PPO_CONFIG['learning_rate'], 'clip_range': PPO_CONFIG['clip_range']}, # Example if LR changed
                **{k: v for k, v in PPO_CONFIG.items() if k not in ['learning_rate', 'clip_range', 'device']} # Pass rest of PPO config
            )
            # **key point**: Reset internal env reference JUST IN CASE (usually handled by load)
            model.set_env(train_env)

            print(f"Loading VecNormalize statistics from: {stats_path}")
            # **key point**: Load stats *into* the existing train_env VecNormalize wrapper
            train_env = VecNormalize.load(stats_path, train_env_no_norm) # Pass the base env here
            train_env.training = True # Ensure loaded stats are used for training updates if needed

            # **key point**: CRITICAL - Sync eval_env stats with loaded train_env stats
            print("Syncing evaluation environment normalization stats with loaded training stats...")
            eval_env.obs_rms = train_env.obs_rms
            eval_env.ret_rms = train_env.ret_rms # Sync reward RMS if norm_reward=True (though it's False here)
            eval_env.training = False # Ensure eval env doesn't update stats

            print(f"Successfully loaded model and stats. Current timesteps: {model.num_timesteps}")
        else:
            print(f"!!! ERROR: Cannot load model. Files not found:")
            if not os.path.exists(model_path): print(f"  - Model: {model_path}")
            if not os.path.exists(stats_path): print(f"  - Stats: {stats_path}")
            print("!!! ABORTING TRAINING !!!")
            return # Exit if loading fails

    else:
        # Initialize a new model if not loading
        print(f"Defining NEW PPO model with MlpPolicy on device: {PPO_CONFIG['device']}...")
        model = PPO(
            "MlpPolicy",
            train_env, # Use the VecNormalize wrapped env
            verbose=1,
            seed=SEED,
            tensorboard_log=f"runs/{int(time.time())}_{RUN_NAME_SUFFIX}", # Log TensorBoard data locally
            **PPO_CONFIG # Pass all PPO hyperparameters
        )

    # --- Initialize WandB ---
    print("Initializing WandB...")
    # You might want a different run name or use wandb resume features if continuing
    wandb_run_name = f"{int(time.time())}_{RUN_NAME_SUFFIX}"
    # Optional: Resume previous run if ID is known
    # wandb_resume_id = "PREVIOUS_WANDB_RUN_ID" if LOAD_PREVIOUS_MODEL else None

    combined_config = {
        "policy": "MlpPolicy",
        "action_space_type": "Box (Flattened)",
        "total_timesteps": TOTAL_TRAINING_TIMESTEPS,
        "num_parallel_envs": NUM_PARALLEL_ENVS,
        "vec_env_type": SubprocVecEnv.__name__,
        "seed": SEED,
        "load_previous_model": LOAD_PREVIOUS_MODEL, # Log whether we loaded
        "previous_model_load_dir": PREVIOUS_MODEL_LOAD_DIR if LOAD_PREVIOUS_MODEL else None,
        **PPO_CONFIG,
        **env_kwargs
    }
    serializable_config = {k: v for k, v in combined_config.items() if not callable(v)}

    run = wandb.init(
        project=WANDB_PROJECT_NAME,
        entity=WANDB_ENTITY,
        name=wandb_run_name,
        config=serializable_config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
        # resume="allow" if wandb_resume_id else None, # Use if resuming
        # id=wandb_resume_id if wandb_resume_id else None, # Use if resuming
    )
    # Log hyperparameters AFTER model is defined/loaded
    # model.logger might not be fully set up before .learn() is called first time
    # HParam callback handles this better.


    # --- Callback Setup ---
    print("Setting up callbacks...")
    # Checkpoint frequency needs context of total steps vs steps per env
    # Save every N *TOTAL* steps across all envs
    checkpoint_save_freq_total_steps = EVAL_FREQ_TOTAL_STEPS * 4 # Save less frequently than eval maybe
    # Convert total steps freq to per-environment steps freq for CheckpointCallback
    # Ensure save_freq is at least n_steps to avoid issues
    save_freq_per_env = max(1, checkpoint_save_freq_total_steps // NUM_PARALLEL_ENVS)
    save_freq_per_env = max(save_freq_per_env, PPO_CONFIG['n_steps']) # MUST be >= n_steps

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq_per_env, # Frequency based on steps per env
        save_path=os.path.join(MODEL_SAVE_DIR, 'checkpoints'),
        name_prefix='ppo_retire_v4_cont',
        save_replay_buffer=False,
        save_vecnormalize=True # Save VecNormalize stats with checkpoints
    )
    print(f"Checkpoint Callback: Save Freq approx every {checkpoint_save_freq_total_steps:,} total steps (per-env freq: {save_freq_per_env})")

    # Evaluation callback uses total steps frequency directly
    evaluation_callback = EvalCallbackWithComponents(
        eval_env=eval_env, # Pass the normalized eval env
        n_eval_episodes=N_EVAL_EPISODES,
        eval_freq=EVAL_FREQ_TOTAL_STEPS, # Trigger based on total steps
        log_path=os.path.join(MODEL_SAVE_DIR, 'eval_logs'),
        best_model_save_path=os.path.join(MODEL_SAVE_DIR, 'best_model'), # Saves best model for *this* run
        deterministic=True,
        verbose=1,
    )
    print(f"Evaluation Callback: Eval Freq every {EVAL_FREQ_TOTAL_STEPS:,} total steps")

    wandb_callback = OfficialWandbCallback(
        gradient_save_freq=0,
        model_save_path=os.path.join(MODEL_SAVE_DIR, f"wandb_models/{run.id}"),
        model_save_freq=0, # Disable SB3 wandb model saving, use EvalCallback
        verbose=2
    )

    # HParam callback logs hyperparameters (will log current config)
    hparam_callback = HParamCallback()

    callback_list = CallbackList([checkpoint_callback, evaluation_callback, hparam_callback, wandb_callback])

    # --- Training ---
    print("="*50)
    if LOAD_PREVIOUS_MODEL:
        print(f"--- CONTINUING TRAINING to {TOTAL_TRAINING_TIMESTEPS:,} total steps ---")
        print(f"--- Starting from step: {model.num_timesteps} ---")
    else:
        print(f"--- STARTING NEW TRAINING ({TOTAL_TRAINING_TIMESTEPS:,} steps) ---")
    print(f"--- Using {NUM_PARALLEL_ENVS} parallel environments ---")
    print(f"--- Monitor CPU ({os.cpu_count()} cores) and GPU ({torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'N/A'}) utilization ---")
    print("="*50)

    start_time = time.time()
    # **key point**: Determine if resetting timesteps
    reset_timesteps_flag = False if LOAD_PREVIOUS_MODEL else True

    try:
        model.learn(
            total_timesteps=TOTAL_TRAINING_TIMESTEPS,
            callback=callback_list,
            log_interval=LOG_INTERVAL,
            # **key point**: CRITICAL for continuing training!
            reset_num_timesteps=reset_timesteps_flag
        )
    except Exception as e:
        print(f"\n!!! TRAINING ERROR: {e} !!!\n")
        import traceback
        traceback.print_exc()
    finally:
        duration = time.time() - start_time
        duration_min = duration / 60
        duration_hr = duration / 3600
        print("="*50)
        print(f"--- TRAINING PHASE FINISHED ---")
        print(f"Duration: {duration:.2f}s | {duration_min:.2f} min | {duration_hr:.2f} hr")
        print(f"Model artifacts saved in: {MODEL_SAVE_DIR}")
        print(f"Final total timesteps reached: {model.num_timesteps}")
        print("="*50)

        # --- Save Final Model and Normalization Stats ---
        final_model_path = os.path.join(MODEL_SAVE_DIR, "final_model.zip")
        final_stats_path = os.path.join(MODEL_SAVE_DIR, "final_vec_normalize.pkl")
        print(f"Saving final model to: {final_model_path}")
        model.save(final_model_path)

        # **key point**: Save stats from the correct VecNormalize object
        # model.get_env() should return the VecNormalize wrapper we passed/set
        _train_env = model.get_env()
        if isinstance(_train_env, VecNormalize):
             print(f"Saving final VecNormalize stats to: {final_stats_path}")
             _train_env.save(final_stats_path)
        else:
             print(f"Warning: Could not save final VecNormalize stats. Expected VecNormalize, got {type(_train_env)}")


        # --- Clean up ---
        print("Closing environments...")
        try:
            # Check if environments exist before closing
            if 'train_env' in locals() and train_env is not None: train_env.close()
            if 'eval_env' in locals() and eval_env is not None: eval_env.close()
        except Exception as e: print(f"Error closing environments: {e}")

        # Finish the WandB run
        if run:
            print("Finishing WandB run...")
            run.finish()
        print("Script finished.")


# ===-----------------------------------------------------------------------===
#    HYPERPARAMETER LOGGING CALLBACK (Add Bequest Gamma)
# ===-----------------------------------------------------------------------===
# (HParamCallback class definition remains unchanged - pasted below for completeness)
class HParamCallback(BaseCallback):
    """
    Saves hyperparameters and metrics at the start of training for WandB's HParams tab.
    """
    def _on_training_start(self) -> None:
        # Get initial value if clip_range is a schedule
        clip_range_val = self.model.clip_range
        if callable(clip_range_val):
             # Provide initial progress remaining (1.0) for schedule call
             try:
                 clip_range_val = clip_range_val(1.0)
             except Exception:
                 # Fallback if schedule fails with 1.0
                 clip_range_val = self.model.clip_range_vf if hasattr(self.model, 'clip_range_vf') else 0.2 # Or a default

        # Attempt to get current learning rate (might be a schedule)
        lr_val = self.model.learning_rate
        if callable(lr_val):
             # Provide initial progress remaining (1.0) for schedule call
             try:
                lr_val = lr_val(1.0)
             except Exception:
                # Fallback if schedule fails with 1.0
                lr_val = PPO_CONFIG.get('learning_rate', 'schedule') # Indicate it's a schedule if direct call fails


        hparam_dict = {
            # Algorithm & PPO params
            "algorithm": self.model.__class__.__name__,
            "learning_rate": lr_val, # Log the initial LR or indicate schedule
            "gamma": self.model.gamma,
            "gae_lambda": self.model.gae_lambda,
            "ent_coef": self.model.ent_coef,
            "vf_coef": self.model.vf_coef,
            "max_grad_norm": self.model.max_grad_norm,
            "n_steps": self.model.n_steps,
            "batch_size": self.model.batch_size,
            "n_epochs": self.model.n_epochs,
            "clip_range": clip_range_val, # Log initial clip range
            "policy": "MlpPolicy", # Explicitly state policy
            "action_space_type": "Box (Flattened)",
            # Env/training params
            "target_total_timesteps": TOTAL_TRAINING_TIMESTEPS, # Target for this run/combined run
            "num_parallel_envs": NUM_PARALLEL_ENVS,
            "vec_env_type": SubprocVecEnv.__name__,
            "device": PPO_CONFIG['device'],
            "seed": SEED,
            "end_age": END_AGE, # Log the adjusted end age
            "loaded_previous_model": LOAD_PREVIOUS_MODEL, # Indicate if loaded
            # Reward coefficients
            "reward_gamma_w": GAMMA_W,
            "reward_gamma_b": GAMMA_B, # key point: Log bequest gamma
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

        # Define placeholder metrics that WandB expects for HParams comparison
        metric_dict = {
            "rollout/ep_rew_mean": 0.0, # From SB3 logger
            "train/value_loss": 0.0, # From SB3 logger
            "eval/mean_reward": 0.0, # Primary metric for best model
            # Key evaluation components
            "eval/mean_qaly_comp": 0.0,
            "eval/mean_terminal_wealth_reward_comp": 0.0,
            "eval/mean_bequest_reward_comp": 0.0, # key point: Add bequest metric
            "eval/mean_final_wealth": 0.0,
            "eval/mean_annuity_bonus_comp": 0.0,
            "eval/frac_bought_opt_annuity_1": 0.0,
            "eval/frac_bought_opt_annuity_2": 0.0,
            # Other eval components
             "eval/mean_dd_comp": 0.0,
             "eval/mean_risky_comp": 0.0,
             "eval/mean_shortfall_comp": 0.0,
             "eval/mean_total_reward_from_components": 0.0,
            # Health eval metrics
            "eval/frac_health_shock_final_step": 0.0,
            "eval/mean_healthcare_cost_final_step": 0.0,
        }
        try:
             # Use the SB3 logger associated with the model to record HParams
             if hasattr(self.model, 'logger') and self.model.logger is not None:
                 self.model.logger.record("hparams", HParam(hparam_dict, metric_dict), exclude=("stdout", "log", "json", "csv"))
                 print("Hyperparameters logged for WandB HParams tab via model logger.")
             else:
                 print("Warning: Model logger not available at HParamCallback._on_training_start. Skipping HParam logging.")
        except Exception as e:
            print(f"Warning: Failed to record HParams via logger.record: {e}")

    def _on_step(self) -> bool:
        """
        This callback only runs once at the beginning of training.
        :return: True to continue training.
        """
        return True

# ===-----------------------------------------------------------------------===
#    SCRIPT EXECUTION GUARD
# ===-----------------------------------------------------------------------===
if __name__ == "__main__":
    # Check CUDA availability before starting potentially parallel processes
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version detected by PyTorch: {torch.version.cuda}")

    # Crucial for SubprocVecEnv multiprocessing safety, especially on Windows
    # It prevents issues with process forking/spawning.
    # Ensure this guard is present and main() is called within it.
    main()
