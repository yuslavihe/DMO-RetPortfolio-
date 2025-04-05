import numpy as np
import os
import torch
import wandb
import time
import gymnasium as gym
from gymnasium import spaces
from scipy.special import softmax # For normalizing actions
from collections import deque # Used in default EvalCallback, ensure available

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
# Import SubprocVecEnv for true parallelism
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.logger import HParam
from wandb.integration.sb3 import WandbCallback as OfficialWandbCallback

# ===-----------------------------------------------------------------------===
#    CONFIGURATION PARAMETERS (Defaults, can be overridden by sweep)
# ===-----------------------------------------------------------------------===

# --- Environment Asset Structure (Keep Fixed) ---
NUM_STOCKS = 25
NUM_COMMODITIES = 7
NUM_ETFS = 3
NUM_GOV_BONDS = 5
NUM_CORP_BONDS = 4
NUM_STABLE_INC = 3
TOTAL_ASSETS = NUM_STOCKS + NUM_COMMODITIES + NUM_ETFS + NUM_GOV_BONDS + NUM_CORP_BONDS + NUM_STABLE_INC

# --- Initial State & Environment Parameters (Keep Fixed) ---
INITIAL_WEALTH = 1_500_000
START_AGE = 65
END_AGE = 120 # Episode ends if agent reaches this age
TIME_HORIZON = END_AGE - START_AGE # T = 55 steps (years)

# --- Survival Probability Parameters (Keep Fixed) ---
SURVIVAL_A = 0.00005
SURVIVAL_K = 5.6
SURVIVAL_G = 0.055
SURVIVAL_C = 0.0001

# --- Annuity Parameters (Keep Fixed) ---
COMPULSORY_ANNUITY_PAYMENT = 10000.0
OPTIONAL_ANNUITY_1_PREMIUM = 50000.0
OPTIONAL_ANNUITY_1_PAYMENT = 5000.0
OPTIONAL_ANNUITY_2_PREMIUM = 75000.0
OPTIONAL_ANNUITY_2_PAYMENT = 8000.0

# --- Reward/Penalty Coefficients (Keep Fixed for sweep consistency) ---
GAMMA_W = 4.0
THETA = 0.15
BETA_D = 1.0
QALY_BASE = 0.15

# --- Numerical Stability/Clipping (Keep Fixed) ---
MAX_PENALTY_MAGNITUDE_PER_STEP = 100.0
MAX_TERMINAL_REWARD_MAGNITUDE = 1000.0
WEALTH_FLOOR = 1.0
EPSILON = 1e-8

# --- Asset Indices Definitions (Keep Fixed) ---
RISKY_ASSET_INDICES = list(range(NUM_STOCKS + NUM_COMMODITIES + NUM_ETFS))
CRITICAL_ASSET_INDEX = NUM_STOCKS + NUM_COMMODITIES

# --- Forecast Noise Parameters (Keep Fixed) ---
FORECAST_RETURN_NOISE_FACTOR_1YR = 0.3
FORECAST_RETURN_NOISE_FACTOR_2YR = 0.6
FORECAST_MORTALITY_NOISE_1YR = 0.01
FORECAST_MORTALITY_NOISE_2YR = 0.03

# --- Training Run Identification ---
# Suffix will be managed by WandB Sweep run names
MODEL_SAVE_DIR_BASE = "sweep_models" # Base directory for sweep artifacts

# --- WandB Configuration ---
WANDB_PROJECT_NAME = "DMO_RetPortfolio_Sweep" # Use a dedicated project for sweeps
WANDB_ENTITY = '' # Your W&B username or team name

# --- Training & PPO Configuration (Defaults, some will be swept) ---
# key point: Define sweep-specific training duration
SWEEP_TRAIN_STEPS = 150_000 # Reduce significantly for quick runs (e.g., 100k-200k)
# key point: Adjust eval frequency to happen within the short run
SWEEP_EVAL_FREQ = 50_000   # Evaluate at least once or twice per run

# Use slightly fewer CPU cores than total for stability
NUM_PARALLEL_ENVS = max(1, os.cpu_count() - 2 if os.cpu_count() else 4) # Reduced default if os.cpu_count is None
SEED = 42 # Use a fixed seed for comparable runs within sweep constraints

# Check for CUDA availability
if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"CUDA available! Using device: {DEVICE} ({torch.cuda.get_device_name(0)})")
else:
    DEVICE = "cpu"
    print(f"CUDA not available, using device: {DEVICE}")


# Default PPO Hyperparameters (will be overridden by wandb.config)
DEFAULT_PPO_CONFIG = {
    "learning_rate": 1e-4,
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 10, # Keep fixed for sweep
    "gamma": 0.99, # Keep fixed
    "gae_lambda": 0.95, # Keep fixed
    "clip_range": 0.2, # Keep fixed
    "ent_coef": 0.005,
    "vf_coef": 0.5, # Keep fixed
    "max_grad_norm": 0.5, # Keep fixed
    "device": DEVICE,
}

# Evaluation and Logging Settings
N_EVAL_EPISODES = 10 # Reduce eval episodes for faster evaluation during sweep
LOG_INTERVAL = 5 # Log more frequently during short runs


# ===-----------------------------------------------------------------------===
#    RETIREMENT ENVIRONMENT DEFINITION (Flattened Action Space Fix)
# ===-----------------------------------------------------------------------===
class RetirementEnv(gym.Env):
    """
    Custom Environment for Retirement Portfolio Optimization with Annuities.
    Action: Flattened Box space (portfolio weights + annuity purchase triggers).
    Observation: Current age, wealth, portfolio, forecasts, annuity ownership.
    Reward: Combination of QALYs, terminal wealth, and penalties.
    """
    metadata = {'render_modes': []}

    def __init__(self, kwargs):
        super().__init__()
        self.verbose = kwargs.get('verbose', 0) # Optional verbosity

        # --- Store parameters ---
        self.initial_wealth = float(kwargs.get('initial_wealth', INITIAL_WEALTH))
        self.start_age = int(kwargs.get('start_age', START_AGE))
        self.end_age = int(kwargs.get('end_age', END_AGE))
        self.time_horizon = self.end_age - self.start_age

        self.num_stocks = int(kwargs.get('num_stocks', NUM_STOCKS))
        self.num_commodities = int(kwargs.get('num_commodities', NUM_COMMODITIES))
        self.num_etfs = int(kwargs.get('num_etfs', NUM_ETFS))
        self.num_gov_bonds = int(kwargs.get('num_gov_bonds', NUM_GOV_BONDS))
        self.num_corp_bonds = int(kwargs.get('num_corp_bonds', NUM_CORP_BONDS))
        self.num_stable_inc = int(kwargs.get('num_stable_inc', NUM_STABLE_INC))
        self.total_assets = self.num_stocks + self.num_commodities + self.num_etfs + \
                            self.num_gov_bonds + self.num_corp_bonds + self.num_stable_inc

        self.survival_a = float(kwargs.get('survival_a', SURVIVAL_A))
        self.survival_k = float(kwargs.get('survival_k', SURVIVAL_K))
        self.survival_g = float(kwargs.get('survival_g', SURVIVAL_G))
        self.survival_c = float(kwargs.get('survival_c', SURVIVAL_C))

        # Load Annuity Parameters
        self.compulsory_annuity_payment = float(kwargs.get('compulsory_annuity_payment', COMPULSORY_ANNUITY_PAYMENT))
        self.opt_annuity_1_premium = float(kwargs.get('opt_annuity_1_premium', OPTIONAL_ANNUITY_1_PREMIUM))
        self.opt_annuity_1_payment = float(kwargs.get('opt_annuity_1_payment', OPTIONAL_ANNUITY_1_PAYMENT))
        self.opt_annuity_2_premium = float(kwargs.get('opt_annuity_2_premium', OPTIONAL_ANNUITY_2_PREMIUM))
        self.opt_annuity_2_payment = float(kwargs.get('opt_annuity_2_payment', OPTIONAL_ANNUITY_2_PAYMENT))

        # Coefficients
        self.qaly_base = float(kwargs.get('qaly_base', QALY_BASE))
        self.gamma_w = float(kwargs.get('gamma_w', GAMMA_W))
        self.theta = float(kwargs.get('theta', THETA))
        self.beta_d = float(kwargs.get('beta_d', BETA_D))

        self.risky_asset_indices = list(kwargs.get('risky_asset_indices', RISKY_ASSET_INDICES))
        self.critical_asset_index = int(kwargs.get('critical_asset_index', CRITICAL_ASSET_INDEX))

        self.forecast_return_noise_1yr = float(kwargs.get('forecast_return_noise_factor_1yr', FORECAST_RETURN_NOISE_FACTOR_1YR))
        self.forecast_return_noise_2yr = float(kwargs.get('forecast_return_noise_factor_2yr', FORECAST_RETURN_NOISE_FACTOR_2YR))
        self.forecast_mortality_noise_1yr = float(kwargs.get('forecast_mortality_noise_1yr', FORECAST_MORTALITY_NOISE_1YR))
        self.forecast_mortality_noise_2yr = float(kwargs.get('forecast_mortality_noise_2yr', FORECAST_MORTALITY_NOISE_2YR))

        # --- Define Action Space ---
        # key point: Modify Action Space - Flatten into a single Box
        # Shape: N (weights) + 1 (buy annuity 1 trigger) + 1 (buy annuity 2 trigger)
        action_dim = self.total_assets + 2
        # Use bounds like [-1, 1] for the trigger actions, we'll threshold them later
        # Keep original bounds for weights part (softmax handles scaling anyway)
        low_bounds = np.array([-10.0] * self.total_assets + [-1.0] * 2, dtype=np.float32)
        high_bounds = np.array([10.0] * self.total_assets + [1.0] * 2, dtype=np.float32)
        self.action_space = spaces.Box(low=low_bounds, high=high_bounds, shape=(action_dim,), dtype=np.float32)

        # --- Define Observation Space ---
        # Add flags indicating ownership of optional annuities
        # Shape: 1 (age) + 1 (log wealth) + N (weights) + N*2 (returns) + 2 (mortality) + 2 (annuity ownership flags)
        obs_dim = 1 + 1 + self.total_assets + self.total_assets * 2 + 2 + 2 # Added 2 for ownership flags

        self.observation_space = spaces.Dict({
            # Core state info (low-dimensional)
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(1 + 1,), dtype=np.float32),  # age, log_wealth
            # Current portfolio allocation
            "weights": spaces.Box(low=0.0, high=1.0, shape=(self.total_assets,), dtype=np.float32),
            # Normalized weights
            # Forecasts (grouped together)
            "forecasts": spaces.Box(low=-np.inf, high=np.inf, shape=(self.total_assets * 2 + 2,), dtype=np.float32),
            # return_1yr, return_2yr, mort_1yr, mort_2yr
            # Annuity ownership status
            "annuity_status": spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
            # opt_annuity_1_owned_flag, opt_annuity_2_owned_flag
        })
        # --- Internal State ---
        self.current_age = self.start_age
        self.current_wealth = self.initial_wealth
        self.current_weights = np.ones(self.total_assets, dtype=np.float32) / self.total_assets
        self.current_step = 0
        self.peak_wealth_critical_asset = 0.0
        self.is_alive = True
        # Add state flags for annuity ownership
        self.has_opt_annuity_1 = False
        self.has_opt_annuity_2 = False

        # --- Store reward components for info dict ---
        self.reward_components = {}

        # Initialize RNG
        self._seed()


    def _seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
             self._seed(seed)

        self.current_age = self.start_age
        self.current_wealth = self.initial_wealth
        self.current_weights = np.ones(self.total_assets, dtype=np.float32) / self.total_assets
        self.current_step = 0
        self.is_alive = True

        # Reset annuity ownership
        self.has_opt_annuity_1 = False
        self.has_opt_annuity_2 = False

        initial_critical_asset_value = self.current_wealth * self.current_weights[self.critical_asset_index]
        self.peak_wealth_critical_asset = max(EPSILON, initial_critical_asset_value)

        self.reward_components = {
            'qaly_comp': 0.0, 'dd_comp': 0.0, 'risky_comp': 0.0,
            'terminal_wealth_reward_comp': 0.0, 'shortfall_comp': 0.0
        }

        observation = self._get_obs()
        info = self._get_info()

        if np.any(np.isnan(observation)) or np.any(np.isinf(observation)):
            print(f"!!! WARNING: NaN/Inf detected in initial observation: {observation} !!!")
            observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        return observation, info

    def step(self, action):
        # key point: Interpret the flattened Box action
        if not isinstance(action, np.ndarray):
             action = np.array(action) # Ensure it's numpy

        # Check if action dimension matches expected dimension
        if action.shape[0] != self.total_assets + 2:
             raise ValueError(f"Received action of shape {action.shape} but expected ({self.total_assets + 2},)")

        portfolio_action_raw = action[:self.total_assets] # First N elements are for weights
        # Last two elements are continuous triggers for annuity purchase
        buy_opt_1_trigger = action[self.total_assets]
        buy_opt_2_trigger = action[self.total_assets + 1]

        # key point: Threshold the triggers to get discrete actions
        # Using > 0.0 as threshold since we used bounds [-1, 1]
        buy_opt_1_action = 1 if buy_opt_1_trigger > 0.0 else 0
        buy_opt_2_action = 1 if buy_opt_2_trigger > 0.0 else 0

        # --- 1. Apply Portfolio Action (Rebalancing) ---
        # Use softmax on the raw portfolio part of the action
        target_weights = softmax(portfolio_action_raw).astype(np.float32)
        target_weights /= (np.sum(target_weights) + EPSILON) # Ensure sum to 1
        self.current_weights = target_weights

        # --- 1.5. Handle Optional Annuity Purchases (using thresholded actions) ---
        purchase_cost = 0.0
        log_purchase_1 = False
        log_purchase_2 = False
        if buy_opt_1_action == 1 and not self.has_opt_annuity_1 and self.current_wealth >= self.opt_annuity_1_premium:
            purchase_cost += self.opt_annuity_1_premium
            self.has_opt_annuity_1 = True
            log_purchase_1 = True

        # Check wealth *after* potential first purchase
        if buy_opt_2_action == 1 and not self.has_opt_annuity_2 and (self.current_wealth - purchase_cost) >= self.opt_annuity_2_premium:
             purchase_cost += self.opt_annuity_2_premium
             self.has_opt_annuity_2 = True
             log_purchase_2 = True

        # Optional logging if verbose
        # if self.verbose > 1:
        #     if log_purchase_1: print(f"Step {self.current_step}: Purchased Opt Annuity 1 (Trigger: {buy_opt_1_trigger:.2f})")
        #     if log_purchase_2: print(f"Step {self.current_step}: Purchased Opt Annuity 2 (Trigger: {buy_opt_2_trigger:.2f})")


        # Deduct purchase cost
        self.current_wealth -= purchase_cost
        self.current_wealth = max(WEALTH_FLOOR, self.current_wealth)

        # --- 2. Simulate Environment Step (Market Returns, Expenses, Survival) ---
        simulated_results = self._simulate_step(self.current_weights)
        asset_returns = simulated_results['asset_returns']
        annual_expenses = simulated_results['expenses']
        survival_prob_step = simulated_results['survival_prob']

        # --- 3. Update Wealth (Market Growth & Annuity Income) ---
        wealth_after_returns = self.current_wealth * np.sum(self.current_weights * (1 + asset_returns))

        # Calculate and add annuity income for the year
        total_annuity_income = self.compulsory_annuity_payment # Compulsory payment
        if self.has_opt_annuity_1:
            total_annuity_income += self.opt_annuity_1_payment # Add payment if owned
        if self.has_opt_annuity_2:
            total_annuity_income += self.opt_annuity_2_payment # Add payment if owned

        wealth_before_expenses = wealth_after_returns + total_annuity_income
        wealth_after_expenses = wealth_before_expenses - annual_expenses
        self.current_wealth = max(WEALTH_FLOOR, wealth_after_expenses) # Apply floor


        # --- 4. Check Survival ---
        if not hasattr(self, 'np_random'): # Ensure RNG is initialized
             self._seed()
        if self.np_random.random() > survival_prob_step:
            self.is_alive = False

        # --- 5. Calculate Reward Components ---
        step_reward = 0.0
        qaly_comp = 0.0
        dd_comp = 0.0
        risky_comp = 0.0
        shortfall_comp = 0.0

        if self.is_alive:
            qaly_comp = self.qaly_base
            step_reward += qaly_comp

            fraction_in_risky = np.sum(self.current_weights[self.risky_asset_indices])
            raw_risky_penalty = -self.theta * fraction_in_risky
            risky_comp = np.clip(raw_risky_penalty, -MAX_PENALTY_MAGNITUDE_PER_STEP, 0)
            step_reward += risky_comp

            critical_asset_value = max(EPSILON, self.current_wealth * self.current_weights[self.critical_asset_index])
            self.peak_wealth_critical_asset = max(self.peak_wealth_critical_asset, critical_asset_value)
            drawdown_pct = (critical_asset_value - self.peak_wealth_critical_asset) / (self.peak_wealth_critical_asset + EPSILON)
            raw_dd_penalty = self.beta_d * drawdown_pct
            dd_comp = np.clip(raw_dd_penalty, -MAX_PENALTY_MAGNITUDE_PER_STEP, 0)
            step_reward += dd_comp

            if wealth_before_expenses < annual_expenses:
                 shortfall_pct = (annual_expenses - wealth_before_expenses) / (annual_expenses + EPSILON)
                 raw_shortfall_penalty = -self.beta_d * shortfall_pct * 5
                 shortfall_comp = np.clip(raw_shortfall_penalty, -MAX_PENALTY_MAGNITUDE_PER_STEP, 0)
                 step_reward += shortfall_comp

        self.reward_components['qaly_comp'] += qaly_comp
        self.reward_components['dd_comp'] += dd_comp
        self.reward_components['risky_comp'] += risky_comp
        self.reward_components['shortfall_comp'] += shortfall_comp

        # --- 6. Update State and Check Termination ---
        self.current_step += 1
        self.current_age += 1
        terminated = not self.is_alive or (self.current_age >= self.end_age)
        truncated = self.current_step >= self.time_horizon

        # --- 7. Calculate Terminal Reward ---
        terminal_wealth_reward_comp = 0.0
        if terminated or truncated:
            if self.is_alive:
                normalized_terminal_wealth = (self.current_wealth / (self.initial_wealth + EPSILON))
                raw_terminal_reward = normalized_terminal_wealth * self.gamma_w
                terminal_wealth_reward_comp = np.clip(raw_terminal_reward, 0, MAX_TERMINAL_REWARD_MAGNITUDE)
                step_reward += terminal_wealth_reward_comp
            self.reward_components['terminal_wealth_reward_comp'] = terminal_wealth_reward_comp

        # --- 8. Get Observation and Info ---
        observation = self._get_obs()
        info = self._get_info(terminated or truncated)

        # Safety checks
        if np.any(np.isnan(observation)) or np.any(np.isinf(observation)):
            print(f"!!! WARNING: NaN/Inf detected in observation: {observation} at step {self.current_step} !!!")
            observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            step_reward = -MAX_PENALTY_MAGNITUDE_PER_STEP * 2

        if np.isnan(step_reward) or np.isinf(step_reward):
            print(f"!!! WARNING: NaN/Inf detected in reward: {step_reward} at step {self.current_step} !!!")
            step_reward = -MAX_PENALTY_MAGNITUDE_PER_STEP * 2

        step_reward = float(step_reward)

        return observation, step_reward, terminated, truncated, info

    def _get_obs(self):
        """Construct the observation dictionary."""
        # Normalize age (e.g., 0 to 1 over the horizon)
        norm_age = (self.current_age - self.start_age) / (self.end_age - self.start_age)
        # Log wealth (prevents extreme values, scale if needed)
        log_wealth = np.log(max(WEALTH_FLOOR, self.current_wealth))

        # Forecasts (Placeholder: Replace with actual forecast logic)
        return_forecast_1yr = self._get_noisy_forecast(base_value=0.05, noise_factor=self.forecast_return_noise_1yr, size=self.total_assets)
        return_forecast_2yr = self._get_noisy_forecast(base_value=0.05, noise_factor=self.forecast_return_noise_2yr, size=self.total_assets)
        mortality_forecast_1yr = self._get_noisy_forecast(base_value=self._survival_prob(self.current_age + 1), noise_factor=self.forecast_mortality_noise_1yr, size=1)
        mortality_forecast_2yr = self._get_noisy_forecast(base_value=self._survival_prob(self.current_age + 2), noise_factor=self.forecast_mortality_noise_2yr, size=1)

        # Annuity ownership flags
        opt_annuity_1_owned_flag = 1.0 if self.has_opt_annuity_1 else 0.0
        opt_annuity_2_owned_flag = 1.0 if self.has_opt_annuity_2 else 0.0

        # **key point**: Construct the dictionary observation
        obs_dict = {
            "state": np.array([norm_age, log_wealth], dtype=np.float32),
            # Ensure weights passed are normalized and correct type
            "weights": self.current_weights.astype(np.float32),
            "forecasts": np.concatenate([
                return_forecast_1yr.astype(np.float32),
                return_forecast_2yr.astype(np.float32),
                mortality_forecast_1yr.astype(np.float32),
                mortality_forecast_2yr.astype(np.float32)
            ]).flatten(),
            "annuity_status": np.array([opt_annuity_1_owned_flag, opt_annuity_2_owned_flag], dtype=np.float32)
        }

        # Optional: Validate shapes just in case (useful for debugging)
        # assert obs_dict["state"].shape == self.observation_space["state"].shape
        # assert obs_dict["weights"].shape == self.observation_space["weights"].shape
        # assert obs_dict["forecasts"].shape == self.observation_space["forecasts"].shape
        # assert obs_dict["annuity_status"].shape == self.observation_space["annuity_status"].shape

        return obs_dict

    def _get_info(self, is_terminal=False):
        """Return supplementary info dictionary."""
        info = {
            "age": self.current_age,
            "wealth": self.current_wealth,
            "weights": self.current_weights.astype(np.float32),
            "is_alive": self.is_alive,
            # Add annuity status
            "has_opt_annuity_1": self.has_opt_annuity_1,
            "has_opt_annuity_2": self.has_opt_annuity_2,
        }
        if is_terminal:
            info["final_qaly_comp"] = float(self.reward_components['qaly_comp'])
            info["final_dd_comp"] = float(self.reward_components['dd_comp'])
            info["final_risky_comp"] = float(self.reward_components['risky_comp'])
            info["final_terminal_wealth_reward_comp"] = float(self.reward_components['terminal_wealth_reward_comp'])
            info["final_shortfall_comp"] = float(self.reward_components['shortfall_comp'])
            info["final_wealth"] = float(self.current_wealth)
        return info

    def _survival_prob(self, age):
        """Calculate survival probability."""
        if age < 0: return 1.0
        hazard_rate = self.survival_a * np.exp(self.survival_g * age) + self.survival_c
        prob = np.exp(-hazard_rate)
        return np.clip(prob, 0.0, 1.0)

    def _get_noisy_forecast(self, base_value, noise_factor, size):
        """Generate a noisy forecast."""
        if not hasattr(self, 'np_random'): self._seed()
        noise = self.np_random.normal(loc=0.0, scale=np.abs(base_value * noise_factor + EPSILON), size=size)
        forecast = base_value + noise
        return forecast

    def _simulate_step(self, weights):
        """Placeholder for the core simulation logic."""
        mean_returns = np.array([0.08, 0.06, 0.07, 0.02, 0.035, 0.01] * int(np.ceil(self.total_assets/6)))[:self.total_assets]
        volatilities = np.array([0.15, 0.18, 0.12, 0.03, 0.05, 0.005] * int(np.ceil(self.total_assets/6)))[:self.total_assets]

        if not hasattr(self, 'np_random'): self._seed()
        asset_returns = self.np_random.normal(loc=mean_returns, scale=volatilities).astype(np.float32)

        fixed_expense = 50000
        wealth_based_expense = 0.01 * self.current_wealth
        annual_expenses = fixed_expense + wealth_based_expense
        survival_prob_step = self._survival_prob(self.current_age)

        return {
            "asset_returns": asset_returns,
            "expenses": float(annual_expenses),
            "survival_prob": float(survival_prob_step)
        }

    def close(self):
        pass

# ===-----------------------------------------------------------------------===
#    CUSTOM EVALUATION CALLBACK DEFINITION (Keep as is)
# ===-----------------------------------------------------------------------===
class EvalCallbackWithComponents(BaseCallback):
    """
    Callback for evaluating the agent and logging reward components.
    Logs components from info dict retrieved from terminal_observation.
    """
    def __init__(self, eval_env, n_eval_episodes=5, eval_freq=10000,
                 log_path=None, best_model_save_path=None,
                 deterministic=True, verbose=1):
        super().__init__(verbose=verbose)

        if not isinstance(eval_env, gym.vector.VectorEnv):
            print("WARNING: eval_env passed to EvalCallbackWithComponents is not a VectorEnv. Check setup.")
        self.eval_env = eval_env

        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq # Total steps frequency
        self.log_path = log_path
        self.best_model_save_path = best_model_save_path
        self.deterministic = deterministic
        self.best_mean_reward = -np.inf
        self.last_eval_step = 0

        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        if self.best_model_save_path is not None:
            os.makedirs(os.path.dirname(self.best_model_save_path), exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval_step >= self.eval_freq :
            self.last_eval_step = self.num_timesteps

            if self.verbose > 0:
                log_msg = f"Evaluating agent at {self.num_timesteps} training steps..."
                if self.logger: self.logger.info(log_msg)
                else: print(log_msg)

            all_episode_rewards = []
            all_episode_lengths = []
            all_final_infos = []

            current_obs = self.eval_env.reset()
            episodes_done = 0
            current_rewards = np.zeros(self.eval_env.num_envs)
            current_lengths = np.zeros(self.eval_env.num_envs)

            while episodes_done < self.n_eval_episodes:
                action, _states = self.model.predict(current_obs, deterministic=self.deterministic)
                new_obs, rewards, dones, infos = self.eval_env.step(action)

                current_rewards += rewards
                current_lengths += 1

                for i in range(self.eval_env.num_envs):
                    if dones[i]:
                        if episodes_done < self.n_eval_episodes:
                            all_episode_rewards.append(current_rewards[i])
                            all_episode_lengths.append(current_lengths[i])

                            # Look for 'final_info' in the info dict for the finished env
                            final_info = None
                            if isinstance(infos, (list, tuple)) and i < len(infos):
                                final_info = infos[i].get("final_info") if infos[i] else None
                            elif isinstance(infos, dict) and "_final_info" in infos and infos["_final_info"][i]:
                                final_info = infos.get("final_info", [None] * self.eval_env.num_envs)[i]

                            all_final_infos.append(final_info if final_info is not None else {})
                            episodes_done += 1

                        current_rewards[i] = 0
                        current_lengths[i] = 0
                current_obs = new_obs

            mean_reward = np.mean(all_episode_rewards) if all_episode_rewards else 0
            std_reward = np.std(all_episode_rewards) if all_episode_rewards else 0
            mean_ep_length = np.mean(all_episode_lengths) if all_episode_lengths else 0

            if self.verbose > 0:
                 log_msg = f"Eval results: Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}, Mean Ep Length: {mean_ep_length:.2f}"
                 if self.logger: self.logger.info(log_msg)
                 else: print(log_msg)

            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/mean_ep_length", mean_ep_length)
            self.logger.record("eval/std_reward", std_reward)
            self.logger.record("eval/count", len(all_episode_rewards))

            valid_infos = [info for info in all_final_infos if info and isinstance(info, dict)]

            qaly_comps = [info.get("final_qaly_comp", np.nan) for info in valid_infos]
            dd_comps = [info.get("final_dd_comp", np.nan) for info in valid_infos]
            risky_comps = [info.get("final_risky_comp", np.nan) for info in valid_infos]
            term_wealth_comps = [info.get("final_terminal_wealth_reward_comp", np.nan) for info in valid_infos]
            shortfall_comps = [info.get("final_shortfall_comp", np.nan) for info in valid_infos]
            final_wealths = [info.get("final_wealth", np.nan) for info in valid_infos]
            # Log annuity purchase status (average number purchased per episode)
            final_opt1 = [1.0 if info.get("has_opt_annuity_1", False) else 0.0 for info in valid_infos]
            final_opt2 = [1.0 if info.get("has_opt_annuity_2", False) else 0.0 for info in valid_infos]


            self.logger.record("eval/mean_qaly_comp", np.nanmean(qaly_comps) if qaly_comps else 0)
            self.logger.record("eval/mean_dd_comp", np.nanmean(dd_comps) if dd_comps else 0)
            self.logger.record("eval/mean_risky_comp", np.nanmean(risky_comps) if risky_comps else 0)
            self.logger.record("eval/mean_terminal_wealth_reward_comp", np.nanmean(term_wealth_comps) if term_wealth_comps else 0)
            self.logger.record("eval/mean_shortfall_comp", np.nanmean(shortfall_comps) if shortfall_comps else 0)
            self.logger.record("eval/mean_final_wealth", np.nanmean(final_wealths) if final_wealths else 0)
            self.logger.record("eval/mean_opt_annuity_1_purchased", np.nanmean(final_opt1) if final_opt1 else 0)
            self.logger.record("eval/mean_opt_annuity_2_purchased", np.nanmean(final_opt2) if final_opt2 else 0)

            self.logger.dump(step=self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.best_model_save_path is not None:
                    save_path_model = os.path.join(self.best_model_save_path, "best_model.zip")
                    save_path_stats = os.path.join(self.best_model_save_path, "best_model_vecnormalize.pkl")
                    log_msg = f"New best model found! Saving model to {save_path_model}"
                    if self.verbose > 0:
                        if self.logger: self.logger.info(log_msg)
                        else: print(log_msg)
                    self.model.save(save_path_model)
                    training_env = self.model.get_env()
                    if isinstance(training_env, VecNormalize):
                        log_msg_stats = f"Saving VecNormalize stats from training env to {save_path_stats}"
                        if self.verbose > 0:
                             if self.logger: self.logger.info(log_msg_stats)
                             else: print(log_msg_stats)
                        training_env.save(save_path_stats)
        return True


# ===-----------------------------------------------------------------------===
#    HYPERPARAMETER LOGGING CALLBACK (Updated for MlpPolicy)
# ===-----------------------------------------------------------------------===
class HParamCallback(BaseCallback):
     """
     Saves hyperparameters and metrics at the start of training.
     """
     def _on_training_start(self) -> None:
         # Read actual values from the initialized model
         hparam_dict = {
             "algorithm": self.model.__class__.__name__,
             # key point: Policy is now MlpPolicy
             "policy": "MlpPolicy",
             "learning_rate": self.model.learning_rate,
             "n_steps": self.model.n_steps,
             "batch_size": self.model.batch_size,
             "ent_coef": self.model.ent_coef,
             "gamma": self.model.gamma,
             "gae_lambda": self.model.gae_lambda,
             "n_epochs": self.model.n_epochs,
             "vf_coef": self.model.vf_coef,
             "max_grad_norm": self.model.max_grad_norm,
             # Add sweep context
             "total_sweep_steps": SWEEP_TRAIN_STEPS,
             "num_parallel_envs": NUM_PARALLEL_ENVS,
             "device": self.model.device.type,
             # Add reward coefficients for context (ensure accessible, e.g., via global)
             "reward_gamma_w": GAMMA_W,
             "reward_theta": THETA,
             "reward_beta_d": BETA_D,
             "reward_qaly_base": QALY_BASE,
             # Add fixed annuity params for context
             "compulsory_annuity_payment": COMPULSORY_ANNUITY_PAYMENT,
             "opt_annuity_1_premium": OPTIONAL_ANNUITY_1_PREMIUM,
             "opt_annuity_2_premium": OPTIONAL_ANNUITY_2_PREMIUM,
         }
         # Resolve callable schedules if needed (e.g., clip_range)
         if callable(self.model.clip_range):
             hparam_dict["clip_range"] = self.model.clip_range(1.0) # Get value at end
         else:
              hparam_dict["clip_range"] = self.model.clip_range

         # Placeholder metrics for WandB HParams tab
         metric_dict = {
             "rollout/ep_rew_mean": 0.0,
             "eval/mean_reward": 0.0,
             "eval/mean_final_wealth": 0.0,
             "eval/mean_opt_annuity_1_purchased": 0.0,
             "eval/mean_opt_annuity_2_purchased": 0.0
         }
         try:
             self.logger.record(
                  "hparams",
                  HParam(hparam_dict, metric_dict),
                  exclude=("stdout", "log", "json", "csv"),
              )
         except Exception as e:
            print(f"Warning: Failed to record HParams in sweep run: {e}")

     def _on_step(self) -> bool:
         return True


# ===-----------------------------------------------------------------------===
#    MAIN TRAINING FUNCTION FOR SWEEP (Using MlpPolicy)
# ===-----------------------------------------------------------------------===
def train_sweep():
    """Function called by wandb.agent for each sweep run."""
    run = None # Initialize run to None
    train_env = None
    eval_env = None
    model = None
    run_save_dir = None

    try:
        # key point: Initialize WandB for this specific run
        run = wandb.init(project=WANDB_PROJECT_NAME, entity=WANDB_ENTITY, sync_tensorboard=True)

        # --- Access Swept Hyperparameters ---
        lr = wandb.config.learning_rate
        n_steps = wandb.config.n_steps
        batch_size = wandb.config.batch_size
        ent_coef = wandb.config.ent_coef

        # --- Create Run-Specific Save Directory ---
        run_save_dir = os.path.join(MODEL_SAVE_DIR_BASE, run.id)
        os.makedirs(run_save_dir, exist_ok=True)

        # --- Configure PPO based on sweep ---
        ppo_config = DEFAULT_PPO_CONFIG.copy()
        ppo_config["learning_rate"] = lr
        ppo_config["n_steps"] = n_steps
        ppo_config["batch_size"] = batch_size
        ppo_config["ent_coef"] = ent_coef

        # Validate batch_size
        total_buffer_size = n_steps * NUM_PARALLEL_ENVS
        if total_buffer_size % batch_size != 0:
             print(f"Warning: Run {run.id}: batch_size {batch_size} does not evenly divide total buffer size {total_buffer_size}. Training might fail or SB3 may adjust.")
             # Simple fix: Adjust batch size down to nearest power of 2 divisor? Or largest divisor?
             # For sweep simplicity, let SB3 handle it or potentially fail.
             # Robust fix: find largest divisor of total_buffer_size <= batch_size


        # --- Environment Setup (Uses fixed params and Flattened Action Env) ---
        env_kwargs = {
            'initial_wealth': INITIAL_WEALTH, 'start_age': START_AGE, 'end_age': END_AGE,
            'num_stocks': NUM_STOCKS, 'num_commodities': NUM_COMMODITIES, 'num_etfs': NUM_ETFS,
            'num_gov_bonds': NUM_GOV_BONDS, 'num_corp_bonds': NUM_CORP_BONDS, 'num_stable_inc': NUM_STABLE_INC,
            'survival_a': SURVIVAL_A, 'survival_k': SURVIVAL_K, 'survival_g': SURVIVAL_G, 'survival_c': SURVIVAL_C,
            'qaly_base': QALY_BASE, 'gamma_w': GAMMA_W, 'theta': THETA, 'beta_d': BETA_D,
            'risky_asset_indices': RISKY_ASSET_INDICES, 'critical_asset_index': CRITICAL_ASSET_INDEX,
            'forecast_return_noise_factor_1yr': FORECAST_RETURN_NOISE_FACTOR_1YR,
            'forecast_return_noise_factor_2yr': FORECAST_RETURN_NOISE_FACTOR_2YR,
            'forecast_mortality_noise_1yr': FORECAST_MORTALITY_NOISE_1YR,
            'forecast_mortality_noise_2yr': FORECAST_MORTALITY_NOISE_2YR,
            # Annuity params
            'compulsory_annuity_payment': COMPULSORY_ANNUITY_PAYMENT,
            'opt_annuity_1_premium': OPTIONAL_ANNUITY_1_PREMIUM,
            'opt_annuity_1_payment': OPTIONAL_ANNUITY_1_PAYMENT,
            'opt_annuity_2_premium': OPTIONAL_ANNUITY_2_PREMIUM,
            'opt_annuity_2_payment': OPTIONAL_ANNUITY_2_PAYMENT,
            'verbose': 0 # Less verbose env during sweep
        }
        train_env = make_vec_env(lambda: RetirementEnv(env_kwargs), n_envs=NUM_PARALLEL_ENVS, seed=SEED, vec_env_cls=SubprocVecEnv)
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.)

        eval_env = make_vec_env(lambda: RetirementEnv(env_kwargs), n_envs=1, seed=SEED + NUM_PARALLEL_ENVS)
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10., training=False)


        # --- Callback Setup ---
        evaluation_callback = EvalCallbackWithComponents(
            eval_env=eval_env,
            n_eval_episodes=N_EVAL_EPISODES,
            eval_freq=SWEEP_EVAL_FREQ,
            log_path=os.path.join(run_save_dir, 'eval_logs'),
            best_model_save_path=os.path.join(run_save_dir, 'best_model'),
            deterministic=True,
            verbose=0,
        )

        wandb_callback = OfficialWandbCallback(
             gradient_save_freq=0,
             model_save_path=None,
             model_save_freq=0,
             verbose=0
        )

        hparam_callback = HParamCallback()

        callback_list = CallbackList([evaluation_callback, hparam_callback, wandb_callback])


        # --- Model Definition ---
        # Corrected the policy name in the print statement and the PPO call as well
        print(f"Defining PPO model with MultiInputPolicy on device: {ppo_config['device']}...")
        model = PPO(
            "MultiInputPolicy", # Use the correct policy for Dict action space
            train_env,
            verbose=0, # Keyword argument
            seed=SEED, # Keyword argument
            tensorboard_log=f"runs/{run.id}", # Keyword argument
            **ppo_config # CORRECTED: Unpack the dictionary into keyword arguments
        )


        # --- Training ---
        print(f"\n--- Starting WandB Sweep Run: {run.name} ({run.id}) ---")
        print(f"--- Hyperparameters: LR={lr:.1E}, N_Steps={n_steps}, Batch={batch_size}, EntCoef={ent_coef:.4f} ---")
        print(f"--- Training for {SWEEP_TRAIN_STEPS:,} steps using {NUM_PARALLEL_ENVS} envs ---")
        start_time = time.time()
        model.learn(
            total_timesteps=SWEEP_TRAIN_STEPS,
            callback=callback_list,
            log_interval=LOG_INTERVAL,
            reset_num_timesteps=True
        )
        end_time = time.time()
        print(f"--- Finished WandB Sweep Run: {run.name} ({run.id}) in {end_time - start_time:.2f} seconds ---")

    except Exception as e:
        print(f"\n!!! SWEEP RUN {run.id if run else 'UNKNOWN'} FAILED: {e} !!!\n")
        import traceback
        traceback.print_exc()
        if run:
            try:
                 wandb.log({"error": traceback.format_exc()}) # Log traceback to wandb
            except Exception as log_e:
                 print(f"Failed to log error to wandb: {log_e}")

    finally:
        # --- Clean up ---
        print(f"Closing environments for run {run.id if run else 'UNKNOWN'}...")
        try:
            if train_env is not None: train_env.close()
        except Exception as e: print(f"Error closing train_env: {e}")
        try:
            if eval_env is not None: eval_env.close()
        except Exception as e: print(f"Error closing eval_env: {e}")

        if run:
            # Save final VecNormalize stats (optional for sweep)
            if model is not None and run_save_dir is not None and isinstance(model.get_env(), VecNormalize):
                 try:
                     stats_path = os.path.join(run_save_dir, "final_vec_normalize.pkl")
                     model.get_env().save(stats_path)
                     print(f"Saved final normalize stats for run {run.id}")
                 except Exception as e:
                     print(f"Could not save final normalize stats for run {run.id}: {e}")

            print(f"Finishing WandB run {run.id}...")
            run.finish()


# ===-----------------------------------------------------------------------===
#    SWEEP CONFIGURATION AND EXECUTION
# ===-----------------------------------------------------------------------===

if __name__ == "__main__":
    # key point: Define the sweep configuration dictionary
    sweep_config = {
        'method': 'bayes',  # Bayesian optimization is efficient
        'metric': {
            'name': 'eval/mean_reward', # Optimize for this metric logged by EvalCallback
            'goal': 'maximize'
        },
        'parameters': {
            'learning_rate': {
                # Logarithmic range for learning rate
                'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 5e-4
            },
            'n_steps': {
                # Choose from a few common values for rollout buffer size
                'distribution': 'categorical',
                'values': [1024, 2048, 4096]
            },
            'batch_size': {
                # Powers of 2 around the previous default
                # Ensure VRAM can handle the largest value selected here
                'distribution': 'categorical',
                'values': [64, 128, 256]
            },
            'ent_coef': {
                # Explore values around the default, including zero
                'distribution': 'categorical',
                'values': [0.0, 0.001, 0.005, 0.01]
            },
            # --- Keep other PPO params fixed as defined in DEFAULT_PPO_CONFIG ---
        }
    } # End sweep_config

    print("--- Starting WandB Sweep ---")
    # key point: Create the sweep
    # Ensure WANDB_ENTITY is set or remove '/{WANDB_ENTITY}' if using default entity
    sweep_id = wandb.sweep(sweep_config, project=WANDB_PROJECT_NAME, entity=WANDB_ENTITY or None) # Use None if entity is empty string

    print(f"--- Sweep ID: {sweep_id} ---")
    entity_str = WANDB_ENTITY or "[YOUR WANDB ENTITY/USERNAME]" # Placeholder if empty
    print(f"--- Run agent with: wandb agent {entity_str}/{WANDB_PROJECT_NAME}/{sweep_id} ---")

    # key point: Run the sweep agent
    # Set 'count' to limit the number of runs (e.g., 4-6 for 10-15 mins)
    num_sweep_runs = 5 # Adjust based on estimated time per run
    print(f"--- Starting WandB agent for {num_sweep_runs} runs ---")
    try:
        # **key point**: Specify sweep_id using the keyword argument
        wandb.agent(sweep_id=sweep_id, function=train_sweep, count=num_sweep_runs)
    except KeyboardInterrupt:
        print("Sweep agent stopped manually.")
    except Exception as e:
        print(f"\nSweep agent encountered an error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("--- WandB Sweep Finished ---")

