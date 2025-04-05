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
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList, EvalCallback # Use SB3's EvalCallback for base
from stable_baselines3.common.logger import HParam
from wandb.integration.sb3 import WandbCallback as OfficialWandbCallback # Use the official one

# ===-----------------------------------------------------------------------===
#    CONFIGURATION PARAMETERS (MODIFIED BASED ON FEEDBACK)
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
END_AGE = 120 # Episode ends if agent reaches this age
TIME_HORIZON = END_AGE - START_AGE # T = 55 steps (years)

# --- Survival Probability Parameters ---
SURVIVAL_A = 0.00005
SURVIVAL_K = 5.6
SURVIVAL_G = 0.055
SURVIVAL_C = 0.0001

# --- Reward/Penalty Coefficients (MODIFIED) ---
# **key point**: Increased terminal wealth reward scaling
GAMMA_W = 6.0    # Terminal Wealth Reward Scaling (was 4.0)
# **key point**: Decreased risky asset penalty
THETA = 0.05     # Risky Asset Holding Penalty Coefficient (was 0.15)
# **key point**: Decreased drawdown/shortfall penalty base
BETA_D = 0.5     # Drawdown & Shortfall Base Penalty Coefficient (was 1.0)
QALY_BASE = 0.3 # QALY Base Value per year
# **key point**: Added direct reward for annuity purchase
ANNUITY_PURCHASE_BONUS = 5.0 # Small reward for buying an optional annuity

# --- Numerical Stability/Clipping ---
MAX_PENALTY_MAGNITUDE_PER_STEP = 100.0
MAX_TERMINAL_REWARD_MAGNITUDE = 1000.0
WEALTH_FLOOR = 1.0
EPSILON = 1e-8

# --- Asset Indices Definitions ---
RISKY_ASSET_INDICES = list(range(NUM_STOCKS + NUM_COMMODITIES + NUM_ETFS))
CRITICAL_ASSET_INDEX = NUM_STOCKS + NUM_COMMODITIES

# --- Forecast Noise Parameters ---
FORECAST_RETURN_NOISE_FACTOR_1YR = 0.3
FORECAST_RETURN_NOISE_FACTOR_2YR = 0.6
FORECAST_MORTALITY_NOISE_1YR = 0.01
FORECAST_MORTALITY_NOISE_2YR = 0.03

# ===-----------------------------------------------------------------------===
#    ANNUITY PARAMETERS
# ===-----------------------------------------------------------------------===

# --- Compulsory Annuity (Cost=0, fixed annual payment) ---
COMPULSORY_ANNUITY_PAYMENT = 10000.0

# --- Optional Annuity 1 ---
OPTIONAL_ANNUITY_1_PREMIUM = 150000.0 # Cost to buy
OPTIONAL_ANNUITY_1_PAYMENT = 7500.0   # Fixed annual payment once bought
# Consider increasing payment or decreasing premium if still not bought:
# OPTIONAL_ANNUITY_1_PAYMENT = 9000.0

# --- Optional Annuity 2 ---
OPTIONAL_ANNUITY_2_PREMIUM = 225000.0
OPTIONAL_ANNUITY_2_PAYMENT = 12375.0
# Consider increasing payment or decreasing premium if still not bought:
# OPTIONAL_ANNUITY_2_PAYMENT = 14000.0


# --- Training Run Identification ---
RUN_NAME_SUFFIX = "Full_Embedded_BalancedReward_Annuities_FlattenedAction_HW_Optimized_V2" # Updated suffix
MODEL_SAVE_DIR = f"models/{int(time.time())}_{RUN_NAME_SUFFIX}"

# --- WandB Configuration ---
WANDB_PROJECT_NAME = "DMO_RetPortfolio_V2" # Maybe new project or V2 tag
WANDB_ENTITY = '' # Your W&B username or team name, or None for default

# --- Training & PPO Configuration (MODIFIED FOR RESOURCE UTILIZATION) ---
TOTAL_TRAINING_TIMESTEPS = 5_000_000
# **key point**: Increase parallel environments to utilize more CPU cores.
# Use all available logical cores, or slightly fewer if system becomes unstable.
NUM_PARALLEL_ENVS = os.cpu_count() if os.cpu_count() else 16
NUM_PARALLEL_ENVS = max(1, NUM_PARALLEL_ENVS) # Ensure at least 1
SEED = 42

# Check for CUDA availability
if torch.cuda.is_available():
    print(f"CUDA (GPU) available! Using device: cuda ({torch.cuda.get_device_name(0)})")
    DEVICE = "cuda"
else:
    print("CUDA not available, using CPU.")
    DEVICE = "cpu"

# PPO Hyperparameters (MODIFIED FOR RESOURCE UTILIZATION)
PPO_CONFIG = {
    "learning_rate": 1e-4,
    "n_steps": 2048, # Standard, can increase to 4096 if desired
    # **key point**: Increased batch size significantly for better GPU utilization
    "batch_size": 512, # Was 256, consider 1024 if VRAM allows (ensure batch_size <= n_steps * NUM_PARALLEL_ENVS)
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.005, # Consider slightly increasing (e.g., 0.01) to encourage exploration
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    # **key point**: Explicitly set device to leverage GPU/CPU
    "device": DEVICE,
}

# Evaluation and Logging Settings
EVAL_FREQ_TOTAL_STEPS = 50_000
N_EVAL_EPISODES = 30
LOG_INTERVAL = 20

# ===-----------------------------------------------------------------------===
#    RETIREMENT ENVIRONMENT DEFINITION (MODIFIED REWARDS)
# ===-----------------------------------------------------------------------===

class RetirementEnv(gym.Env):
    """
    Custom Environment for Retirement Portfolio Optimization with Annuities.
    Action: Flattened Box space (Portfolio weights + Annuity purchase signals).
    Observation: Box space (Age, Wealth, Weights, Forecasts, Annuity Ownership).
    Reward: Combination of QALYs, terminal wealth, purchase bonus, and penalties.
    """
    metadata = {'render_modes': []}

    def __init__(self, **kwargs):
        super().__init__()

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

        self.qaly_base = float(kwargs.get('qaly_base', QALY_BASE))
        self.gamma_w = float(kwargs.get('gamma_w', GAMMA_W))
        self.theta = float(kwargs.get('theta', THETA))
        self.beta_d = float(kwargs.get('beta_d', BETA_D))
        # **key point**: Added annuity purchase bonus parameter
        self.annuity_purchase_bonus = float(kwargs.get('annuity_purchase_bonus', ANNUITY_PURCHASE_BONUS))

        self.risky_asset_indices = list(kwargs.get('risky_asset_indices', RISKY_ASSET_INDICES))
        self.critical_asset_index = int(kwargs.get('critical_asset_index', CRITICAL_ASSET_INDEX))

        self.forecast_return_noise_1yr = float(kwargs.get('forecast_return_noise_factor_1yr', FORECAST_RETURN_NOISE_FACTOR_1YR))
        self.forecast_return_noise_2yr = float(kwargs.get('forecast_return_noise_factor_2yr', FORECAST_RETURN_NOISE_FACTOR_2YR))
        self.forecast_mortality_noise_1yr = float(kwargs.get('forecast_mortality_noise_1yr', FORECAST_MORTALITY_NOISE_1YR))
        self.forecast_mortality_noise_2yr = float(kwargs.get('forecast_mortality_noise_2yr', FORECAST_MORTALITY_NOISE_2YR))

        self.compulsory_annuity_payment = float(kwargs.get('compulsory_annuity_payment', COMPULSORY_ANNUITY_PAYMENT))
        self.opt_annuity_1_premium = float(kwargs.get('opt_annuity_1_premium', OPTIONAL_ANNUITY_1_PREMIUM))
        self.opt_annuity_1_payment = float(kwargs.get('opt_annuity_1_payment', OPTIONAL_ANNUITY_1_PAYMENT))
        self.opt_annuity_2_premium = float(kwargs.get('opt_annuity_2_premium', OPTIONAL_ANNUITY_2_PREMIUM))
        self.opt_annuity_2_payment = float(kwargs.get('opt_annuity_2_payment', OPTIONAL_ANNUITY_2_PAYMENT))

        self.verbose = int(kwargs.get('verbose', 0)) # Get verbose flag for debugging

        # --- Define Action Space ---
        action_dim = self.total_assets + 2 # Portfolio weights + 2 annuity decisions
        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(action_dim,), dtype=np.float32)

        # --- Define Observation Space ---
        obs_dim = 1 + 1 + self.total_assets + self.total_assets * 2 + 2 + 2 # Added 2 for ownership flags
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # --- Internal State ---
        self.current_age = self.start_age
        self.current_wealth = self.initial_wealth
        self.current_weights = np.ones(self.total_assets, dtype=np.float32) / self.total_assets
        self.current_step = 0
        self.peak_wealth_critical_asset = 0.0
        self.is_alive = True
        self.has_opt_annuity_1 = False
        self.has_opt_annuity_2 = False
        self.reward_components = {}


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_age = self.start_age
        self.current_wealth = self.initial_wealth
        self.current_weights = np.ones(self.total_assets, dtype=np.float32) / self.total_assets
        self.current_step = 0
        self.is_alive = True

        initial_critical_asset_value = self.current_wealth * self.current_weights[self.critical_asset_index]
        self.peak_wealth_critical_asset = max(EPSILON, initial_critical_asset_value)

        self.has_opt_annuity_1 = False
        self.has_opt_annuity_2 = False

        # **key point**: Initialize all potential reward components
        self.reward_components = {
            'qaly_comp': 0.0, 'dd_comp': 0.0, 'risky_comp': 0.0,
            'terminal_wealth_reward_comp': 0.0, 'shortfall_comp': 0.0,
            'annuity_bonus_comp': 0.0 # Add bonus component tracking
        }

        observation = self._get_obs()
        info = self._get_info()

        if np.any(np.isnan(observation)) or np.any(np.isinf(observation)):
            print(f"!!! WARNING: NaN/Inf detected in initial observation: {observation} !!!")
            observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        return observation, info

    def step(self, action):
        # Initialize step reward and components for this step
        step_reward = 0.0
        qaly_comp = 0.0
        dd_comp = 0.0
        risky_comp = 0.0
        shortfall_comp = 0.0
        annuity_bonus_this_step = 0.0 # Track bonus specifically for this step

        # **key point**: Unpack the flattened action vector
        flat_action = action
        portfolio_action = flat_action[:self.total_assets]
        buy_opt_1_signal = flat_action[self.total_assets]
        buy_opt_2_signal = flat_action[self.total_assets + 1]

        # --- 1. Apply Portfolio Action (Rebalancing) ---
        target_weights = softmax(portfolio_action).astype(np.float32)
        target_weights /= np.sum(target_weights) + EPSILON
        self.current_weights = target_weights

        # --- 1.5. Handle Optional Annuity Purchases ---
        purchase_cost = 0.0
        if buy_opt_1_signal > 0.0 and not self.has_opt_annuity_1 and self.current_wealth >= self.opt_annuity_1_premium:
            purchase_cost += self.opt_annuity_1_premium
            self.has_opt_annuity_1 = True
            # **key point**: Add annuity purchase bonus to this step's reward
            annuity_bonus_this_step += self.annuity_purchase_bonus
            if self.verbose > 1: print(f"Step {self.current_step}: Purchased Optional Annuity 1 (Signal: {buy_opt_1_signal:.2f}, Bonus: {self.annuity_purchase_bonus})")

        if buy_opt_2_signal > 0.0 and not self.has_opt_annuity_2 and self.current_wealth >= self.opt_annuity_2_premium:
            if self.current_wealth - purchase_cost >= self.opt_annuity_2_premium: # Check wealth AFTER potential first purchase
                purchase_cost += self.opt_annuity_2_premium
                self.has_opt_annuity_2 = True
                # **key point**: Add annuity purchase bonus to this step's reward
                annuity_bonus_this_step += self.annuity_purchase_bonus
                if self.verbose > 1: print(f"Step {self.current_step}: Purchased Optional Annuity 2 (Signal: {buy_opt_2_signal:.2f}, Bonus: {self.annuity_purchase_bonus})")

        # Apply bonus to step reward and accumulate total bonus
        step_reward += annuity_bonus_this_step
        self.reward_components['annuity_bonus_comp'] += annuity_bonus_this_step

        # Deduct purchase cost immediately
        self.current_wealth -= purchase_cost
        self.current_wealth = max(WEALTH_FLOOR, self.current_wealth)

        # --- 2. Simulate Environment Step (Market Returns, Expenses, Survival) ---
        simulated_results = self._simulate_step(self.current_weights)
        asset_returns = simulated_results['asset_returns']
        annual_expenses = simulated_results['expenses']
        survival_prob_step = simulated_results['survival_prob']

        # --- 3. Update Wealth (Market Growth & Annuity Income) ---
        wealth_after_returns = self.current_wealth * np.sum(self.current_weights * (1 + asset_returns))

        total_annuity_income = self.compulsory_annuity_payment
        if self.has_opt_annuity_1:
            total_annuity_income += self.opt_annuity_1_payment
        if self.has_opt_annuity_2:
            total_annuity_income += self.opt_annuity_2_payment

        wealth_before_expenses = wealth_after_returns + total_annuity_income
        wealth_after_expenses = wealth_before_expenses - annual_expenses
        self.current_wealth = max(WEALTH_FLOOR, wealth_after_expenses)

        # --- 4. Check Survival ---
        if not hasattr(self, 'np_random'):
             self.np_random, _ = gym.utils.seeding.np_random(None)
        if self.np_random.random() > survival_prob_step:
            self.is_alive = False

        # --- 5. Calculate Other Reward Components ---
        terminal_wealth_reward_comp = 0.0 # Initialize here for clarity

        if self.is_alive:
            # a) QALY Reward
            qaly_comp = self.qaly_base
            step_reward += qaly_comp

            # b) Risky Asset Holding Penalty (using MODIFIED self.theta)
            fraction_in_risky = np.sum(self.current_weights[self.risky_asset_indices])
            raw_risky_penalty = -self.theta * fraction_in_risky
            risky_comp = np.clip(raw_risky_penalty, -MAX_PENALTY_MAGNITUDE_PER_STEP, 0)
            step_reward += risky_comp

            # c) Drawdown Penalty (using MODIFIED self.beta_d)
            critical_asset_value = self.current_wealth * self.current_weights[self.critical_asset_index]
            critical_asset_value = max(EPSILON, critical_asset_value)
            self.peak_wealth_critical_asset = max(self.peak_wealth_critical_asset, critical_asset_value)
            drawdown_pct = (critical_asset_value - self.peak_wealth_critical_asset) / (self.peak_wealth_critical_asset + EPSILON)
            raw_dd_penalty = self.beta_d * drawdown_pct # Use modified beta_d
            dd_comp = np.clip(raw_dd_penalty, -MAX_PENALTY_MAGNITUDE_PER_STEP, 0)
            step_reward += dd_comp

            # d) Shortfall Check (using MODIFIED self.beta_d and multiplier)
            if wealth_before_expenses < annual_expenses:
                 shortfall_pct = (annual_expenses - wealth_before_expenses) / (annual_expenses + EPSILON)
                 # **key point**: Reduced shortfall penalty multiplier
                 raw_shortfall_penalty = -self.beta_d * shortfall_pct * 2 # Was * 5
                 shortfall_comp = np.clip(raw_shortfall_penalty, -MAX_PENALTY_MAGNITUDE_PER_STEP, 0)
                 step_reward += shortfall_comp

        # Accumulate components for the episode (excluding terminal reward for now)
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
        if terminated or truncated:
            if self.is_alive: # Only give terminal wealth reward if agent survived to the end
                normalized_terminal_wealth = (self.current_wealth / (self.initial_wealth + EPSILON))
                # Use MODIFIED self.gamma_w
                raw_terminal_reward = normalized_terminal_wealth * self.gamma_w
                terminal_wealth_reward_comp = np.clip(raw_terminal_reward, 0, MAX_TERMINAL_REWARD_MAGNITUDE)
                step_reward += terminal_wealth_reward_comp
            # Store the final terminal reward component regardless of whether it was added to step_reward
            self.reward_components['terminal_wealth_reward_comp'] = terminal_wealth_reward_comp

        # --- 8. Get Observation and Info ---
        observation = self._get_obs()
        info = self._get_info(terminated or truncated) # Pass terminal flag

        # Safety checks
        if np.any(np.isnan(observation)) or np.any(np.isinf(observation)):
            print(f"!!! WARNING: NaN/Inf detected in observation: {observation} at step {self.current_step} !!!")
            observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            step_reward = -MAX_PENALTY_MAGNITUDE_PER_STEP * 2 # Heavy penalty

        if np.isnan(step_reward) or np.isinf(step_reward):
            print(f"!!! WARNING: NaN/Inf detected in reward: {step_reward} at step {self.current_step} !!!")
            step_reward = -MAX_PENALTY_MAGNITUDE_PER_STEP * 2 # Heavy penalty

        step_reward = float(step_reward)

        return observation, step_reward, terminated, truncated, info

    def _get_obs(self):
        """Construct the observation array including annuity ownership."""
        norm_age = (self.current_age - self.start_age) / (self.end_age - self.start_age)
        log_wealth = np.log(max(WEALTH_FLOOR, self.current_wealth))

        return_forecast_1yr = self._get_noisy_forecast(base_value=0.05, noise_factor=self.forecast_return_noise_1yr, size=self.total_assets)
        return_forecast_2yr = self._get_noisy_forecast(base_value=0.05, noise_factor=self.forecast_return_noise_2yr, size=self.total_assets)
        mortality_forecast_1yr = self._get_noisy_forecast(base_value=self._survival_prob(self.current_age + 1), noise_factor=self.forecast_mortality_noise_1yr, size=1)
        mortality_forecast_2yr = self._get_noisy_forecast(base_value=self._survival_prob(self.current_age + 2), noise_factor=self.forecast_mortality_noise_2yr, size=1)

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
        """Return supplementary info dictionary including accumulated rewards."""
        info = {
            "age": self.current_age,
            "wealth": self.current_wealth,
            "weights": self.current_weights.astype(np.float32),
            "is_alive": self.is_alive,
            "has_opt_annuity_1": self.has_opt_annuity_1,
            "has_opt_annuity_2": self.has_opt_annuity_2,
        }
        if is_terminal:
            # **key point**: Added QALY debug print here
            qaly_val = self.reward_components.get('qaly_comp', 0.0)
            if self.verbose > 1: # Only print if verbose debugging is on
                 print(f"DEBUG (_get_info): Final Step {self.current_step}, Age {self.current_age}, Alive: {self.is_alive}, Accumulated QALY: {qaly_val}")

            # Populate final components, using .get for safety
            info["final_qaly_comp"] = float(qaly_val)
            info["final_dd_comp"] = float(self.reward_components.get('dd_comp', 0.0))
            info["final_risky_comp"] = float(self.reward_components.get('risky_comp', 0.0))
            info["final_terminal_wealth_reward_comp"] = float(self.reward_components.get('terminal_wealth_reward_comp', 0.0))
            info["final_shortfall_comp"] = float(self.reward_components.get('shortfall_comp', 0.0))
            # **key point**: Added annuity bonus component logging
            info["final_annuity_bonus_comp"] = float(self.reward_components.get('annuity_bonus_comp', 0.0))
            info["final_wealth"] = float(self.current_wealth)

            # Calculate final total reward from components for verification
            total_comp_reward = sum(self.reward_components.values())
            info["final_total_reward_from_components"] = float(total_comp_reward)

        return info

    def _survival_prob(self, age):
        """Calculate survival probability."""
        if age < 0: return 1.0
        hazard_rate = self.survival_a * np.exp(self.survival_g * age) + self.survival_c
        prob = np.exp(-hazard_rate)
        return np.clip(prob, 0.0, 1.0)

    def _get_noisy_forecast(self, base_value, noise_factor, size):
        """Generate a noisy forecast."""
        if not hasattr(self, 'np_random'):
             self.np_random, _ = gym.utils.seeding.np_random(None)
        noise = self.np_random.normal(loc=0.0, scale=np.abs(base_value * noise_factor + EPSILON), size=size)
        forecast = base_value + noise
        return forecast

    def _simulate_step(self, weights):
        """Placeholder for the core simulation logic."""
        # Example placeholder - REPLACE WITH YOUR FINANCIAL MODEL
        mean_returns = np.array([0.08, 0.06, 0.07, 0.02, 0.035, 0.01] * int(np.ceil(self.total_assets/6)))[:self.total_assets]
        volatilities = np.array([0.15, 0.18, 0.12, 0.03, 0.05, 0.005] * int(np.ceil(self.total_assets/6)))[:self.total_assets]

        if not hasattr(self, 'np_random'):
             self.np_random, _ = gym.utils.seeding.np_random(None)

        asset_returns = self.np_random.normal(loc=mean_returns, scale=volatilities).astype(np.float32)

        # Example Expenses: Fixed + % of wealth (Ensure this is realistic)
        fixed_expense = 50000 # Example: $50k/year base
        variable_expense = 0.005 * self.current_wealth # Example: 0.5% of current wealth
        annual_expenses = fixed_expense + variable_expense

        survival_prob_step = self._survival_prob(self.current_age)

        return {
            "asset_returns": asset_returns,
            "expenses": float(annual_expenses),
            "survival_prob": float(survival_prob_step)
        }

    def close(self):
        pass # Cleanup if needed

# ===-----------------------------------------------------------------------===
#    CUSTOM EVALUATION CALLBACK DEFINITION (MODIFIED LOGGING & DEBUG)
# ===-----------------------------------------------------------------------===

class EvalCallbackWithComponents(BaseCallback):
    """
    Callback for evaluating the agent and logging reward components.
    Assumes eval_env is a VecEnv. Logs components from info dict. Adds Debug prints.
    """
    def __init__(self, eval_env, n_eval_episodes=5, eval_freq=10000,
                 log_path=None, best_model_save_path=None,
                 deterministic=True, verbose=1, debug_qaly=False): # Added debug flag
        super().__init__(verbose=verbose)

        if not isinstance(eval_env, gym.vector.VectorEnv):
             print(f"WARNING in EvalCallbackWithComponents: eval_env is of type {type(eval_env)}, not VectorEnv. Ensure it's wrapped correctly (e.g., DummyVecEnv).")
        self.eval_env = eval_env

        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq # TOTAL training timesteps frequency
        self.log_path = log_path
        self.best_model_save_path = best_model_save_path
        self.deterministic = deterministic
        self.best_mean_reward = -np.inf
        self.last_eval_step = 0
        self.debug_qaly = debug_qaly # Store debug flag

        if self.log_path is not None: os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        if self.best_model_save_path is not None: os.makedirs(os.path.dirname(self.best_model_save_path), exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval_step >= self.eval_freq :
            self.last_eval_step = self.num_timesteps
            if self.verbose > 0: print(f"Evaluating agent at {self.num_timesteps} training steps...")

            all_episode_rewards = []
            all_episode_lengths = []
            all_final_infos = []

            # Reset evaluation env
            current_obs = self.eval_env.reset()
            # Manually handle VecEnv state tracking for episodes
            episodes_done = 0
            num_envs = self.eval_env.num_envs
            current_rewards = np.zeros(num_envs)
            current_lengths = np.zeros(num_envs)
            episode_starts = np.ones(num_envs, dtype=bool)

            while episodes_done < self.n_eval_episodes:
                action, _states = self.model.predict(current_obs, deterministic=self.deterministic)
                new_obs, rewards, dones, infos = self.eval_env.step(action)

                current_rewards += rewards
                current_lengths += 1

                for i in range(num_envs):
                    if dones[i]:
                        # Only record if we haven't finished the required number of episodes
                        if episodes_done < self.n_eval_episodes:
                            if self.verbose > 1: print(f"Eval Episode {episodes_done+1} finished. Reward: {current_rewards[i]}, Length: {current_lengths[i]}")
                            all_episode_rewards.append(current_rewards[i])
                            all_episode_lengths.append(current_lengths[i])
                            # Correctly extract final info from VecEnv
                            if isinstance(infos, (list, tuple)):
                                final_info = infos[i].get("final_info", None)
                            else: # Handle older gym/SB3 structure if necessary
                                final_info = infos.get("final_info", [None]*num_envs)[i]

                            all_final_infos.append(final_info if final_info is not None else {})
                            episodes_done += 1

                        # Reset the state for this env
                        current_rewards[i] = 0
                        current_lengths[i] = 0
                        episode_starts[i] = True # Mark for potential reset in next step if needed by VecEnv wrapper

                # Important: Use the new observations
                current_obs = new_obs


            mean_reward = np.mean(all_episode_rewards) if all_episode_rewards else 0
            std_reward = np.std(all_episode_rewards) if all_episode_rewards else 0
            mean_ep_length = np.mean(all_episode_lengths) if all_episode_lengths else 0

            if self.verbose > 0:
                print(f"Eval results: Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}, Mean Ep Length: {mean_ep_length:.2f}")

            # Log standard metrics
            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/mean_ep_length", mean_ep_length)
            self.logger.record("eval/std_reward", std_reward)
            self.logger.record("eval/count", len(all_episode_rewards))

            # Log components from final info dictionaries
            valid_infos = [info for info in all_final_infos if info and isinstance(info, dict)]

            # Helper function to safely extract and calculate mean
            def get_mean_component(key, default_val=np.nan):
                vals = [info.get(key, default_val) for info in valid_infos]
                # **key point**: Add debug print for QALY if requested
                if self.debug_qaly and key == "final_qaly_comp":
                     print(f"DEBUG (EvalCallback): Raw qaly_comps list: {vals}")
                mean_val = np.nanmean(vals) if vals else 0
                if self.debug_qaly and key == "final_qaly_comp":
                     print(f"DEBUG (EvalCallback): Calculated mean_qaly_comp: {mean_val}")
                return mean_val if not np.isnan(mean_val) else 0 # Return 0 if result is NaN

            # Log all components
            self.logger.record("eval/mean_qaly_comp", get_mean_component("final_qaly_comp"))
            self.logger.record("eval/mean_dd_comp", get_mean_component("final_dd_comp"))
            self.logger.record("eval/mean_risky_comp", get_mean_component("final_risky_comp"))
            self.logger.record("eval/mean_terminal_wealth_reward_comp", get_mean_component("final_terminal_wealth_reward_comp"))
            self.logger.record("eval/mean_shortfall_comp", get_mean_component("final_shortfall_comp"))
            self.logger.record("eval/mean_final_wealth", get_mean_component("final_wealth"))
            # **key point**: Log new annuity bonus component
            self.logger.record("eval/mean_annuity_bonus_comp", get_mean_component("final_annuity_bonus_comp"))

            # Log annuity purchase status during evaluation
            opt1_bought_final = [info.get("has_opt_annuity_1", False) for info in valid_infos]
            opt2_bought_final = [info.get("has_opt_annuity_2", False) for info in valid_infos]
            self.logger.record("eval/frac_bought_opt_annuity_1", np.mean(opt1_bought_final) if opt1_bought_final else 0)
            self.logger.record("eval/frac_bought_opt_annuity_2", np.mean(opt2_bought_final) if opt2_bought_final else 0)

            # Log total reward from components for comparison
            self.logger.record("eval/mean_total_reward_from_components", get_mean_component("final_total_reward_from_components"))


            self.logger.dump(step=self.num_timesteps)

            # Save best model logic
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.best_model_save_path is not None:
                    save_path_model = os.path.join(self.best_model_save_path, "best_model.zip")
                    save_path_stats = os.path.join(self.best_model_save_path, "best_model_vecnormalize.pkl")
                    if self.verbose > 0: print(f"New best model found! Saving model to {save_path_model}")
                    self.model.save(save_path_model)
                    training_env = self.model.get_env()
                    # Check if the environment is wrapped with VecNormalize
                    if isinstance(training_env, VecNormalize):
                        if self.verbose > 0: print(f"Saving VecNormalize stats from training env to {save_path_stats}")
                        training_env.save(save_path_stats)
                    elif hasattr(training_env, 'venv') and isinstance(training_env.venv, VecNormalize):
                         # Sometimes the wrapper is one level down
                         if self.verbose > 0: print(f"Saving VecNormalize stats from training env.venv to {save_path_stats}")
                         training_env.venv.save(save_path_stats)

        return True

# ===-----------------------------------------------------------------------===
#    MAIN TRAINING EXECUTION SCRIPT (MODIFIED CONFIG & PARALLELISM)
# ===-----------------------------------------------------------------------===

def main():
    """Main function to set up and run the training."""
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    print(f"Created save directory: {MODEL_SAVE_DIR}")

    # --- Print Configuration Summary ---
    print("="*50)
    print("--- TRAINING CONFIGURATION (V2 - Adjusted Rewards & Resources) ---") # Updated Title
    print("="*50)
    print(f"Run Suffix: {RUN_NAME_SUFFIX}")
    print(f"Total Timesteps: {TOTAL_TRAINING_TIMESTEPS:,}")
    # **key point**: Print actual number of parallel envs being used
    print(f"**key point**: Parallel Envs: {NUM_PARALLEL_ENVS} (Using SubprocVecEnv)")
    print(f"Initial Wealth: {INITIAL_WEALTH}")
    print(f"Time Horizon: {START_AGE}-{END_AGE} ({TIME_HORIZON} years)")
    print(f"Assets: S={NUM_STOCKS}, C={NUM_COMMODITIES}, E={NUM_ETFS}, GB={NUM_GOV_BONDS}, CB={NUM_CORP_BONDS}, SI={NUM_STABLE_INC} (Total={TOTAL_ASSETS})")
    # **key point**: Print MODIFIED Reward Coeffs
    print(f"Reward Coeffs: GAMMA_W={GAMMA_W}, THETA={THETA}, BETA_D={BETA_D}, QALY_BASE={QALY_BASE}, ANNUITY_BONUS={ANNUITY_PURCHASE_BONUS}")
    print(f"Compulsory Annuity Payment: {COMPULSORY_ANNUITY_PAYMENT}")
    print(f"Optional Annuity 1: Premium={OPTIONAL_ANNUITY_1_PREMIUM}, Payment={OPTIONAL_ANNUITY_1_PAYMENT}")
    print(f"Optional Annuity 2: Premium={OPTIONAL_ANNUITY_2_PREMIUM}, Payment={OPTIONAL_ANNUITY_2_PAYMENT}")
    # **key point**: Print MODIFIED PPO Config
    print(f"PPO Steps: {PPO_CONFIG['n_steps']}, **key point**: Batch Size: {PPO_CONFIG['batch_size']}, LR: {PPO_CONFIG['learning_rate']}, EntCoef: {PPO_CONFIG['ent_coef']}")
    print(f"**key point**: PPO Device: {PPO_CONFIG['device']}")
    print(f"PPO Policy: MlpPolicy (Action Space Flattened)")
    print(f"Eval Freq (Total Steps): {EVAL_FREQ_TOTAL_STEPS}, Eval Episodes: {N_EVAL_EPISODES}")
    print("="*50)

    # --- Environment Setup ---
    env_kwargs = {
        'initial_wealth': INITIAL_WEALTH, 'start_age': START_AGE, 'end_age': END_AGE,
        'num_stocks': NUM_STOCKS, 'num_commodities': NUM_COMMODITIES, 'num_etfs': NUM_ETFS,
        'num_gov_bonds': NUM_GOV_BONDS, 'num_corp_bonds': NUM_CORP_BONDS, 'num_stable_inc': NUM_STABLE_INC,
        'survival_a': SURVIVAL_A, 'survival_k': SURVIVAL_K, 'survival_g': SURVIVAL_G, 'survival_c': SURVIVAL_C,
        'qaly_base': QALY_BASE, 'gamma_w': GAMMA_W, 'theta': THETA, 'beta_d': BETA_D,
        'annuity_purchase_bonus': ANNUITY_PURCHASE_BONUS, # Pass new bonus
        'risky_asset_indices': RISKY_ASSET_INDICES, 'critical_asset_index': CRITICAL_ASSET_INDEX,
        'forecast_return_noise_factor_1yr': FORECAST_RETURN_NOISE_FACTOR_1YR,
        'forecast_return_noise_factor_2yr': FORECAST_RETURN_NOISE_FACTOR_2YR,
        'forecast_mortality_noise_1yr': FORECAST_MORTALITY_NOISE_1YR,
        'forecast_mortality_noise_2yr': FORECAST_MORTALITY_NOISE_2YR,
        'compulsory_annuity_payment': COMPULSORY_ANNUITY_PAYMENT,
        'opt_annuity_1_premium': OPTIONAL_ANNUITY_1_PREMIUM,
        'opt_annuity_1_payment': OPTIONAL_ANNUITY_1_PAYMENT,
        'opt_annuity_2_premium': OPTIONAL_ANNUITY_2_PREMIUM,
        'opt_annuity_2_payment': OPTIONAL_ANNUITY_2_PAYMENT,
        'verbose': 0 # Set env verbosity (e.g., 2 for detailed annuity purchase prints)
    }

    print(f"Creating {NUM_PARALLEL_ENVS} parallel training environments using SubprocVecEnv...")
    train_env = make_vec_env(
        lambda: RetirementEnv(**env_kwargs),
        n_envs=NUM_PARALLEL_ENVS,
        seed=SEED,
        vec_env_cls=SubprocVecEnv # Use Subproc for true parallelism
    )
    print("Normalizing training environment observations and potentially rewards...")
    # You might consider normalizing rewards if their scale changes significantly due to adjustments
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10., gamma=PPO_CONFIG['gamma'])

    # --- Evaluation Environment Setup ---
    print("Creating evaluation environment (n_envs=1)...")
    # Use DummyVecEnv for the single evaluation environment
    eval_env = make_vec_env(lambda: RetirementEnv(**env_kwargs), n_envs=1, seed=SEED + NUM_PARALLEL_ENVS, vec_env_cls=DummyVecEnv)
    print("Normalizing evaluation environment observations using stats from training env...")
    # Crucially, load stats from training_env for consistent evaluation
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10., training=False, gamma=PPO_CONFIG['gamma'])
    # Link stats: eval_env.obs_rms = train_env.obs_rms requires careful handling or loading from checkpoint

    # --- Initialize WandB ---
    print("Initializing WandB...")
    combined_config = {
        "policy": "MlpPolicy",
        "action_space_type": "Box (Flattened)",
        "total_timesteps": TOTAL_TRAINING_TIMESTEPS,
        "num_parallel_envs": NUM_PARALLEL_ENVS,
        "vec_env_type": "SubprocVecEnv",
        "seed": SEED,
        **PPO_CONFIG,
        **env_kwargs # Log all env params
    }
    run = wandb.init(
        project=WANDB_PROJECT_NAME,
        entity=WANDB_ENTITY,
        name=f"{int(time.time())}_{RUN_NAME_SUFFIX}",
        config=combined_config,
        sync_tensorboard=True,
        monitor_gym=True, # Auto-log video/renderings if available
        save_code=True,
    )

    # --- Callback Setup ---
    print("Setting up callbacks...")
    # Calculate eval_freq based on total steps, ensuring it's a multiple of n_steps*n_envs for clean alignment
    # Make eval_freq relative to TOTAL timesteps, not per-environment steps
    eval_freq_steps = max(EVAL_FREQ_TOTAL_STEPS // NUM_PARALLEL_ENVS, PPO_CONFIG['n_steps'])

    checkpoint_save_freq_per_env = eval_freq_steps * 2 # Save checkpoints less frequently than eval
    print(f"Checkpoint Callback: Save Freq Per Env = {checkpoint_save_freq_per_env} steps (approx every {checkpoint_save_freq_per_env * NUM_PARALLEL_ENVS:,} total steps)")
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_save_freq_per_env,
        save_path=os.path.join(MODEL_SAVE_DIR, 'checkpoints'),
        name_prefix='ppo_retirement',
        save_replay_buffer=False, # PPO doesn't use a replay buffer
        save_vecnormalize=True # Save VecNormalize stats with checkpoints
    )

    print(f"Evaluation Callback: Eval Freq = {EVAL_FREQ_TOTAL_STEPS} total steps ({eval_freq_steps} steps per env)")
    # Use our custom evaluation callback
    evaluation_callback = EvalCallbackWithComponents(
        eval_env=eval_env,
        n_eval_episodes=N_EVAL_EPISODES,
        eval_freq=eval_freq_steps, # Frequency based on per-env steps
        log_path=os.path.join(MODEL_SAVE_DIR, 'eval_logs'),
        best_model_save_path=os.path.join(MODEL_SAVE_DIR, 'best_model'),
        deterministic=True,
        verbose=1,
        debug_qaly=True # Enable QALY debug prints in callback
    )

    wandb_callback = OfficialWandbCallback(
        gradient_save_freq=0, # Don't save gradients
        model_save_path=os.path.join(MODEL_SAVE_DIR, f"wandb_models/{run.id}"),
        model_save_freq=0, # Use CheckpointCallback and EvalCallback for saving models
        verbose=2
    )
    hparam_callback = HParamCallback() # Use updated HParamCallback below

    callback_list = CallbackList([checkpoint_callback, evaluation_callback, hparam_callback, wandb_callback])

    # --- Model Definition ---
    # **key point**: Optionally define larger network architecture
    # policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128])) # Example larger network
    policy_kwargs = None # Use default MLP size initially

    print(f"Defining PPO model with MlpPolicy on device: {PPO_CONFIG['device']}...")
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        seed=SEED,
        tensorboard_log=f"runs/{run.id}", # Log TensorBoard data for WandB
        policy_kwargs=policy_kwargs, # Pass network architecture if defined
        **PPO_CONFIG # Pass all other PPO hyperparameters
    )

    # --- Training ---
    print("="*50)
    print(f"--- STARTING TRAINING ({TOTAL_TRAINING_TIMESTEPS:,} steps) ---")
    print(f"--- Using {NUM_PARALLEL_ENVS} parallel environments ---")
    print(f"--- Monitor CPU ({os.cpu_count()} cores) and GPU ({torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'N/A'}) utilization ---")
    print("="*50)
    start_time = time.time()
    try:
        model.learn(
            total_timesteps=TOTAL_TRAINING_TIMESTEPS,
            callback=callback_list,
            log_interval=LOG_INTERVAL, # Log training stats frequency
            reset_num_timesteps=True # Start timesteps from 0 for this run
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
        print(f"--- TRAINING FINISHED --- (Duration: {duration:.2f}s | {duration_min:.2f} min | {duration_hr:.2f} hr)")
        print("="*50)

        # --- Save Final Model and Normalization Stats ---
        final_model_path = os.path.join(MODEL_SAVE_DIR, "final_model.zip")
        final_stats_path = os.path.join(MODEL_SAVE_DIR, "final_vec_normalize.pkl") # Changed name slightly
        print(f"Saving final model to: {final_model_path}")
        model.save(final_model_path)
        print(f"Saving final VecNormalize stats to: {final_stats_path}")
        # Get the potentially wrapped env
        _train_env = model.get_env()
        if _train_env and isinstance(_train_env, VecNormalize):
             _train_env.save(final_stats_path)
        elif _train_env and hasattr(_train_env, 'venv') and isinstance(_train_env.venv, VecNormalize):
             # If wrapped, save the underlying VecNormalize
             _train_env.venv.save(final_stats_path)
        else:
             print(f"Warning: Could not save VecNormalize stats (env type: {type(_train_env)})")

        # --- Clean up ---
        print("Closing environments...")
        try:
            # Check if environments exist before closing
            if 'train_env' in locals() and train_env is not None: train_env.close()
            if 'eval_env' in locals() and eval_env is not None: eval_env.close()
        except Exception as e: print(f"Error closing environments: {e}")

        if run:
            print("Finishing WandB run...")
            run.finish()
        print("Script finished.")

# ===-----------------------------------------------------------------------===
#    HYPERPARAMETER LOGGING CALLBACK (Updated with new params)
# ===-----------------------------------------------------------------------===
class HParamCallback(BaseCallback):
    """Saves hyperparameters and metrics at the start of training for WandB."""
    def _on_training_start(self) -> None:
        # Clip range can be a schedule, get initial value
        clip_range_val = self.model.clip_range
        if callable(clip_range_val):
             clip_range_val = clip_range_val(0.0) # Initial value at progress 0

        hparam_dict = {
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
            "clip_range": clip_range_val,
            "policy": "MlpPolicy",
            "action_space_type": "Box (Flattened)",
            # Env/training params
            "total_timesteps": TOTAL_TRAINING_TIMESTEPS,
            "num_parallel_envs": NUM_PARALLEL_ENVS,
            "vec_env_type": "SubprocVecEnv",
            "device": PPO_CONFIG['device'],
            "seed": SEED,
            # **key point**: Log updated reward coefficients
            "reward_gamma_w": GAMMA_W,
            "reward_theta": THETA,
            "reward_beta_d": BETA_D,
            "reward_qaly_base": QALY_BASE,
            "reward_annuity_bonus": ANNUITY_PURCHASE_BONUS, # Added bonus
            # Annuity hparams
            "compulsory_annuity_payment": COMPULSORY_ANNUITY_PAYMENT,
            "opt_annuity_1_premium": OPTIONAL_ANNUITY_1_PREMIUM,
            "opt_annuity_1_payment": OPTIONAL_ANNUITY_1_PAYMENT,
            "opt_annuity_2_premium": OPTIONAL_ANNUITY_2_PREMIUM,
            "opt_annuity_2_payment": OPTIONAL_ANNUITY_2_PAYMENT,
        }

        # Define placeholder metrics that WandB expects
        metric_dict = {
            "rollout/ep_rew_mean": 0.0,
            "train/value_loss": 0.0,
            "eval/mean_reward": 0.0, # Primary metric for best model
            # Key evaluation components
            "eval/mean_qaly_comp": 0.0,
            "eval/mean_terminal_wealth_reward_comp": 0.0,
            "eval/mean_final_wealth": 0.0,
            "eval/mean_annuity_bonus_comp": 0.0, # Added bonus metric
            "eval/frac_bought_opt_annuity_1": 0.0,
            "eval/frac_bought_opt_annuity_2": 0.0,
            # Other eval components
             "eval/mean_dd_comp": 0.0,
             "eval/mean_risky_comp": 0.0,
             "eval/mean_shortfall_comp": 0.0,
             "eval/mean_total_reward_from_components": 0.0,
        }
        try:
             # Log hparams and metrics to WandB
             # Note: SB3 logger automatically syncs Tensorboard logs which WandB picks up.
             # This HParam logging ensures they are directly associated in WandB's HParam tab.
             # Using logger.record directly is cleaner than manipulating WandB run object here.
             self.logger.record("hparams", HParam(hparam_dict, metric_dict), exclude=("stdout", "log", "json", "csv"))
             print("Hyperparameters logged to WandB HParams tab.")
        except Exception as e: print(f"Warning: Failed to record HParams directly via logger.record: {e}")

    def _on_step(self) -> bool:
        # Only needs to run once at the start
        return True

# ===-----------------------------------------------------------------------===
#    SCRIPT EXECUTION GUARD
# ===-----------------------------------------------------------------------===
if __name__ == "__main__":
    # Ensure PyTorch CUDA is checked *before* potentially spawning subprocesses
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version detected by PyTorch: {torch.version.cuda}")

    # Crucial for SubprocVecEnv multiprocessing safety, especially on Windows
    # Needs to be inside the main guard
    main()
