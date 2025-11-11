---
date: 2026-07-14T11:00:59-04:00
description: "https://verl.readthedocs.io/en/latest/start/quickstart.html"
featured_image: "/images/PPO_GSM8K/jaz.png"
tags: ["RL"]
title: "verl - PPO_GSM8K 源码阅读"
---

## 1. Prepare the dataset

[examples/data_preprocess/gsm8k.py](examples/data_preprocess/gsm8k.py)





## 2. PPO training

### Reward Model/Function

+ #### [verl/utils/reward_score/gsm8k.py](https://github.com/volcengine/verl/blob/v0.4.1/verl/utils/reward_score/gsm8k.py)

  ```python
  def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
      """Args:
          solution_str: the solution text
          ground_truth: the ground truth
          method: the method to extract the solution, choices are 'strict' and 'flexible'
          format_score: the score for the format
          score: the score for the correct answer
      """
      answer = extract_solution(solution_str=solution_str, method=method)	# 使用正则表达式匹配提取最终答案
      if answer is None:									# 没有找到答案：返回0分
          return 0
      else:
          if answer == ground_truth:			# 答案正确：返回score
              return score
          else:														# 答案错误：返回format_score
              return format_score
  ```

+ #### [examples/ppo_trainer/run_deepseek7b_llm.sh](examples/ppo_trainer/run_deepseek7b_llm.sh)

  ```bash
  set -x
  
  python3 -m verl.trainer.main_ppo \
     data.train_files=$HOME/data/gsm8k/train.parquet \
     data.val_files=$HOME/data/gsm8k/test.parquet \
     data.train_batch_size=1024 \
     data.max_prompt_length=512 \
     data.max_response_length=512 \
     actor_rollout_ref.model.path=deepseek-ai/deepseek-llm-7b-chat \
     actor_rollout_ref.actor.optim.lr=1e-6 \
     actor_rollout_ref.model.use_remove_padding=True \
     actor_rollout_ref.actor.ppo_mini_batch_size=256 \
     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
     actor_rollout_ref.actor.fsdp_config.param_offload=False \
     actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
     actor_rollout_ref.model.enable_gradient_checkpointing=True \
     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
     actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
     actor_rollout_ref.rollout.name=vllm \
     actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
     actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
     actor_rollout_ref.ref.fsdp_config.param_offload=True \
     critic.optim.lr=1e-5 \
     critic.model.use_remove_padding=True \
     critic.model.path=deepseek-ai/deepseek-llm-7b-chat \
     critic.model.enable_gradient_checkpointing=True \
     critic.ppo_micro_batch_size_per_gpu=32 \
     critic.model.fsdp_config.param_offload=False \
     critic.model.fsdp_config.optimizer_offload=False \
     algorithm.kl_ctrl.kl_coef=0.001 \
     trainer.critic_warmup=0 \
     trainer.logger='["console","wandb"]' \
     trainer.project_name='verl_example_gsm8k' \
     trainer.experiment_name='deepseek_llm_7b_function_rm' \
     trainer.n_gpus_per_node=8 \
     trainer.nnodes=1 \
     trainer.save_freq=-1 \
     trainer.test_freq=1 \
     trainer.total_epochs=15 $@
  ```

+ #### [verl/trainer/main_ppo.py](verl/trainer/main_ppo.py)

  + 调用 TaskRunner 类进行远程任务执行

    ```python
    def run_ppo(config) -> None:
        """Initialize Ray cluster and run distributed PPO training process.
    
        Args:
            config: Training configuration object containing all necessary parameters
                    for distributed PPO training including Ray initialization settings,
                    model paths, and training hyperparameters.
        """
        # Check if Ray is not initialized
        if not ray.is_initialized():
            # Initialize Ray with a local cluster configuration
            # Set environment variables in the runtime environment to control tokenizer parallelism,
            # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
            # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
            ray.init(
                runtime_env=get_ppo_ray_runtime_env(),
                num_cpus=config.ray_init.num_cpus,
            )
    
        # Create a remote instance of the TaskRunner class, and
        # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
        if (
            is_cuda_available
            and config.trainer.get("profile_steps") is not None
            and len(config.trainer.get("profile_steps", [])) > 0
        ):
            from verl.utils.import_utils import is_nvtx_available
    
            assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
            nsight_options = OmegaConf.to_container(config.trainer.controller_nsight_options)
            runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
        else:
            runner = TaskRunner.remote()
        ray.get(runner.run.remote(config))
    
        # [Optional] get the path of the timeline trace file from the configuration, default to None
        # This file is used for performance analysis
        timeline_json_file = config.ray_init.get("timeline_json_file", None)
        if timeline_json_file:
            ray.timeline(filename=timeline_json_file)
    ```

  + Task Runner

    ```python
    @ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
    class TaskRunner:
        """Ray remote class for executing distributed PPO training tasks.
    
        This class encapsulates the main training logic and runs as a Ray remote actor
        to enable distributed execution across multiple nodes and GPUs.
        """
    
        def run(self, config):
            """Execute the main PPO training workflow.
    
            This method sets up the distributed training environment, initializes
            workers, datasets, and reward functions, then starts the training process.
    
            Args:
                config: Training configuration object containing all parameters needed
                       for setting up and running the PPO training process.
            """
            # Print the initial configuration. `resolve=True` will evaluate symbolic values.
    ```

    ```python
    				# Load the reward manager for training and validation.
            reward_fn = load_reward_manager(
                config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
            )
            val_reward_fn = load_reward_manager(
                config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
            )
            resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    ```

    ```python
    				from verl.utils.dataset.rl_dataset import collate_fn
            # Create training and validation datasets.
            train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor, is_train=True)
            val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor, is_train=False)
            train_sampler = create_rl_sampler(config.data, train_dataset)
    ```

    ```python
            # Initialize the PPO trainer.
            trainer = RayPPOTrainer(
                config=config,
                tokenizer=tokenizer,
                processor=processor,
                role_worker_mapping=role_worker_mapping,
                resource_pool_manager=resource_pool_manager,
                ray_worker_group_cls=ray_worker_group_cls,
                reward_fn=reward_fn,
                val_reward_fn=val_reward_fn,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                collate_fn=collate_fn,
                train_sampler=train_sampler,
            )
            # Initialize the workers of the trainer.
            trainer.init_workers()
            # Start the training process.
            trainer.fit()
    ```

+ #### [verl/trainer/ppo/reward.py](verl/trainer/ppo/reward.py)

  + 异步奖励计算

    ```python
    @ray.remote(num_cpus=1)
    def compute_reward_async(data: DataProto, config=None, tokenizer=None, reward_fn=None):
        """
        Load the reward manager and compute the reward for a batch of data.
        This is meant to be run in a separate Ray worker.
        """
        if reward_fn is None:
            assert config is not None and tokenizer is not None, (
                "config and tokenizer must not be None when reward_fn is None"
            )
            warnings.warn("using config and tokenizer with compute_reward_async is deprecated", stacklevel=2)
            # 配置奖励管理器
            reward_fn = load_reward_manager(
                config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
            )
    
        return compute_reward(data, reward_fn)
    ```

    ```python
    def compute_reward(data: DataProto, reward_fn: AbstractRewardManager) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Compute reward for a batch of data.
        """
        try:
            reward_result = reward_fn(data, return_dict=True)
            reward_tensor = reward_result["reward_tensor"]
            reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
        except Exception as e:
            print(f"Error in reward_fn: {e}")
            reward_tensor = reward_fn(data)
            reward_extra_infos_dict = {}
    
        return reward_tensor, reward_extra_infos_dict
    ```

  + **核心**：配置奖励管理器

    ```python
    def load_reward_manager(
        config: DictConfig, tokenizer: Any, num_examine: int, **reward_kwargs: Any
    ) -> AbstractRewardManager:
        """
        Load and initialize a reward manager based on the configuration.
        """
    
        # The list of pre-defined reward managers are defined in `verl/workers/reward_manager/`:
        	# naive: NaiveRewardManager
        	# prime: PrimeRewardManager
        	# batch: BatchRewardManager
        	# dapo: DAPORewardManager
        # Note(haibin.lin): For custom reward managers, please make sure they are imported and
        # registered via `verl.workers.reward_manager.register`
        # By default reward_manager is set to naive (NaiveRewardManager)
        reward_manager_name = config.reward_model.get("reward_manager", "naive")
        reward_manager_cls = get_reward_manager_cls(reward_manager_name)
    
        # Try to get a custom reward function based on the configuration
        compute_score = get_custom_reward_fn(config)					# 奖励函数设置
        final_compute_score = compute_score
    
        if compute_score is None:
            sandbox_config = config.reward_model.get("sandbox_fusion")
            sandbox_url = sandbox_config.get("url") if sandbox_config else None
            memory_limit_mb = sandbox_config.get("memory_limit_mb", 1024)
            if sandbox_url:
                sandbox_manager = multiprocessing.Manager()
                # Create a semaphore to control concurrent access to the sandbox
                _concurrent_semaphore = sandbox_manager.Semaphore(sandbox_config.get("max_concurrent", 64))
                final_compute_score = partial(								# 使用带沙箱的默认奖励计算函数
                    default_compute_score,
                    sandbox_fusion_url=sandbox_url,
                    concurrent_semaphore=_concurrent_semaphore,
                    memory_limit_mb=memory_limit_mb,
                )
            else:
                final_compute_score = default_compute_score		# 使用无沙箱的默认奖励计算函数
    
        # Instantiate and return the reward manager with the specified parameters
        return reward_manager_cls(
            tokenizer=tokenizer,
            num_examine=num_examine,
            compute_score=final_compute_score,
            reward_fn_key=config.data.reward_fn_key,
            **reward_kwargs,
        )
    ```

  + 动态加载自定义奖励函数

    ```python
    def get_custom_reward_fn(config: DictConfig) -> Optional[RawRewardFn]:
        """Load and return a custom reward function from external file.
        Dynamically imports a reward function from a specified file path and wraps
        it with additional keyword arguments from the configuration.
        """
    
        reward_fn_config = config.get("custom_reward_function") or {}
        file_path = reward_fn_config.get("path")
        if not file_path:
            return None
    
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Reward function file '{file_path}' not found.")
    
        spec = importlib.util.spec_from_file_location("custom_module", file_path)
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_module"] = module
            assert spec.loader is not None
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Error loading module from '{file_path}': {e}") from e
    
        function_name = reward_fn_config.get("name")
        assert function_name is not None
        if not hasattr(module, function_name):
            raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")
    
        print(f"using customized reward function '{function_name}' from '{file_path}'")
        raw_fn = getattr(module, function_name)
    
        reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))
    
        return partial(_call_with_kwargs, raw_fn, reward_kwargs)
    ```

  
