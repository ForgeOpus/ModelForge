import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { config, getSystemInfo, validateDatasetPath, startTraining } from '../services/api';

const FinetuneSettings = ({ defaultValues, updateSettings }) => {
  const navigate = useNavigate();
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [datasetPath, setDatasetPath] = useState('');
  const [datasetValidated, setDatasetValidated] = useState(false);
  const [formState, setFormState] = useState({});
  const [settingsUpdated, setSettingsUpdated] = useState(false);
  const [activeTooltip, setActiveTooltip] = useState(null);
  
  // New state for dynamic options
  const [availableProviders, setAvailableProviders] = useState([]);
  const [availableStrategies, setAvailableStrategies] = useState([]);
  const [systemInfo, setSystemInfo] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Fetch system info on component mount
  useEffect(() => {
    const fetchSystemInfo = async () => {
      try {
        setIsLoading(true);
        const info = await getSystemInfo();
        console.log("Fetched system info:", info);
        
        setSystemInfo(info);
        setAvailableProviders(info.available_providers || ['huggingface']);
        setAvailableStrategies(info.available_strategies || ['sft']);
        
        setIsLoading(false);
      } catch (err) {
        console.error("Error fetching system info:", err);
        setError("Failed to load system information");
        // Fallback to defaults
        setAvailableProviders(['huggingface']);
        setAvailableStrategies(['sft']);
        setIsLoading(false);
      }
    };

    fetchSystemInfo();
  }, []);

  useEffect(() => {
    const fetchDefaultSettings = async () => {
      try {
        // Fetch default settings
        const settingsResponse = await fetch(`${config.baseURL}/finetune/load_settings`);
        if (!settingsResponse.ok) throw new Error('Failed to fetch settings');
        const settingsData = await settingsResponse.json();
        console.log("Fetched default values:", settingsData.default_values);

        // Fetch session data (task, model_name)
        const sessionResponse = await fetch(`${config.baseURL}/finetune/session`);
        if (!sessionResponse.ok) throw new Error('Failed to fetch session');
        const sessionData = await sessionResponse.json();
        console.log("Fetched session data:", sessionData);

        // Merge defaults with session data
        const mergedState = {
          ...settingsData.default_values,
          task: sessionData.task,
          model_name: sessionData.selected_model,
          compute_specs: settingsData.compute_profile || 'low_end',
        };

        console.log("Merged form state:", mergedState);
        defaultValues = mergedState;
        setFormState(mergedState);
      } catch (err) {
        console.error("Error fetching settings:", err);
      }
    };

    fetchDefaultSettings();
  }, []);

  // Sync with props when they change
  useEffect(() => {
    console.log("defaultValues changed in FinetuneSettings:", defaultValues);
    if (defaultValues) {
      // Create a deep copy to break any references
      const values = JSON.parse(JSON.stringify(defaultValues));
      
      // Set defaults for new fields if not present
      if (!values.provider) values.provider = 'huggingface';
      if (!values.strategy) values.strategy = 'sft';
      if (values.eval_split === undefined) values.eval_split = 0.2;
      if (values.eval_steps === undefined) values.eval_steps = 100;
      
      console.log("Setting form values to:", values);
      setFormState(values);
    }
  }, [defaultValues]);

  const handleDatasetPathChange = (e) => {
    setDatasetPath(e.target.value);
    setDatasetValidated(false);
  };

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    const newValue = type === 'checkbox' ? checked : value;

    // Create a completely new object for state update
    const updatedState = {
      ...formState,
      [name]: type === 'number' ? Number(newValue) : newValue
    };

    console.log(`Changing ${name} to:`, newValue);
    console.log("New form values:", updatedState);

    // If task or model is changing, log it prominently
    if (name === 'task' || name === 'model_name') {
      console.log(`⚠️ IMPORTANT: ${name} changed to "${newValue}"`);
    }

    setFormState(updatedState);
  };

  const handleQuantizationChange = (value) => {
    const updatedState = {
      ...formState,
      use_4bit: value === '4bit',
      use_8bit: value === '8bit',
      quantization: value
    };

    setFormState(updatedState);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    try {
      setError(null);
      
      // Validate required fields
      if (!formState.task || !formState.model_name) {
        setError("Missing required fields. Please go back and complete previous steps.");
        return;
      }
      
      // Step 1: Validate dataset path
      if (datasetPath.trim()) {
        console.log("Validating dataset path...");
        const validateResponse = await validateDatasetPath(datasetPath.trim());
        console.log("Dataset validated:", validateResponse);

        // Update form state with the validated file path
        formState.dataset = validateResponse.file_path;
        setDatasetValidated(true);
      } else {
        throw new Error("Please enter the path to your dataset file");
      }

      // Step 2: Prepare training configuration
      const trainingConfig = {
        // Required fields
        task: formState.task,
        model_name: formState.model_name,
        dataset: formState.dataset,
        compute_specs: formState.compute_specs || 'low_end',
        
        // Provider and strategy (NEW)
        provider: formState.provider || 'huggingface',
        strategy: formState.strategy || 'sft',
        
        // LoRA settings
        lora_r: formState.lora_r || 16,
        lora_alpha: formState.lora_alpha || 32,
        lora_dropout: formState.lora_dropout || 0.1,
        
        // Quantization settings
        use_4bit: formState.use_4bit !== undefined ? formState.use_4bit : true,
        use_8bit: formState.use_8bit || false,
        bnb_4bit_compute_dtype: formState.bnb_4bit_compute_dtype || 'float16',
        bnb_4bit_quant_type: formState.bnb_4bit_quant_type || 'nf4',
        use_nested_quant: formState.use_nested_quant || false,
        
        // Training precision
        fp16: formState.fp16 || false,
        bf16: formState.bf16 || false,
        
        // Training hyperparameters
        num_train_epochs: formState.num_train_epochs || 1,
        per_device_train_batch_size: formState.per_device_train_batch_size || 1,
        per_device_eval_batch_size: formState.per_device_eval_batch_size || 1,
        gradient_accumulation_steps: formState.gradient_accumulation_steps || 4,
        gradient_checkpointing: formState.gradient_checkpointing !== undefined ? formState.gradient_checkpointing : true,
        max_grad_norm: formState.max_grad_norm || 0.3,
        learning_rate: formState.learning_rate || 0.0002,
        weight_decay: formState.weight_decay || 0.001,
        optim: formState.optim || 'paged_adamw_32bit',
        lr_scheduler_type: formState.lr_scheduler_type || 'cosine',
        max_steps: formState.max_steps || -1,
        warmup_ratio: formState.warmup_ratio || 0.03,
        group_by_length: formState.group_by_length !== undefined ? formState.group_by_length : true,
        packing: formState.packing || false,
        
        // Sequence settings
        max_seq_length: formState.max_seq_length || null,
        
        // Evaluation settings (NEW)
        eval_split: formState.eval_split !== undefined ? formState.eval_split : 0.2,
        eval_steps: formState.eval_steps || 100,
      };

      console.log("Starting training with config:", trainingConfig);

      // Step 3: Start training
      const trainingResponse = await startTraining(trainingConfig);
      console.log("Training started:", trainingResponse);

      setSettingsUpdated(true);
      
      // Navigate to loading page after short delay
      setTimeout(() => {
        navigate('/finetune/loading');
      }, 1000);

    } catch (error) {
      console.error("Error submitting form:", error);
      setError(error.message || "Failed to start training");
      alert(`Failed to start training: ${error.message}`);
    }
  };

  // Tooltip definitions
  const tooltips = {
    task: "What you want your AI to be good at doing",
    model_name: "The base AI model you're customizing",
    provider: "Model provider (HuggingFace for standard, Unsloth for 2x faster training)",
    strategy: "Training method (SFT for basic, RLHF for preference learning, DPO for direct optimization, QLoRA for memory efficiency)",
    gpu: "The graphics card that will run your training",
    ram: "Computer memory available for training",
    num_train_epochs: "How many times the AI will see your training data",
    learning_rate: "How quickly the model adapts to new information",
    per_device_train_batch_size: "How many examples are processed at once",
    max_seq_length: "Maximum text length the model can handle",
    dataset_file: "Path to your local training data file (JSON or JSONL format)",
    lora_r: "Controls model capacity and training speed",
    lora_alpha: "Controls how much the model changes during training",
    quantization: "Reduces model size to fit in memory",
    bnb_4bit_compute_dtype: "Technical setting for calculation precision",
    optim: "Method used to update the model during training",
    lr_scheduler_type: "How learning speed changes during training",
    eval_split: "Percentage of data to use for validation (0-1)",
    eval_steps: "How often to evaluate the model during training"
  };

  // Provider descriptions
  const providerDescriptions = {
    huggingface: "Standard HuggingFace Hub models",
    unsloth: "Optimized for 2x faster training with lower memory"
  };

  // Strategy descriptions
  const strategyDescriptions = {
    sft: "Supervised Fine-Tuning - Standard approach with LoRA",
    rlhf: "Reinforcement Learning from Human Feedback - Preference-based training",
    dpo: "Direct Preference Optimization - Simpler alternative to RLHF",
    qlora: "Quantized LoRA - Memory-efficient training with 4-bit quantization"
  };

  // Tooltip display component
  const Tooltip = ({ id, children }) => (
    <div className="flex justify-between items-center">
      <div className="flex-grow">{children}</div>
      <div className="relative">
        <span 
          className="cursor-help text-orange-500 font-bold text-sm"
          onMouseEnter={() => setActiveTooltip(id)}
          onMouseLeave={() => setActiveTooltip(null)}
        >
          ?
        </span>
        {activeTooltip === id && (
          <div className="absolute z-10 w-64 p-2 text-sm text-white bg-gray-900 rounded-md shadow-lg border border-orange-500 right-0 top-6">
            {tooltips[id]}
          </div>
        )}
      </div>
    </div>
  );

  if (isLoading) {
    return (
      <div className="container mx-auto px-4 py-8 max-w-4xl">
        <div className="text-center">
          <div className="animate-spin h-12 w-12 border-4 border-orange-500 rounded-full border-t-transparent mx-auto"></div>
          <p className="text-gray-400 mt-4">Loading system information...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8 max-w-4xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white">Finetuning Settings</h1>
        <p className="text-gray-400 mt-2">Configure your model training parameters</p>
        
        {systemInfo && (
          <div className="mt-4 p-3 bg-gray-800 rounded-lg text-sm text-gray-300">
            <strong>Available:</strong> {availableProviders.length} provider(s), {availableStrategies.length} strategy(ies)
          </div>
        )}
      </div>

      {error && (
        <div className="bg-red-700 text-white p-4 rounded-lg mb-6">
          <strong>Error:</strong> {error}
        </div>
      )}

      {settingsUpdated && (
        <div className="bg-green-700 text-white p-4 rounded-lg mb-6 flex items-center justify-between">
          <div>Settings updated successfully! Redirecting...</div>
          <div className="animate-spin h-5 w-5 border-2 border-white rounded-full border-t-transparent"></div>
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-8">
        {/* Configuration Summary */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-white mb-4">Configuration Summary</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <Tooltip id="task">
                <label className="block text-sm font-medium text-gray-400 mb-1">Task</label>
              </Tooltip>
              <div className="bg-gray-900 border border-gray-700 rounded-lg p-3 text-white">
                {formState.task || defaultValues.task || 'Not set'}
              </div>
            </div>
            <div>
              <Tooltip id="model_name">
                <label className="block text-sm font-medium text-gray-400 mb-1">Model Name</label>
              </Tooltip>
              <div className="bg-gray-900 border border-gray-700 rounded-lg p-3 text-white">
                {formState.model_name || defaultValues.model_name || 'Not set'}
              </div>
            </div>
            <div>
              <Tooltip id="gpu">
                <label className="block text-sm font-medium text-gray-400 mb-1">GPU</label>
              </Tooltip>
              <div className="bg-gray-900 border border-gray-700 rounded-lg p-3 text-white">
                {formState.hardware_config?.gpu || defaultValues.hardware_config?.gpu || 'N/A'}
              </div>
            </div>
            <div>
              <Tooltip id="ram">
                <label className="block text-sm font-medium text-gray-400 mb-1">RAM</label>
              </Tooltip>
              <div className="bg-gray-900 border border-gray-700 rounded-lg p-3 text-white">
                {formState.hardware_config?.ram || defaultValues.hardware_config?.ram ?
                  `${formState.hardware_config?.ram || defaultValues.hardware_config?.ram} GB` : 'N/A'}
              </div>
            </div>
          </div>
        </div>

        {/* Provider and Strategy Selection */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-white mb-4">Provider & Strategy</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <Tooltip id="provider">
                <label htmlFor="provider" className="block text-sm font-medium text-gray-400 mb-1">
                  Model Provider
                </label>
              </Tooltip>
              <select
                id="provider"
                name="provider"
                value={formState.provider || 'huggingface'}
                onChange={handleInputChange}
                className="bg-gray-900 border border-gray-700 rounded-lg p-3 w-full text-white focus:border-orange-500 focus:outline-none"
              >
                {availableProviders.map(provider => (
                  <option key={provider} value={provider}>
                    {provider.charAt(0).toUpperCase() + provider.slice(1)}
                  </option>
                ))}
              </select>
              {formState.provider && providerDescriptions[formState.provider] && (
                <p className="mt-2 text-xs text-gray-400">
                  {providerDescriptions[formState.provider]}
                </p>
              )}
            </div>

            <div>
              <Tooltip id="strategy">
                <label htmlFor="strategy" className="block text-sm font-medium text-gray-400 mb-1">
                  Training Strategy
                </label>
              </Tooltip>
              <select
                id="strategy"
                name="strategy"
                value={formState.strategy || 'sft'}
                onChange={handleInputChange}
                className="bg-gray-900 border border-gray-700 rounded-lg p-3 w-full text-white focus:border-orange-500 focus:outline-none"
              >
                {availableStrategies.map(strategy => (
                  <option key={strategy} value={strategy}>
                    {strategy.toUpperCase()}
                  </option>
                ))}
              </select>
              {formState.strategy && strategyDescriptions[formState.strategy] && (
                <p className="mt-2 text-xs text-gray-400">
                  {strategyDescriptions[formState.strategy]}
                </p>
              )}
            </div>
          </div>
        </div>

        {/* Basic Settings */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-white mb-4">Basic Settings</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <Tooltip id="num_train_epochs">
                <label htmlFor="num_train_epochs" className="block text-sm font-medium text-gray-400 mb-1">
                  Training Epochs
                </label>
              </Tooltip>
              <input
                type="number"
                id="num_train_epochs"
                name="num_train_epochs"
                min="1"
                max="100"
                value={formState.num_train_epochs || 3}
                onChange={handleInputChange}
                className="bg-gray-900 border border-gray-700 rounded-lg p-3 w-full text-white focus:border-orange-500 focus:outline-none"
              />
            </div>

            <div>
              <Tooltip id="learning_rate">
                <label htmlFor="learning_rate" className="block text-sm font-medium text-gray-400 mb-1">
                  Learning Rate
                </label>
              </Tooltip>
              <input
                type="number"
                id="learning_rate"
                name="learning_rate"
                step="0.000001"
                value={formState.learning_rate || 0.0002}
                onChange={handleInputChange}
                className="bg-gray-900 border border-gray-700 rounded-lg p-3 w-full text-white focus:border-orange-500 focus:outline-none"
              />
            </div>

            <div>
              <Tooltip id="per_device_train_batch_size">
                <label htmlFor="per_device_train_batch_size" className="block text-sm font-medium text-gray-400 mb-1">
                  Batch Size (Train)
                </label>
              </Tooltip>
              <input
                type="number"
                id="per_device_train_batch_size"
                name="per_device_train_batch_size"
                min="1"
                value={formState.per_device_train_batch_size || 2}
                onChange={handleInputChange}
                className="bg-gray-900 border border-gray-700 rounded-lg p-3 w-full text-white focus:border-orange-500 focus:outline-none"
              />
            </div>

            <div>
              <Tooltip id="max_seq_length">
                <label htmlFor="max_seq_length" className="block text-sm font-medium text-gray-400 mb-1">
                  Max Sequence Length
                </label>
              </Tooltip>
              <input
                type="number"
                id="max_seq_length"
                name="max_seq_length"
                value={formState.max_seq_length || 512}
                onChange={handleInputChange}
                className="bg-gray-900 border border-gray-700 rounded-lg p-3 w-full text-white focus:border-orange-500 focus:outline-none"
              />
            </div>

            <div className="col-span-full">
              <Tooltip id="dataset_file">
                <label htmlFor="dataset_path" className="block text-sm font-medium text-gray-400 mb-1">
                  Dataset File Path
                </label>
              </Tooltip>
              <input
                type="text"
                id="dataset_path"
                name="dataset_path"
                placeholder="C:\path\to\your\dataset.json"
                value={datasetPath}
                onChange={handleDatasetPathChange}
                className="bg-gray-900 border border-gray-700 rounded-lg p-3 w-full text-white focus:border-orange-500 focus:outline-none"
              />
              <p className="mt-2 text-sm text-gray-400">
                Enter the full path to your local JSON or JSONL dataset file
              </p>
            </div>
          </div>
        </div>

        {/* Evaluation Settings */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-white mb-4">Evaluation Settings</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <Tooltip id="eval_split">
                <label htmlFor="eval_split" className="block text-sm font-medium text-gray-400 mb-1">
                  Validation Split
                </label>
              </Tooltip>
              <input
                type="number"
                id="eval_split"
                name="eval_split"
                min="0"
                max="1"
                step="0.05"
                value={formState.eval_split !== undefined ? formState.eval_split : 0.2}
                onChange={handleInputChange}
                className="bg-gray-900 border border-gray-700 rounded-lg p-3 w-full text-white focus:border-orange-500 focus:outline-none"
              />
              <p className="mt-2 text-xs text-gray-400">
                {Math.round((formState.eval_split || 0.2) * 100)}% of data for validation
              </p>
            </div>

            <div>
              <Tooltip id="eval_steps">
                <label htmlFor="eval_steps" className="block text-sm font-medium text-gray-400 mb-1">
                  Evaluation Steps
                </label>
              </Tooltip>
              <input
                type="number"
                id="eval_steps"
                name="eval_steps"
                min="10"
                step="10"
                value={formState.eval_steps || 100}
                onChange={handleInputChange}
                className="bg-gray-900 border border-gray-700 rounded-lg p-3 w-full text-white focus:border-orange-500 focus:outline-none"
              />
              <p className="mt-2 text-xs text-gray-400">
                Evaluate every {formState.eval_steps || 100} training steps
              </p>
            </div>
          </div>
        </div>

        {/* Advanced Settings */}
        <div className="flex justify-center mb-6">
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center text-orange-500 hover:text-orange-400 transition"
          >
            <span className="mr-2">Advanced Settings</span>
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className={`h-5 w-5 transform transition-transform ${showAdvanced ? 'rotate-180' : ''}`}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
        </div>

        {showAdvanced && (
          <div className="space-y-6">
            {/* LoRA Settings */}
            <div className="bg-gray-800 rounded-lg p-6">
              <h3 className="text-lg font-medium text-white mb-4">LoRA Configuration</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <Tooltip id="lora_r">
                  <label htmlFor="lora_r" className="block text-sm font-medium text-gray-400 mb-1">
                    LoRA Rank (r)
                  </label>
                </Tooltip>
                <select
                  id="lora_r"
                  name="lora_r"
                  value={formState.lora_r || 16}
                  onChange={handleInputChange}
                  className="bg-gray-900 border border-gray-700 rounded-lg p-3 w-full text-white focus:border-orange-500 focus:outline-none" >
                  <option value="4">4</option>
                  <option value="8">8</option>
                  <option value="16">16</option>
                  <option value="32">32</option>
                  <option value="64">64</option>
                </select>
              </div>
              <div>
                <Tooltip id="lora_alpha">
                  <label htmlFor="lora_alpha" className="block text-sm font-medium text-gray-400 mb-1">
                    LoRA Alpha
                  </label>
                </Tooltip>
                <input
                  type="number"
                  id="lora_alpha"
                  name="lora_alpha"
                  min="1"
                  value={formState.lora_alpha || 32}
                  onChange={handleInputChange}
                  className="bg-gray-900 border border-gray-700 rounded-lg p-3 w-full text-white focus:border-orange-500 focus:outline-none"
                />
              </div>
              </div>
            </div>

            {/* Quantization Settings */}
            <div className="bg-gray-800 rounded-lg p-6">
              <h3 className="text-lg font-medium text-white mb-4">Quantization Settings</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <Tooltip id="quantization">
                    <label className="block text-sm font-medium text-gray-400 mb-2">Precision</label>
                  </Tooltip>
                  <div className="space-y-3">
                    <label className="flex items-center">
                      <input
                        type="radio"
                        name="quantization"
                        value="4bit"
                        checked={formState.quantization === '4bit'}
                        onChange={() => handleQuantizationChange('4bit')}
                        className="rounded-full bg-gray-900 border-gray-700 text-orange-500 focus:ring-orange-500"
                      />
                      <span className="ml-2 text-sm text-gray-400">4-bit Quantization</span>
                    </label>
                    <label className="flex items-center">
                      <input
                        type="radio"
                        name="quantization"
                        value="8bit"
                        checked={formState.quantization === '8bit'}
                        onChange={() => handleQuantizationChange('8bit')}
                        className="rounded-full bg-gray-900 border-gray-700 text-orange-500 focus:ring-orange-500"
                      />
                      <span className="ml-2 text-sm text-gray-400">8-bit Quantization</span>
                    </label>
                    <label className="flex items-center">
                      <input
                        type="radio"
                        name="quantization"
                        value="none"
                        checked={formState.quantization === 'none'}
                        onChange={() => handleQuantizationChange('none')}
                        className="rounded-full bg-gray-900 border-gray-700 text-orange-500 focus:ring-orange-500"
                      />
                      <span className="ml-2 text-sm text-gray-400">No Quantization</span>
                    </label>
                  </div>
                </div>

                <div>
                  <Tooltip id="bnb_4bit_compute_dtype">
                    <label htmlFor="bnb_4bit_compute_dtype" className="block text-sm font-medium text-gray-400 mb-1">
                      Compute Dtype
                    </label>
                  </Tooltip>
                  <select
                    id="bnb_4bit_compute_dtype"
                    name="bnb_4bit_compute_dtype"
                    value={formState.bnb_4bit_compute_dtype || 'float16'}
                    onChange={handleInputChange}
                    className="bg-gray-900 border border-gray-700 rounded-lg p-3 w-full text-white focus:border-orange-500 focus:outline-none"
                  >
                    <option value="float32">float32</option>
                    <option value="bfloat16">bfloat16</option>
                    <option value="float16">float16</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Optimization Settings */}
            <div className="bg-gray-800 rounded-lg p-6">
              <h3 className="text-lg font-medium text-white mb-4">Optimization Settings</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <Tooltip id="optim">
                    <label htmlFor="optim" className="block text-sm font-medium text-gray-400 mb-1">
                      Optimizer
                    </label>
                  </Tooltip>
                  <select
                    id="optim"
                    name="optim"
                    value={formState.optim || 'paged_adamw_32bit'}
                    onChange={handleInputChange}
                    className="bg-gray-900 border border-gray-700 rounded-lg p-3 w-full text-white focus:border-orange-500 focus:outline-none"
                  >
                    <option value="paged_adamw_32bit">Paged AdamW 32bit</option>
                    <option value="adamw_torch">AdamW Torch</option>
                    <option value="adamw_bnb_8bit">AdamW 8-bit</option>
                  </select>
                </div>
                <div>
                  <Tooltip id="lr_scheduler_type">
                    <label htmlFor="lr_scheduler_type" className="block text-sm font-medium text-gray-400 mb-1">
                      Learning Rate Scheduler
                    </label>
                  </Tooltip>
                  <select
                    id="lr_scheduler_type"
                    name="lr_scheduler_type"
                    value={formState.lr_scheduler_type || 'cosine'}
                    onChange={handleInputChange}
                    className="bg-gray-900 border border-gray-700 rounded-lg p-3 w-full text-white focus:border-orange-500 focus:outline-none"
                  >
                    <option value="cosine">Cosine</option>
                    <option value="linear">Linear</option>
                    <option value="constant">Constant</option>
                  </select>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Submit Button */}
        <div className="text-center">
          <button
            type="submit"
            className="bg-orange-500 hover:bg-orange-600 px-6 py-3 rounded-lg font-medium transition inline-flex items-center"
          >
            Start Finetuning
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-5 w-5 ml-2"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M13 10V3L4 14h7v7l9-11h-7z"
              />
            </svg>
          </button>
        </div>
      </form>
    </div>
  );
};

export default FinetuneSettings;
