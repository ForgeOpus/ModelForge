import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { config } from '../services/api';

const HardwareDetection = ({ currentSettings, updateSettings }) => {
  const navigate = useNavigate();
  const [showResults, setShowResults] = useState(false);
  const [selectedTask, setSelectedTask] = useState(currentSettings.task || '');
  const [selectedModel, setSelectedModel] = useState(currentSettings.model_name || '');
  const [hardwareData, setHardwareData] = useState(null);
  const [stateUpdated, setStateUpdated] = useState(false);
  
  // Custom model states
  const [useCustomModel, setUseCustomModel] = useState(false);
  const [customRepoName, setCustomRepoName] = useState('');
  const [validationStatus, setValidationStatus] = useState(null); // null, 'validating', 'valid', 'invalid'
  const [validationResult, setValidationResult] = useState(null);
  const [acceptRisks, setAcceptRisks] = useState(false);

  useEffect(() => {
    setShowResults(false);
  }, [selectedTask]);
  // Synchronize local state with props
  useEffect(() => {
    console.log("Current settings from props:", currentSettings);
    setSelectedTask(currentSettings.task || '');
    setSelectedModel(currentSettings.model_name || '');
  }, [currentSettings]);

  // Reset custom model validation when switching between modes
  useEffect(() => {
    if (!useCustomModel) {
      setCustomRepoName('');
      setValidationStatus(null);
      setValidationResult(null);
      setAcceptRisks(false);
    }
  }, [useCustomModel]);

  // Custom model validation function
  const validateCustomModel = async () => {
    if (!customRepoName.trim()) {
      setValidationStatus('invalid');
      setValidationResult({ error: 'Repository name cannot be empty' });
      return;
    }

    setValidationStatus('validating');
    try {
      const response = await fetch(`${config.baseURL}/finetune/validate_custom_model`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ repo_name: customRepoName.trim() }),
      });

      const result = await response.json();
      
      if (response.ok && result.validation.valid) {
        setValidationStatus('valid');
        setValidationResult(result.validation);
      } else {
        setValidationStatus('invalid');
        setValidationResult(result.validation || { error: 'Validation failed' });
      }
    } catch (error) {
      console.error('Validation error:', error);
      setValidationStatus('invalid');
      setValidationResult({ error: 'Unable to validate model. Check your connection.' });
    }
  };

  const tasks = [
    {
      id: 'text-generation',
      title: 'Text Generation',
      description: 'Create models for generating coherent and relevant answers to queries based on old knowledge, like Customer Support Bots'
    },
    {
      id: 'summarization',
      title: 'Summarization',
      description: 'Train models that can condense long documents into concise summaries.'
    },
    {
      id: 'extractive-question-answering',
      title: 'Extractive Question Answering',
      description: 'Build models that extract relevant answers from large texts based on user queries, like for RAG.'
    },
  ];

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    try {
        console.log('Selected Task:', selectedTask);
        const resp = await fetch(`${config.baseURL}/finetune/detect`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ task: selectedTask, provider: currentSettings.provider })
        });

        const mockResponse = await resp.json();
        if (!resp.ok){
            throw new Error('Failed to fetch hardware data');
        }

        setHardwareData(mockResponse);
        setShowResults(true);
      
      // Save both the task and hardware config in the same update
      const updatedSettings = {
        task: selectedTask,
        hardware_config: {
          gpu: mockResponse.gpu_name,
          ram: mockResponse.ram_total_gb,
          disk: mockResponse.available_diskspace_gb,
          cpu_cores: mockResponse.cpu_cores,
        }
      };
      
      console.log('Updating both task and hardware config:', updatedSettings);
      
      updateSettings((prevSettings) => {
        console.log('Previous Settings in handleSubmit:', prevSettings);
        const newSettings = {
          ...prevSettings,
          ...updatedSettings
        };
        console.log('New Settings in handleSubmit:', newSettings);
        return newSettings;
      });
    } catch (error) {
      console.error("Error:", error.message);
    }
  };

  const handleConfigureTraining = async () => {
    console.log('Selected Task before update:', selectedTask);
    console.log('Selected Model before update:', selectedModel);
    console.log('Use Custom Model:', useCustomModel);
    
    let modelName = selectedModel;
    
    // Handle custom model case
    if (useCustomModel) {
      if (validationStatus !== 'valid' || !acceptRisks) {
        alert('Please validate your custom model and accept the risks before proceeding.');
        return;
      }
      
      try {
        // Set the custom model on the backend
        const response = await fetch(`${config.baseURL}/finetune/set_custom_model`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ repo_name: customRepoName.trim() }),
        });
        
        if (!response.ok) {
          throw new Error('Failed to set custom model');
        }
        
        modelName = customRepoName.trim();
      } catch (error) {
        console.error('Error setting custom model:', error);
        alert('Failed to set custom model. Please try again.');
        return;
      }
    }
    
    // Direct update (not using a function) to ensure immediate update
    const updatedSettings = {
      task: selectedTask,
      model_name: modelName,
      is_custom_model: useCustomModel
    };
    
    console.log('Updating task settings (direct update):', updatedSettings);
    
    // Update parent component state using direct object instead of function
    updateSettings(updatedSettings);
    
    // Set state updated flag and navigate
    setStateUpdated(true);
    setTimeout(() => {
      navigate('/finetune/load_settings');
    }, 1000);
  };

  return (
    <div className="max-w-4xl mx-auto p-4">
      <div className="text-center mb-10">
        <h1 className="text-3xl md:text-4xl font-bold mb-4">Hardware Detection</h1>
        <p className="text-lg text-gray-300 mb-6">
          Let's check your system to find the optimal configuration for LLM finetuning
        </p>
      </div>

      {stateUpdated && (
        <div className="bg-green-700 text-white p-4 rounded-lg mb-6 flex items-center justify-between">
          <div>Settings updated successfully! Redirecting to configuration page...</div>
          <div className="animate-spin h-5 w-5 border-2 border-white rounded-full border-t-transparent"></div>
        </div>
      )}

      <div className="bg-gray-800 rounded-xl p-6 md:p-8 shadow-lg mb-12">
        <h2 className="text-2xl font-semibold mb-6 flex items-center">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-6 w-6 mr-2 text-orange-500"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"
            />
          </svg>
          System Hardware Configuration
        </h2>

        <form onSubmit={handleSubmit} className="mb-8">
          <h3 className="text-xl font-medium mb-4">Select Task</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
  {tasks.map((task) => (
    <div 
      key={task.id}
      onClick={() => setSelectedTask(task.id)}
      className={`
        task-card p-5 rounded-lg cursor-pointer transition-all
        ${selectedTask === task.id 
          ? 'bg-gray-800 border-2 border-orange-500' 
          : 'bg-gray-800 border border-gray-700'}
        hover:border-orange-500 hover:bg-gray-700
      `}
    >
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-lg font-medium text-orange-500">{task.title}</h4>
        {selectedTask === task.id && (
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-orange-500" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
          </svg>
        )}
      </div>
      <p className="text-gray-400 text-sm">{task.description}</p>
    </div>
  ))}
</div>

          <div className="mt-6 flex flex-col md:flex-row md:items-center md:justify-between">
            <button
              type="submit"
              className="bg-orange-500 hover:bg-orange-600 px-6 py-3 rounded-lg text-white font-medium transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
              disabled={!selectedTask}
            >
              Detect Hardware
            </button>
          </div>
        </form>

        {showResults && hardwareData && (
          <div className="border-t border-gray-700 pt-6 mt-6">
            <h3 className="text-xl font-medium mb-4">Hardware Detection Results</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-400 mb-1">System Profile</label>
                  <div className="p-3 bg-gray-900 rounded-lg text-white">{hardwareData.profile}</div>
                </div>
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-400 mb-1">GPU</label>
                  <div className="p-3 bg-gray-900 rounded-lg text-white">{hardwareData.gpu_name}</div>
                </div>
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-400 mb-1">GPU Memory</label>
                  <div className="p-3 bg-gray-900 rounded-lg text-white">{hardwareData.gpu_total_memory_gb} GB</div>
                </div>
              </div>

              <div>
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-400 mb-1">System RAM</label>
                  <div className="p-3 bg-gray-900 rounded-lg text-white">{hardwareData.ram_total_gb} GB</div>
                </div>
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-400 mb-1">Available Disk Space</label>
                  <div className="p-3 bg-gray-900 rounded-lg text-white">{hardwareData.available_diskspace_gb} GB</div>
                </div>
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-400 mb-1">CPU Cores</label>
                  <div className="p-3 bg-gray-900 rounded-lg text-white">{hardwareData.cpu_cores}</div>
                </div>
              </div>
            </div>

            <div className="mt-4">
              <label className="block text-sm font-medium text-gray-400 mb-1">Recommended Model</label>
              <div className="p-3 bg-gray-900 rounded-lg font-medium text-orange-500">
                {hardwareData.model_recommendation}
              </div>
            </div>

            <div className="mt-6">
              <label className="block text-sm font-medium text-gray-400 mb-3">
                Model Selection
              </label>
              
              {/* Model selection radio buttons */}
              <div className="mb-4">
                <div className="flex flex-col space-y-2">
                  <label className="flex items-center">
                    <input
                      type="radio"
                      name="modelType"
                      checked={!useCustomModel}
                      onChange={() => setUseCustomModel(false)}
                      className="mr-3 text-orange-500 focus:ring-orange-500"
                    />
                    <span className="text-white">Use Recommended Models</span>
                  </label>
                  <label className="flex items-center">
                    <input
                      type="radio"
                      name="modelType"
                      checked={useCustomModel}
                      onChange={() => setUseCustomModel(true)}
                      className="mr-3 text-orange-500 focus:ring-orange-500"
                    />
                    <span className="text-white">Use Custom Model</span>
                  </label>
                </div>
              </div>

              {/* Recommended models selector */}
              {!useCustomModel && (
                <div>
                  <label htmlFor="model-select" className="block text-sm font-medium text-gray-400 mb-1">
                    Select Your Model
                  </label>
                  <select
                    id="model-select"
                    value={selectedModel}
                    onChange={async (e) => {
                      console.log('Setting model to:', e.target.value);
                      setSelectedModel(e.target.value);
                      await fetch(`${config.baseURL}/finetune/set_model`, {
                        method: 'POST',
                        headers: {
                          'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                          selected_model: e.target.value,
                        }),
                      });
                    }}
                    className="bg-gray-900 border border-gray-700 rounded-lg p-3 w-full text-white focus:border-orange-500 focus:outline-none"
                  >
                    <option value="">Select a model</option>
                    {hardwareData.possible_options?.map((model) => (
                      <option key={model} value={model}>
                        {model}
                      </option>
                    ))}
                  </select>
                </div>
              )}

              {/* Custom model input */}
              {useCustomModel && (
                <div className="space-y-4">
                  <div>
                    <label htmlFor="custom-repo" className="block text-sm font-medium text-gray-400 mb-1">
                      HuggingFace Repository Name
                    </label>
                    <div className="flex space-x-2">
                      <input
                        id="custom-repo"
                        type="text"
                        value={customRepoName}
                        onChange={(e) => {
                          setCustomRepoName(e.target.value);
                          // Reset validation when user types
                          if (validationStatus) {
                            setValidationStatus(null);
                            setValidationResult(null);
                          }
                        }}
                        placeholder="meta-llama/Llama-3.2-1B"
                        className="bg-gray-900 border border-gray-700 rounded-lg p-3 flex-1 text-white focus:border-orange-500 focus:outline-none"
                      />
                      <button
                        onClick={validateCustomModel}
                        disabled={!customRepoName.trim() || validationStatus === 'validating'}
                        className="bg-orange-500 hover:bg-orange-600 disabled:bg-gray-600 disabled:cursor-not-allowed px-4 py-3 rounded-lg text-white font-medium transition flex items-center justify-center min-w-[100px]"
                      >
                        {validationStatus === 'validating' ? (
                          <div className="animate-spin h-4 w-4 border-2 border-white rounded-full border-t-transparent"></div>
                        ) : (
                          'Validate'
                        )}
                      </button>
                    </div>
                  </div>

                  {/* Validation status */}
                  {validationStatus && (
                    <div className={`p-3 rounded-lg ${
                      validationStatus === 'valid' ? 'bg-green-800 border border-green-600' :
                      validationStatus === 'invalid' ? 'bg-red-800 border border-red-600' :
                      'bg-gray-800 border border-gray-600'
                    }`}>
                      {validationStatus === 'valid' && (
                        <div className="flex items-center text-green-300">
                          <svg className="h-5 w-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                          </svg>
                          <span className="font-medium">Model validated successfully!</span>
                        </div>
                      )}
                      {validationStatus === 'invalid' && (
                        <div className="flex items-center text-red-300">
                          <svg className="h-5 w-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                          </svg>
                          <span className="font-medium">
                            {validationResult?.error || 'Validation failed'}
                          </span>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Warning box for custom models */}
                  <div className="bg-yellow-600 border border-yellow-500 rounded-lg p-4">
                    <h4 className="font-bold mb-2 flex items-center text-yellow-100">
                      <svg className="h-5 w-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                      </svg>
                      Custom Model Warning
                    </h4>
                    <ul className="text-sm space-y-1 text-yellow-100 mb-3">
                      <li>• You are responsible for ensuring the model fits your hardware</li>
                      <li>• Fine-tuning may fail if the model is incompatible</li>
                      <li>• Memory usage and training time are not estimated</li>
                      <li>• Check the model's license before use</li>
                    </ul>
                    <label className="flex items-center">
                      <input 
                        type="checkbox" 
                        checked={acceptRisks}
                        onChange={(e) => setAcceptRisks(e.target.checked)}
                        className="mr-2" 
                      />
                      <span className="text-sm text-yellow-100">I understand and accept these risks</span>
                    </label>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      <div className="text-center mb-8">
        <a
          href="/"
          className="text-orange-500 hover:text-orange-400 mr-6 inline-flex items-center"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-5 w-5 mr-2"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M11 17l-5-5m0 0l5-5m-5 5h12"
            />
          </svg>
          Back to Home
        </a>
        <button
          onClick={handleConfigureTraining}
          className="bg-orange-500 hover:bg-orange-600 px-6 py-3 rounded-lg font-medium transition inline-flex items-center disabled:opacity-50 disabled:cursor-not-allowed"
          disabled={
            !showResults || 
            (useCustomModel ? (validationStatus !== 'valid' || !acceptRisks) : !selectedModel)
          }
        >
          Configure Training
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
              d="M13 7l5 5m0 0l-5 5m5-5H6"
            />
          </svg>
        </button>
      </div>
    </div>
  );
};

export default HardwareDetection;
