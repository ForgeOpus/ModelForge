import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { config } from '../services/api';

const ProviderSelectionPage = ({ currentSettings, updateSettings }) => {
  const navigate = useNavigate();
  const [providers, setProviders] = useState([]);
  const [loadingProviders, setLoadingProviders] = useState(true);
  const [selectedProvider, setSelectedProvider] = useState(currentSettings.provider || 'huggingface');
  const [hoveredProvider, setHoveredProvider] = useState(null);

  useEffect(() => {
    const fetchProviders = async () => {
      try {
        const response = await fetch(`${config.baseURL}/finetune/providers`);
        if (!response.ok) throw new Error('Failed to fetch providers');
        
        const data = await response.json();
        console.log("Fetched providers:", data);
        setProviders(data.providers || []);
        setLoadingProviders(false);
      } catch (err) {
        console.error("Error fetching providers:", err);
        setLoadingProviders(false);
      }
    };

    fetchProviders();
  }, []);

  const handleProviderSelect = (providerName) => {
    setSelectedProvider(providerName);
  };

  const handleContinue = () => {
    // Update settings with selected provider
    if (updateSettings) {
      updateSettings({ provider: selectedProvider });
    }
    
    // Navigate to task selection (hardware detection page)
    navigate('/finetune/detect');
  };

  const providerDetails = {
    huggingface: {
      features: [
        'Standard HuggingFace transformers',
        'PEFT/LoRA fine-tuning',
        '4-bit/8-bit quantization',
        'Maximum compatibility',
        'Established ecosystem'
      ],
      performance: 'Baseline speed & memory',
      bestFor: 'General use, maximum compatibility'
    },
    unsloth: {
      features: [
        'Optimized FastLanguageModel',
        'Enhanced LoRA/QLoRA',
        'Memory-efficient attention',
        '~2x faster training',
        '~30% less memory usage'
      ],
      performance: '2x faster, 30% less memory',
      bestFor: 'Faster training, limited hardware',
      installCommand: 'pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"'
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-black to-gray-900 text-white">
      <div className="container mx-auto px-4 py-8 max-w-6xl">
        {/* Header */}
        <div className="mb-12 text-center">
          <h1 className="text-4xl md:text-5xl font-bold mb-4">
            Choose Your Fine-tuning Provider
          </h1>
          <p className="text-xl text-gray-400">
            Select the backend that best suits your needs
          </p>
        </div>

        {/* Loading State */}
        {loadingProviders && (
          <div className="flex justify-center items-center py-16">
            <div className="animate-spin h-12 w-12 border-4 border-orange-500 border-t-transparent rounded-full"></div>
          </div>
        )}

        {/* Provider Cards */}
        {!loadingProviders && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-12">
            {providers.map((provider) => {
              const isSelected = selectedProvider === provider.name;
              const isAvailable = provider.available;
              const details = providerDetails[provider.name] || {};

              return (
                <div
                  key={provider.name}
                  onClick={() => isAvailable && handleProviderSelect(provider.name)}
                  onMouseEnter={() => setHoveredProvider(provider.name)}
                  onMouseLeave={() => setHoveredProvider(null)}
                  className={`
                    relative rounded-xl p-6 border-2 transition-all duration-300 transform
                    ${isAvailable ? 'cursor-pointer hover:scale-105' : 'cursor-not-allowed opacity-60'}
                    ${isSelected && isAvailable 
                      ? 'border-orange-500 bg-gradient-to-br from-orange-500/20 to-transparent shadow-xl shadow-orange-500/20' 
                      : 'border-gray-700 bg-gray-800/50 hover:border-gray-600'
                    }
                  `}
                >
                  {/* Selected Badge */}
                  {isSelected && isAvailable && (
                    <div className="absolute -top-3 -right-3 bg-orange-500 text-white px-4 py-1 rounded-full text-sm font-bold">
                      Selected
                    </div>
                  )}

                  {/* Not Installed Badge */}
                  {!isAvailable && (
                    <div className="absolute -top-3 -right-3 bg-yellow-600 text-white px-4 py-1 rounded-full text-sm font-bold">
                      Not Installed
                    </div>
                  )}

                  {/* Provider Header */}
                  <div className="flex items-center mb-4">
                    <div className={`
                      w-12 h-12 rounded-lg flex items-center justify-center mr-4
                      ${isAvailable ? 'bg-orange-500/20' : 'bg-gray-700/20'}
                    `}>
                      <svg className={`w-6 h-6 ${isAvailable ? 'text-orange-500' : 'text-gray-500'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                      </svg>
                    </div>
                    <div>
                      <h3 className="text-2xl font-bold capitalize">{provider.name}</h3>
                      <p className="text-sm text-gray-400">{provider.description}</p>
                    </div>
                  </div>

                  {/* Performance Badge */}
                  {details.performance && (
                    <div className="mb-4 inline-block bg-gray-900 px-3 py-1 rounded-full text-sm text-orange-400 border border-orange-500/30">
                      ⚡ {details.performance}
                    </div>
                  )}

                  {/* Features List */}
                  {details.features && (
                    <div className="mb-4">
                      <h4 className="text-sm font-semibold text-gray-400 mb-2">Features:</h4>
                      <ul className="space-y-2">
                        {details.features.map((feature, idx) => (
                          <li key={idx} className="flex items-start text-sm">
                            <svg className="w-5 h-5 mr-2 text-orange-500 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                            </svg>
                            <span className="text-gray-300">{feature}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Best For */}
                  {details.bestFor && (
                    <div className="mt-4 pt-4 border-t border-gray-700">
                      <p className="text-sm text-gray-400">
                        <span className="font-semibold">Best for:</span> {details.bestFor}
                      </p>
                    </div>
                  )}

                  {/* Installation Instructions */}
                  {!isAvailable && details.installCommand && hoveredProvider === provider.name && (
                    <div className="mt-4 p-3 bg-gray-900 border border-yellow-600 rounded-lg">
                      <p className="text-xs text-yellow-400 font-semibold mb-2">Installation Required:</p>
                      <code className="block bg-gray-800 p-2 rounded text-xs text-white overflow-x-auto">
                        {details.installCommand}
                      </code>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {/* Comparison Table */}
        <div className="bg-gray-800/50 rounded-xl p-6 mb-8">
          <h3 className="text-xl font-bold mb-4 text-center">Provider Comparison</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-left">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="pb-3 text-gray-400 font-medium">Feature</th>
                  <th className="pb-3 text-center text-gray-400 font-medium">HuggingFace</th>
                  <th className="pb-3 text-center text-gray-400 font-medium">Unsloth</th>
                </tr>
              </thead>
              <tbody className="text-sm">
                <tr className="border-b border-gray-700/50">
                  <td className="py-3 text-gray-300">Training Speed</td>
                  <td className="py-3 text-center">1x (baseline)</td>
                  <td className="py-3 text-center text-orange-400 font-semibold">~2x faster</td>
                </tr>
                <tr className="border-b border-gray-700/50">
                  <td className="py-3 text-gray-300">Memory Usage</td>
                  <td className="py-3 text-center">1x (baseline)</td>
                  <td className="py-3 text-center text-orange-400 font-semibold">~30% less</td>
                </tr>
                <tr className="border-b border-gray-700/50">
                  <td className="py-3 text-gray-300">Compatibility</td>
                  <td className="py-3 text-center text-green-400">Excellent</td>
                  <td className="py-3 text-center text-yellow-400">Good</td>
                </tr>
                <tr>
                  <td className="py-3 text-gray-300">Installation</td>
                  <td className="py-3 text-center text-green-400">Pre-installed</td>
                  <td className="py-3 text-center text-yellow-400">Optional</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Continue Button */}
        <div className="text-center">
          <button
            onClick={handleContinue}
            disabled={!selectedProvider || loadingProviders}
            className={`
              px-8 py-4 rounded-lg text-lg font-medium transition inline-flex items-center
              ${selectedProvider && !loadingProviders
                ? 'bg-orange-500 hover:bg-orange-600 text-white shadow-lg hover:shadow-orange-500/30 transform hover:-translate-y-1'
                : 'bg-gray-600 text-gray-400 cursor-not-allowed'
              }
            `}
          >
            Continue with {selectedProvider || 'Selected Provider'}
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

        {/* Info Footer */}
        <div className="mt-8 text-center text-sm text-gray-500">
          <p>You can change the provider later in the advanced settings</p>
        </div>
      </div>
    </div>
  );
};

export default ProviderSelectionPage;
