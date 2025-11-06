import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { config } from '../services/api';

const ProviderSelectionPage = ({ currentSettings, updateSettings }) => {
  const navigate = useNavigate();
  const [providers, setProviders] = useState([]);
  const [selectedProvider, setSelectedProvider] = useState('huggingface');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Load provider from settings if already selected
    if (currentSettings?.provider) {
      setSelectedProvider(currentSettings.provider);
    }
    
    // Fetch available providers from backend
    fetchProviders();
  }, [currentSettings]);

  const fetchProviders = async () => {
    try {
      // Fetch available providers from backend
      // We can get this from a lightweight endpoint or define it locally
      // For now, we'll check if they're available by making a test request
      
      const baseProviders = [
        {
          name: 'huggingface',
          displayName: 'HuggingFace',
          available: true,
          description: 'Industry-standard fine-tuning with comprehensive task support',
          benefits: [
            'Supports all task types (text-generation, summarization, Q&A)',
            'Full control over quantization (4-bit and 8-bit)',
            'Extensive hyperparameter customization',
            'Battle-tested and widely adopted',
            'Compatible with all HuggingFace models'
          ],
          recommended: true,
          logo: '🤗'
        },
        {
          name: 'unsloth',
          displayName: 'Unsloth AI',
          available: false, // Will be updated from backend
          description: 'Optimized fine-tuning with 2x faster training and reduced memory usage',
          benefits: [
            '2x faster training speed',
            '60% less memory consumption',
            'Optimized for text-generation tasks',
            'Advanced 4-bit quantization',
            'Efficient gradient checkpointing',
            'Best for resource-constrained environments'
          ],
          recommended: false,
          logo: '⚡',
          badge: 'OPTIMIZED'
        }
      ];
      
      // Try to fetch from backend to check availability
      try {
        // Make a dummy hardware detection to get provider info
        const response = await fetch(`${config.baseURL}/finetune/detect`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ task: 'text-generation' })
        });
        
        if (response.ok) {
          const data = await response.json();
          if (data.available_providers && Array.isArray(data.available_providers)) {
            // Update availability based on backend response
            baseProviders.forEach(provider => {
              provider.available = data.available_providers.includes(provider.name);
            });
          }
        }
      } catch (backendError) {
        console.log('Could not fetch provider availability from backend, using defaults');
      }
      
      setProviders(baseProviders);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching providers:', error);
      setLoading(false);
    }
  };

  const handleProviderSelect = (providerName) => {
    setSelectedProvider(providerName);
  };

  const handleContinue = () => {
    // Update settings with selected provider
    updateSettings({
      ...currentSettings,
      provider: selectedProvider
    });
    
    // Navigate to model selection/hardware detection
    navigate('/finetune/detect');
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-400">Loading providers...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-white mb-4">
            Choose Your Fine-Tuning Provider
          </h1>
          <p className="text-xl text-gray-400 max-w-3xl mx-auto">
            Select the provider that best fits your needs. Each provider offers unique optimizations and capabilities.
          </p>
        </div>

        {/* Provider Cards */}
        <div className="grid md:grid-cols-2 gap-8 mb-12">
          {providers.map((provider) => (
            <div
              key={provider.name}
              onClick={() => provider.available && handleProviderSelect(provider.name)}
              className={`
                relative rounded-2xl p-8 cursor-pointer transition-all duration-300 transform
                ${selectedProvider === provider.name
                  ? 'bg-gradient-to-br from-blue-600 to-blue-800 shadow-2xl scale-105 ring-4 ring-blue-400'
                  : 'bg-gray-800 hover:bg-gray-750 shadow-lg hover:shadow-xl hover:scale-102'
                }
                ${!provider.available && 'opacity-50 cursor-not-allowed'}
              `}
            >
              {/* Badge */}
              {provider.badge && (
                <div className="absolute top-4 right-4 bg-yellow-500 text-gray-900 px-3 py-1 rounded-full text-xs font-bold">
                  {provider.badge}
                </div>
              )}
              
              {/* Recommended Badge */}
              {provider.recommended && (
                <div className="absolute top-4 right-4 bg-green-500 text-white px-3 py-1 rounded-full text-xs font-bold">
                  RECOMMENDED
                </div>
              )}

              {/* Logo and Title */}
              <div className="flex items-center mb-6">
                <div className="text-6xl mr-4">{provider.logo}</div>
                <div>
                  <h2 className="text-3xl font-bold text-white mb-1">
                    {provider.displayName}
                  </h2>
                  {!provider.available && (
                    <span className="text-sm text-red-400 font-semibold">
                      Not Installed
                    </span>
                  )}
                </div>
              </div>

              {/* Description */}
              <p className="text-gray-300 mb-6 text-lg">
                {provider.description}
              </p>

              {/* Benefits List */}
              <div className="space-y-3">
                <h3 className="text-white font-semibold text-sm uppercase tracking-wide mb-3">
                  Key Benefits
                </h3>
                {provider.benefits.map((benefit, idx) => (
                  <div key={idx} className="flex items-start">
                    <svg
                      className={`w-5 h-5 mr-3 flex-shrink-0 mt-0.5 ${
                        selectedProvider === provider.name ? 'text-white' : 'text-blue-400'
                      }`}
                      fill="currentColor"
                      viewBox="0 0 20 20"
                    >
                      <path
                        fillRule="evenodd"
                        d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                        clipRule="evenodd"
                      />
                    </svg>
                    <span className={`text-sm ${
                      selectedProvider === provider.name ? 'text-white' : 'text-gray-300'
                    }`}>
                      {benefit}
                    </span>
                  </div>
                ))}
              </div>

              {/* Selection Indicator */}
              {selectedProvider === provider.name && (
                <div className="mt-6 flex items-center justify-center">
                  <div className="bg-white text-blue-600 px-4 py-2 rounded-lg font-bold text-sm">
                    ✓ Selected
                  </div>
                </div>
              )}

              {/* Installation Note */}
              {!provider.available && (
                <div className="mt-6 p-4 bg-gray-900 rounded-lg">
                  <p className="text-sm text-gray-400">
                    To use this provider, install it with:
                    <code className="block mt-2 bg-black p-2 rounded text-green-400 text-xs">
                      pip install modelforge-finetuning[{provider.name}]
                    </code>
                  </p>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Continue Button */}
        <div className="flex justify-center">
          <button
            onClick={handleContinue}
            disabled={!selectedProvider}
            className={`
              px-8 py-4 rounded-xl font-bold text-lg transition-all duration-300 transform
              ${selectedProvider
                ? 'bg-blue-600 hover:bg-blue-700 text-white shadow-lg hover:shadow-xl hover:scale-105'
                : 'bg-gray-700 text-gray-500 cursor-not-allowed'
              }
            `}
          >
            Continue with {providers.find(p => p.name === selectedProvider)?.displayName}
            <svg
              className="inline-block w-5 h-5 ml-2"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
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

        {/* Info Box */}
        <div className="mt-12 max-w-4xl mx-auto bg-gray-800 rounded-xl p-6 border border-gray-700">
          <div className="flex items-start">
            <svg
              className="w-6 h-6 text-blue-400 mr-3 flex-shrink-0 mt-1"
              fill="currentColor"
              viewBox="0 0 20 20"
            >
              <path
                fillRule="evenodd"
                d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
                clipRule="evenodd"
              />
            </svg>
            <div>
              <h3 className="text-white font-semibold mb-2">Need Help Choosing?</h3>
              <ul className="text-gray-400 text-sm space-y-1">
                <li>• Choose <strong className="text-white">HuggingFace</strong> for maximum compatibility and all task types</li>
                <li>• Choose <strong className="text-white">Unsloth AI</strong> for faster text-generation with limited GPU memory</li>
                <li>• You can always change providers later by starting a new fine-tuning session</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProviderSelectionPage;
