import React, { useEffect, useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import '../styles/Loading.css';
import { config } from '../services/api';

const Loading = () => {
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState(0);
  const [isIdle, setIsIdle] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState('running'); // 'running', 'completed', 'error', 'idle'
  const [statusMessage, setStatusMessage] = useState('');
  const [backendProgress, setBackendProgress] = useState(null);
  const statusIntervalRef = useRef(null);
  const navigate = useNavigate();
  
  const steps = [
    "Initializing model...",
    "Loading dataset...",
    "Configuring training parameters...",
    "Preparing LoRA adapters...",
    "Starting training loop...",
    "Training in progress..."
  ];
  
  // Function to manually set idle status (for demo purposes)
  const setTrainingComplete = () => {
    setIsIdle(true);
    setProgress(100);
    setCurrentStep(steps.length - 1);
    if (statusIntervalRef.current) {
      clearInterval(statusIntervalRef.current);
    }
  };

  // Function to check backend status
  useEffect(() => {
    const checkStatus = async () => {
      try {
        const responseGet = await fetch(`${config.baseURL}/finetune/status`, {
          method: 'GET',
        });
        const result = await responseGet.json();
        
        console.log('Status API response:', result);
        
        // Update training status
        setTrainingStatus(result.status || 'running');
        setStatusMessage(result.message || '');
        
        // Check if training completed successfully
        if (result.status === 'completed') {
          console.log('Training completed successfully!');
          setIsIdle(true);
          setProgress(100); // Force progress to 100%
          setCurrentStep(steps.length - 1); // Set to final step
          if (statusIntervalRef.current) {
            clearInterval(statusIntervalRef.current);
          }
          return; // Exit early to avoid conflicting updates
        }
        
        // Check if training encountered an error
        if (result.status === 'error') {
          console.error('Training failed:', result.message);
          setProgress(0);
          if (statusIntervalRef.current) {
            clearInterval(statusIntervalRef.current);
          }
          return;
        }
        
        // Only update progress if training is running and progress is provided
        if (result.status === 'running' && result.progress !== undefined) {
          console.log('Backend progress:', result.progress);
          setBackendProgress(Number(result.progress));
        }
      } catch (error) {
        console.error('Error checking status:', error);
      }
    };

    // Initial check
    checkStatus();
    
    // Setup interval for checking status - reduce to 5 seconds for more responsive updates
    statusIntervalRef.current = setInterval(checkStatus, 5000);

    return () => {
      if (statusIntervalRef.current) {
        clearInterval(statusIntervalRef.current);
      }
    };
  }, [steps.length]);

  // Block window/tab closure
  useEffect(() => {
    const handleBeforeUnload = (e) => {
      if (!isIdle) {
        e.preventDefault();
        e.returnValue = '';
        return '';
      }
    };
  
    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [isIdle]);

  const handleChatWithAI = async () => {
    try {
      // First, send GET request to get the model path
      const modelPathResponse = await fetch(`${config.baseURL}/playground/model_path`, {
        method: 'GET'
      });
  
      if (!modelPathResponse.ok) {
        throw new Error('Failed to get model path');
      }
  
      // Extract the model path from the response
      const modelPathData = await modelPathResponse.json();
      const modelPath = modelPathData.model_path;
  
      console.log("Received model path:", modelPath);
  
      // Now send POST request with the model path
      const response = await fetch(`${config.baseURL}/playground/new`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model_path: modelPath
        })
      });
  
      // After successful POST request, redirect to homepage
      navigate('/');
      
    } catch (error) {
      console.error('Error starting AI chat:', error);
      // Show an error message to the user
      alert('There was an error starting the AI chat. Please try again.');
    }
  };

  // Simulate progress with more realistic surging and shorter pausing
  useEffect(() => {
    if (isIdle) return; // Don't animate if we're done
    
    // Define realistic progress behavior with stops and surges
    const simulateRealisticProgress = () => {
      // Random surge speed between 0.2 and 0.7
      const baseSpeed = Math.random() * 0.5 + 0.2;
      
      // Define strategic stopping points
      const stopPoints = [32, 67, 88]; // Stop at these percentages
      const stopDurations = [2000, 3000, 3500]; // Shorter pauses (milliseconds)
      
      // Check if we need to stop at a checkpoint
      const isAtStopPoint = (progress) => {
        return stopPoints.some((point, index) => {
          // If we're within 0.5% of a stop point and haven't passed it yet
          return Math.abs(progress - point) < 0.5 && progress < point + 0.5;
        });
      };
      
      // Get the stop duration if we're at a stop point
      const getStopDuration = (progress) => {
        for (let i = 0; i < stopPoints.length; i++) {
          if (Math.abs(progress - stopPoints[i]) < 0.5) {
            return stopDurations[i];
          }
        }
        return 0;
      };
      
      // Surge function that accelerates and decelerates randomly
      const getSurgeSpeed = (progress) => {
        // Random fluctuation in speed
        const fluctuation = Math.sin(Date.now() / 1000) * 0.15 + 
                          Math.cos(Date.now() / 800) * 0.1;
        
        // Slow down as we approach stop points
        const nearestStopPoint = stopPoints.find(point => progress < point);
        if (nearestStopPoint && (nearestStopPoint - progress < 5)) {
          return baseSpeed * 0.5; // Slow down before stopping
        }
        
        // Surge after stop points
        const lastStopPoint = stopPoints.filter(point => progress > point).pop();
        if (lastStopPoint && (progress - lastStopPoint < 3)) {
          return baseSpeed * 1.5; // Surge after stopping
        }
        
        return baseSpeed + fluctuation;
      };
      
      setProgress(oldProgress => {
        // If we have backend progress, use that instead
        if (backendProgress !== null && !isNaN(backendProgress)) {
          // Ensure backendProgress is a valid number between 0-100
          const validProgress = Math.max(0, Math.min(100, Number(backendProgress)));
          
          // If backendProgress is very close to previous value, apply smoothing
          if (Math.abs(validProgress - oldProgress) < 0.5) {
            return oldProgress + 0.1; // Small increment for visual feedback
          }
          
          // If backendProgress jumps backward (shouldn't happen but just in case)
          if (validProgress < oldProgress) {
            return oldProgress; // Don't go backward
          }
          
          // Update step based on progress
          const newStep = Math.floor(validProgress / (100 / steps.length));
          if (newStep !== currentStep && newStep < steps.length) {
            setCurrentStep(newStep);
          }
          
          return validProgress;
        }
        
        // Don't exceed 95% for simulated progress
        if (oldProgress >= 95) {
          return 95;
        }
        
        // Check if we should stop at a checkpoint
        if (isAtStopPoint(oldProgress)) {
          const stopDuration = getStopDuration(oldProgress);
          setTimeout(() => {
            // Force a state update to resume progress
            setProgress(prev => prev + 0.1);
          }, stopDuration);
          
          // Stay at the same progress during stop
          return oldProgress;
        }
        
        // Calculate new progress with surge patterns
        const surgeSpeed = getSurgeSpeed(oldProgress);
        const newProgress = oldProgress + surgeSpeed;
        
        // Update step based on progress
        const newStep = Math.floor(newProgress / (100 / steps.length));
        if (newStep !== currentStep && newStep < steps.length) {
          setCurrentStep(newStep);
        }
        
        return newProgress;
      });
    };
    
    const timer = setInterval(simulateRealisticProgress, 100);

    return () => {
      clearInterval(timer);
    };
  }, [currentStep, steps.length, isIdle, backendProgress]);

  return (
    <div className="loading-container relative">
      {/* Training completion demo button
      {!isIdle && (
        <button 
          onClick={setTrainingComplete}
          className="absolute top-4 right-4 bg-gray-700 hover:bg-gray-600 text-white px-3 py-2 rounded-md text-sm font-medium transition"
        >
          Demo: Complete Training
        </button>
      )} */}
    
      <div className="loading-card">
        {trainingStatus === 'error' ? (
          /* Error view */
          <div className="error-container text-center">
            <div className="mb-6">
              <svg className="mx-auto h-24 w-24 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
            </div>
            <h2 className="text-2xl text-red-500 font-bold mb-4">Training Failed</h2>
            <p className="text-gray-300 mb-2">{statusMessage || 'An error occurred during training.'}</p>
            <p className="text-gray-400 text-sm mb-6">Please check your configuration and try again.</p>
            <button
              onClick={() => navigate('/finetune/settings')}
              className="bg-orange-500 hover:bg-orange-600 px-6 py-3 rounded-lg text-white font-semibold transition"
            >
              ← Back to Settings
            </button>
          </div>
        ) : !isIdle ? (
          <>
            <h1 className="loading-title">Finetuning Your Model</h1>
            
            {/* Loading animation */}
            <div className="circular-loader-container">
              <div className="orbital-loader">
                <div className="center-core"></div>
                <div className="orbital-ring ring-1"></div>
                <div className="orbital-ring ring-2"></div>
                <div className="orbital-ring ring-3"></div>
                
                <div className="particle particle-1"></div>
                <div className="particle particle-2"></div>
                <div className="particle particle-3"></div>
                <div className="particle particle-4"></div>
                <div className="particle particle-5"></div>
                
                <div className="orbital-trail trail-1"></div>
                <div className="orbital-trail trail-2"></div>
                <div className="orbital-trail trail-3"></div>
              </div>
            </div>
            
            <div className="step-indicator">
              <p className="current-step">{steps[currentStep]}</p>
            </div>
            
            <div className="progress-container">
              <div 
                className="progress-bar" 
                style={{ width: `${progress}%` }}
              ></div>
              <div className="progress-text">{Math.floor(progress)}%</div>
            </div>
          </>
        ) : (
          /* Success view after completion - no title text */
          <div className="success-container">
            {/* New 3D cube animation */}
            <div className="celebration-container mb-6">
              <div className="ai-network-animation">
                <div className="network-core"></div>
                <div className="network-layer layer-1"></div>
                <div className="network-layer layer-2"></div>
                <div className="network-layer layer-3"></div>
                
                <div className="connection connection-1"></div>
                <div className="connection connection-2"></div>
                <div className="connection connection-3"></div>
                <div className="connection connection-4"></div>
                <div className="connection connection-5"></div>
                <div className="connection connection-6"></div>
                
                <div className="data-node node-1"></div>
                <div className="data-node node-2"></div>
                <div className="data-node node-3"></div>
                <div className="data-node node-4"></div>
                <div className="data-node node-5"></div>
                <div className="data-node node-6"></div>
              </div>
            </div>
            
            <div className="text-center mb-8">
              <h2 className="text-3xl text-orange-500 font-bold mb-4 glow-text">BOOM! Your AI is alive! 🎉</h2>
              <p className="text-gray-300 text-lg bouncing-text">
                You just leveled up in the AI world! Your custom model is ready to blow minds!
              </p>
              {statusMessage && (
                <p className="text-gray-400 text-sm mt-2 italic">{statusMessage}</p>
              )}
              <div className="mt-4 inline-block px-4 py-2 bg-green-500/20 border border-green-500 rounded-lg">
                <p className="text-green-400 text-sm font-semibold">✓ Training Completed Successfully</p>
              </div>
            </div>
            
            <div className="flex justify-center">
              <button
                onClick={handleChatWithAI} 
                className="bg-orange-500 hover:bg-orange-600 px-8 py-4 rounded-lg text-white text-lg font-bold transition flex items-center pulse-button"
              >
                <span className="mr-2">🚀</span>
                Let's Chat With Your AI!
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 ml-2 animate-bounce" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                </svg>
              </button>
              
              <style jsx>{`
                .pulse-button {
                  animation: pulse 1.5s infinite;
                  box-shadow: 0 0 0 0 rgba(249, 115, 22, 0.7);
                }
                
                @keyframes pulse {
                  0% {
                    transform: scale(0.95);
                    box-shadow: 0 0 0 0 rgba(249, 115, 22, 0.7);
                  }
                  
                  70% {
                    transform: scale(1);
                    box-shadow: 0 0 0 10px rgba(249, 115, 22, 0);
                  }
                  
                  100% {
                    transform: scale(0.95);
                    box-shadow: 0 0 0 0 rgba(249, 115, 22, 0);
                  }
                }
              `}</style>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// Add CSS for the 3D cube animation
const successAnimationStyles = `
/* Success container */
.celebration-container {
  position: relative;
  height: 200px;
  display: flex;
  justify-content: center;
  align-items: center;
  overflow: hidden;
}

/* AI Network Animation */
.ai-network-animation {
  position: relative;
  width: 230px;
  height: 230px;
  margin: 0 auto;
  perspective: 1000px;
  transform-style: preserve-3d;
}

/* Core and layers */
.network-core {
  position: absolute;
  width: 24px;
  height: 24px;
  background-color: #ff8033;
  border-radius: 50%;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  box-shadow: 0 0 15px rgba(255, 128, 51, 0.7);
  animation: pulse-core 2s infinite ease-in-out;
  z-index: 30;
}

.network-layer {
  position: absolute;
  border: 2px solid rgba(255, 128, 51, 0.5);
  border-radius: 50%;
  top: 50%;
  left: 50%;
  transform-style: preserve-3d;
  animation: rotate-layer 20s infinite linear;
}

.layer-1 {
  width: 80px;
  height: 80px;
  transform: translate(-50%, -50%) rotateX(60deg) rotateY(0deg);
  animation-direction: normal;
}

.layer-2 {
  width: 140px;
  height: 140px;
  transform: translate(-50%, -50%) rotateX(30deg) rotateY(30deg);
  animation-direction: reverse;
  animation-duration: 25s;
}

.layer-3 {
  width: 200px;
  height: 200px;
  transform: translate(-50%, -50%) rotateX(70deg) rotateY(10deg);
  animation-duration: 30s;
}

/* Connection lines */
.connection {
  position: absolute;
  top: 50%;
  left: 50%;
  height: 2px;
  background: linear-gradient(90deg, rgba(255, 128, 51, 0.8), rgba(255, 128, 51, 0));
  transform-origin: left center;
  z-index: 10;
}

.connection-1 {
  width: 100px;
  transform: translate(0, -1px) rotate(30deg);
}

.connection-2 {
  width: 80px;
  transform: translate(0, -1px) rotate(150deg);
}

.connection-3 {
  width: 120px;
  transform: translate(0, -1px) rotate(210deg);
}

.connection-4 {
  width: 90px;
  transform: translate(0, -1px) rotate(270deg);
}

.connection-5 {
  width: 110px;
  transform: translate(0, -1px) rotate(330deg);
}

.connection-6 {
  width: 80px;
  transform: translate(0, -1px) rotate(90deg);
}

/* Data nodes */
.data-node {
  position: absolute;
  width: 10px;
  height: 10px;
  background-color: #ff8033;
  border-radius: 50%;
  box-shadow: 0 0 8px rgba(255, 128, 51, 0.6);
  animation: pulse-node 2s infinite alternate;
}

.node-1 {
  top: calc(50% - 55px);
  left: calc(50% + 65px);
  animation-delay: 0s;
}

.node-2 {
  top: calc(50% + 45px);
  left: calc(50% + 85px);
  animation-delay: 0.4s;
}

.node-3 {
  top: calc(50% + 80px);
  left: calc(50% - 30px);
  animation-delay: 0.8s;
}

.node-4 {
  top: calc(50% + 25px);
  left: calc(50% - 90px);
  animation-delay: 1.2s;
}

.node-5 {
  top: calc(50% - 70px);
  left: calc(50% - 60px);
  animation-delay: 1.6s;
}

.node-6 {
  top: calc(50% - 20px);
  left: calc(50% + 95px);
  animation-delay: 2s;
}

/* Animations */
@keyframes pulse-core {
  0% { transform: translate(-50%, -50%) scale(1); opacity: 1; }
  50% { transform: translate(-50%, -50%) scale(1.2); opacity: 0.8; }
  100% { transform: translate(-50%, -50%) scale(1); opacity: 1; }
}

@keyframes rotate-layer {
  from { transform: translate(-50%, -50%) rotateX(60deg) rotateY(0deg) rotateZ(0deg); }
  to { transform: translate(-50%, -50%) rotateX(60deg) rotateY(0deg) rotateZ(360deg); }
}

@keyframes pulse-node {
  0% { transform: scale(1); opacity: 0.7; }
  100% { transform: scale(1.3); opacity: 1; }
}

.glow-text {
  text-shadow: 0 0 10px rgba(249, 115, 22, 0.7);
  animation: glow 2s ease-in-out infinite alternate;
}

.bouncing-text {
  animation: bounce 0.7s ease infinite alternate;
}

/* Progress bar surging animation */
.progress-bar {
  background: linear-gradient(90deg, #f97316, #ffaa70);
  animation: gradient-shift 3s ease infinite;
  background-size: 200% 200%;
}

@keyframes glow {
  0% { filter: brightness(1); }
  100% { filter: brightness(1.3); }
}

@keyframes bounce {
  from { transform: translateY(0); }
  to { transform: translateY(-5px); }
}

@keyframes gradient-shift {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}
`;

// Add the styles to the document
if (typeof document !== 'undefined') {
  const styleElement = document.createElement('style');
  styleElement.textContent = successAnimationStyles;
  document.head.appendChild(styleElement);
}

export default Loading;