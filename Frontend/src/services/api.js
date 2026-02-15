export const config = {
    baseURL: '/api',
    headers: {
        'Content-Type': 'application/json',
    },
}

// API Service Functions

/**
 * Get system information including available providers and strategies
 * @returns {Promise<Object>} System info with providers and strategies
 */
export const getSystemInfo = async () => {
    try {
        const response = await fetch(`${config.baseURL}/info`, {
            method: 'GET',
            headers: config.headers,
        });

        if (!response.ok) {
            throw new Error(`Failed to fetch system info: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('Error fetching system info:', error);
        throw error;
    }
};

/**
 * Validate a local dataset file path
 * @param {string} datasetPath - Local file path to the dataset
 * @returns {Promise<Object>} Validation response with file_path
 */
export const validateDatasetPath = async (datasetPath) => {
    try {
        const response = await fetch(`${config.baseURL}/finetune/validate_dataset_path`, {
            method: 'POST',
            headers: config.headers,
            body: JSON.stringify({ dataset_path: datasetPath }),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `Validation failed: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('Error validating dataset path:', error);
        throw error;
    }
};

/**
 * Start model training with configuration
 * @param {Object} config - Complete training configuration
 * @returns {Promise<Object>} Training start response
 */
export const startTraining = async (trainingConfig) => {
    try {
        const response = await fetch(`${config.baseURL}/finetune/start_training`, {
            method: 'POST',
            headers: config.headers,
            body: JSON.stringify(trainingConfig),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `Training start failed: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('Error starting training:', error);
        throw error;
    }
};

/**
 * Get training status
 * @returns {Promise<Object>} Training status
 */
export const getTrainingStatus = async () => {
    try {
        const response = await fetch(`${config.baseURL}/finetune/status`, {
            method: 'GET',
            headers: config.headers,
        });

        if (!response.ok) {
            throw new Error(`Failed to fetch training status: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('Error fetching training status:', error);
        throw error;
    }
};

/**
 * Get hardware specifications
 * @returns {Promise<Object>} Hardware specs
 */
export const getHardwareSpecs = async () => {
    try {
        const response = await fetch(`${config.baseURL}/finetune/hardware_specs`, {
            method: 'GET',
            headers: config.headers,
        });

        if (!response.ok) {
            throw new Error(`Failed to fetch hardware specs: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('Error fetching hardware specs:', error);
        throw error;
    }
};

/**
 * Get recommended models for a task
 * @param {string} task - Task type
 * @returns {Promise<Object>} Model recommendations
 */
export const getRecommendedModels = async (task) => {
    try {
        const response = await fetch(`${config.baseURL}/finetune/recommended_models/${task}`, {
            method: 'GET',
            headers: config.headers,
        });

        if (!response.ok) {
            throw new Error(`Failed to fetch recommendations: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('Error fetching model recommendations:', error);
        throw error;
    }
};
