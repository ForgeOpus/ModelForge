import React from 'react';
import { useState, useEffect } from 'react';

const ListModelsPage = () => {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await fetch('/api/models/all');
        if (!response.ok) {
          throw new Error('Failed to fetch models');
        }
        const data = await response.json();
        setModels(data.models);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchModels();
  }, []);

  if (loading) {
    return <div className="text-center text-gray-400">Loading models...</div>;
  }

  if (error) {
    return <div className="text-center text-red-500">Error: {error}</div>;
  }

  return (
    <div className="p-6">
      <h3 className="text-xl font-medium mb-4">Fine-Tuned Models</h3>
      {models.length === 0 ? (
        <div className="text-gray-400">No models found.</div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {models.map((model) => (
            <div
              key={model.model_id}
              className="p-5 bg-gray-800 rounded-lg border border-gray-700 hover:border-orange-500 transition"
            >
              <h4 className="text-lg font-medium text-orange-500 mb-2">{model.model_name}</h4>
              <p className="text-gray-400 text-sm mb-1">
                <span className="font-medium text-white">Task:</span> {model.pipeline_task}
              </p>
              <p className="text-gray-400 text-sm mb-1">
                <span className="font-medium text-white">Compute Specs:</span> {model.compute_specs}
              </p>
              <p className="text-gray-400 text-sm">
                <span className="font-medium text-white">Model Path:</span> {model.model_path}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default ListModelsPage;