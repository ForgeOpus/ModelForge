import React, { useEffect, useState } from "react";
import { config } from "../services/api";
import PushToHubForm from "../components/PushToHubForm";
import ErrorDialog from "../components/ErrorDialog";

const ListAllModels = () => {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedModel, setSelectedModel] = useState(null);
  const [showPushForm, setShowPushForm] = useState(false);
  const [showErrorDialog, setShowErrorDialog] = useState(false);
  const [errorDetails, setErrorDetails] = useState(null);

  useEffect(() => {
    fetch(config.baseURL + "/models/")
      .then((res) => {
        if (!res.ok) {
          throw new Error(`Failed to fetch models: HTTP ${res.status} ${res.statusText}`);
        }
        return res.json();
      })
      .then((data) => {
        setModels(data.models || []);
        setLoading(false);
      })
      .catch((err) => {
        setError("Failed to load models");
        setErrorDetails(err.message);
        setShowErrorDialog(true);
        setLoading(false);
      });
  }, []);

  const handlePushSuccess = (message) => {
    setShowPushForm(false);
    setSelectedModel(null);
  };

  const handleCloseForm = () => {
    setShowPushForm(false);
    setSelectedModel(null);
  };

  const handleCloseErrorDialog = () => {
    setShowErrorDialog(false);
    setError(null);
    setErrorDetails(null);
  };

  const handleOpenPlayground = async (modelPath) => {
    try {
      const response = await fetch(config.baseURL + "/playground/new", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ model_path: modelPath }),
      });

      if (!response.ok) {
        const data = await response.json();
        setError("Failed to open playground");
        setErrorDetails(data.error || `HTTP ${response.status}: ${response.statusText}`);
        setShowErrorDialog(true);
      }
    } catch (err) {
      setError("Network error while opening playground");
      setErrorDetails(err.message);
      setShowErrorDialog(true);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-black to-gray-900 py-12 px-2">
      <div className="max-w-6xl mx-auto bg-gray-900/80 rounded-2xl shadow-2xl p-8">
        <h1 className="text-3xl md:text-4xl font-bold text-orange-400 mb-8 text-center tracking-tight">
          All Models
        </h1>

        {loading ? (
          <div className="flex justify-center items-center min-h-[200px] text-orange-400 text-lg">
            <svg className="animate-spin h-6 w-6 mr-2 text-orange-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"></path>
            </svg>
            Loading...
          </div>
        ) : (
          <>
            {/* TABLE VIEW */}
            <div className="hidden md:block overflow-x-auto rounded-xl shadow-lg">
              <table className="min-w-full bg-gray-800 rounded-xl overflow-hidden">
                <thead>
                  <tr>
                    <th className="px-4 py-3 bg-orange-500/20 text-orange-400 font-semibold text-left">ID</th>
                    <th className="px-4 py-3 bg-orange-500/20 text-orange-400 font-semibold text-left">Model Name</th>
                    <th className="px-4 py-3 bg-orange-500/20 text-orange-400 font-semibold text-left">Base Model</th>
                    <th className="px-4 py-3 bg-orange-500/20 text-orange-400 font-semibold text-left">Task</th>
                    <th className="px-4 py-3 bg-orange-500/20 text-orange-400 font-semibold text-left">Description</th>
                    <th className="px-4 py-3 bg-orange-500/20 text-orange-400 font-semibold text-left">Created At</th>
                    <th className="px-4 py-3 bg-orange-500/20 text-orange-400 font-semibold text-left">Model Path</th>
                    <th className="px-4 py-3 bg-orange-500/20 text-orange-400 font-semibold text-left">Playground</th>
                    <th className="px-4 py-3 bg-orange-500/20 text-orange-400 font-semibold text-left">Push to Hub</th>
                  </tr>
                </thead>
                <tbody>
                  {models.length === 0 ? (
                    <tr>
                      <td colSpan={9} className="text-center text-gray-400 py-8">
                        No models found.
                      </td>
                    </tr>
                  ) : (
                    models.map((model, idx) => (
                      <tr key={model.id} className={idx % 2 === 0 ? "bg-gray-900/60" : "bg-gray-800/80"}>
                        <td className="px-4 py-3 text-gray-200">{model.id}</td>
                        <td className="px-4 py-3 text-orange-300 break-all">{model.model_name}</td>
                        <td className="px-4 py-3 text-gray-300 break-all">{model.base_model}</td>
                        <td className="px-4 py-3 text-gray-300">{model.task}</td>
                        <td className="px-4 py-3 text-gray-400">{model.description}</td>
                        <td className="px-4 py-3 text-gray-400">{new Date(model.creation_date).toLocaleString()}</td>
                        <td className="px-4 py-3 text-gray-400 break-all max-w-xs">{model.model_path}</td>
                        <td className="px-4 py-3">
                          <button onClick={() => handleOpenPlayground(model.model_path)} className="text-orange-400 underline hover:text-orange-300 transition cursor-pointer">
                            Open
                          </button>
                        </td>
                        <td className="px-4 py-3">
                          <button onClick={() => { setSelectedModel(model); setShowPushForm(true); }} className="bg-orange-500 text-white rounded-md px-4 py-2 hover:bg-orange-400 transition">
                            Push
                          </button>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>

            {/* CARD VIEW */}
            <div className="md:hidden space-y-4">
              {models.length === 0 ? (
                <p className="text-center text-gray-400 py-8">No models found.</p>
              ) : (
                models.map((model) => (
                  <div key={model.id} className="bg-gray-800 rounded-xl p-4 shadow-md">
                    <div className="text-orange-300 font-semibold text-lg break-all">{model.model_name}</div>
                    <div className="text-gray-400 text-sm mb-2">{model.description}</div>
                    <div className="text-sm text-gray-300">
                      <div><span className="font-medium text-orange-400">Base:</span> {model.base_model}</div>
                      <div><span className="font-medium text-orange-400">Task:</span> {model.task}</div>
                      <div><span className="font-medium text-orange-400">Created:</span> {new Date(model.creation_date).toLocaleString()}</div>
                      <div className="break-all"><span className="font-medium text-orange-400">Path:</span> {model.model_path}</div>
                    </div>
                    <div className="mt-4 flex flex-wrap gap-2">
                      <button onClick={() => handleOpenPlayground(model.model_path)} className="text-sm bg-transparent text-orange-400 underline hover:text-orange-300">
                        Open Playground
                      </button>
                      <button onClick={() => { setSelectedModel(model); setShowPushForm(true); }} className="text-sm bg-orange-500 text-white rounded-md px-3 py-1 hover:bg-orange-400">
                        Push to Hub
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </>
        )}
      </div>

      {showPushForm && (
        <PushToHubForm model={selectedModel} onClose={handleCloseForm} onPush={handlePushSuccess} />
      )}


      <ErrorDialog
        isOpen={showErrorDialog}
        onClose={handleCloseErrorDialog}
        title="Application Error"
        message={error}
        details={errorDetails}
      />
    </div>
  );
};

export default ListAllModels;
