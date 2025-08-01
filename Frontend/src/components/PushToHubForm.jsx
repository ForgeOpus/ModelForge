import React, { useState } from "react";
import { config } from "../services/api";
import ErrorDialog from "./ErrorDialog";

// Large, prominent, centered modal for status/success
const StatusModal = ({ isOpen, message, onClose, isSuccess = false }) => {
  if (!isOpen) return null;

  // Extract HuggingFace URL from message if present
  const urlRegex = /(https:\/\/huggingface\.co\/[^\s]+)/g;
  const urls = message ? message.match(urlRegex) : [];

  // Split message by URLs to render text and links separately
  const renderMessage = () => {
    if (!urls || urls.length === 0) {
      return <p className="text-orange-300 text-xl mb-8 text-center">{message}</p>;
    }

    let parts = message.split(urlRegex);
    return (
      <div className="text-orange-300 text-xl mb-8 text-center">
        {parts.map((part, index) => {
          if (urls.includes(part)) {
            return (
              <a
                key={index}
                href={part}
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-400 underline hover:text-blue-300 break-all"
              >
                {part}
              </a>
            );
          }
          return <span key={index}>{part}</span>;
        })}
      </div>
    );
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center z-50 p-4">
      <div className="bg-gray-900 rounded-3xl shadow-2xl w-full max-w-2xl p-10 flex flex-col items-center border-4 border-orange-500">
        {isSuccess && <h2 className="text-3xl font-bold text-orange-400 mb-6 text-center">Success!</h2>}
        {renderMessage()}
        {onClose && (
          <button
            onClick={onClose}
            className="px-8 py-3 bg-orange-500 text-white rounded-xl hover:bg-orange-600 transition-colors font-bold text-lg shadow-lg"
          >
            Close
          </button>
        )}
      </div>
    </div>
  );
};

const PushToHubForm = ({ model, onClose, onPush }) => {
  const [repoName, setRepoName] = useState("");
  const [isPrivate, setIsPrivate] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [validationError, setValidationError] = useState("");
  const [showErrorDialog, setShowErrorDialog] = useState(false);
  const [errorDetails, setErrorDetails] = useState(null);
  const [showSuccessModal, setShowSuccessModal] = useState(false);
  const [successMessage, setSuccessMessage] = useState("");

  // Validate repo name format
  const validateRepoName = (name) => {
    const pattern = /^([A-Za-z_0-9-]+\/[A-Za-z_0-9-]+)$/;
    return pattern.test(name);
  };

  const handleRepoNameChange = (e) => {
    const value = e.target.value;
    setRepoName(value);

    if (value && !validateRepoName(value)) {
      setValidationError("Repository name must be in format 'username/repo-name' (letters, numbers, hyphens and underscores only)");
    } else {
      setValidationError("");
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!repoName.trim()) {
      setError("Repository name is required");
      setShowErrorDialog(true);
      return;
    }

    if (!validateRepoName(repoName)) {
      setError("Invalid repository name format");
      setShowErrorDialog(true);
      return;
    }

    setLoading(true);
    setError("");

    try {
      const response = await fetch(config.baseURL + "/hub/push", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          repo_name: repoName,
          model_path: model.model_path,
          private: isPrivate,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        setLoading(false); // Stop loading immediately on error
        const errorMessage = data.error || "Failed to push model to hub";
        setError(errorMessage);
        setErrorDetails(data.details || `HTTP ${response.status}: ${response.statusText}`);
        setShowErrorDialog(true);
        return;
      }

      // Success case - show success modal after a brief delay
      setTimeout(() => {
        setLoading(false);
        setSuccessMessage(data.message || "Model pushed to Hub successfully!");
        setShowSuccessModal(true);
      }, 2000);

      onPush && onPush(data.message);
    } catch (err) {
      setLoading(false); // Stop loading immediately on network error
      setError("Network error or unexpected failure");
      setErrorDetails(err.message);
      setShowErrorDialog(true);
    }
  };

  const handleCloseErrorDialog = () => {
    setShowErrorDialog(false);
    setError("");
    setErrorDetails(null);
  };

  return (
    <>
      {/* Loading modal */}
      <StatusModal
        isOpen={loading}
        message="Pushing to Hub. This might take a while."
      />

      {/* Success modal */}
      <StatusModal
        isOpen={showSuccessModal}
        message={successMessage}
        isSuccess={true}
        onClose={() => {
          setShowSuccessModal(false);
          onClose();
        }}
      />

      {/* Main form modal */}
      {!loading && !showSuccessModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-gray-900 rounded-2xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6 border-b border-gray-700">
              <h2 className="text-2xl font-bold text-orange-400">Push Model to HuggingFace Hub</h2>
              <button
                onClick={onClose}
                className="absolute top-4 right-4 text-gray-400 hover:text-white transition-colors"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            <form onSubmit={handleSubmit} className="p-6 space-y-6">
              {/* Repository Name */}
              <div>
                <label className="block text-orange-400 font-semibold mb-2">
                  Repository Name *
                </label>
                <input
                  type="text"
                  value={repoName}
                  onChange={handleRepoNameChange}
                  placeholder="username/model-name or model-name"
                  className="w-full px-4 py-3 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-orange-500 focus:ring-1 focus:ring-orange-500"
                  required
                />
                {validationError && (
                  <p className="text-red-400 text-sm mt-1">{validationError}</p>
                )}
                <p className="text-gray-400 text-sm mt-1">
                  Format: 'username/repo-name' or 'repo-name' (letters and underscores only)
                </p>
              </div>

              {/* Privacy Setting */}
              <div>
                <label className="flex items-center space-x-3">
                  <input
                    type="checkbox"
                    checked={isPrivate}
                    onChange={(e) => setIsPrivate(e.target.checked)}
                    className="w-5 h-5 text-orange-500 bg-gray-800 border-gray-600 rounded focus:ring-orange-500 focus:ring-2"
                  />
                  <span className="text-orange-400 font-semibold">Make repository private</span>
                </label>
                <p className="text-gray-400 text-sm mt-1 ml-8">
                  {isPrivate ? "Repository will be private" : "Repository will be public"}
                </p>
              </div>

              {/* Read-only Model Information */}
              <div className="space-y-4 border-t border-gray-700 pt-6">
                <h3 className="text-lg font-semibold text-orange-400">Model Information</h3>

                <div>
                  <label className="block text-gray-300 font-medium mb-1">Model Name</label>
                  <div className="px-4 py-3 bg-gray-800 border border-gray-600 rounded-lg text-gray-200">
                    {model.model_name}
                  </div>
                </div>

                <div>
                  <label className="block text-gray-300 font-medium mb-1">Base Model</label>
                  <div className="px-4 py-3 bg-gray-800 border border-gray-600 rounded-lg text-gray-200 break-all">
                    {model.base_model}
                  </div>
                </div>

                <div>
                  <label className="block text-gray-300 font-medium mb-1">Task Type</label>
                  <div className="px-4 py-3 bg-gray-800 border border-gray-600 rounded-lg text-gray-200">
                    {model.task}
                  </div>
                </div>

                <div>
                  <label className="block text-gray-300 font-medium mb-1">Local Model Path</label>
                  <div className="px-4 py-3 bg-gray-800 border border-gray-600 rounded-lg text-gray-200 break-all text-sm">
                    {model.model_path}
                  </div>
                </div>

                <div>
                  <label className="block text-gray-300 font-medium mb-1">Description</label>
                  <div className="px-4 py-3 bg-gray-800 border border-gray-600 rounded-lg text-gray-200">
                    {model.description || "No description provided"}
                  </div>
                </div>
              </div>

              {/* Error Display */}
              {error && (
                <div className="text-red-400 bg-red-900/20 border border-red-700 rounded-lg p-3">
                  {error}
                </div>
              )}

              {/* Action Buttons */}
              <div className="flex gap-4 pt-4">
                <button
                  type="button"
                  onClick={onClose}
                  className="flex-1 px-6 py-3 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition-colors font-medium"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={loading || !!validationError || !repoName.trim()}
                  className="flex-1 px-6 py-3 bg-orange-500 text-white rounded-lg hover:bg-orange-600 transition-colors font-medium disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
                >
                  {loading ? (
                    <>
                      <svg className="animate-spin h-5 w-5 mr-2" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"></path>
                      </svg>
                      Pushing...
                    </>
                  ) : (
                    "Push to Hub"
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      <ErrorDialog
        isOpen={showErrorDialog}
        onClose={handleCloseErrorDialog}
        title="Push to Hub Error"
        message={error}
        details={errorDetails}
      />
    </>
  );
};

export default PushToHubForm;
