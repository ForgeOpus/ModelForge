import logging
import re
from typing import Dict, Any
from huggingface_hub import model_info, HfApi
from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError


class ModelValidator:
    """
    Simple validator for HuggingFace model repositories.
    Performs basic checks without complex memory or compatibility analysis.
    """
    
    def __init__(self):
        """Initialize the ModelValidator with HuggingFace API client."""
        self.hf_api = HfApi()
        
    def validate_repo_name(self, repo_name: str) -> Dict[str, Any]:
        """
        Validate a HuggingFace repository name format and existence.
        
        Args:
            repo_name: Repository name (e.g., "meta-llama/Llama-3.2-1B" or "gpt2")
            
        Returns:
            Dictionary with validation results
        """
        result = {
            "valid": False,
            "exists": False,
            "accessible": False,
            "repo_name": repo_name,
            "model_info": None,
            "error": None,
            "warnings": []
        }
        
        try:
            # Basic format validation
            if not self._is_valid_repo_format(repo_name):
                result["error"] = "Invalid repository name format. Use 'organization/model-name' or 'model-name'"
                return result
            
            # Check if model exists and is accessible
            exists_result = self.check_model_exists(repo_name)
            result.update(exists_result)
            
            if result["exists"] and result["accessible"]:
                # Get basic model info
                info_result = self.get_basic_model_info(repo_name)
                result["model_info"] = info_result
                result["valid"] = True
                
        except Exception as e:
            result["error"] = f"Validation failed: {str(e)}"
            logging.error(f"Model validation error for {repo_name}: {e}")
            
        return result
    
    def check_model_exists(self, repo_name: str) -> Dict[str, Any]:
        """
        Check if a model exists on HuggingFace Hub.
        
        Args:
            repo_name: Repository name to check
            
        Returns:
            Dictionary with existence and accessibility info
        """
        result = {
            "exists": False,
            "accessible": False,
            "error": None,
            "is_gated": False
        }
        
        try:
            # Try to get model info
            info = model_info(repo_name)
            result["exists"] = True
            result["accessible"] = True
            result["is_gated"] = getattr(info, 'gated', False)
            
        except RepositoryNotFoundError:
            result["error"] = "Model not found on HuggingFace Hub"
            
        except GatedRepoError:
            result["exists"] = True
            result["accessible"] = False
            result["is_gated"] = True
            result["error"] = "Model exists but requires access approval"
            
        except Exception as e:
            result["error"] = f"Unable to check model: {str(e)}"
            logging.error(f"Error checking model {repo_name}: {e}")
            
        return result
    
    def get_basic_model_info(self, repo_name: str) -> Dict[str, Any]:
        """
        Get basic information about a model.
        
        Args:
            repo_name: Repository name
            
        Returns:
            Dictionary with basic model information
        """
        info = {
            "architecture": None,
            "model_type": None,
            "library": None,
            "tags": [],
            "pipeline_tag": None
        }
        
        try:
            model_data = model_info(repo_name)
            
            # Extract basic info from model metadata
            if hasattr(model_data, 'config') and model_data.config:
                config = model_data.config
                info["model_type"] = config.get("model_type")
                info["architecture"] = config.get("architectures", [None])[0] if config.get("architectures") else None
                
            if hasattr(model_data, 'library_name'):
                info["library"] = model_data.library_name
                
            if hasattr(model_data, 'tags'):
                info["tags"] = model_data.tags or []
                
            if hasattr(model_data, 'pipeline_tag'):
                info["pipeline_tag"] = model_data.pipeline_tag
                
        except Exception as e:
            logging.warning(f"Could not get detailed info for {repo_name}: {e}")
            
        return info
    
    def _is_valid_repo_format(self, repo_name: str) -> bool:
        """
        Check if repository name follows valid HuggingFace format.
        
        Args:
            repo_name: Repository name to validate
            
        Returns:
            True if format is valid
        """
        if not repo_name or not isinstance(repo_name, str):
            return False
            
        # Remove any leading/trailing whitespace
        repo_name = repo_name.strip()
        
        # Empty after stripping
        if not repo_name:
            return False
            
        # Check for valid characters and structure
        # Allow: letters, numbers, hyphens, underscores, dots, and one forward slash
        pattern = r'^[a-zA-Z0-9]([a-zA-Z0-9._-]*[a-zA-Z0-9])?(/[a-zA-Z0-9]([a-zA-Z0-9._-]*[a-zA-Z0-9])?)?$'
        
        return bool(re.match(pattern, repo_name))
    
    def get_default_warnings(self) -> list:
        """
        Get default warnings for custom model usage.
        
        Returns:
            List of warning messages
        """
        return [
            "Custom model compatibility not guaranteed",
            "Memory and performance not estimated", 
            "User responsible for hardware limitations",
            "Check model license before use"
        ] 