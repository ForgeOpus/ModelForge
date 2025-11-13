import logging
from typing import Any, Dict, List, Tuple
from .config_manager import ConfigurationManager


class ModelRecommendationEngine:
    """
    Engine for recommending models based on hardware profiles and tasks.
    Uses performance-optimized strategy to select the best model for given hardware.
    """
    
    def __init__(self, config_manager: ConfigurationManager = None):
        """
        Initialize the model recommendation engine.
        
        Args:
            config_manager: ConfigurationManager instance. If None, creates a new one.
        """
        try:
            self.config_manager = config_manager if config_manager else ConfigurationManager()
            logging.info("ModelRecommendationEngine initialized successfully")
        except Exception as e:
            error_msg = f"Failed to initialize ModelRecommendationEngine: {str(e)}"
            logging.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def get_recommendation(self, hardware_profile: str, task: str) -> Tuple[str, List[str]]:
        """
        Get model recommendation for given hardware profile and task.
        
        Args:
            hardware_profile: Hardware profile (low_end, mid_range, high_end)
            task: Task name (text-generation, summarization, etc.)
            
        Returns:
            Tuple of (primary_model, alternative_models)
            
        Raises:
            ValueError: If profile or task is not supported
            RuntimeError: If recommendation fails
        """
        try:
            # Validate inputs
            self._validate_profile(hardware_profile)
            self._validate_task(task)
            
            logging.info(f"Getting recommendation for profile: {hardware_profile}, task: {task}")
            
            # Get model configuration for this profile and task
            model_config = self.config_manager.get_models_for_profile_and_task(hardware_profile, task)
            
            # Performance-optimized strategy: return primary model and alternatives
            primary_model = model_config.get("primary")
            alternative_models = model_config.get("alternatives", [])
            
            if not primary_model:
                raise ValueError(f"No primary model found for profile '{hardware_profile}' and task '{task}'")
            
            logging.info(f"Recommendation: primary='{primary_model}', alternatives={alternative_models}")
            
            return primary_model, alternative_models
            
        except ValueError:
            # Re-raise ValueError as is
            raise
        except Exception as e:
            error_msg = f"Failed to get recommendation for profile '{hardware_profile}' and task '{task}': {str(e)}"
            logging.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def get_recommendation_with_custom_model(self, custom_model: str, hardware_profile: str, task: str) -> Tuple[str, List[str]]:
        """
        Get recommendation when using a custom model, keeping alternatives as fallbacks.
        
        Args:
            custom_model: Custom model repository name
            hardware_profile: Hardware profile (low_end, mid_range, high_end)
            task: Task name (text-generation, summarization, etc.)
            
        Returns:
            Tuple of (custom_model, alternative_models_from_profile)
            
        Raises:
            ValueError: If profile or task is not supported
        """
        try:
            # Validate inputs
            self._validate_profile(hardware_profile)
            self._validate_task(task)
            
            logging.info(f"Getting custom model recommendation: {custom_model} for profile: {hardware_profile}, task: {task}")
            
            # Get recommended alternatives for fallback
            try:
                _, alternative_models = self.get_recommendation(hardware_profile, task)
            except Exception as e:
                logging.warning(f"Could not get alternatives for custom model: {e}")
                alternative_models = []
            
            logging.info(f"Custom model: {custom_model}, fallback alternatives: {alternative_models}")
            
            return custom_model, alternative_models
            
        except ValueError:
            # Re-raise ValueError as is
            raise
        except Exception as e:
            error_msg = f"Failed to get custom model recommendation for '{custom_model}': {str(e)}"
            logging.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def get_all_recommendations_for_profile(self, hardware_profile: str) -> Dict[str, Dict[str, Any]]:
        """
        Get all model recommendations for a given hardware profile across all tasks.
        
        Args:
            hardware_profile: Hardware profile (low_end, mid_range, high_end)
            
        Returns:
            Dictionary mapping task names to their model configurations
            
        Raises:
            ValueError: If profile is not supported
            RuntimeError: If retrieval fails
        """
        try:
            self._validate_profile(hardware_profile)
            
            logging.info(f"Getting all recommendations for profile: {hardware_profile}")
            
            profiles = self.config_manager.get_model_profiles()
            profile_config = profiles.get(hardware_profile, {})
            
            if not profile_config:
                raise ValueError(f"No model configurations found for profile '{hardware_profile}'")
            
            return profile_config
            
        except ValueError:
            # Re-raise ValueError as is
            raise
        except Exception as e:
            error_msg = f"Failed to get all recommendations for profile '{hardware_profile}': {str(e)}"
            logging.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def get_supported_tasks_for_profile(self, hardware_profile: str) -> List[str]:
        """
        Get list of supported tasks for a given hardware profile.
        
        Args:
            hardware_profile: Hardware profile (low_end, mid_range, high_end)
            
        Returns:
            List of supported task names
            
        Raises:
            ValueError: If profile is not supported
            RuntimeError: If retrieval fails
        """
        try:
            self._validate_profile(hardware_profile)
            
            profile_config = self.get_all_recommendations_for_profile(hardware_profile)
            supported_tasks = list(profile_config.keys())
            
            logging.info(f"Supported tasks for profile '{hardware_profile}': {supported_tasks}")
            
            return supported_tasks
            
        except ValueError:
            # Re-raise ValueError as is
            raise
        except Exception as e:
            error_msg = f"Failed to get supported tasks for profile '{hardware_profile}': {str(e)}"
            logging.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _validate_profile(self, hardware_profile: str) -> None:
        """
        Validate that the hardware profile is supported.
        
        Args:
            hardware_profile: Hardware profile to validate
            
        Raises:
            ValueError: If profile is not supported
        """
        if not hardware_profile or not isinstance(hardware_profile, str):
            raise ValueError(f"Invalid hardware profile: {hardware_profile}")
        
        profiles = self.config_manager.get_model_profiles()
        if hardware_profile not in profiles:
            supported_profiles = list(profiles.keys())
            raise ValueError(f"Unsupported hardware profile '{hardware_profile}'. Supported profiles: {supported_profiles}")
    
    def _validate_task(self, task: str) -> None:
        """
        Validate that the task is supported.
        
        Args:
            task: Task name to validate
            
        Raises:
            ValueError: If task is not supported
        """
        if not task or not isinstance(task, str):
            raise ValueError(f"Invalid task: {task}")
        
        if not self.config_manager.is_task_supported(task):
            supported_tasks = self.config_manager.get_task_names()
            raise ValueError(f"Unsupported task '{task}'. Supported tasks: {supported_tasks}")
    
    def get_recommendation_with_fallback(self, hardware_profile: str, task: str) -> Tuple[str, List[str]]:
        """
        Get model recommendation with fallback to lower-tier profiles if needed.
        
        Args:
            hardware_profile: Hardware profile (low_end, mid_range, high_end)
            task: Task name (text-generation, summarization, etc.)
            
        Returns:
            Tuple of (primary_model, alternative_models)
            
        Raises:
            RuntimeError: If no recommendation can be found even with fallback
        """
        fallback_order = ["high_end", "mid_range", "low_end"]
        
        # Start with the requested profile
        profiles_to_try = [hardware_profile]
        
        # Add fallback profiles if different from requested
        for fallback_profile in fallback_order:
            if fallback_profile != hardware_profile:
                profiles_to_try.append(fallback_profile)
        
        last_error = None
        
        for profile in profiles_to_try:
            try:
                result = self.get_recommendation(profile, task)
                if profile != hardware_profile:
                    logging.warning(f"Used fallback profile '{profile}' for original request '{hardware_profile}'")
                return result
            except Exception as e:
                last_error = e
                logging.warning(f"Failed to get recommendation for profile '{profile}': {str(e)}")
        
        # If all profiles failed, raise the last error
        error_msg = f"Failed to get recommendation for task '{task}' with any profile. Last error: {str(last_error)}"
        logging.error(error_msg)
        raise RuntimeError(error_msg) from last_error 