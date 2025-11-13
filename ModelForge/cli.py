"""
Refactored CLI for ModelForge.
Clean entry point with improved structure.
"""
import sys
import subprocess
from huggingface_hub import login, whoami
from .logging_config import logger


def check_huggingface_login():
    """
    Check if user is logged into HuggingFace.
    If not, prompt for login.
    """
    try:
        whoami()
        logger.info("HuggingFace authentication verified")
        return True
    except Exception:
        logger.warning("Not logged into HuggingFace")
        print("\n" + "=" * 80)
        print("HuggingFace Login Required")
        print("=" * 80)
        print("\nTo use ModelForge, you need to be logged into HuggingFace.")
        print("This allows you to access and fine-tune models from the HuggingFace Hub.")
        print("\nOptions:")
        print("1. Login now (will prompt for token)")
        print("2. Exit and login manually with: huggingface-cli login")
        print("=" * 80 + "\n")

        choice = input("Enter your choice (1 or 2): ").strip()

        if choice == "1":
            try:
                login()
                logger.info("HuggingFace login successful")
                return True
            except Exception as e:
                logger.error(f"Login failed: {e}")
                print(f"\nLogin failed: {e}")
                print("Please try again or login manually with: huggingface-cli login")
                return False
        else:
            print("\nPlease login with: huggingface-cli login")
            print("Then run ModelForge again.")
            return False


def main():
    """
    Main entry point for ModelForge CLI.
    """
    print("\n" + "=" * 80)
    print("  __  __           _      _ _____                     ")
    print(" |  \\/  |         | |    | |  ___|                    ")
    print(" | \\  / | ___   __| | ___| | |_ ___  _ __ __ _  ___  ")
    print(" | |\\/| |/ _ \\ / _` |/ _ \\ |  _/ _ \\| '__/ _` |/ _ \\ ")
    print(" | |  | | (_) | (_| |  __/ | || (_) | | | (_| |  __/ ")
    print(" |_|  |_|\\___/ \\__,_|\\___|_|_| \\___/|_|  \\__, |\\___| ")
    print("                                          __/ |      ")
    print("                                         |___/       ")
    print("\n ModelForge v2.0 - Modular Fine-Tuning Platform")
    print("=" * 80 + "\n")

    # Check HuggingFace login
    if not check_huggingface_login():
        sys.exit(1)

    # Import and run app
    try:
        import uvicorn
        from .app import app

        logger.info("Starting ModelForge server...")
        print("\nStarting ModelForge server...")
        print("Server will be available at: http://localhost:8000")
        print("API documentation: http://localhost:8000/docs")
        print("Press Ctrl+C to stop\n")

        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8000,
            log_level="info",
        )

    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        print("\n\nServer stopped. Goodbye!")

    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        print(f"\nError starting server: {e}")
        print("Please check the logs for more details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
