import os
import sys

# configuration
REPO_URL = "https://github.com/fbourgey/fre-gy-7773-mlfe"
REPO_NAME = "fre-gy-7773-mlfe"

# check if we are running in Google Colab
IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    print("Running in Colab. Setting up environment...")

    # clone if the folder doesn't exist
    if not os.path.exists(REPO_NAME):
        os.system(f"git clone {REPO_URL}")

    # navigate into the repo
    os.chdir(REPO_NAME)

    print(f"Current directory: {os.getcwd()}")
else:
    print("Running locally. Skipping git clone.")
