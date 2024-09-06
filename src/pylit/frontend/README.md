### 🏗️ DIY
How to implement [Methods](#), [Models](#) and [Optimizers](#) by yourself.

### 📚 Documentation Guide
Please follow the steps below:

1. Make sure that ``pandoc`` is installed on your local device:
``https://pandoc.org/installing.html``

2. Open Anaconda prompt and navigate to `Pylit`.

3. Clean the build directory:
```bash
make clean
```

4. Build the documentation:
```bash
make html
```

5. Open the index.html file located in the `docs/build/html/` directory.

### 🚀 Usage Guide

Follow these steps to get started with your workspace and run your experiments:

1. **Create a Workspace Folder**
   - Create a dedicated workspace folder on your local file system for your projects.

2. **Navigate to the Frontend Directory**
   - Open your terminal and run:
     ```bash
     cd pylit/src/pylit/frontend/
     ```

3. **Activate the Conda Environment**
   - Activate the environment with the following command:
     ```bash
     conda activate pylit
     ```

4. **Start the Application**
   - Run the Streamlit app:
     ```bash
     streamlit run 👋_hello.py
     ```

5. **Configure Settings**
   - Go to the settings within the app.
   - Change the workspace variable to point to your workspace directory.
   - Save the settings.

6. **Create Your Experiment**
   - Navigate to the "Experiment" section in the app.
   - Start creating your experiment!

7. **Finalize Experiment Setup**
   - Once you've finished setting up your experiment, go back to your terminal.

8. **Navigate to Your Workspace**
   - In the terminal, navigate to your workspace directory.

9. **Reactivate the Conda Environment**
   - Ensure the environment is active by running:
     ```bash
     conda activate pylit
     ```

10. **Run the Experiment Script**
    - Execute your experiment script by running:
      ```bash
      python <exp_name>/run.py
      ```

11. **View Results**
    - Wait for the experiment to complete and view your results. Voilà!

Happy experimenting! 🎉
