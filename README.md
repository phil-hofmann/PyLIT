# Python Laplace Inverse Transformation
`Pylit`

This software provides  a Python implementation of the inverse Laplace Transformation.

## 📜 License

The project is licensed under the [MIT License](LICENSE.txt).

## 💬 Citations

- Inverse Laplace transform in quantum many-body theory. [Alexander Benedix Robles](a.benedix-robles@hzdr.de), [Phil-Alexander Hofmann](mailto:philhofmann@outlook.com), [Tobias Dornheim](t.dornheim@hzdr.de), [Michael Hecht](m.hecht@hzdr.de)

## 👥 Team and Support

- [Phil-Alexander Hofmann](https://github.com/philippocalippo/)
- [Alexander Benedix Robles](https://github.com/alexanderbenedix/)

## 🙏 Acknowledgments

We would like to acknowledge:
- [Dr. Tobias Dornheim](https://www.casus.science/de-de/team-members/dr-tobias-dornheim/) - [CASUS](https://www.casus.science/) ([HZDR](https://www.hzdr.de/)),
- [Prof. Dr. Michael Hecht](https://www.casus.science/de-de/team-members/michael-hecht/) - [CASUS](https://www.casus.science/) ([HZDR](https://www.hzdr.de/)),

and the support and resources provided by the [Center for Advanced Systems Understanding](https://www.casus.science/) ([Helmholtz-Zentrum Dreden-Rossendorf](https://www.hzdr.de/)) where the development of this project took place.

## 📝 Remarks

This project originated from a prototype developed by Alexander Benedix Robles. His initial work laid the foundation for the current implementation.

## 💻 Installation Guide
Please follow the steps below:

1. Clone the project:
```bash
git clone https://github.com/phil-hofmann/pylit.git
```

2. Open Anaconda prompt.

3. Create a virtual environment:
```bash
conda env create -f environment.yml
```

4. Activate environment:
```bash
conda activate pylit
```

5. Install using pip:
```bash
pip install -e .
```

6. If you want to deactivate the environment:
```bash
conda deactivate
```

## 📚 Documentation Guide
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

## 🚀 Usage Guide

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
