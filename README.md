# Python Laplace Inverse Transformation

`Pylit`

This software provides a Python implementation of the inverse Laplace Transformation.

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

### 💻 Installation Guide

1. **Clone the repository**

```bash
git clone https://github.com/phil-hofmann/pylit.git
cd pylit
```

2. **Make the shell scripts executable**

```bash
chmod +x app_setup.sh app_start.sh
```

3. **Run the setup**

```bash
./setup.sh
```

## 🚀 Usage Guide

Follow these steps to get started with your workspace and run your experiments.

1. **Create a workspace folder**
   Create a dedicated workspace folder on your local file system for your projects.

2. **Start Pylit**

   ```bash
   ./app_start.sh
   ```

3. **Configure settings**

   - Navigate to 'settings'
   - Change the workspace variable to point to your created workspace directory
   - Save the settings and refresh the page

4. **Create your first experiment**

   - Navigate to 'experiment'
   - Begin by typing a name and hit enter

5. **Finalize Experiment Setup**

   - Once you've finished setting up your experiment, save your experiment
   - Afterwards, go back to your terminal

Update:

6. **Navigate to your workspace**

   Open a new tab and navigate to your workspace directory and activate the 'venv' again

   ```bash
   source venv/bin/activate
   ```

7. **Reactivate the virtual environment**

   - Ensure the virtual environment is active ..

8. **Run the experiment script**

   - Execute your experiment script by running:
     ```bash
     python <exp_name>/run.py
     ```

9. **View Results**
   - Wait for the experiment to complete and view your results. Voilà!

Happy experimenting! 🎉
