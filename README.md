This is the repo for the course to run stuff locally without any
crap software (jupiter, anacoda, etc.)

Setup.

1) Install python3:

> pacman -S python

Or

> apt install python3

or whatever...

2) Create virtual environment for that project:

> mkdir ~/dat257x_venv
> python3 -m venv ~/dat257x_venv

3) Source it:

> source ~/dat257x_venv/bin/activate

You can create alias for that in .bashrc

4) Install dependencies:

> pip install -r requirements.txt

5) Test your setup:

> cd LabFiles
> python3 00_rooms_random_agent.py

FAQ.

I don't see any plots. What do I do?

Install Tkinter (your python is probably built without it):

apt install python3-tk
